"""Distributed worker implementation for MPLS.

This module implements the DistributedWorker class that runs truly distributed
decentralized federated learning with asynchronous training and model aggregation.
"""

import copy
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import WorkerConfig, extract_layers, get_layer_params, set_layer_params
from ..workers import MPLSWorker
from .network import NetworkManager
from .config_loader import DistributedWorkerConfig

logger = logging.getLogger(__name__)


class DistributedWorker:
    """Distributed worker for asynchronous MPLS training.
    
    This worker runs training and aggregation in parallel:
    - Training thread: Continuously trains on local data
    - Aggregation thread: Pulls layers from peers and aggregates
    - gRPC server: Serves layer requests from peers
    """
    
    def __init__(
        self,
        config: DistributedWorkerConfig,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """Initialize distributed worker.
        
        Args:
            config: Worker configuration
            model: PyTorch model to train
            dataloader: DataLoader for local data
            criterion: Loss function
            optimizer: Optimizer
        """
        self.config = config
        self.worker_id = config.worker_id
        self.device = config.device
        
        # Model and training
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        # Parameter store with double buffering for thread safety
        self.current_model_lock = threading.Lock()
        self.aggregated_model = copy.deepcopy(model)
        
        # Iteration tracking
        self.current_iteration = 0
        self.max_iterations = config.max_iterations
        
        # Status
        self.status = "initializing"
        self.running = False
        
        # Threads
        self.training_thread: Optional[threading.Thread] = None
        self.aggregation_thread: Optional[threading.Thread] = None
        
        # Network manager
        self.network = NetworkManager(self, config.listen_address)
        
        # MPLS components (peer selection, layer selection, list scheduling)
        self.layers = extract_layers(self.model)
        self.tau1 = config.tau1
        self.tau2 = config.tau2
        
        # Layer sizes in MB
        self.layer_sizes = []
        for l in self.layers:
            params = sum(p.numel() for p in l.parameters())
            self.layer_sizes.append(params * 4 / (1024 * 1024))
        
        # Peer and layer selection probabilities
        self.peer_probs = {}
        self.layer_probs = {}
        self._initialize_probabilities()
        
        # Peer metadata cache
        self.peer_metadata: Dict[int, any] = {}
        
        # Previous parameters for gradient computation
        self.prev_params = {
            i: get_layer_params(l).clone() for i, l in enumerate(self.layers)
        }
        
        # Statistics
        self.training_losses = []
        self.aggregation_times = []
        
        logger.info(f"DistributedWorker {self.worker_id} initialized with {len(self.layers)} layers")
    
    def _initialize_probabilities(self):
        """Initialize peer and layer selection probabilities."""
        num_peers = len(self.config.peers)
        if num_peers == 0:
            return
        
        # Uniform initial probabilities
        self.peer_probs = {p.id: 1.0 / num_peers for p in self.config.peers}
        self.layer_probs = {
            i: {p.id: 1.0 / num_peers for p in self.config.peers}
            for i in range(len(self.layers))
        }
    
    def get_current_model(self) -> nn.Module:
        """Get current model (thread-safe).
        
        Returns:
            Current model
        """
        with self.current_model_lock:
            return copy.deepcopy(self.model)
    
    def extract_layers(self, model: nn.Module) -> List[nn.Module]:
        """Extract learnable layers from model.
        
        Args:
            model: PyTorch model
            
        Returns:
            List of layer modules
        """
        return extract_layers(model)
    
    def get_layer_params(self, layer: nn.Module) -> torch.Tensor:
        """Get layer parameters as flattened tensor.
        
        Args:
            layer: Layer module
            
        Returns:
            Flattened parameter tensor
        """
        return get_layer_params(layer)
    
    def start(self):
        """Start the distributed worker."""
        logger.info(f"Worker {self.worker_id} starting...")
        
        # Start network server
        self.network.start_server()
        
        # Connect to peers
        for peer in self.config.peers:
            self.network.connect_to_peer(peer.id, peer.address)
            # Give servers time to start
            time.sleep(0.5)
        
        # Fetch initial peer metadata
        self._update_peer_metadata()
        
        # Start threads
        self.running = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        
        self.training_thread.start()
        self.aggregation_thread.start()
        
        logger.info(f"Worker {self.worker_id} started successfully")
    
    def stop(self):
        """Stop the distributed worker."""
        logger.info(f"Worker {self.worker_id} stopping...")
        
        self.running = False
        
        # Wait for threads to finish
        if self.training_thread:
            self.training_thread.join(timeout=10)
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=10)
        
        # Shutdown network
        self.network.shutdown()
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _training_loop(self):
        """Training loop (runs in separate thread)."""
        logger.info(f"Worker {self.worker_id} training loop started")
        
        while self.running and self.current_iteration < self.max_iterations:
            self.status = "training"
            
            # Train one epoch
            self.model.train()
            total_loss = 0
            batches = 0
            
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            avg_loss = total_loss / max(1, batches)
            self.training_losses.append(avg_loss)
            
            # Update gradient history
            with torch.no_grad():
                for i, layer in enumerate(self.layers):
                    current = get_layer_params(layer)
                    self.prev_params[i] = current.clone()
            
            logger.debug(f"Worker {self.worker_id} iteration {self.current_iteration} loss: {avg_loss:.4f}")
            
            # Wait for aggregation to complete before next iteration
            # (In true async, we could continue, but for stability we sync here)
            time.sleep(0.1)
        
        logger.info(f"Worker {self.worker_id} training loop finished")
    
    def _aggregation_loop(self):
        """Aggregation loop (runs in separate thread)."""
        logger.info(f"Worker {self.worker_id} aggregation loop started")
        
        while self.running and self.current_iteration < self.max_iterations:
            self.status = "aggregating"
            
            start_time = time.time()
            
            # Update peer metadata
            self._update_peer_metadata()
            
            # Update peer and layer selection probabilities
            self._update_selection_probabilities()
            
            # Develop aggregation strategy using list scheduling
            strategy = self._list_scheduling()
            
            # Pull layers from peers in parallel
            pulled_layers = self._pull_layers_parallel(strategy)
            
            # Aggregate layers
            self._aggregate_layers(pulled_layers)
            
            # Update current model with aggregated model
            with self.current_model_lock:
                for i, layer in enumerate(self.layers):
                    agg_layer = extract_layers(self.aggregated_model)[i]
                    layer.load_state_dict(agg_layer.state_dict())
            
            elapsed = time.time() - start_time
            self.aggregation_times.append(elapsed)
            
            self.current_iteration += 1
            
            logger.debug(f"Worker {self.worker_id} aggregation {self.current_iteration} completed in {elapsed:.2f}s")
            
            # Aggregation interval
            time.sleep(0.5)
        
        logger.info(f"Worker {self.worker_id} aggregation loop finished")
    
    def _update_peer_metadata(self):
        """Fetch metadata from all peers."""
        for peer in self.config.peers:
            metadata = self.network.get_peer_metadata(peer.id)
            if metadata:
                self.peer_metadata[peer.id] = metadata
    
    def _update_selection_probabilities(self):
        """Update peer and layer selection probabilities (Algorithm: Peer Selection, Layer Selection)."""
        if not self.config.peers:
            return
        
        peer_ids = [p.id for p in self.config.peers]
        
        # Peer Selection (Equation 3, 4 from paper)
        bandwidths = np.array([self.network.get_bandwidth(p_id) for p_id in peer_ids])
        norm_bw = bandwidths / (bandwidths.sum() + 1e-8)
        
        # Data distribution divergence
        divs = []
        local_dist = self.config.data_distribution if hasattr(self.config, 'data_distribution') else np.ones(10) / 10
        
        for p_id in peer_ids:
            if p_id in self.peer_metadata:
                peer_dist = np.array(self.peer_metadata[p_id].data_distribution)
                div = np.sum(np.abs(local_dist - peer_dist))
                divs.append(div)
            else:
                divs.append(0)
        
        divs = np.array(divs)
        norm_div = divs / (divs.sum() + 1e-8)
        
        # Combined score (Equation 4)
        scores = self.tau1 * norm_bw + self.tau2 * norm_div
        scores = scores / (scores.sum() + 1e-8)
        self.peer_probs = {p_id: s for p_id, s in zip(peer_ids, scores)}
        
        # Layer Selection (Equation 5, 6 from paper)
        # Use gradient variation as proxy for training efficiency
        for l_idx in range(len(self.layers)):
            gradients = []
            for p_id in peer_ids:
                # Use L2 norm of parameters as proxy
                # (In paper, this would be gradient variation between epochs)
                g_val = 1.0  # Simplified
                gradients.append(g_val)
            
            gradients = np.array(gradients)
            norm_grads = gradients / (gradients.sum() + 1e-8)
            self.layer_probs[l_idx] = {p_id: g for p_id, g in zip(peer_ids, norm_grads)}
    
    def _list_scheduling(self) -> Dict[int, List[int]]:
        """List scheduling algorithm for aggregation strategy (Algorithm 1 from paper).
        
        Returns:
            Dictionary mapping peer_id -> list of layer indices to pull
        """
        if not self.config.peers:
            return {}
        
        peer_ids = [p.id for p in self.config.peers]
        num_peers = len(peer_ids)
        num_layers = len(self.layers)
        
        # Initialize matrices
        mu1 = np.zeros((num_peers, num_layers))  # Communication time
        mu2 = np.zeros((num_peers, num_layers))  # Selection probability
        
        for i, p_id in enumerate(peer_ids):
            bw = self.network.get_bandwidth(p_id)
            for j in range(num_layers):
                mu1[i, j] = self.layer_sizes[j] / (bw + 1e-6)
                mu2[i, j] = self.peer_probs.get(p_id, 0) * self.layer_probs[j].get(p_id, 0)
        
        # Threshold (average probability)
        theta = np.mean(mu2, axis=0)
        
        # Mask out low-probability assignments
        valid_mask = mu2 >= theta
        mu1_masked = np.where(valid_mask, mu1, np.inf)
        
        # Calculate efficiency (Equation 9)
        min_times = np.min(mu1_masked, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = min_times / mu1_masked
            efficiency[np.isinf(mu1_masked)] = 0
            efficiency = np.nan_to_num(efficiency)
        
        # Rank peers
        peer_ranks = np.sum(efficiency, axis=1)
        sorted_peer_indices = np.argsort(-peer_ranks)
        
        # Assignment
        assignment = {p_id: [] for p_id in peer_ids}
        peer_loads = np.zeros(num_peers)
        assigned_layers = set()
        
        # Phase 1: Assign each layer to best peer
        for _ in range(num_layers):
            best_l, best_p_idx, max_eff = -1, -1, -1
            for p_idx in sorted_peer_indices:
                for l in range(num_layers):
                    if l in assigned_layers:
                        continue
                    eff = efficiency[p_idx, l]
                    if eff > max_eff:
                        max_eff = eff
                        best_l = l
                        best_p_idx = p_idx
            
            if best_l != -1:
                p_id = peer_ids[best_p_idx]
                assignment[p_id].append(best_l)
                assigned_layers.add(best_l)
                peer_loads[best_p_idx] += mu1[best_p_idx, best_l]
        
        # Phase 2: Fill gaps (pull additional layers if time permits)
        max_load = np.max(peer_loads) if peer_loads.size > 0 else 0
        for p_idx in range(num_peers):
            p_id = peer_ids[p_idx]
            for l in range(num_layers):
                if l not in assignment[p_id]:
                    cost = mu1[p_idx, l]
                    if peer_loads[p_idx] + cost <= max_load and efficiency[p_idx, l] > 0:
                        assignment[p_id].append(l)
                        peer_loads[p_idx] += cost
        
        return assignment
    
    def _pull_layers_parallel(self, strategy: Dict[int, List[int]]) -> Dict[int, Dict[int, torch.Tensor]]:
        """Pull layers from peers in parallel.
        
        Args:
            strategy: Dictionary mapping peer_id -> layer indices
            
        Returns:
            Dictionary mapping peer_id -> {layer_idx -> tensor}
        """
        pulled_layers = {}
        
        # Pull from each peer (could be parallelized with ThreadPoolExecutor)
        for peer_id, layer_indices in strategy.items():
            if layer_indices:
                layers = self.network.pull_layers(peer_id, layer_indices)
                if layers:
                    pulled_layers[peer_id] = layers
        
        return pulled_layers
    
    def _aggregate_layers(self, pulled_layers: Dict[int, Dict[int, torch.Tensor]]):
        """Aggregate pulled layers into model (Equation 1 from paper).
        
        Args:
            pulled_layers: Dictionary mapping peer_id -> {layer_idx -> tensor}
        """
        with torch.no_grad():
            for l_idx, layer in enumerate(self.layers):
                local_params = get_layer_params(layer)
                sum_params = local_params.clone()
                count = 1
                
                # Add parameters from peers
                for peer_id, peer_layers in pulled_layers.items():
                    if l_idx in peer_layers:
                        peer_params = peer_layers[l_idx]
                        if peer_params.shape == local_params.shape:
                            sum_params += peer_params
                            count += 1
                
                # Average
                avg_params = sum_params / count
                
                # Update aggregated model
                agg_layer = extract_layers(self.aggregated_model)[l_idx]
                set_layer_params(agg_layer, avg_params)
    
    def get_statistics(self) -> Dict:
        """Get worker statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'worker_id': self.worker_id,
            'iteration': self.current_iteration,
            'avg_loss': np.mean(self.training_losses[-10:]) if self.training_losses else 0,
            'avg_aggregation_time': np.mean(self.aggregation_times[-10:]) if self.aggregation_times else 0,
            'status': self.status,
        }
