import copy

import numpy as np
import torch
import torch.nn as nn

from .config import WorkerConfig, extract_layers, get_layer_params, set_layer_params


class MPLSWorker:
    def __init__(self, config: WorkerConfig, model: nn.Module, tau1=0.5, tau2=0.5):
        self.config = config
        self.model = model
        self.tau1 = tau1  # Weight for Bandwidth
        self.tau2 = tau2  # Weight for Data Divergence
        self.layers = extract_layers(self.model)

        # Calculate layer sizes in MB
        self.layer_sizes = []
        for l in self.layers:
            params = sum(p.numel() for p in l.parameters())
            self.layer_sizes.append(params * 4 / (1024 * 1024))  # Float32 = 4 bytes

        self.peer_probs = (
            {p: 1.0 / len(config.peers) for p in config.peers} if config.peers else {}
        )
        self.layer_probs = (
            {
                i: {p: 1.0 / len(config.peers) for p in config.peers}
                for i in range(len(self.layers))
            }
            if config.peers
            else {}
        )

        self.prev_params = {
            i: get_layer_params(l).clone() for i, l in enumerate(self.layers)
        }
        self.comm_delays = []

    def train_epoch(self, dataloader, optimizer, criterion):
        """Standard Local Training"""
        self.model.train()
        total_loss = 0
        batches = 0
        for data, target in dataloader:
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        # Update gradient history for Layer Selection
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                current = get_layer_params(layer)
                self.prev_params[i] = current.clone()

        return total_loss / max(1, batches)

    def update_probabilities(self, peer_distributions, peer_models):
        """Update Peer Selection and Layer Selection probabilities."""
        if not self.config.peers:
            return

        # --- 1. Peer Selection ---
        bws = np.array([self.config.bandwidth.get(p, 1.0) for p in self.config.peers])
        norm_bw = bws / (bws.sum() + 1e-8)

        divs = []
        for p in self.config.peers:
            if p in peer_distributions:
                d = np.sum(
                    np.abs(self.config.data_distribution - peer_distributions[p])
                )
                divs.append(d)
            else:
                divs.append(0)
        divs = np.array(divs)
        norm_div = divs / (divs.sum() + 1e-8)

        scores = self.tau1 * norm_bw + self.tau2 * norm_div
        scores = scores / (scores.sum() + 1e-8)
        self.peer_probs = {p: s for p, s in zip(self.config.peers, scores)}

        # --- 2. Layer Selection ---
        for l_idx in range(len(self.layers)):
            grads = []
            for p in self.config.peers:
                if p in peer_models:
                    p_layers = extract_layers(peer_models[p])
                    if l_idx < len(p_layers):
                        p_param = get_layer_params(p_layers[l_idx])
                        # Simplified proxy for variation: L2 norm
                        g_val = torch.norm(p_param).item()
                        grads.append(g_val)
                    else:
                        grads.append(0)
                else:
                    grads.append(0)

            grads = np.array(grads)
            norm_grads = grads / (grads.sum() + 1e-8)
            self.layer_probs[l_idx] = {
                p: g for p, g in zip(self.config.peers, norm_grads)
            }

    def list_scheduling(self):
        """Algorithm 1: Aggregation Strategy Development"""
        if not self.config.peers:
            return {}

        num_layers = len(self.layers)
        peers = self.config.peers

        mu1 = np.zeros((len(peers), num_layers))
        mu2 = np.zeros((len(peers), num_layers))

        for i, p in enumerate(peers):
            bw = self.config.bandwidth.get(p, 1.0)
            for j in range(num_layers):
                mu1[i, j] = self.layer_sizes[j] / (bw + 1e-6)
                mu2[i, j] = self.peer_probs.get(p, 0) * self.layer_probs[j].get(p, 0)

        theta = np.mean(mu2, axis=0)
        valid_mask = mu2 >= theta
        mu1_masked = np.where(valid_mask, mu1, np.inf)

        min_times = np.min(mu1_masked, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = min_times / mu1_masked
            efficiency[np.isinf(mu1_masked)] = 0
            efficiency = np.nan_to_num(efficiency)

        peer_ranks = np.sum(efficiency, axis=1)
        sorted_peer_indices = np.argsort(-peer_ranks)

        assignment = {p: [] for p in peers}
        peer_loads = np.zeros(len(peers))
        assigned_layers = set()

        # Phase 1: Assign unassigned layers
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
                p_id = peers[best_p_idx]
                assignment[p_id].append(best_l)
                assigned_layers.add(best_l)
                peer_loads[best_p_idx] += mu1[best_p_idx, best_l]

        # Phase 2: Fill gaps
        max_load = np.max(peer_loads)
        for p_idx in range(len(peers)):
            for l in range(num_layers):
                if l not in assignment[peers[p_idx]]:
                    cost = mu1[p_idx, l]
                    if (
                        peer_loads[p_idx] + cost <= max_load
                        and efficiency[p_idx, l] > 0
                    ):
                        assignment[peers[p_idx]].append(l)
                        peer_loads[p_idx] += cost

        self.comm_delays.append(max_load)
        return assignment

    def aggregate(self, peer_models):
        """Execute Aggregation Strategy"""
        strategy = self.list_scheduling()

        with torch.no_grad():
            for l_idx, layer in enumerate(self.layers):
                local_params = get_layer_params(layer)
                sum_params = local_params.clone()
                count = 1

                for p_id, layers_to_pull in strategy.items():
                    if l_idx in layers_to_pull and p_id in peer_models:
                        p_model = peer_models[p_id]
                        p_layers = extract_layers(p_model)
                        if l_idx < len(p_layers):
                            p_params = get_layer_params(p_layers[l_idx])
                            if p_params.shape == local_params.shape:
                                sum_params += p_params
                                count += 1

                avg_params = sum_params / count
                set_layer_params(layer, avg_params)


class APPGWorker(MPLSWorker):
    def aggregate(self, peer_models):
        with torch.no_grad():
            for l_idx, layer in enumerate(self.layers):
                local_params = get_layer_params(layer)
                sum_params = local_params.clone()
                count = 1
                for p_id in self.config.peers:
                    if p_id in peer_models:
                        p_layers = extract_layers(peer_models[p_id])
                        if l_idx < len(p_layers):
                            p_params = get_layer_params(p_layers[l_idx])
                            if p_params.shape == local_params.shape:
                                sum_params += p_params
                                count += 1
                set_layer_params(layer, sum_params / count)
        if self.config.peers:
            max_delay = max(
                [
                    sum(self.layer_sizes) / (self.config.bandwidth.get(p, 1) + 1e-6)
                    for p in self.config.peers
                ]
            )
            self.comm_delays.append(max_delay)


class NetMaxWorker(MPLSWorker):
    def aggregate(self, peer_models):
        if not self.config.peers:
            return
        best_peer = max(
            self.config.peers, key=lambda x: self.config.bandwidth.get(x, 0)
        )

        with torch.no_grad():
            if best_peer in peer_models:
                p_layers = extract_layers(peer_models[best_peer])
                for l_idx, layer in enumerate(self.layers):
                    local_params = get_layer_params(layer)
                    p_params = get_layer_params(p_layers[l_idx])
                    if local_params.shape == p_params.shape:
                        avg = (local_params + p_params) / 2
                        set_layer_params(layer, avg)

        delay = sum(self.layer_sizes) / (self.config.bandwidth.get(best_peer, 1) + 1e-6)
        self.comm_delays.append(delay)
