"""Configuration management for distributed MPLS."""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PeerConfig:
    """Configuration for a peer worker."""
    id: int
    address: str


@dataclass
class DistributedWorkerConfig:
    """Configuration for a distributed worker."""
    
    # Worker identification
    worker_id: int
    listen_address: str
    
    # Peer connections
    peers: List[PeerConfig] = field(default_factory=list)
    
    # Data configuration
    partition_id: int = 0
    dataset: str = "MNIST"
    non_iid_level: float = 0.5
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 0.01
    local_epochs: int = 1
    max_iterations: int = 100
    
    # MPLS parameters
    tau1: float = 0.5  # Bandwidth weight
    tau2: float = 0.5  # Data divergence weight
    
    # Model configuration
    model_name: str = "SimpleMLP"
    
    # System configuration
    compute_speed: float = 1.0
    device: str = "cpu"
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DistributedWorkerConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            DistributedWorkerConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse peer configurations
        peers = []
        if 'peers' in config_dict:
            for peer_dict in config_dict['peers']:
                peers.append(PeerConfig(
                    id=peer_dict['id'],
                    address=peer_dict['address']
                ))
        
        # Flatten nested dictionaries
        flat_config = {
            'worker_id': config_dict.get('worker_id', 0),
            'listen_address': config_dict.get('listen_address', '0.0.0.0:50051'),
            'peers': peers,
        }
        
        # Data configuration
        if 'data' in config_dict:
            flat_config['partition_id'] = config_dict['data'].get('partition_id', 0)
            flat_config['dataset'] = config_dict['data'].get('dataset', 'MNIST')
            flat_config['non_iid_level'] = config_dict['data'].get('non_iid_level', 0.5)
        
        # Training configuration
        if 'training' in config_dict:
            flat_config['batch_size'] = config_dict['training'].get('batch_size', 32)
            flat_config['learning_rate'] = config_dict['training'].get('learning_rate', 0.01)
            flat_config['local_epochs'] = config_dict['training'].get('local_epochs', 1)
            flat_config['max_iterations'] = config_dict['training'].get('max_iterations', 100)
        
        # MPLS configuration
        if 'mpls' in config_dict:
            flat_config['tau1'] = config_dict['mpls'].get('tau1', 0.5)
            flat_config['tau2'] = config_dict['mpls'].get('tau2', 0.5)
        
        # Model configuration
        if 'model' in config_dict:
            flat_config['model_name'] = config_dict['model'].get('name', 'SimpleMLP')
        
        # System configuration
        if 'system' in config_dict:
            flat_config['compute_speed'] = config_dict['system'].get('compute_speed', 1.0)
            flat_config['device'] = config_dict['system'].get('device', 'cpu')
        
        # Logging configuration
        if 'logging' in config_dict:
            flat_config['log_level'] = config_dict['logging'].get('level', 'INFO')
            flat_config['log_dir'] = config_dict['logging'].get('dir', 'logs')
        
        return cls(**flat_config)
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        config_dict = {
            'worker_id': self.worker_id,
            'listen_address': self.listen_address,
            'peers': [{'id': p.id, 'address': p.address} for p in self.peers],
            'data': {
                'partition_id': self.partition_id,
                'dataset': self.dataset,
                'non_iid_level': self.non_iid_level,
            },
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'local_epochs': self.local_epochs,
                'max_iterations': self.max_iterations,
            },
            'mpls': {
                'tau1': self.tau1,
                'tau2': self.tau2,
            },
            'model': {
                'name': self.model_name,
            },
            'system': {
                'compute_speed': self.compute_speed,
                'device': self.device,
            },
            'logging': {
                'level': self.log_level,
                'dir': self.log_dir,
            },
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {output_path}")


@dataclass
class TopologyConfig:
    """Configuration for network topology."""
    
    topology_type: str = "ring"  # ring, random, fully_connected
    num_workers: int = 5
    base_port: int = 50051
    base_address: str = "localhost"
    random_graph_prob: float = 0.5
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TopologyConfig':
        """Load topology configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            TopologyConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            topology_type=config_dict.get('topology_type', 'ring'),
            num_workers=config_dict.get('num_workers', 5),
            base_port=config_dict.get('base_port', 50051),
            base_address=config_dict.get('base_address', 'localhost'),
            random_graph_prob=config_dict.get('random_graph_prob', 0.5),
        )
    
    def to_yaml(self, output_path: str):
        """Save topology configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        config_dict = {
            'topology_type': self.topology_type,
            'num_workers': self.num_workers,
            'base_port': self.base_port,
            'base_address': self.base_address,
            'random_graph_prob': self.random_graph_prob,
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class ConfigGenerator:
    """Generate worker configurations from topology specification."""
    
    @staticmethod
    def generate_topology(topology_config: TopologyConfig) -> Dict[int, List[int]]:
        """Generate adjacency list for network topology.
        
        Args:
            topology_config: Topology configuration
            
        Returns:
            Dictionary mapping worker_id -> list of peer_ids
        """
        import numpy as np
        
        num_workers = topology_config.num_workers
        adjacency = {i: [] for i in range(num_workers)}
        
        if topology_config.topology_type == "ring":
            # Ring topology: each worker connected to neighbors
            for i in range(num_workers):
                prev_peer = (i - 1) % num_workers
                next_peer = (i + 1) % num_workers
                adjacency[i] = [prev_peer, next_peer] if prev_peer != next_peer else [prev_peer]
        
        elif topology_config.topology_type == "fully_connected":
            # Fully connected: all workers connected to all others
            for i in range(num_workers):
                adjacency[i] = [j for j in range(num_workers) if j != i]
        
        elif topology_config.topology_type == "random":
            # Random graph: connect with probability
            import random
            random.seed(42)  # For reproducibility
            for i in range(num_workers):
                for j in range(i + 1, num_workers):
                    if random.random() < topology_config.random_graph_prob:
                        adjacency[i].append(j)
                        adjacency[j].append(i)
        
        else:
            raise ValueError(f"Unknown topology type: {topology_config.topology_type}")
        
        logger.info(f"Generated {topology_config.topology_type} topology with {num_workers} workers")
        return adjacency
    
    @staticmethod
    def generate_worker_configs(
        topology_config: TopologyConfig,
        output_dir: str = "configs",
        dataset: str = "MNIST",
        non_iid_level: float = 0.5,
    ) -> List[str]:
        """Generate worker configuration files.
        
        Args:
            topology_config: Topology configuration
            output_dir: Directory to save configuration files
            dataset: Dataset name
            non_iid_level: Non-IID level for data partitioning
            
        Returns:
            List of generated configuration file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate topology
        adjacency = ConfigGenerator.generate_topology(topology_config)
        
        config_paths = []
        
        for worker_id in range(topology_config.num_workers):
            # Create peer configurations
            peers = []
            for peer_id in adjacency[worker_id]:
                peer_address = f"{topology_config.base_address}:{topology_config.base_port + peer_id}"
                peers.append(PeerConfig(id=peer_id, address=peer_address))
            
            # Create worker configuration
            listen_address = f"0.0.0.0:{topology_config.base_port + worker_id}"
            
            worker_config = DistributedWorkerConfig(
                worker_id=worker_id,
                listen_address=listen_address,
                peers=peers,
                partition_id=worker_id,
                dataset=dataset,
                non_iid_level=non_iid_level,
            )
            
            # Save to file
            config_path = os.path.join(output_dir, f"worker_{worker_id}.yaml")
            worker_config.to_yaml(config_path)
            config_paths.append(config_path)
            
            logger.info(f"Generated configuration for worker {worker_id}: {config_path}")
        
        return config_paths
