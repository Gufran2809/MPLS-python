#!/usr/bin/env python3
"""Run distributed MPLS training with multiple workers on a single machine."""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_configs import ConfigGenerator, TopologyConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_worker(config_path: str, worker_id: int):
    """Run a single worker process.
    
    Args:
        config_path: Path to worker configuration file
        worker_id: Worker ID for logging
    """
    # Import here to avoid issues with multiprocessing
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    
    from src.distributed.config_loader import DistributedWorkerConfig
    from src.distributed.worker import DistributedWorker
    from src.model import create_model
    
    # Set up logging for this worker
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"worker_{worker_id}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    worker_logger = logging.getLogger(f"Worker{worker_id}")
    worker_logger.addHandler(file_handler)
    worker_logger.setLevel(logging.INFO)
    
    try:
        worker_logger.info(f"Worker {worker_id} starting...")
        
        # Load configuration
        config = DistributedWorkerConfig.from_yaml(config_path)
        worker_logger.info(f"Loaded configuration from {config_path}")
        
        # Load dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        if config.dataset == "MNIST":
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        elif config.dataset == "CIFAR10":
            dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")
        
        # Partition data for this worker
        labels = np.array([int(y) for _, y in dataset])
        indices = []
        
        # Simple partitioning based on partition_id
        num_workers = config.worker_id + 1  # Approximate
        partition_size = len(dataset) // (num_workers * 2)  # Rough estimate
        start_idx = config.partition_id * partition_size
        end_idx = min(start_idx + partition_size, len(dataset))
        indices = list(range(start_idx, end_idx))
        
        worker_logger.info(f"Worker {worker_id} has {len(indices)} data samples")
        
        # Create dataloader
        if len(indices) > 0:
            subset = Subset(dataset, indices)
            dataloader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
        else:
            # Empty dataloader
            dataloader = DataLoader(dataset, batch_size=1)
        
        # Compute data distribution for MPLS
        if len(indices) > 0:
            labels_subset = labels[indices]
            num_classes = 10
            data_distribution = np.array([
                np.sum(labels_subset == c) / len(labels_subset) for c in range(num_classes)
            ])
        else:
            data_distribution = np.ones(10) / 10
        
        config.data_distribution = data_distribution
        
        # Create model
        model = create_model()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        
        # Create distributed worker
        worker = DistributedWorker(
            config=config,
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
        )
        
        # Start worker
        worker.start()
        
        # Run until max iterations
        while worker.current_iteration < worker.max_iterations:
            time.sleep(1)
            
            # Log statistics periodically
            if worker.current_iteration % 10 == 0:
                stats = worker.get_statistics()
                worker_logger.info(
                    f"Iteration {stats['iteration']}: "
                    f"Loss={stats['avg_loss']:.4f}, "
                    f"AggTime={stats['avg_aggregation_time']:.2f}s"
                )
        
        # Stop worker
        worker.stop()
        worker_logger.info(f"Worker {worker_id} finished")
        
    except KeyboardInterrupt:
        worker_logger.info(f"Worker {worker_id} interrupted")
    except Exception as e:
        worker_logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='Run distributed MPLS training')
    parser.add_argument('--config-dir', type=str, default='configs',
                       help='Directory containing worker configurations')
    parser.add_argument('--num-workers', type=int, default=5,
                       help='Number of workers to run')
    parser.add_argument('--topology', type=str, default='ring',
                       choices=['ring', 'random', 'fully_connected'],
                       help='Network topology (auto-generates configs if needed)')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of training rounds')
    parser.add_argument('--generate-configs', action='store_true',
                       help='Generate configurations before running')
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    
    # Generate configurations if requested
    if args.generate_configs or not config_dir.exists():
        logger.info("Generating worker configurations...")
        config_dir.mkdir(exist_ok=True)
        
        topology_config = TopologyConfig(
            topology_type=args.topology,
            num_workers=args.num_workers,
            base_port=50051,
            base_address='localhost',
        )
        
        ConfigGenerator.generate_worker_configs(
            topology_config=topology_config,
            output_dir=str(config_dir),
        )
        logger.info(f"Generated configurations in {config_dir}")
    
    # Find worker configuration files
    config_files = sorted(config_dir.glob("worker_*.yaml"))
    
    if not config_files:
        logger.error(f"No worker configuration files found in {config_dir}")
        return
    
    logger.info(f"Found {len(config_files)} worker configurations")
    
    # Update max_iterations in all configs
    from src.distributed.config_loader import DistributedWorkerConfig
    for config_file in config_files:
        config = DistributedWorkerConfig.from_yaml(str(config_file))
        config.max_iterations = args.rounds
        config.to_yaml(str(config_file))
    
    # Start worker processes
    processes = []
    
    try:
        for i, config_file in enumerate(config_files[:args.num_workers]):
            logger.info(f"Starting worker {i} with config {config_file}")
            
            p = multiprocessing.Process(
                target=run_worker,
                args=(str(config_file), i)
            )
            p.start()
            processes.append(p)
            
            # Stagger starts to avoid race conditions
            time.sleep(1)
        
        logger.info(f"Started {len(processes)} worker processes")
        logger.info("Press Ctrl+C to stop all workers")
        
        # Wait for all processes
        for p in processes:
            p.join()
        
        logger.info("All workers finished")
        
    except KeyboardInterrupt:
        logger.info("Interrupted! Stopping all workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
        logger.info("All workers stopped")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
