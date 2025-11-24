#!/usr/bin/env python3
"""Single worker runner for standalone operation."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.distributed.config_loader import DistributedWorkerConfig
from src.distributed.worker import DistributedWorker
from src.model import create_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run a single MPLS worker')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to worker configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DistributedWorkerConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration for worker {config.worker_id}")
    
    # Set up logging
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"worker_{config.worker_id}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Load dataset
    logger.info(f"Loading {config.dataset} dataset...")
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
    
    # Partition data
    labels = np.array([int(y) for _, y in dataset])
    num_classes = 10
    
    # Simple partitioning: divide dataset into chunks
    total_samples = len(dataset)
    samples_per_worker = total_samples // 10  # Assuming max 10 workers
    start_idx = config.partition_id * samples_per_worker
    end_idx = min(start_idx + samples_per_worker, total_samples)
    indices = list(range(start_idx, end_idx))
    
    logger.info(f"Worker {config.worker_id} has {len(indices)} samples (indices {start_idx} to {end_idx})")
    
    # Create dataloader
    if len(indices) > 0:
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
        
        # Compute data distribution
        labels_subset = labels[indices]
        data_distribution = np.array([
            np.sum(labels_subset == c) / len(labels_subset) for c in range(num_classes)
        ])
    else:
        logger.warning("No data samples for this worker!")
        dataloader = DataLoader(dataset, batch_size=1)
        data_distribution = np.ones(num_classes) / num_classes
    
    config.data_distribution = data_distribution
    logger.info(f"Data distribution: {data_distribution}")
    
    # Create model
    model = create_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    
    # Create distributed worker
    logger.info("Creating distributed worker...")
    worker = DistributedWorker(
        config=config,
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
    )
    
    # Start worker
    logger.info(f"Starting worker {config.worker_id}...")
    worker.start()
    
    # Run until max iterations
    try:
        while worker.current_iteration < worker.max_iterations:
            time.sleep(2)
            
            # Log statistics periodically
            if worker.current_iteration > 0 and worker.current_iteration % 5 == 0:
                stats = worker.get_statistics()
                logger.info(
                    f"Iteration {stats['iteration']}/{config.max_iterations}: "
                    f"Loss={stats['avg_loss']:.4f}, "
                    f"AggTime={stats['avg_aggregation_time']:.2f}s, "
                    f"Status={stats['status']}"
                )
        
        logger.info(f"Worker {config.worker_id} completed {config.max_iterations} iterations")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop worker
        logger.info("Stopping worker...")
        worker.stop()
        logger.info("Worker stopped successfully")


if __name__ == "__main__":
    main()
