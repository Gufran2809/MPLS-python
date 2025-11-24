#!/usr/bin/env python3
"""Generate worker configurations from topology specification."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distributed.config_loader import TopologyConfig, ConfigGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate worker configurations')
    parser.add_argument('--topology', type=str, default='ring',
                       choices=['ring', 'random', 'fully_connected'],
                       help='Network topology type')
    parser.add_argument('--num-workers', type=int, default=5,
                       help='Number of workers')
    parser.add_argument('--base-port', type=int, default=50051,
                       help='Base port number')
    parser.add_argument('--base-address', type=str, default='localhost',
                       help='Base address for workers')
    parser.add_argument('--random-prob', type=float, default=0.5,
                       help='Connection probability for random topology')
    parser.add_argument('--output-dir', type=str, default='configs',
                       help='Output directory for configuration files')
    parser.add_argument('--dataset', type=str, default='MNIST',
                       help='Dataset name')
    parser.add_argument('--non-iid-level', type=float, default=0.5,
                       help='Non-IID level for data partitioning')
    
    args = parser.parse_args()
    
    # Create topology configuration
    topology_config = TopologyConfig(
        topology_type=args.topology,
        num_workers=args.num_workers,
        base_port=args.base_port,
        base_address=args.base_address,
        random_graph_prob=args.random_prob,
    )
    
    # Generate worker configurations
    logger.info(f"Generating configurations for {args.num_workers} workers with {args.topology} topology")
    
    config_paths = ConfigGenerator.generate_worker_configs(
        topology_config=topology_config,
        output_dir=args.output_dir,
        dataset=args.dataset,
        non_iid_level=args.non_iid_level,
    )
    
    logger.info(f"Generated {len(config_paths)} configuration files in {args.output_dir}")
    for path in config_paths:
        logger.info(f"  - {path}")
    
    # Also save topology configuration
    topology_path = Path(args.output_dir) / "topology.yaml"
    topology_config.to_yaml(str(topology_path))
    logger.info(f"Saved topology configuration to {topology_path}")


if __name__ == "__main__":
    main()
