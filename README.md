# Distributed MPLS: Decentralized Federated Learning

A production-ready implementation of MPLS (Multi-Path Learning System) for truly distributed, asynchronous decentralized federated learning based on the Euro-Par 2025 research paper.

## Features

🚀 **True P2P Communication** - gRPC-based peer-to-peer layer exchange  
⚡ **Asynchronous Training** - Overlapping local training and model aggregation  
🎯 **All MPLS Algorithms** - Peer selection, layer selection, list scheduling  
🐳 **Docker Support** - Easy multi-machine deployment  
📊 **Monitoring** - Built-in logging and statistics  
🔧 **Flexible Deployment** - Single machine (multi-process) or distributed  

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Distributed Worker                     │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   Training   │◄──────►│ Aggregation  │                  │
│  │    Thread    │  Async │    Thread    │                  │
│  └──────────────┘        └──────────────┘                  │
│         │                        │                          │
│         │                        ▼                          │
│         │              ┌──────────────────┐                 │
│         └─────────────►│ Parameter Store  │                 │
│                        └──────────────────┘                 │
│                                 │                           │
│                        ┌──────────────────┐                 │
│                        │  gRPC Server     │                 │
│                        └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │ Worker  │  │ Worker  │  │ Worker  │
              │    1    │  │    2    │  │    3    │
              └─────────┘  └─────────┘  └─────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_distributed.txt
```

### 2. Run Distributed Training (Single Machine)

```bash
# Generate configurations for 5 workers with ring topology
python scripts/generate_configs.py --num-workers 5 --topology ring

# Run all workers
python scripts/run_distributed.py --num-workers 5 --rounds 100
```

### 3. Check Results

```bash
# View logs
tail -f logs/worker_0.log

# Monitor all workers
watch -n 1 'tail -n 3 logs/worker_*.log'
```

## Usage

### Generate Configurations

```bash
python scripts/generate_configs.py \
  --num-workers 10 \
  --topology random \
  --random-prob 0.5 \
  --output-dir configs
```

**Topology Options:**
- `ring`: Each worker connects to 2 neighbors
- `random`: Random graph with specified connection probability
- `fully_connected`: All workers connect to all others

### Run Distributed Training

```bash
python scripts/run_distributed.py \
  --config-dir configs \
  --num-workers 5 \
  --rounds 100
```

### Docker Deployment

```bash
# Build image
docker build -t mpls-worker .

# Run with docker-compose
docker-compose up --scale worker=5
```

## Configuration

Example worker configuration (`configs/worker_0.yaml`):

```yaml
worker_id: 0
listen_address: "0.0.0.0:50051"

peers:
  - id: 1
    address: "localhost:50052"
  - id: 4
    address: "localhost:50055"

data:
  partition_id: 0
  dataset: "MNIST"
  non_iid_level: 0.5

training:
  batch_size: 32
  learning_rate: 0.01
  local_epochs: 1
  max_iterations: 100

mpls:
  tau1: 0.5  # Bandwidth weight
  tau2: 0.5  # Data divergence weight

system:
  compute_speed: 1.0
  device: "cpu"
```

## Algorithms Implemented

### 1. Peer Selection (Equation 4)
Workers select peers based on:
- **Bandwidth** (τ₁ weight): Prefer high-bandwidth peers
- **Data Divergence** (τ₂ weight): Prefer peers with different data distributions

### 2. Layer Selection (Equation 6)
Workers select which layers to pull from each peer based on:
- Training efficiency (gradient variation)
- Historical aggregation performance

### 3. List Scheduling (Algorithm 1)
Optimally assign layers to peers to:
- Minimize communication delay
- Balance load across peers
- Maximize bandwidth utilization

### 4. Model Aggregation (Equation 1)
Average pulled layers with local model:
```
w_i^{k+1}(l) = (Σ y_s^k(l) w_s^k(l) + w_i^k(l)) / (Σ y_s^k(l) + 1)
```

## Project Structure

```
mpls_dfl_python/
├── src/
│   ├── distributed/
│   │   ├── network.py          # gRPC communication
│   │   ├── worker.py           # Distributed worker
│   │   ├── config_loader.py    # Configuration management
│   │   └── mpls_service_pb2.py # Generated protobuf
│   ├── workers.py              # MPLS/APPG/NetMax algorithms
│   ├── model.py                # Neural network models
│   └── config.py               # Worker configuration
├── proto/
│   └── mpls_service.proto      # gRPC service definition
├── scripts/
│   ├── generate_configs.py     # Config generator
│   └── run_distributed.py      # Multi-process runner
├── configs/                    # Generated worker configs
├── logs/                       # Worker logs
├── Dockerfile                  # Container image
└── docker-compose.yml          # Multi-container orchestration
```

## Differences from Simulation

| Aspect | Simulation (`src/runner.py`) | Distributed |
|--------|------------------------------|-------------|
| **Workers** | Python objects in single process | Independent processes |
| **Communication** | In-memory object sharing | gRPC over network |
| **Execution** | Synchronous (wait for all) | Asynchronous (independent) |
| **Model Exchange** | Direct memory copy | Serialization + network transfer |
| **Deployment** | Single machine only | Single or multi-machine |

## Multi-Machine Deployment

### Option 1: Manual Deployment

On each machine, create a worker configuration with peer addresses:

```yaml
# Machine 1: worker_0.yaml
worker_id: 0
listen_address: "0.0.0.0:50051"
peers:
  - id: 1
    address: "192.168.1.101:50051"  # Machine 2
  - id: 2
    address: "192.168.1.102:50051"  # Machine 3
```

Run on each machine:
```bash
python -m src.distributed.worker --config configs/worker_0.yaml
```

### Option 2: Docker Swarm/Kubernetes

Deploy using container orchestration for automatic scaling and management.

## Monitoring

### Real-time Logs

```bash
# All workers
tail -f logs/worker_*.log

# Specific worker
tail -f logs/worker_0.log
```

### Statistics

Each worker logs:
- Training loss
- Aggregation time
- Communication delays
- Peer connectivity status

## Performance Tips

1. **Bandwidth**: Use `tau1=0.7, tau2=0.3` for faster convergence in good networks
2. **Data Heterogeneity**: Use `tau1=0.3, tau2=0.7` for highly non-IID data
3. **Topology**: Fully-connected achieves best accuracy but highest traffic
4. **Workers**: 5-10 workers is optimal for single machine

## Troubleshooting

### Workers can't connect to peers

```bash
# Check if ports are open
netstat -tulpn | grep 5005

# Try pinging peer
nc -zv localhost 50051
```

### Out of memory

Reduce batch size or number of workers:
```yaml
training:
  batch_size: 16  # Default is 32
```

### Slow convergence

Increase `tau2` to aggregate from more diverse peers:
```yaml
mpls:
  tau1: 0.3
  tau2: 0.7
```

## Citation

If you use this implementation, please cite the original MPLS paper:

```bibtex
@inproceedings{xu2025mpls,
  title={MPLS: Stacking Diverse Layers Into One Model for Decentralized Federated Learning},
  author={Xu, Yang and Yao, Zhiwei and Xu, Hongli and Liao, Yunming and Xie, Zuan},
  booktitle={Euro-Par 2025},
  year={2025}
}
```

## License

This implementation is provided for research purposes.

## Acknowledgments

Based on the MPLS paper from Euro-Par 2025. Original simulation code refactored for distributed deployment.
