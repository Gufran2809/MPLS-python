# Quick Start Guide

Get up and running with distributed MPLS in under 5 minutes!

## Prerequisites

- Python 3.8+
- pip
- 2GB RAM minimum

## Step-by-Step Guide

### 1. Install Dependencies (1 minute)

```bash
cd /home/alien/Desktop/mpls_dfl_python

# Activate virtual environment  
source .venv/bin/activate

# Install distributed requirements (if not already done)
pip install -r requirements_distributed.txt
```

### 2. Generate Configurations (30 seconds)

```bash
# Create configs for 3 workers with ring topology
python scripts/generate_configs.py \
  --num-workers 3 \
  --topology ring \
  --output-dir configs
```

This creates:
- `configs/worker_0.yaml`
- `configs/worker_1.yaml`
- `configs/worker_2.yaml`
- `configs/topology.yaml`

### 3. Run Distributed Training (2 minutes)

```bash
# Run 3 workers for 50 rounds
python scripts/run_distributed.py \
  --config-dir configs \
  --num-workers 3 \
  --rounds 50
```

You'll see output like:
```
2025-11-24 19:30:00 - INFO - Generating worker configurations...
2025-11-24 19:30:01 - INFO - Starting worker 0 with config configs/worker_0.yaml
2025-11-24 19:30:02 - INFO - Starting worker 1 with config configs/worker_1.yaml
2025-11-24 19:30:03 - INFO - Starting worker 2 with config configs/worker_2.yaml
2025-11-24 19:30:03 - INFO - Started 3 worker processes
```

### 4. Monitor Progress (realtime)

Open a new terminal and run:

```bash
# Watch all worker logs
watch -n 1 'tail -n 2 logs/worker_*.log'
```

Or check individual worker logs:

```bash
tail -f logs/worker_0.log
```

### 5. Wait for Completion

Training will run for 50 rounds (about 2-3 minutes). Workers will automatically stop when done.

Press `Ctrl+C` to stop early if needed.

## What Just Happened?

1. **3 workers** started on `localhost:50051`, `50052`, `50053`
2. **Ring topology**: Worker 0 ↔ Worker 1 ↔ Worker 2 ↔ Worker 0
3. **MNIST dataset** partitioned across workers (non-IID)
4. **Asynchronous training**: Each worker trains independently
5. **Layer exchange**: Workers pull layers from neighbors via gRPC
6. **MPLS algorithms**: Peer selection + layer selection + list scheduling

## Next Steps

### Try Different Topologies

**Fully Connected** (all workers connected):
```bash
python scripts/generate_configs.py --num-workers 3 --topology fully_connected
python scripts/run_distributed.py --num-workers 3 --rounds 50
```

**Random Graph**:
```bash
python scripts/generate_configs.py --num-workers 5 --topology random --random-prob 0.5
python scripts/run_distributed.py --num-workers 5 --rounds 50
```

### Scale Up

Run with more workers:
```bash
python scripts/generate_configs.py --num-workers 10 --topology ring
python scripts/run_distributed.py --num-workers 10 --rounds 100
```

### Tune MPLS Parameters

Edit `configs/worker_0.yaml`:

```yaml
mpls:
  tau1: 0.7  # Favor high-bandwidth peers
  tau2: 0.3  # Less weight on data divergence
```

Then run:
```bash
python scripts/run_distributed.py --num-workers 3 --rounds 50
```

### Compare with Baselines

Run the original simulation for comparison:

```bash
python -m src.runner
```

## Troubleshooting

**"No module named grpc"**
```bash
pip install -r requirements_distributed.txt
```

**"Address already in use"**
```bash
# Kill existing workers
pkill -f run_distributed

# Wait 5 seconds and try again
sleep 5
python scripts/run_distributed.py --num-workers 3 --rounds 50
```

**Workers not connecting**
```bash
# Check if ports are open
netstat -tulpn | grep 5005

# Restart with clean configs
rm -rf configs
python scripts/generate_configs.py --num-workers 3 --topology ring
python scripts/run_distributed.py --num-workers 3 --rounds 50
```

## Understanding the Logs

```
2025-11-24 19:30:15 - Worker0 - INFO - Worker 0 gRPC server started on 0.0.0.0:50051
```
✅ Worker server is listening

```
2025-11-24 19:30:16 - Worker0 - INFO - Worker 0 connected to peer 1 at localhost:50052
```
✅ Connection established

```
2025-11-24 19:30:20 - Worker0 - INFO - Iteration 10: Loss=0.4521, AggTime=0.34s
```
✅ Training progress (loss going down is good!)

```
2025-11-24 19:30:21 - Worker0 - DEBUG - Pulled 2 layers from peer 1
```
✅ Layer exchange working

## Congratulations! 🎉

You've successfully run distributed MPLS training. The system is now truly peer-to-peer with:
- ✅ Independent worker processes
- ✅ gRPC network communication
- ✅ Asynchronous training and aggregation
- ✅ Dynamic peer/layer selection

Ready for multi-machine deployment? See `README.md` for details!
