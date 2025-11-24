# Quick Deployment Guide for Your 3 Servers

## 🎯 Your Server IPs
- **Server 1**: 10.96.0.255 (Worker 0)
- **Server 2**: 10.96.0.87 (Worker 1)  
- **Server 3**: 10.96.0.2 (Worker 2)

## Option 1: Automated Deployment (Recommended)

Run this **ONE command** from your local machine:

```bash
./scripts/deploy_remote.sh 10.96.0.255 10.96.0.87 10.96.0.2
```

This will:
1. ✅ Copy code to all 3 servers
2. ✅ Install dependencies
3. ✅ Setup configurations  
4. ✅ Test connectivity

Then start workers (one command per server):

```bash
# Server 1
ssh user@10.96.0.255 'cd ~/mpls_dfl && source .venv/bin/activate && python -m src.distributed.worker_runner --config configs/worker.yaml'

# Wait 10 seconds, then Server 2
ssh user@10.96.0.87 'cd ~/mpls_dfl && source .venv/bin/activate && python -m src.distributed.worker_runner --config configs/worker.yaml'

# Wait 10 seconds, then Server 3
ssh user@10.96.0.2 'cd ~/mpls_dfl && source .venv/bin/activate && python -m src.distributed.worker_runner --config configs/worker.yaml'
```

## Option 2: Manual Step-by-Step

### Step 1: Copy to each server

```bash
# From your local machine
scp -r /home/alien/Desktop/mpls_dfl_python user@10.96.0.255:~/mpls_dfl
scp -r /home/alien/Desktop/mpls_dfl_python user@10.96.0.87:~/mpls_dfl
scp -r /home/alien/Desktop/mpls_dfl_python user@10.96.0.2:~/mpls_dfl
```

### Step 2: Install on each server

SSH into each and run:

```bash
cd ~/mpls_dfl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt requirements_distributed.txt
python -m grpc_tools.protoc -I./proto --python_out=./src/distributed --grpc_python_out=./src/distributed ./proto/mpls_service.proto
```

### Step 3: Copy configs

```bash
# From local machine
scp configs/remote/worker_0.yaml user@10.96.0.255:~/mpls_dfl/configs/
scp configs/remote/worker_1.yaml user@10.96.0.87:~/mpls_dfl/configs/
scp configs/remote/worker_2.yaml user@10.96.0.2:~/mpls_dfl/configs/
```

### Step 4: Start workers

```bash
# On Server 1 (10.96.0.255)
cd ~/mpls_dfl && source .venv/bin/activate
python -m src.distributed.worker_runner --config configs/worker_0.yaml

# Wait 10 seconds, then on Server 2 (10.96.0.87)
cd ~/mpls_dfl && source .venv/bin/activate
python -m src.distributed.worker_runner --config configs/worker_1.yaml

# Wait 10 seconds, then on Server 3 (10.96.0.2)
cd ~/mpls_dfl && source .venv/bin/activate
python -m src.distributed.worker_runner --config configs/worker_2.yaml
```

## Monitor Progress

```bash
# From any server
tail -f ~/mpls_dfl/logs/worker_*.log
```

Look for:
- ✅ "gRPC server started"
- ✅ "Connected to peer X"
- ✅ "Iteration X: Loss=..."

## Verify It's Working

You should see messages like:
```
Worker 0 connected to peer 1 at 10.96.0.87:50051
Worker 0 connected to peer 2 at 10.96.0.2:50051
Pulled 2 layers from peer 1
Iteration 10: Loss=0.45, AggTime=0.5s
```

## Troubleshooting

**Can't connect?**
```bash
# On each server, allow firewall
sudo ufw allow 50051

# Test connectivity
ping 10.96.0.255
nc -zv 10.96.0.87 50051
```

**Import errors?**
```bash
cd ~/mpls_dfl
export PYTHONPATH=~/mpls_dfl:$PYTHONPATH
```

## Full Details

See [DEPLOYMENT.md](file:///home/alien/Desktop/mpls_dfl_python/DEPLOYMENT.md) for complete guide with systemd services, troubleshooting, etc.
