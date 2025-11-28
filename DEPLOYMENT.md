# Multi-Server Deployment Guide

## Your Server Setup

| Server | IP Address | Worker ID | Port |
|--------|------------|-----------|------|
| Server 1 | 10.96.0.255 | 0 | 50051 |
| Server 2 | 10.96.0.87 | 1 | 50051 |
| Server 3 | 10.96.0.62 | 2 | 50051 |

**Topology**: Ring (0 ↔ 1 ↔ 2 ↔ 0)

## Step-by-Step Deployment

### Step 1: Prepare All Servers

On **each server**, run:

```bash
# Install Python dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Create project directory
mkdir -p ~/mpls_dfl
cd ~/mpls_dfl

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Copy Code to Servers

From your **local machine**, copy the project to each server:

```bash
# Server 1 (10.96.0.255)
scp -r /home/alien/Desktop/mpls_dfl_python/* user@10.96.0.255:~/mpls_dfl/

# Server 2 (10.96.0.87)
scp -r /home/alien/Desktop/mpls_dfl_python/* user@10.96.0.87:~/mpls_dfl/

# Server 3 (10.96.0.62)
scp -r /home/alien/Desktop/mpls_dfl_python/* user@10.96.0.62:~/mpls_dfl/
```

**Note**: Replace `user` with your actual SSH username.

### Step 3: Install Dependencies on Each Server

SSH into each server and run:

```bash
cd ~/mpls_dfl
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install -r requirements_distributed.txt

# Generate gRPC stubs
python -m grpc_tools.protoc \
  -I./proto \
  --python_out=./src/distributed \
  --grpc_python_out=./src/distributed \
  ./proto/mpls_service.proto
```

### Step 4: Copy Pre-Generated Configs

I've generated worker configurations for you in `configs/remote/`. Copy the appropriate config to each server:

```bash
# From your local machine:

# Server 1
scp configs/remote/worker_0.yaml user@10.96.0.255:~/mpls_dfl/configs/

# Server 2  
scp configs/remote/worker_1.yaml user@10.96.0.87:~/mpls_dfl/configs/

# Server 3
scp configs/remote/worker_2.yaml user@10.96.0.62:~/mpls_dfl/configs/
```

### Step 5: Verify Network Connectivity

On **each server**, test connectivity to peers:

```bash
# From Server 1 (10.96.0.255)
nc -zv 10.96.0.87 50051   # Test Server 2
nc -zv 10.96.0.62 50051    # Test Server 3

# From Server 2 (10.96.0.87)
nc -zv 10.96.0.255 50051  # Test Server 1
nc -zv 10.96.0.62 50051    # Test Server 3

# From Server 3 (10.96.0.62)
nc -zv 10.96.0.255 50051  # Test Server 1
nc -zv 10.96.0.87 50051   # Test Server 2
```

If `nc` is not installed: `sudo apt-get install netcat`

**Important**: Ensure firewall allows port 50051:
```bash
# On each server
sudo ufw allow 50051
```

### Step 6: Start Workers

Start workers in this order (to avoid connection errors):

**Server 1** (10.96.0.255):
```bash
cd ~/mpls_dfl
source .venv/bin/activate
python -m src.distributed.worker_runner --config configs/worker_0.yaml
```

Wait 5 seconds, then **Server 2** (10.96.0.87):
```bash
cd ~/mpls_dfl
source .venv/bin/activate
python -m src.distributed.worker_runner --config configs/worker_1.yaml
```

Wait 5 seconds, then **Server 3** (10.96.0.62):
```bash
cd ~/mpls_dfl
source .venv/bin/activate
python -m src.distributed.worker_runner --config configs/worker_2.yaml
```

### Step 7: Monitor Training

On each server, check logs:

```bash
tail -f ~/mpls_dfl/logs/worker_*.log
```

You should see:
- ✅ "gRPC server started on 0.0.0.0:50051"
- ✅ "Connected to peer X at IP:50051"
- ✅ "Iteration 10: Loss=0.45, AggTime=0.5s"
- ✅ "Pulled 2 layers from peer X"

## Alternative: Using systemd (Run as Service)

To run workers automatically on boot, create systemd services:

**On each server**, create `/etc/systemd/system/mpls-worker.service`:

```ini
[Unit]
Description=MPLS Distributed Worker
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/mpls_dfl
Environment="PATH=/home/your_username/mpls_dfl/.venv/bin"
ExecStart=/home/your_username/mpls_dfl/.venv/bin/python -m src.distributed.worker_runner --config configs/worker_X.yaml
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace:
- `your_username` with your actual username
- `worker_X.yaml` with the correct worker ID

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mpls-worker
sudo systemctl start mpls-worker
sudo systemctl status mpls-worker
```

## Troubleshooting

### Workers can't connect to peers

```bash
# Check if port is listening
sudo netstat -tulpn | grep 50051

# Check firewall
sudo ufw status

# Allow port
sudo ufw allow 50051
```

### "Connection refused"

One server hasn't started yet. Start workers in order (Server 1 → 2 → 3) with 5-second delays.

### "No module named 'src'"

```bash
# Make sure you're in the project directory
cd ~/mpls_dfl

# Add to PYTHONPATH
export PYTHONPATH=/home/your_username/mpls_dfl:$PYTHONPATH
```

### Check connectivity between servers

```bash
# Ping test
ping -c 3 10.96.0.255
ping -c 3 10.96.0.87
ping -c 3 10.96.0.62

# Port test
telnet 10.96.0.255 50051
```

## Quick Deploy Script

I've created `scripts/deploy_remote.sh` for automated deployment. Run from your local machine:

```bash
./scripts/deploy_remote.sh 10.96.0.255 10.96.0.87 10.96.0.62
```

## Expected Timeline

- **Setup (Steps 1-4)**: ~10 minutes per server
- **Network verification**: ~2 minutes
- **Training (100 rounds)**: ~10-15 minutes
- **Total**: ~45 minutes first time, ~15 minutes for subsequent runs

## Success Indicators

✅ All workers show "Connected to peer X" in logs  
✅ Loss decreasing over iterations  
✅ "Pulled X layers from peer Y" messages  
✅ No "Connection refused" or timeout errors  

## Data Partitioning

Each server will automatically get different data:
- **Server 1** (worker 0): samples 0 to N/3
- **Server 2** (worker 1): samples N/3 to 2N/3  
- **Server 3** (worker 2): samples 2N/3 to N

Data is partitioned with Non-IID level 0.5 by default (configurable in YAML).
