#!/bin/bash
# Automated deployment script for remote servers

set -e  # Exit on error

SERVERS=("$@")
NUM_SERVERS=${#SERVERS[@]}

if [ $NUM_SERVERS -eq 0 ]; then
    echo "Usage: $0 <server1_ip> <server2_ip> <server3_ip> ..."
    echo "Example: $0 10.96.0.255 10.96.0.87 10.96.0.2"
    exit 1
fi

echo "=========================================="
echo "MPLS Distributed Deployment"
echo "=========================================="
echo "Servers: ${SERVERS[@]}"
echo "Number of servers: $NUM_SERVERS"
echo ""

# Configuration
PROJECT_DIR="/home/alien/Desktop/mpls_dfl_python"
REMOTE_DIR="~/mpls_dfl"
SSH_USER=${SSH_USER:-"alien"}  # Default to alien, can be overridden

read -p "SSH username [$SSH_USER]: " input_user
if [ ! -z "$input_user" ]; then
    SSH_USER=$input_user
fi

echo ""
echo "Step 1: Copying project to servers..."
echo "=========================================="

for i in "${!SERVERS[@]}"; do
    server="${SERVERS[$i]}"
    echo "[$((i+1))/$NUM_SERVERS] Copying to $server..."
    
    # Create directory on remote
    ssh ${SSH_USER}@${server} "mkdir -p ${REMOTE_DIR}" || {
        echo "ERROR: Could not connect to $server"
        exit 1
    }
    
    # Copy project files
    rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
          --exclude 'logs' --exclude 'data' \
          ${PROJECT_DIR}/ ${SSH_USER}@${server}:${REMOTE_DIR}/
    
    # Copy specific worker config
    scp ${PROJECT_DIR}/configs/remote/worker_${i}.yaml \
        ${SSH_USER}@${server}:${REMOTE_DIR}/configs/worker.yaml
    
    echo "✓ Copied to $server"
done

echo ""
echo "Step 2: Installing dependencies on servers..."
echo "=========================================="

for i in "${!SERVERS[@]}"; do
    server="${SERVERS[$i]}"
    echo "[$((i+1))/$NUM_SERVERS] Installing on $server..."
    
    ssh ${SSH_USER}@${server} "cd ${REMOTE_DIR} && \
        python3 -m venv .venv && \
        source .venv/bin/activate && \
        pip install --upgrade pip && \
        pip install -r requirements.txt && \
        pip install -r requirements_distributed.txt && \
        python -m grpc_tools.protoc -I./proto --python_out=./src/distributed --grpc_python_out=./src/distributed ./proto/mpls_service.proto && \
        mkdir -p logs data" || {
        echo "ERROR: Installation failed on $server"
        exit 1
    }
    
    echo "✓ Installed on $server"
done

echo ""
echo "Step 3: Testing connectivity..."
echo "=========================================="

for i in "${!SERVERS[@]}"; do
    server="${SERVERS[$i]}"
    echo "Testing from ${SERVERS[$i]}:"
    
    # Test ping to other servers
    for j in "${!SERVERS[@]}"; do
        if [ $i -ne $j ]; then
            target="${SERVERS[$j]}"
            ssh ${SSH_USER}@${server} "ping -c 1 -W 2 $target > /dev/null 2>&1" && \
                echo "  ✓ Can reach $target" || \
                echo "  ✗ Cannot reach $target"
        fi
    done
done

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "To start training, run on each server in order:"
echo ""

for i in "${!SERVERS[@]}"; do
    server="${SERVERS[$i]}"
    echo "Server $((i+1)) (${server}):"
    echo "  ssh ${SSH_USER}@${server}"
    echo "  cd ${REMOTE_DIR}"
    echo "  source .venv/bin/activate"
    echo "  python -m src.distributed.worker_runner --config configs/worker.yaml"
    echo ""
done

echo "Or use this automated command:"
echo ""
echo "# Start all workers (run from local machine)"
for i in "${!SERVERS[@]}"; do
    server="${SERVERS[$i]}"
    echo "ssh ${SSH_USER}@${server} 'cd ${REMOTE_DIR} && source .venv/bin/activate && nohup python -m src.distributed.worker_runner --config configs/worker.yaml > worker.log 2>&1 &'"
    [ $i -lt $((NUM_SERVERS-1)) ] && echo "sleep 5"
done

echo ""
echo "To monitor:"
echo "ssh ${SSH_USER}@${SERVERS[0]} 'tail -f ${REMOTE_DIR}/logs/worker_*.log'"
