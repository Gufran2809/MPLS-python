# MPLS Distributed System Architecture

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "Distributed MPLS System"
        subgraph "Server 1 (10.96.0.255)"
            W1[Worker 0 Process]
            W1 --> T1[Training Thread]
            W1 --> A1[Aggregation Thread]
            W1 --> G1[gRPC Server:50051]
            W1 --> P1[Parameter Store]
            T1 -.->|Write Model| P1
            A1 -.->|Read/Update| P1
        end
        
        subgraph "Server 2 (10.96.0.87)"
            W2[Worker 1 Process]
            W2 --> T2[Training Thread]
            W2 --> A2[Aggregation Thread]
            W2 --> G2[gRPC Server:50051]
            W2 --> P2[Parameter Store]
            T2 -.->|Write Model| P2
            A2 -.->|Read/Update| P2
        end
        
        subgraph "Server 3 (10.96.0.62)"
            W3[Worker 2 Process]
            W3 --> T3[Training Thread]
            W3 --> A3[Aggregation Thread]
            W3 --> G3[gRPC Server:50051]
            W3 --> P3[Parameter Store]
            T3 -.->|Write Model| P3
            A3 -.->|Read/Update| P3
        end
    end
    
    G1 <-->|gRPC: Layer Exchange| G2
    G2 <-->|gRPC: Layer Exchange| G3
    G3 <-->|gRPC: Layer Exchange| G1
    
    style W1 fill:#e1f5ff
    style W2 fill:#e1f5ff
    style W3 fill:#e1f5ff
    style G1 fill:#ffe1e1
    style G2 fill:#ffe1e1
    style G3 fill:#ffe1e1
```

## 2. Single Worker Internal Architecture

```mermaid
graph TB
    subgraph "Worker Node"
        subgraph "Configuration"
            CFG[Worker Config<br/>- ID, Peers<br/>- MPLS params<br/>- Data partition]
        end
        
        subgraph "Data Layer"
            DATA[Local Dataset<br/>MNIST/CIFAR<br/>Non-IID Partition]
            LOADER[DataLoader<br/>Batch: 32]
            DATA --> LOADER
        end
        
        subgraph "Training Thread"
            TRAIN[Training Loop]
            FWD[Forward Pass]
            BACK[Backward Pass]
            OPT[SGD Optimizer]
            
            TRAIN --> FWD --> BACK --> OPT
            OPT -->|Update| MODEL
        end
        
        subgraph "Model & Storage"
            MODEL[Neural Network<br/>SimpleMLP<br/>784→64→10]
            PARAMS[Parameter Store<br/>Thread-Safe<br/>Double Buffer]
            
            MODEL -.->|Clone| PARAMS
        end
        
        subgraph "Aggregation Thread"
            AGG_LOOP[Aggregation Loop]
            PEER_SEL[Peer Selection<br/>Bandwidth + Divergence]
            LAYER_SEL[Layer Selection<br/>Gradient Based]
            LIST_SCHED[List Scheduling<br/>Algorithm 1]
            PULL[Pull Layers<br/>Parallel from Peers]
            AVERAGE[Aggregate Layers<br/>Weighted Average]
            
            AGG_LOOP --> PEER_SEL --> LAYER_SEL --> LIST_SCHED
            LIST_SCHED --> PULL --> AVERAGE
            AVERAGE -->|Update| PARAMS
        end
        
        subgraph "Network Layer"
            GRPC_SRV[gRPC Server<br/>Serve Layer Requests]
            GRPC_CLI[gRPC Client<br/>Pull from Peers]
            NET_MGR[Network Manager<br/>- Connections<br/>- Bandwidth Monitor<br/>- Serialization]
            
            PULL --> GRPC_CLI
            GRPC_CLI --> NET_MGR
            NET_MGR --> GRPC_SRV
        end
        
        LOADER --> TRAIN
        CFG --> PEER_SEL
        CFG --> LAYER_SEL
        
        PARAMS -.->|Read| GRPC_SRV
        PARAMS -.->|Latest Model| TRAIN
    end
    
    EXT1[Peer 1<br/>gRPC] <-->|Layer Exchange| NET_MGR
    EXT2[Peer 2<br/>gRPC] <-->|Layer Exchange| NET_MGR
    
    style TRAIN fill:#d4f1d4
    style AGG_LOOP fill:#ffd4d4
    style NET_MGR fill:#d4d4ff
    style MODEL fill:#fff4d4
```

## 3. MPLS Algorithms Flow

```mermaid
graph LR
    subgraph "MPLS Core Algorithms"
        START[Aggregation Round] --> UPDATE_META[Update Peer Metadata]
        
        UPDATE_META --> PEER_ALG[Peer Selection Algorithm]
        
        subgraph "Peer Selection (Eq. 4)"
            PEER_ALG --> BW[Measure Bandwidth<br/>B_s for each peer]
            PEER_ALG --> DIV[Compute Data Divergence<br/>DD_s for each peer]
            BW --> NORM1[Normalize: B̄_s]
            DIV --> NORM2[Normalize: D̄D_s]
            NORM1 --> COMBINE[Score = τ₁·B̄_s + τ₂·D̄D_s]
            NORM2 --> COMBINE
            COMBINE --> PROB_P[Peer Probabilities p_s]
        end
        
        PROB_P --> LAYER_ALG[Layer Selection Algorithm]
        
        subgraph "Layer Selection (Eq. 6)"
            LAYER_ALG --> GRAD[Gradient Variation<br/>g_s^(t',t)(l)]
            GRAD --> NORM3[Normalize per layer]
            NORM3 --> PROB_L[Layer Probabilities q_s(l)]
        end
        
        PROB_L --> SCHED_ALG[List Scheduling Algorithm]
        
        subgraph "List Scheduling (Alg. 1)"
            SCHED_ALG --> INIT[Initialize μ₁, μ₂ matrices]
            INIT --> EFFICIENCY[Compute Efficiency E(s,l)]
            EFFICIENCY --> RANK[Rank Peers by Σ E(s,l)]
            RANK --> ASSIGN1[Phase 1: Assign each layer<br/>to best peer]
            ASSIGN1 --> ASSIGN2[Phase 2: Fill gaps<br/>without exceeding max load]
            ASSIGN2 --> STRATEGY[Aggregation Strategy<br/>π: layer → peer]
        end
        
        STRATEGY --> PULL_EXEC[Execute: Pull Layers<br/>in Parallel]
        PULL_EXEC --> AGG_EXEC[Aggregate Layers]
        
        subgraph "Model Aggregation (Eq. 1)"
            AGG_EXEC --> AVG[w_i^(k+1)(l) = <br/>(Σ y_s^k(l)·w_s^k(l) + w_i^k(l))<br/>/ (Σ y_s^k(l) + 1)]
        end
        
        AVG --> END[Updated Model]
    end
    
    style PEER_ALG fill:#e3f2fd
    style LAYER_ALG fill:#f3e5f5
    style SCHED_ALG fill:#e8f5e9
    style AGG_EXEC fill:#fff3e0
```

## 4. Communication Protocol

```mermaid
sequenceDiagram
    participant W0 as Worker 0
    participant W1 as Worker 1
    participant W2 as Worker 2
    
    Note over W0,W2: Initialization Phase
    W0->>W0: Start gRPC Server (50051)
    W1->>W1: Start gRPC Server (50051)
    W2->>W2: Start gRPC Server (50051)
    
    W0->>W1: Connect (gRPC Channel)
    W0->>W2: Connect (gRPC Channel)
    W1->>W0: Connect (gRPC Channel)
    W1->>W2: Connect (gRPC Channel)
    W2->>W0: Connect (gRPC Channel)
    W2->>W1: Connect (gRPC Channel)
    
    Note over W0,W2: Training Phase (Continuous)
    
    loop Every Iteration
        par Training (Parallel)
            W0->>W0: Train on Local Data
            W1->>W1: Train on Local Data
            W2->>W2: Train on Local Data
        end
        
        Note over W0,W2: Aggregation Phase
        
        W0->>W0: Run List Scheduling
        W0->>W0: Strategy: Pull L1,L2 from W1, L3 from W2
        
        par Layer Pulling
            W0->>W1: PullLayers([L1, L2])
            W1-->>W0: LayerResponse(L1, L2)
            
            W0->>W2: PullLayers([L3])
            W2-->>W0: LayerResponse(L3)
        end
        
        W0->>W0: Aggregate Layers
        W0->>W0: Update Model
        
        par Serve Peer Requests
            W1->>W0: PullLayers([...])
            W0-->>W1: LayerResponse([...])
            
            W2->>W0: PullLayers([...])
            W0-->>W2: LayerResponse([...])
        end
    end
```

## 5. Data Flow Architecture

```mermaid
graph TB
    subgraph "Data Pipeline"
        DS[Dataset: MNIST/CIFAR<br/>60,000 samples]
        
        subgraph "Non-IID Partitioning"
            PART[Dirichlet Distribution<br/>α = 0.5]
            DS --> PART
            PART --> D0[Worker 0 Data<br/>~20,000 samples<br/>Skewed to classes 0,1,2]
            PART --> D1[Worker 1 Data<br/>~20,000 samples<br/>Skewed to classes 3,4,5]
            PART --> D2[Worker 2 Data<br/>~20,000 samples<br/>Skewed to classes 6,7,8,9]
        end
        
        D0 --> DL0[DataLoader 0<br/>Batch: 32]
        D1 --> DL1[DataLoader 1<br/>Batch: 32]
        D2 --> DL2[DataLoader 2<br/>Batch: 32]
        
        DL0 --> T0[Training Thread 0]
        DL1 --> T1[Training Thread 1]
        DL2 --> T2[Training Thread 2]
        
        T0 --> M0[Model 0]
        T1 --> M1[Model 1]
        T2 --> M2[Model 2]
    end
    
    subgraph "Model Exchange"
        M0 <-->|Layers| M1
        M1 <-->|Layers| M2
        M2 <-->|Layers| M0
    end
    
    subgraph "Convergence"
        M0 --> CONV[Global Model<br/>Converges to<br/>All 10 Classes]
        M1 --> CONV
        M2 --> CONV
    end
    
    style D0 fill:#ffebee
    style D1 fill:#e3f2fd
    style D2 fill:#e8f5e9
    style CONV fill:#fff9c4
```

## 6. Network Topology (Ring)

```mermaid
graph TB
    subgraph "Ring Topology"
        W0[Worker 0<br/>10.96.0.255:50051<br/>Peers: 1, 2]
        W1[Worker 1<br/>10.96.0.87:50051<br/>Peers: 0, 2]
        W2[Worker 2<br/>10.96.0.62:50051<br/>Peers: 0, 1]
        
        W0 <-->|BW: 10-15 MB/s| W1
        W1 <-->|BW: 10-15 MB/s| W2
        W2 <-->|BW: 10-15 MB/s| W0
    end
    
    style W0 fill:#e1f5ff
    style W1 fill:#ffe1e1
    style W2 fill:#e1ffe1
```

## 7. Thread Interaction & Synchronization

```mermaid
stateDiagram-v2
    [*] --> Initializing
    
    Initializing --> ReadyToStart: Load Config & Data
    ReadyToStart --> Running: Start Threads
    
    state Running {
        [*] --> TrainingActive
        [*] --> AggregationActive
        
        state TrainingActive {
            [*] --> LoadBatch
            LoadBatch --> Forward
            Forward --> Backward
            Backward --> UpdateWeights
            UpdateWeights --> WriteParams: thread-safe write
            WriteParams --> LoadBatch
        }
        
        state AggregationActive {
            [*] --> Wait: Sleep interval
            Wait --> ReadParams: thread-safe read
            ReadParams --> ComputeStrategy
            ComputeStrategy --> PullLayers: gRPC calls
            PullLayers --> AggregateLayers
            AggregateLayers --> WriteParams: thread-safe write
            WriteParams --> IncrementIter
            IncrementIter --> Wait
        }
    }
    
    Running --> Stopping: Max iterations reached
    Stopping --> Cleanup: Join threads
    Cleanup --> [*]
```

## 8. File System Architecture

```
mpls_dfl_python/
│
├── src/                          # Source code
│   ├── distributed/              # Distributed implementation
│   │   ├── worker.py            # DistributedWorker class
│   │   ├── network.py           # gRPC communication
│   │   ├── config_loader.py     # Configuration management
│   │   ├── worker_runner.py     # Standalone runner
│   │   └── mpls_service_pb2.py  # Generated protobuf
│   ├── workers.py               # MPLS/APPG/NetMax algorithms
│   ├── model.py                 # Neural network models
│   ├── config.py                # Base configuration
│   └── runner.py                # Simulation runner
│
├── proto/
│   └── mpls_service.proto       # gRPC service definition
│
├── scripts/
│   ├── generate_configs.py      # Config generator
│   ├── run_distributed.py       # Multi-process runner
│   └── deploy_remote.sh         # Remote deployment
│
├── configs/
│   ├── remote/                  # Remote server configs
│   │   ├── worker_0.yaml       # Server 1 config
│   │   ├── worker_1.yaml       # Server 2 config
│   │   └── worker_2.yaml       # Server 3 config
│   └── example_*.yaml          # Example configs
│
├── logs/                        # Worker logs (runtime)
├── data/                        # MNIST/CIFAR datasets (runtime)
├── Dockerfile                   # Container image
├── docker-compose.yml           # Multi-container orchestration
└── requirements*.txt            # Dependencies
```

## 9. Deployment Architecture

```mermaid
graph TB
    subgraph "Multi-Server Deployment"
        subgraph "Server 1: 10.96.0.255"
            S1_OS[Linux OS]
            S1_OS --> S1_PY[Python 3.10 + venv]
            S1_PY --> S1_CODE[Project Code]
            S1_CODE --> S1_W[Worker 0 Process]
            S1_W --> S1_PORT[Port 50051]
        end
        
        subgraph "Server 2: 10.96.0.87"
            S2_OS[Linux OS]
            S2_OS --> S2_PY[Python 3.10 + venv]
            S2_PY --> S2_CODE[Project Code]
            S2_CODE --> S2_W[Worker 1 Process]
            S2_W --> S2_PORT[Port 50051]
        end
        
        subgraph "Server 3: 10.96.0.62"
            S3_OS[Linux OS]
            S3_OS --> S3_PY[Python 3.10 + venv]
            S3_PY --> S3_CODE[Project Code]
            S3_CODE --> S3_W[Worker 2 Process]
            S3_W --> S3_PORT[Port 50051]
        end
    end
    
    subgraph "Network Communication"
        NET[TCP/IP Network<br/>gRPC Protocol]
    end
    
    S1_PORT <--> NET
    S2_PORT <--> NET
    S3_PORT <--> NET
    
    subgraph "Local Development"
        DEV[Developer Machine]
        DEV -->|Deploy Script| S1_OS
        DEV -->|Deploy Script| S2_OS
        DEV -->|Deploy Script| S3_OS
        DEV -->|Monitor Logs| S1_W
    end
    
    style DEV fill:#fff9c4
    style NET fill:#e1f5ff
```

## Key Components Summary

| Component | Description | Location |
|-----------|-------------|----------|
| **DistributedWorker** | Main worker class with async threads | `src/distributed/worker.py` |
| **NetworkManager** | gRPC client/server management | `src/distributed/network.py` |
| **MPLS Algorithms** | Peer/layer selection, list scheduling | `src/distributed/worker.py:350-489` |
| **gRPC Service** | Protocol definitions | `proto/mpls_service.proto` |
| **Config System** | YAML-based configuration | `src/distributed/config_loader.py` |
| **Deployment** | Remote server deployment | `scripts/deploy_remote.sh` |

## Technology Stack

- **Communication**: gRPC (Protocol Buffers)
- **Deep Learning**: PyTorch
- **Data**: torchvision (MNIST, CIFAR)
- **Serialization**: NumPy + Protobuf
- **Configuration**: PyYAML
- **Deployment**: Python multiprocessing / Docker
- **Monitoring**: File-based logging
