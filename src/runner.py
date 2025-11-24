import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .config import WorkerConfig
from .model import create_model
from .workers import APPGWorker, MPLSWorker, NetMaxWorker


def run_experiment(mode="MPLS", num_workers=5, rounds=10):
    print(f"\n--- Starting {mode} Experiment ---")

    # 1. Setup Data
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )
        print("Loaded MNIST dataset.")
    except Exception as e:
        print(f"MNIST download failed ({e}). Using random dummy data.")
        dataset = TensorDataset(
            torch.randn(1000, 1, 28, 28), torch.randint(0, 10, (1000,))
        )
        test_dataset = TensorDataset(
            torch.randn(200, 1, 28, 28), torch.randint(0, 10, (200,))
        )

    # 2. Split Data (Fixed Dirichlet Logic)
    indices = [[] for _ in range(num_workers)]
    # extract labels safely from dataset
    labels = np.array([int(y) for _, y in dataset])

    print("Partitioning data (Non-IID)...")
    for k in range(10):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # Dirichlet split
        proportions = np.random.dirichlet(np.repeat(0.5, num_workers))
        proportions = np.array([p for p in proportions])
        proportions = proportions / proportions.sum()

        # Calculate split indices
        split_indices = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = (
            np.split(idx_k, split_indices)
            if len(idx_k)
            else [np.array([], dtype=int) for _ in range(num_workers)]
        )

        for i in range(num_workers):
            indices[i].extend(splits[i])

    # 3. Setup Workers
    workers = []
    criterion = torch.nn.CrossEntropyLoss()
    test_loader = DataLoader(test_dataset, batch_size=1000)

    for i in range(num_workers):
        peers = [((i - 1) % num_workers), ((i + 1) % num_workers)]  # Ring Topology
        bw = {p: np.random.uniform(5.0, 15.0) for p in peers}  # Random Bandwidth MB/s

        # Calculate Data Distribution
        if len(indices[i]) > 0:
            labels_i = labels[indices[i]]
            dist = np.array([np.sum(labels_i == c) for c in range(10)]) / len(labels_i)
        else:
            dist = np.zeros(10)

        conf = WorkerConfig(i, peers, bw, dist)
        model = create_model()

        if mode == "MPLS":
            w = MPLSWorker(conf, model)
        elif mode == "APPG":
            w = APPGWorker(conf, model)
        else:
            w = NetMaxWorker(conf, model)

        loader = None
        if len(indices[i]) > 0:
            loader = DataLoader(
                Subset(dataset, indices[i]), batch_size=32, shuffle=True
            )

        workers.append(
            {
                "worker": w,
                "loader": loader,
                "opt": torch.optim.SGD(model.parameters(), lr=0.01),
            }
        )

    # 4. Training Loop
    history = {"loss": [], "acc": [], "time": []}
    start_time = time.time()

    for r in range(rounds):
        round_loss = 0
        active_workers = 0

        # Local Training
        for w_data in workers:
            if w_data["loader"]:
                loss = w_data["worker"].train_epoch(
                    w_data["loader"], w_data["opt"], criterion
                )
                round_loss += loss
                active_workers += 1

        # Collect Models
        peer_models = {
            i: copy.deepcopy(w["worker"].model) for i, w in enumerate(workers)
        }
        peer_dists = {
            i: w["worker"].config.data_distribution for i, w in enumerate(workers)
        }

        # Aggregation
        for w_data in workers:
            w = w_data["worker"]
            if mode == "MPLS":
                w.update_probabilities(peer_dists, peer_models)
            w.aggregate(peer_models)

        # Evaluate (using worker 0 as representative)
        acc = 0
        with torch.no_grad():
            correct = 0
            total = 0
            model = workers[0]["worker"].model
            model.eval()
            for x, y in test_loader:
                out = model(x)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
            acc = 100 * correct / total

        avg_round_loss = round_loss / max(1, active_workers)
        elapsed = time.time() - start_time
        history["loss"].append(avg_round_loss)
        history["acc"].append(acc)
        history["time"].append(elapsed)

        print(
            f"Round {r + 1}/{rounds} | Loss: {avg_round_loss:.4f} | Acc: {acc:.2f}% | Time: {elapsed:.2f}s"
        )

    return history


if __name__ == "__main__":
    results = {}
    # Run experiments
    results["MPLS"] = run_experiment("MPLS", rounds=15)
    results["APPG"] = run_experiment("APPG", rounds=15)
    results["NetMax"] = run_experiment("NetMax", rounds=15)

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, hist in results.items():
        plt.plot(hist["acc"], label=name)
    plt.title("Accuracy vs Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    for name, hist in results.items():
        plt.plot(hist["time"], hist["acc"], label=name, marker="o")
    plt.title("Accuracy vs Wall-Clock Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("mpls_results.png")
    print("\nResults saved to mpls_results.png")
