import numpy as np
import torch
import torch.nn as nn


class WorkerConfig:
    """Configuration for a DFL Worker"""

    def __init__(
        self, worker_id, peers, bandwidth, data_distribution, compute_speed=1.0
    ):
        self.worker_id = worker_id
        self.peers = peers
        self.bandwidth = bandwidth  # Map: peer_id -> speed (MB/s)
        self.data_distribution = data_distribution
        self.compute_speed = compute_speed


def get_layer_params(layer: nn.Module) -> torch.Tensor:
    """Flatten layer parameters into a 1D tensor."""
    params = []
    for p in layer.parameters():
        params.append(p.data.view(-1))
    if not params:
        return torch.tensor([])
    return torch.cat(params)


def set_layer_params(layer: nn.Module, params: torch.Tensor):
    """Restore layer parameters from a 1D tensor."""
    if params.numel() == 0:
        return
    offset = 0
    for p in layer.parameters():
        numel = p.numel()
        if offset + numel > params.numel():
            break
        p.data.copy_(params[offset : offset + numel].view_as(p))
        offset += numel


def extract_layers(model: nn.Module):
    """Extract learnable layers (Conv2d, Linear) for MPLS."""
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append(module)
    return layers
