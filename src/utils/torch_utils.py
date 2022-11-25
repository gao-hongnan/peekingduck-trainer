"""Torch Utilities."""
import torch


def load(path: str, device: str = "cpu") -> torch.nn.Module:
    """Loads PyTorch model from path."""
    checkpoint = torch.load(path, map_location=torch.device(device))
    return checkpoint
