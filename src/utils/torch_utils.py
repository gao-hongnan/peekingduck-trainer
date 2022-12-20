"""Torch Utilities."""
import torch
import matplotlib.pyplot as plt


def load(path: str, device: str = "cpu") -> torch.nn.Module:
    """Loads PyTorch model from path."""
    checkpoint = torch.load(path, map_location=torch.device(device))
    return checkpoint


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.
    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        img = img.detach().numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
########### Dataset Utils ###########

import hashlib

# Reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
