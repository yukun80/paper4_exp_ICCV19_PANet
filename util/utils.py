"""Utility helpers for experiment reproducibility."""

import random

import torch


def set_seed(seed: int) -> None:
    """Set the random seeds for Python and PyTorch backends."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
