from __future__ import annotations

import random
import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed the relevant random number generators.

    TODO: complete this function so that Python, NumPy and PyTorch are
    controlled. Improve CUDA/cuDNN repeatability when CUDA is available.
    """
    random.seed(seed)
    # TODO: NumPy
    # TODO: PyTorch CPU
    # TODO: CUDA, if available
    # TODO: cuDNN deterministic / benchmark flags


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""
    # TODO: implement worker seeding for NumPy and Python random.
    pass


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded torch.Generator used for deterministic shuffling."""
    # TODO
    return torch.Generator()
