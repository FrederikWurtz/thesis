"""master.train package

Exports core training utilities from the split modules.
"""
from .trainer_core import DEMDataset, FluidDEMDataset, normalize_inputs
from .checkpoints import save_checkpoint, load_checkpoint
from .cli import main as train_main

__all__ = ['DEMDataset', 'FluidDEMDataset', 'normalize_inputs',
           'save_checkpoint', 'load_checkpoint', 'train_main']
