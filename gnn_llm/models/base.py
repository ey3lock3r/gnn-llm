import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class GigaModel(nn.Module, ABC):
    """
    Abstract Base Class for GigaGraph Models (v10.0).
    Ensures all algorithms follow the same interface for the Unified Trainer.
    """
    def __init__(self, vocab_size, d_model, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

    @abstractmethod
    def train_step(self, x, y, **kwargs):
        """Perform a single training step (BP, local, or denoising)."""
        pass

    @abstractmethod
    def save_checkpoint(self, path, step):
        """Save model weights and metadata."""
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """Load model weights and return the resumed step."""
        pass
