from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from dwg.embedder import Embedder

from .config import Config


class Discriminator(nn.Module):
    """Waveform discriminator.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configurations.
        """
        super().__init__()
        self.proj_signal = nn.utils.weight_norm(
            nn.Conv1d(2, config.channels, 1))
        self.embedder = nn.Sequential(
            Embedder(
                config.pe, config.embeddings, config.steps, config.mappers),
            nn.Linear(config.embeddings, config.channels))

        self.disc = nn.Sequential(
            *[
                nn.Sequential(
                    nn.utils.weight_norm(nn.Conv1d(
                        config.channels, config.channels, config.kernels,
                        padding=(config.kernels - 1) * i // 2, dilation=i)),
                    nn.LeakyReLU(config.leak))
                for i in range(1, config.layers + 1)],
            nn.utils.weight_norm(nn.Conv1d(config.channels, 1, 1)))

    def forward(self,
                prev: torch.Tensor,
                signal: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Discriminating inputs.
        Args:
            prev: [torch.float32; [B, T]], x_{t-1}, previous signal.
            signal: [torch.float32; [B, T]], x_t, denoised signal.
            steps: [torch.long; [B]], diffusion steps.
        Returns:
            [torch.float32; [B, T]], pointwise discrminated.
        """
        # [B, C, T]
        x = self.proj_signal(torch.stack([prev, signal], dim=1))
        # [B, T]
        return self.disc(x + self.embedder(steps)[..., None]).squeeze(1)

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict()}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    def load(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])
