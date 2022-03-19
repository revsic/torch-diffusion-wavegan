from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.leak = config.leak
        self.proj_signal = nn.utils.weight_norm(
            nn.Conv1d(2, config.channels, 1))
        self.embedder = Embedder(
                config.pe, config.embeddings, config.steps, config.mappers)

        self.disc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.embeddings, config.channels),
                nn.utils.weight_norm(nn.Conv1d(
                    config.channels, config.channels, config.kernels,
                    padding=(config.kernels - 1) * i // 2, dilation=i))])
            for i in range(1, config.layers + 1)])
        
        self.proj_out = nn.utils.weight_norm(nn.Conv1d(config.channels, 1, 1))

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
        # [B, E]
        embed = self.embedder(steps)
        for proj_embed, conv in self.disc:
            # [B, C, T]
            x = F.leaky_relu(conv(x + proj_embed(embed)[..., None]), self.leak)
        # [B, T]
        return self.proj_out(x).squeeze(1)

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
