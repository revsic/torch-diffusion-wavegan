from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dwg.embedder import Embedder
from dwg.upsampler import Upsampler

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
            nn.Conv1d(2, config.channels * 2, 1))
        self.embedder = Embedder(
                config.pe, config.embeddings, config.steps, config.mappers)

        self.upsampler = Upsampler(
            config.mel, config.upkernels, config.upscales, config.leak)

        self.disc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.embeddings, config.channels * 2),
                nn.utils.weight_norm(nn.Conv1d(
                    config.mel, config.channels, 1)),
                nn.utils.weight_norm(nn.Conv1d(
                    config.channels * 2, config.channels * 2, config.kernels,
                    padding=(config.kernels - 1) * i // 2, dilation=i, groups=2))])
            for i in range(1, config.layers + 1)])

        self.proj_out = nn.utils.weight_norm(nn.Conv1d(
            config.channels * 2, 2, 1, groups=2))

    def forward(self,
                prev: torch.Tensor,
                signal: torch.Tensor,
                mel: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Discriminating inputs.
        Args:
            prev: [torch.float32; [B, T]], x_{t-1}, previous signal.
            signal: [torch.float32; [B, T]], x_t, denoised signal.
            mel: [torch.float32; [B, T // hop, mel]], mel spectrogram.
            steps: [torch.long; [B]], diffusion steps.
        Returns:
            [torch.float32; [B, T]], pointwise discrminated.
        """
        # [B, C x 2, T]
        x = self.proj_signal(torch.stack([prev, signal], dim=1))
        # [B, E]
        embed = self.embedder(steps)
        # [B, mel, T]
        mel = self.upsampler(mel.transpose(1, 2))
        for proj_embed, proj_mel, conv in self.disc:
            # [B, C, T], unconditional 2-group conv
            c, u = conv(x + proj_embed(embed)[..., None]).chunk(2, dim=1)
            # [B, C x 2, T], half condition
            x = F.leaky_relu(torch.cat([c + proj_mel(mel), u], dim=1), self.leak)
        # [B, 2, T]
        return self.proj_out(x)

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
