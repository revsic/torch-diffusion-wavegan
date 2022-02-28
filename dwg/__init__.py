from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import Config
from .embedder import Embedder
from .upsampler import Upsampler
from .wavenet import WaveNetBlock


class DiffusionWaveGAN(nn.Module):
    """Parallel spectrogram-conditioned waveform generation with DiffusionGAN.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configurations.
        """
        super().__init__()
        self.proj_signal = nn.utils.weight_norm(
            nn.Conv1d(1, config.channels, 1))
        self.proj_latent = nn.Sequential(
            nn.Conv1d(1, config.mapchannels, 1), nn.SiLU(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        config.mapchannels, config.mapchannels, config.mapkernels,
                        padding=config.mapkernels // 2),
                    nn.SiLU())
                for _ in range(config.maplayers - 1)],
            nn.Conv1d(config.mapchannels, config.channels, 1))

        self.embedder = Embedder(
            config.pe, config.embeddings, config.steps, config.mappings)

        self.upsampler = Upsampler(
            config.mel, config.upkernels, config.upscales, config.leak)

        self.blocks = nn.ModuleList([
            WaveNetBlock(
                config.channels, config.embeddings, config.mel,
                config.kernels, config.dilations ** j)
            for _ in range(config.cycles)
            for j in range(config.layers)])

        self.proj_out = nn.Sequential(
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(config.channels, config.channels, 1)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Conv1d(config.channels, 1, 1)))

    def denoise(self,
                signal: torch.Tensor,
                latent: torch.Tensor,
                mel: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Denoised waveform conditioned on mel-spectrogram.
        Args:
            signal: [torch.float32; [B, T]], input signal.
            latent: [torch.float32; [B, T]], latent variable.
            mel: [torch.float32; [B, mel, T / prod(scales)]], mel-spectrogram.
            steps: [torch.long; [B]], diffusion steps.
        Returns:
            [torch.float32; [B, T]], denoised waveform. 
        """
        # [B, C, T]
        x = self.proj_signal(signal[:, None]) + self.proj_latent(latent[:, None])
        # [B, mel, T]
        mel = self.upsampler(mel)
        # [B, E]
        embed = self.embedder(steps)
        # L x [B, C, T]
        skips = []
        for block in self.blocks:
            # [B, C, T], [B, C, T]
            x, skip = block(x, embed, mel)
            skips.append(skip)
        # [B, T]
        return self.proj_out(
                torch.sum(skips, dim=1) * (len(self.blocks) ** -0.5)
            ).squeeze(dim=1)
