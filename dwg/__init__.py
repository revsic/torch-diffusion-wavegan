from typing import Any, Dict, Optional, Tuple

import numpy as np
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
        self.steps = config.steps
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
        # [S + 1], 0 ~ S
        self.register_buffer('betas', config.betas())
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim=-1))

    def forward(self,
                mel: torch.Tensor,
                signal: Optional[torch.Tensor] = None,
                latent: Optional[torch.Tensor] = None,
                sample: bool = True) -> torch.Tensor:
        """Generated waveform conditioned on mel-spectrogram.
        Args:
            mel: [torch.float32; [B, T / prod(scales)], mel], mel-spectrogram.
            signal: [torch.float32; [B, T]], initial noise.
            latent: [torch.float32; [B, T]], provided latent variable.
            sample: whether sample the inverse process or not.
        Returns:
            [torch.float32; [B, T]], generated waveform.
        """
        # [B, T]
        signal = signal or torch.randn(
            mel.shape[0], mel.shape[-1] * np.prod(self.config.upscales),
            device=mel.device)
        latent = latent or torch.randn_like(signal)
        # zero-based step
        for step in range(self.steps - 1, -1, -1):
            # [B, T], [B]
            mean, std = self.inverse(
                signal, latent, mel, torch.tensor([step], device=mel.device))
            # [B, T]
            signal = mean + torch.randn_like(mean) * std[:, None] \
                if sample else mean
        # [B, T]
        return signal

    def diffusion(self,
                  signal: torch.Tensor,
                  steps: torch.Tensor,
                  next_: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion process.
        Args:
            signal: [torch.float32; [B, T]], input signal.
            steps: [torch.long; [B]], t, target diffusion steps, zero-based.
            next_: whether move single steps or multiple steps.
                if next_, signal is z_{t - 1}, otherwise signal is z_0.
        Returns:
            [torch.float32; [B, T]], z_{t}, diffused mean.
            [torch.float32; [B]], standard deviation.
        """
        if next_:
            # [B], one-based sample
            beta = self.betas[steps + 1]
            # [B, T], [B]
            return (1. - beta[None]).sqrt() * signal, beta.sqrt()
        # [B], one-based sample
        alpha_bar = self.alphas_bar[steps + 1]
        # [B, T], [B]
        return alpha_bar.sqrt() * signal, (1 - alpha_bar).sqrt()

    def inverse(self,
                signal: torch.Tensor,
                latent: torch.Tensor,
                mel: torch.Tensor,
                steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse process, single step denoise.
        Args:
            signal: [torch.float32; [B, T]], input signal, z_{t}.
            latent: [torch.float32; [B, T]], latent variable.
            mel: [torch.float32; [B, T / prod(scales), mel]], mel-spectrogram.
            steps: [torch.long; [B]], t, diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, T]], waveform mean, z_{t - 1}
            [torch.float32; [B]], waveform std.
        """
        # [B, T]
        denoised = self.denoise(signal, latent, mel, steps)
        # [B], make one-based
        prev, steps = steps, steps + 1
        # [B, T]
        mean = self.alphas_bar[prev].sqrt() * self.betas[steps] / (
                1 - self.alphas_bar[steps]) * denoised + \
            self.alphas[steps].sqrt() * (1. - self.alphas_bar[prev]) / (
                1 - self.alphas_bar[steps]) * signal
        # [B]
        var = (1 - self.alphas_bar[prev]) / (
            1 - self.alphas_bar[steps]) * self.betas[steps]
        return mean, var.sqrt()

    def denoise(self,
                signal: torch.Tensor,
                latent: torch.Tensor,
                mel: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Denoise waveform conditioned on mel-spectrogram.
        Args:
            signal: [torch.float32; [B, T]], input signal.
            latent: [torch.float32; [B, T]], latent variable.
            mel: [torch.float32; [B, T / prod(scales), mel]], mel-spectrogram.
            steps: [torch.long; [B]], diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, T]], denoised waveform. 
        """
        # [B, C, T]
        x = self.proj_signal(signal[:, None]) + self.proj_latent(latent[:, None])
        # [B, mel, T]
        mel = self.upsampler(mel.transpose(1, 2))
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
