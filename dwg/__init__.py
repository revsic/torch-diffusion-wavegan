from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .embedder import Embedder
from .scheduler import Scheduler
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
            nn.utils.weight_norm(nn.Conv1d(config.channels, 1, 1)),
            nn.Tanh())

        self.scheduler = Scheduler(
            config.steps, config.internals, config.logit_min, config.logit_max)

    def forward(self,
                mel: torch.Tensor,
                signal: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Generated waveform conditioned on mel-spectrogram.
        Args:
            mel: [torch.float32; [B, T / prod(scales)], mel], mel-spectrogram.
            signal: [torch.float32; [B, T]], initial noise.
        Returns:
            [torch.float32; [B, T]], generated waveform.
            [np.float32; [B, T]], intermediate representations.
        """
        factor = np.prod(self.upsampler.scales)
        # [B, T]
        signal = signal or torch.randn(
            mel.shape[0], mel.shape[1] * factor, device=mel.device)
        # S x [B, T]
        ir = [signal.cpu().detach().numpy()]
        # zero-based step
        for step in range(self.steps - 1, -1, -1):
            # [B, T], [B]
            mean, std = self.inverse(
                signal, mel, torch.tensor([step], device=mel.device))
            # [B, T]
            signal = mean + torch.randn_like(mean) * std[:, None]
            ir.append(signal.cpu().detach().numpy())
        # [B, T]
        return signal, ir

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
        # [S + 1]
        logsnr, betas = self.scheduler()
        if next_:
            # [B], one-based sample
            beta = betas[steps + 1]
            # [B, T], [B]
            return (1. - beta[:, None]).sqrt() * signal, beta.sqrt()
        # [S + 1]
        alphas_bar = torch.sigmoid(logsnr)
        # [B], one-based sample
        alpha_bar = alphas_bar[steps + 1]
        # [B, T], [B]
        return alpha_bar[:, None].sqrt() * signal, (1 - alpha_bar).sqrt()

    def inverse(self,
                signal: torch.Tensor,
                mel: torch.Tensor,
                steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse process, single step denoise.
        Args:
            signal: [torch.float32; [B, T]], input signal, z_{t}.
            mel: [torch.float32; [B, T / prod(scales), mel]], mel-spectrogram.
            steps: [torch.long; [B]], t, diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, T]], waveform mean, z_{t - 1}
            [torch.float32; [B]], waveform std.
        """
        # [S + 1]
        logsnr, betas = self.scheduler()
        # [S + 1]
        alphas, alphas_bar = 1. - betas, torch.sigmoid(logsnr)
        # [B, T]
        denoised = self.denoise(signal, mel, steps)
        # [B], make one-based
        prev, steps = steps, steps + 1
        # [B, T]
        mean = alphas_bar[prev, None].sqrt() * betas[steps, None] / (
                1 - alphas_bar[steps, None]) * denoised + \
            alphas[steps, None].sqrt() * (1. - alphas_bar[prev, None]) / (
                1 - alphas_bar[steps, None]) * signal
        # [B]
        var = (1 - alphas_bar[prev]) / (1 - alphas_bar[steps]) * betas[steps]
        return mean, var.sqrt()

    def denoise(self,
                signal: torch.Tensor,
                mel: torch.Tensor,
                steps: torch.Tensor) -> torch.Tensor:
        """Denoise waveform conditioned on mel-spectrogram.
        Args:
            signal: [torch.float32; [B, T]], input signal.
            mel: [torch.float32; [B, T / prod(scales), mel]], mel-spectrogram.
            steps: [torch.long; [B]], diffusion steps, zero-based.
        Returns:
            [torch.float32; [B, T]], denoised waveform. 
        """
        # [B, C, T]
        x = self.proj_signal(signal[:, None])
        # [B, mel, T]
        mel = self.upsampler(mel.transpose(1, 2))
        # [B, E]
        embed = self.embedder(steps)
        # L x [B, C, T]
        skips = 0.
        for block in self.blocks:
            # [B, C, T], [B, C, T]
            x, skip = block(x, embed, mel)
            skips = skips + skip
        # [B, T]
        return self.proj_out(
            skips * (len(self.blocks) ** -0.5)).squeeze(dim=1)

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
