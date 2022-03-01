from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from disc import Discriminator
from dwg import DiffusionWaveGAN


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self,
                 model: DiffusionWaveGAN,
                 disc: Discriminator,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: diffuion-wavegan model.
            disc: discriminator.
            config: training configurations.
            device: torch device.
        """
        self.model = model
        self.disc = disc
        self.config = config
        self.device = device

    def wrap(self, bunch: List[np.ndarray]) -> List[torch.Tensor]:
        """Wrap the array to torch tensor.
        Args:
            bunch: input tensors.
        Returns:
            wrapped.
        """
        return [torch.tensor(array, device=self.device) for array in bunch]

    def random_segment(self, bunch: List[np.ndarray]) -> List[np.ndarray]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                mel: [np.float32; [B, T, mel]], mel-spectrogram.
                speech: [np.float32; [B, T x H]], speech audio signal.
                mellen: [np.long; [B]], spectrogram lengths.
                speechlen: [np.long; [B]], speech lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B, T, mel], [B, T x H], [B], [B]
        mel, speech, mellen, _ = bunch
        # [B]
        start = np.random.randint(mellen - self.config.train.seglen)
        # [B, S, mel]
        mel = np.array(
            [m[s:s + self.config.train.seglen] for m, s in zip(mel, start)])
        # [B]
        start = start * self.config.data.hop
        seglen = self.config.train.seglen * self.config.data.hop
        # [B, S x H]
        speech = np.array(
            [n[s:s + seglen] for n, s in zip(speech, start)])
        return [mel, speech]

    def loss_discriminator(self, mel: torch.Tensor, speech: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the discriminator loss.
        Args:
            mel: [torch.float32; [B, S, M]], segmented spectrogram.
            speech: [torch.float32; [B, S x H]], segmented speech.
        Returns:
            loss and disctionaries.
        """
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (mel.shape[0],), device=mel.device)
        # [B, S x H], [B]
        prev_mean, prev_std = self.model.diffusion(speech, steps - 1)
        # [B, S x H]
        prev = prev_mean + torch.randn_like(prev_mean) * prev_std[:, None]
        # [B, S x H], [B]
        base_mean, base_std = self.model.diffusion(prev, steps, next_=True)
        # [B, S x H]
        base = base_mean + torch.randn_like(base_mean) * base_std[:, None]
        # [B, S x H]
        disc_gt = self.disc(prev, base, steps)
        # []
        loss_d = F.binary_cross_entropy_with_logits(
            disc_gt, torch.ones_like(disc_gt))

        # [B, S x H]
        denoised = self.model.denoise(base, torch.randn_like(base), mel, steps)
        # [B, S x H], [B]
        pred_mean, pred_std = self.model.diffusion(denoised, steps - 1)
        # [B, S x H]
        pred = pred_mean + torch.randn_like(pred_mean) * pred_std[:, None]
        # [B, S x H]
        disc_pred = self.disc(pred, base, steps)
        # []
        loss_g = F.binary_cross_entropy_with_logits(
            disc_pred, torch.zeros_like(disc_pred))
        # least square loss
        loss = loss_d + loss_g
        losses = {
            'dloss': loss.item(),
            'dloss_d': loss_d.item(), 'dloss_g': loss_g.item()}
        return loss, losses, {
            'base': base.cpu().detach().numpy(),
            'prev': prev.cpu().detach().numpy(),
            'denoised': denoised.cpu().detach().numpy(),
            'pred': pred.cpu().detach().numpy()}

    def loss_generator(self, mel: torch.Tensor, speech: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the generator loss.
        Args:
            mel: [torch.float32; [B, S, M]], segmented spectrogram.
            speech: [torch.float32; [B, S x H]], segmented speech.
        Returns:
            loss and disctionaries.
        """
        # [B], zero-based
        steps = torch.randint(
            self.config.model.steps, (mel.shape[0],), device=mel.device)
        # [B, S x H], [B]
        base_mean, base_std = self.model.diffusion(speech, steps)
        # [B, S x H]
        base = base_mean + torch.randn_like(base_mean) * base_std[:, None]
        # [B, S x H]
        denoised = self.model.denoise(base, torch.randn_like(base), mel, steps)
        # [B, S x H], [B]
        pred_mean, pred_std = self.model.diffusion(denoised, steps - 1)
        # [B, S x H]
        pred = pred_mean + torch.randn_like(pred_mean) * pred_std[:, None]
        # [B, S x H]
        disc_pred = self.disc(pred, base, steps)
        # []
        loss = F.binary_cross_entropy_with_logits(
            disc_pred, torch.ones_like(disc_pred))
        losses = {'gloss': loss.item()}
        return loss, losses, {
            'base': base.cpu().detach().numpy(),
            'denoised': denoised.cpu().detach().numpy(),
            'pred': pred.cpu().detach().numpy()}
