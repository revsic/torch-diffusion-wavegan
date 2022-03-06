import numpy as np


class Config:
    """DiffusionWaveGAN configurations.
    """
    def __init__(self, mel: int):
        """Initializer.
        Args:
            mel: spectrogram channels.
        """
        self.mel = mel

        # diffusion steps
        self.steps = 4

        # schedules
        self.internals = 1024
        self.logit_max = 10
        self.logit_min = -10

        # block
        self.channels = 64
        self.kernels = 3
        self.dilations = 2

        # embedder
        self.pe = 128
        self.embeddings = 512
        self.mappings = 2

        # latent mapper
        self.mapchannels = 8
        self.mapkernels = 5
        self.maplayers = 2

        # upsampler
        self.upkernels = 5
        self.upscales = [4, 4, 4, 4]
        self.leak = 0.2

        # wavenet
        self.cycles = 3
        self.layers = 10

    def betas(self) -> np.ndarray:
        """Beta values.
        """
        steps = np.arange(1, self.steps + 1)
        # [S]
        betas = 1 - np.exp(
            -self.beta_min / self.steps - 0.5 * (
                self.beta_max - self.beta_min
            ) * (2 * steps - 1) * self.steps ** -2)
        # [S + 1]
        return np.concatenate([[0.], betas])
