class Config:
    """Discriminator configurations.
    """
    def __init__(self, mel: int, steps: int):
        """Initializer.
        Args:
            mel: size of the mel-scale filterbank.
            steps: diffusion steps.
        """
        self.mel = mel
        self.steps = steps

        # embedding
        self.pe = 128
        self.embeddings = 512
        self.mappers = 2

        # upsampler
        self.upkernels = 5
        self.upscales = [4, 4, 4, 4]
        self.leak = 0.2

        # block
        self.channels = 64
        self.kernels = 3
        self.layers = 10
        self.leak = 0.2
