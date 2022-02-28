class Config:
    """Discriminator configurations.
    """
    def __init__(self, steps: int):
        """Initializer.
        Args:
            steps: diffusion steps.
        """
        self.steps = steps

        # embedding
        self.pe = 128
        self.embeddings = 512
        self.mappers = 2

        # block
        self.channels = 64
        self.kernels = 3
        self.layers = 10
        self.leak = 0.2
