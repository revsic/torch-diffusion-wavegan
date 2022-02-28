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
