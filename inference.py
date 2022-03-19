import argparse
import os
import json

import librosa
import torch

from config import Config
from dwg import DiffusionWaveGAN
from speechset.utils.melstft import MelSTFT


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None)
parser.add_argument('--ckpt', default=None)
parser.add_argument('--wav', default=None)
args = parser.parse_args()

# load config
with open(args.config) as f:
    config = Config.load(json.load(f))

dwg = DiffusionWaveGAN(config.model)

# load checkpoint
ckpt = torch.load(args.ckpt, map_location='cpu')
dwg.load(ckpt)

device = torch.device('cuda:0')
dwg.to(device)
dwg.eval()

# load wav
wav, _ = librosa.load(args.wav, sr=config.data.sr)

# generate spectrogram
stft = MelSTFT(config.data)
# [T, mel]
spec = torch.tensor(stft(wav), device=device)
with torch.no_grad():
    # [1, T x hop]
    wav, _ = dwg(spec[None])
    # [T x hop]
    wav = wav.squeeze(0).cpu().numpy()

librosa.output.write_wav('output.wav', wav, sr=config.data.sr)
