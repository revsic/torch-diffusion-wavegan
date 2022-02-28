# torch-diffusion-wavegan

Parallel waveform generation with DiffusionGAN

- DiffWave: A Versatile Diffusion Model for Audio Synthesis, Zhifeng Kong et al., 2020. [[arXiv:2009.09761](https://arxiv.org/abs/2009.09761)]
- Tackling the Generative Learning Trilemma with Denoising Diffusion GANs, Xiao et al., 2021. [[2112.07804](https://arxiv.org/abs/2112.07804)]

## Requirements

Tested in python 3.7.9 conda environment.

## Usage

Download LJSpeech dataset from [official:keithito](https://keithito.com/LJ-Speech-Dataset/).

To train model, run [train.py](./train.py)

```bash
python -m utils.dump \
    --data-dir /datasets/ljspeech \
    --output-dir /datasets/ljspeech/vocdump \
    --num-proc 8

python train.py \
    --data-dir /datasets/ljspeech/vocdump \
    --from-dump
```
