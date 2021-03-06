import argparse
import json
import os

import git
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import speechset
from config import Config
from disc import Discriminator
from dwg import DiffusionWaveGAN
from speechset.utils.melstft import MelSTFT
from utils.wrapper import TrainingWrapper


class Trainer:
    """TacoSpawn trainer.
    """
    LOG_IDX = 0

    def __init__(self,
                 model: DiffusionWaveGAN,
                 disc: Discriminator,
                 dataset: speechset.VocoderDataset,
                 config: Config,
                 device: torch.device):
        """Initializer.
        Args:
            model: diffusion-wavegan model.
            disc: discriminator.
            dataset: dataset.
            config: unified configurations.
            device: target computing device.
        """
        self.model = model
        self.disc = disc
        self.dataset = dataset
        self.config = config
        # train-test split
        self.testset = self.dataset.split(config.train.split)

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train.batch,
            shuffle=config.train.shuffle,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=config.train.batch,
            collate_fn=self.dataset.collate,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory)

        # training wrapper
        self.wrapper = TrainingWrapper(model, disc, config, device)

        self.optim_g = torch.optim.Adam(
            self.model.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2))

        self.optim_d = torch.optim.Adam(
            self.disc.parameters(),
            config.train.learning_rate,
            (config.train.beta1, config.train.beta2))

        self.train_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = SummaryWriter(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.melspec = MelSTFT(config.data)
        self.cmap = np.array(plt.get_cmap('viridis').colors)

    def train(self, epoch: int = 0):
        """Train wavegrad.
        Args:
            epoch: starting step.
        """
        self.model.train()
        step = epoch * len(self.loader)
        for epoch in tqdm.trange(epoch, self.config.train.epoch):
            with tqdm.tqdm(total=len(self.loader), leave=False) as pbar:
                for it, bunch in enumerate(self.loader):
                    mel, speech = self.wrapper.wrap(
                        self.wrapper.random_segment(bunch))
                    loss_g, losses_g, aux_g = \
                        self.wrapper.loss_generator(mel, speech)
                    # update
                    self.optim_g.zero_grad()
                    loss_g.backward()
                    self.optim_g.step()

                    loss_d, losses_d, aux_d = \
                        self.wrapper.loss_discriminator(mel, speech)
                    # update
                    self.optim_d.zero_grad()
                    loss_d.backward()
                    self.optim_d.step()

                    step += 1
                    pbar.update()
                    pbar.set_postfix({'loss': loss_d.item(), 'step': step})

                    for key, val in {**losses_g, **losses_d}.items():
                        self.train_log.add_scalar(f'loss/{key}', val, step)

                    with torch.no_grad():
                        grad_norm = np.mean([
                            torch.norm(p.grad).item()
                            for p in self.model.parameters() if p.grad is not None])
                        param_norm = np.mean([
                            torch.norm(p).item()
                            for p in self.model.parameters() if p.dtype == torch.float32])

                    self.train_log.add_scalar('common/grad-norm', grad_norm, step)
                    self.train_log.add_scalar('common/param-norm', param_norm, step)
                    self.train_log.add_scalar(
                        'common/learning-rate-g', self.optim_g.param_groups[0]['lr'], step)
                    self.train_log.add_scalar(
                        'common/learning-rate-d', self.optim_d.param_groups[0]['lr'], step)

                    if (it + 1) % (len(self.loader) // 50) == 0:
                        speech = speech[Trainer.LOG_IDX].cpu().numpy()
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/gt', self.mel_img(speech), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/q(z_{t-1}|z_{0})', self.mel_img(aux_d['prev'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/q(z_{t}|z_{t-1})', self.mel_img(aux_d['base'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/p(z_{0}|z_{t})', self.mel_img(aux_d['denoised'][Trainer.LOG_IDX]), step)
                        self.train_log.add_image(
                            # [3, M, T]
                            'train/p(z_{t-1}|z_{0})', self.mel_img(aux_d['pred'][Trainer.LOG_IDX]), step)
                        # scheduler plot
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.plot(aux_g['alphas_bar'])
                        self.train_log.add_figure('train/alphas_bar', fig, step)
            losses = {
                key: [] for key in {**losses_d, **losses_g}}
            with torch.no_grad():
                for bunch in self.testloader:
                    mel, speech = self.wrapper.wrap(
                        self.wrapper.random_segment(bunch))
                    _, losses_g, _ = self.wrapper.loss_generator(mel, speech)
                    _, losses_d, _ = self.wrapper.loss_discriminator(mel, speech)
                    for key, val in {**losses_g, **losses_d}.items():
                        losses[key].append(val)
                # test log
                for key, val in losses.items():
                    self.test_log.add_scalar(f'loss/{key}', np.mean(val), step)

                # wrap last bunch
                mel, speech, mellen, speechlen = bunch
                # [T x H], gt plot
                speech = speech[Trainer.LOG_IDX, :speechlen[Trainer.LOG_IDX]]
                self.test_plot('gt', speech, step)

                # inference
                self.model.eval()
                # [T, mel]
                mel = mel[Trainer.LOG_IDX, :mellen[Trainer.LOG_IDX]]
                # [1, T x H]
                _, [noise, *ir] = self.model(torch.tensor(mel[None], device=device))
                self.model.train()

                # noise plot
                self.test_plot(
                    f'z_{{{self.config.model.steps}}}', noise.squeeze(0), step)
                # intermediate representation
                for i, signal in enumerate(ir[::-1]):
                    self.test_plot(
                        f'p(z_{{{i}}}|z_{{{i + 1}}}))', signal.squeeze(0), step)

            self.model.save(f'{self.ckpt_path}_{epoch}.ckpt', self.optim_g)
            self.disc.save(f'{self.ckpt_path}_{epoch}.ckpt-disc', self.optim_d)

    def mel_img(self, signal: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram images.
        Args:
            signal: [float32; [T x H]], speech signal.
        Returns:
            [float32; [3, M, T]], mel-spectrogram in viridis color map.
        """
        # [T, M]
        mel = self.melspec(signal)
        # minmax norm in range(0, 1)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
        # in range(0, 255)
        mel = (mel * 255).astype(np.uint8)
        # [T, M, 3]
        mel = self.cmap[mel]
        # [3, M, T], make origin lower
        mel = np.flip(mel, axis=1).transpose(2, 1, 0)
        return mel

    def test_plot(self, name: str, signal: np.ndarray, step: int):
        """Plot signals.
        Args:
            name: plot name.
            signal: [np.float32; [T]], audio signal.
            step: current training steps.
        """
        self.test_log.add_image(
            f'test/image/{name}', self.mel_img(signal), step)
        self.test_log.add_audio(
            f'test/audio/{name}', signal[None], step, sample_rate=self.config.data.sr)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-epoch', default=0, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--name', default=None)
    parser.add_argument('--auto-rename', default=False, action='store_true')
    parser.add_argument('--from-dump', default=False, action='store_true')
    args = parser.parse_args()

    # seed setting
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # configurations
    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    if args.name is not None:
        config.train.name = args.name

    log_path = os.path.join(config.train.log, config.train.name)
    # auto renaming
    if args.auto_rename and os.path.exists(log_path):
        config.train.name = next(
            f'{config.train.name}_{i}' for i in range(1024)
            if not os.path.exists(f'{log_path}_{i}'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # prepare datasets
    ljspeech = speechset.utils.DumpDataset(
            speechset.VocoderDataset, args.data_dir) \
        if args.from_dump \
        else speechset.VocoderDataset(
            speechset.datasets.LJSpeech(args.data_dir), config.data)

    # model definition
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DiffusionWaveGAN(config.model)
    model.to(device)

    disc = Discriminator(config.disc)
    disc.to(device)

    trainer = Trainer(model, disc, ljspeech, config, device)

    # loading
    if args.load_epoch > 0:
        # find checkpoint
        ckpt_path = os.path.join(
            config.train.ckpt,
            config.train.name,
            f'{config.train.name}_{args.load_epoch}.ckpt')
        # load checkpoint
        ckpt = torch.load(ckpt_path)
        model.load(ckpt, trainer.optim_g)
        # discriminator checkpoint
        ckpt_disc = torch.load(f'{ckpt_path}-disc')
        disc.load(ckpt_disc, trainer.optim_d)
        print('[*] load checkpoint: ' + ckpt_path)
        # since epoch starts with 0
        args.load_epoch += 1

    # git configuration
    repo = git.Repo()
    config.train.hash = repo.head.object.hexsha
    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    # start train
    trainer.train(args.load_epoch)
