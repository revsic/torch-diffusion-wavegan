import argparse

from speechset import VocoderDataset, Config
from speechset.datasets import LJSpeech
from speechset.utils import mp_dump

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-proc', default=4, type=int)
    args = parser.parse_args()

    config = Config(batch=None)
    ljspeech = VocoderDataset(LJSpeech(args.data_dir), config)
    # dump
    mp_dump(ljspeech, args.output_dir, args.num_proc)
