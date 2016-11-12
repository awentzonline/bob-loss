#!/usr/bin/env python
import numpy as np
from bobloss.gan import GAN


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Bob Loss GAN model')
    GAN.add_to_arg_parser(arg_parser)
    args = arg_parser.parse_args()

    gan = GAN(args)
    gan.load_dataset()
    gan.build_models()
    print('pretraining discriminator')
    gan.train(num_epochs=100, train_generator=False)
    print('training')
    d_losses, g_losses = gan.train()
    for losses in (d_losses, g_losses):
        print('Losses:')
        print(losses[:20])
        print(losses[-20:])
