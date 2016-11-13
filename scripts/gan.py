#!/usr/bin/env python
import numpy as np
from bobloss.fdcgan import DCGAN


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Bob Loss GAN model')
    DCGAN.add_to_arg_parser(arg_parser)
    args = arg_parser.parse_args()

    model = DCGAN(args)
    model.load_dataset()
    model.build_models()
    if args.num_pretrain_batches:
        print('pretraining discriminator')
        model.train_discriminator(
            args.batch_size, num_batches=args.num_pretrain_batches, verbose=True
        )
        #model.train(num_epochs=args.num_pretrain_batches, train_generator=False)
    print('training')
    d_losses, g_losses = model.train()
    for losses in (d_losses, g_losses):
        print('Losses:')
        print(losses[:20])
        print(losses[-20:])
