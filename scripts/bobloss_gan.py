#!/usr/bin/env python
import numpy as np
from bobloss.dcgansr import DCGANSR
from keras.datasets import mnist
from keras.layers import (
    Activation, BatchNormalization, Convolution2D, Deconvolution2D, Dense,
    Dropout, Flatten, GlobalAveragePooling2D, Input, LeakyReLU, Permute,
    Reshape, UpSampling2D
)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import activity_l1, activity_l2, l1, l2


class BobLossGAN(DCGANSR):
    def _load_dataset(self):
        data = np.load(
            self.config.dataset
        )
        for name in ('frame_data', 'frame_ids', 'episode_ids'):
            setattr(self, name, data[name])
        #self.frame_data = (self.frame_data / 128.) - 1.
        self.frame_data = self.frame_data / 256.

    def _build_models(self):
        # generator
        activation = 'relu'
        kernel_size = 5
        max_channels = 1024
        cnn_layers = ((max_channels // 2, 3, 3), (max_channels // 4, 3, 3), (max_channels // 8, 3, 3))
        cnn_layers = ((max_channels // 2, 3, 3), (max_channels // 4, 3, 3))
        num_cnn_layers = len(cnn_layers)
        scale = 2 ** num_cnn_layers
        img_height, img_width, img_channels = self.frame_shape
        generator_input = Input(shape=(self.config.generator_input_dim,))
        x = Dense(
            max_channels * (img_height // scale) * (img_width // scale)
        )(generator_input)
        x = BatchNormalization(mode=2)(x)
        x = Activation(activation)(x)
        x = Reshape((img_height // scale, img_width // scale, max_channels))(x)
        for channels, w, h in cnn_layers:
            x = UpSampling2D((2, 2))(x)
            x = Convolution2D(channels, kernel_size, kernel_size, border_mode='same')(x)
            x = batchnorm_tf(x)
            x = Activation(activation)(x)
            x = Convolution2D(channels, 1, 1, border_mode='same')(x)
            x = batchnorm_tf(x)
            x = Activation(activation)(x)
        x = Convolution2D(img_channels, 1, 1, activation='sigmoid')(x)
        generator = Model(generator_input, x)
        generator_optimizer = Adam(lr=1e-4)
        generator.compile(
            loss='categorical_crossentropy',
            optimizer=generator_optimizer
        )
        generator.summary()

        # discriminator
        max_channels = 1024
        #cnn_layers = ((max_channels // 8, 3, 3), (max_channels // 4, 3, 3), (max_channels // 2, 3, 3))
        cnn_layers = ((max_channels // 4, 3, 3), (max_channels // 2, 3, 3))
        num_cnn_layers = len(cnn_layers)
        print self.frame_shape
        x = discriminator_input = Input(shape=self.frame_shape)
        for channels, w, h in cnn_layers:
            x = Convolution2D(
                channels, kernel_size, kernel_size, subsample=(2, 2), border_mode='same'
            )(x)
            x = batchnorm_tf(x)
            if activation == 'tanh':
                x = Activation('tanh')(x)
            else:
                x = LeakyReLU(0.2)(x)
            x = Convolution2D(channels, 1, 1, border_mode='same')(x)
            x = batchnorm_tf(x)
            if activation == 'tanh':
                x = Activation('tanh')(x)
            else:
                x = LeakyReLU(0.2)(x)
            x = Dropout(self.config.dropout)(x)
        x = Flatten()(x)
        x = Dense(max_channels)(x)
        # x = Convolution2D(max_channels, 1, 1, border_mode='same')(x)
        # x = batchnorm_tf(x)
        if activation == 'tanh':
            x = Activation('tanh')(x)
        else:
            x = LeakyReLU(0.2)(x)
        x = Dropout(self.config.dropout)(x)
        # x = Convolution2D(2, 1, 1, border_mode='same')(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Activation('softmax')(x)
        x = Dense(2, activation='softmax')(x)
        discriminator = Model(discriminator_input, x)
        discriminator_optimizer = Adam(lr=1e-4)
        discriminator.compile(
            loss='categorical_crossentropy',
            optimizer=discriminator_optimizer
        )

        def make_trainable(net, val):
            net.trainable = val
            for l in net.layers:
               l.trainable = val
        # end-to-end model for training generator
        make_trainable(discriminator, False)
        gan_input = Input(shape=(self.config.generator_input_dim,))
        g = generator(gan_input)
        d = discriminator(g)
        gan = Model(gan_input, d)
        gan.compile(
            loss='categorical_crossentropy',
            optimizer=generator_optimizer
        )
        gan.summary()
        make_trainable(discriminator, True)
        discriminator.summary()
        return generator, discriminator, gan


def batchnorm_tf(x):
    # work-around apparent theano/tf-dim_ordering bug
    x = Permute((2, 3, 1))(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Permute((3, 1, 2))(x)
    return x


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Bob Loss GAN model')
    BobLossGAN.add_to_arg_parser(arg_parser)
    args = arg_parser.parse_args()

    args.generator_input_dim = 256
    model = BobLossGAN(args)
    model.load_dataset()
    model.build_models()
    # print('pretraining gen')
    # model.train_generator(
    #     args.batch_size, num_batches=args.num_pretrain_batches, verbose=True
    # )
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
