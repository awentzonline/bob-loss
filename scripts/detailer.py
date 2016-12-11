#!/usr/bin/env python
import numpy as np
from bobloss.fcgan import FCGAN
from bobloss.subpixel import SubPixelUpscaling
from keras.datasets import mnist
from keras.layers import (
    Activation, Convolution2D, Deconvolution2D, Dense,
    Dropout, Flatten, GlobalAveragePooling2D, Input, LeakyReLU, Permute,
    Reshape, UpSampling2D
)
from keras import backend as K
from keras.models import Model
from keras.layers import merge
from keras.optimizers import Adam, SGD


class GANModel(FCGAN):
    def _load_dataset(self):
        data = np.load(
            self.config.dataset
        )
        for name in ('frame_data', 'frame_ids', 'episode_ids'):
            setattr(self, name, data[name])
        if K.image_dim_ordering() == 'th' and self.frame_data.shape[-1] in (1, 3):
            self.frame_data = np.transpose(self.frame_data, (0, 3, 1, 2))
        self.frame_data = (self.frame_data / 127.5) - 1.

    def _build_models(self):
        # generator
        activation = 'relu'
        kernel_size = 3
        num_channels = 16
        num_layers = 5
        if K.image_dim_ordering() == 'tf':
            img_height, img_width, img_channels = self.frame_shape
        else:
            img_channels, img_height, img_width = self.frame_shape
        x = generator_input = Input(shape=self.frame_shape)
        #last_layer = Convolution2D(num_channels, 1, 1, border_mode='same')(x)
        for layer_i in range(num_layers):
            x = Convolution2D(num_channels * (layer_i + 1), kernel_size, kernel_size, border_mode='same')(x)
            x = Activation(activation)(x)
            # x = Convolution2D(num_channels, kernel_size, kernel_size, border_mode='same')(x)
            # x = Activation(activation)(x)
            # x = merge([last_layer, x], mode='sum')
            # last_layer = x
        x = Convolution2D(
            img_channels, 1, 1, activation=self.config.generator_activation,
            border_mode='same'
        )(x)
        generator = Model(generator_input, x)
        generator_optimizer = Adam(lr=2e-3, beta_1=0.5)
        generator.compile(
            loss='categorical_crossentropy',
            optimizer=generator_optimizer
        )
        generator.summary()

        # discriminator
        kernel_size = 3
        max_channels = 256
        #cnn_layers = ((max_channels // 4, 3, 3), (max_channels // 2, 3, 3))
        cnn_layers = ((max_channels // 2, 3, 3),)
        if K.image_dim_ordering() == 'tf':
            disc_input_shape = (img_height, img_width, img_channels * 2)
        else:
            disc_input_shape = (img_channels * 2, img_height, img_width)
        x = discriminator_input = Input(shape=disc_input_shape)
        for channels, w, h in cnn_layers:
            x = Convolution2D(
                channels, kernel_size, kernel_size, subsample=(2, 2), border_mode='same'
            )(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(self.config.dropout)(x)
        x = Flatten()(x)
        x = Dense(max_channels)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(2, activation='softmax')(x)
        discriminator = Model(discriminator_input, x)
        discriminator_optimizer = SGD()#Adam(lr=2e-4, beta_1=0.5)
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=discriminator_optimizer
        )

        def make_trainable(net, val):
            net.trainable = val
            for l in net.layers:
               l.trainable = val
        # end-to-end model for training generator
        make_trainable(discriminator, False)
        gan_input = Input(shape=self.frame_shape)
        g = generator(gan_input)
        d = discriminator(merge([g, gan_input], mode='concat', concat_axis=1))
        gan = Model(gan_input, d)
        gan.compile(
            loss='binary_crossentropy',
            optimizer=generator_optimizer
        )
        gan.summary()
        make_trainable(discriminator, True)
        discriminator.summary()
        return generator, discriminator, gan


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Detailer GAN')
    GANModel.add_to_arg_parser(arg_parser)
    args = arg_parser.parse_args()

    model = GANModel(args)
    model.load_dataset()
    model.build_models()
    if args.num_pretrain_batches:
        print('pretraining discriminator')
        model.train_discriminator(
            args.batch_size, num_batches=args.num_pretrain_batches, verbose=True
        )
    print('training')
    d_losses, g_losses = model.train()
    for losses in (d_losses, g_losses):
        print('Losses:')
        print(losses[:20])
        print(losses[-20:])
