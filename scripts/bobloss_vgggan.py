#!/usr/bin/env python
import numpy as np
from bobloss.dcgansr import DCGANSR
from bobloss.subpixel import SubPixelUpscaling
from keras.datasets import mnist
from keras.layers import (
    Activation, BatchNormalization, Convolution2D, Deconvolution2D, Dense,
    Dropout, Flatten, GlobalAveragePooling2D, Input, Lambda, Layer, LeakyReLU, Permute,
    Reshape, UpSampling2D
)
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img


class GANModel(DCGANSR):
    def _load_dataset(self):
        data = np.load(
            self.config.dataset
        )
        for name in ('frame_data', 'frame_ids', 'episode_ids'):
            setattr(self, name, data[name])
        if self.frame_data.shape[-1] in (1, 3) and K.image_dim_ordering() == 'th':
            self.frame_data = np.transpose(self.frame_data, (0, 3, 1, 2))
        #self.frame_data = imagenet_utils.preprocess_input(self.frame_data)
        self.frame_data = (self.frame_data / 128.) - 1.
        #print self.frame_data.min(), self.frame_data.mean(), self.frame_data.max()

    def _build_models(self):
        # generator
        activation = 'relu'
        kernel_size = 3
        max_channels = 256
        #cnn_layers = ((max_channels // 2, 3, 3), (max_channels // 4, 3, 3))#, (max_channels // 8, 3, 3))
        cnn_layers = ((max_channels // 2, 3, 3), (max_channels // 4, 3, 3), (max_channels // 8, 3, 3))
        #cnn_layers = ((max_channels // 2, 3, 3), (max_channels // 4, 3, 3),)
        num_cnn_layers = len(cnn_layers)
        scale = 2 ** num_cnn_layers
        if K.image_dim_ordering() == 'tf':
            img_height, img_width, img_channels = self.frame_shape
        else:
            img_channels, img_height, img_width = self.frame_shape
        generator_input = Input(shape=(self.config.generator_input_dim,))
        x = Dense(
            max_channels * (img_height // scale) * (img_width // scale)
        )(generator_input)
        x = BatchNormalization(mode=2)(x)
        x = Activation(activation)(x)
        if K.image_dim_ordering() == 'tf':
            reshape_order = (img_height // scale, img_width // scale, max_channels)
        else:
            reshape_order = (max_channels, img_height // scale, img_width // scale)
        x = Reshape(reshape_order)(x)
        for channels, w, h in cnn_layers:
            #x = UpSampling2D((2, 2))(x)
            x = SubPixelUpscaling(2, channels // 2)(x)
            x = Activation(activation)(x)
            x = Convolution2D(channels, kernel_size, kernel_size, border_mode='same')(x)
            x = batchnorm_tf(x)
            x = Activation(activation)(x)
        # x = Convolution2D(
        #     img_channels, 1, 1, activation='relu',#self.config.generator_activation,
        #     border_mode='same'
        # )(x)
        x = Convolution2D(
            img_channels, 1, 1, activation='tanh',#self.config.generator_activation,
            border_mode='same'
        )(x)
        #x = Denormalize()(x)
        generator = Model(generator_input, x)
        generator_optimizer = Adam(lr=2e-4, beta_1=0.5)
        generator.compile(
            loss='categorical_crossentropy',
            optimizer=generator_optimizer
        )
        generator.summary()

        # discriminator
        kernel_size = 5
        max_channels = 256
        x = discriminator_input = Input(shape=self.frame_shape)
        #x = Normalize()(x)
        vgg16 = VGG16(include_top=False, input_tensor=discriminator_input)
        make_trainable(vgg16, False)
        x = vgg16.get_layer(name='block3_pool')(x)
        x = Flatten()(x)
        x = Dense(max_channels)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(2, activation='softmax')(x)
        discriminator = Model(discriminator_input, x)
        discriminator_optimizer = Adam(lr=2e-4, beta_1=0.5)
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=discriminator_optimizer
        )
        # end-to-end model for training generator
        make_trainable(discriminator, False)
        gan_input = Input(shape=(self.config.generator_input_dim,))
        g = generator(gan_input)
        d = discriminator(g)
        gan = Model(gan_input, d)
        gan.compile(
            loss='binary_crossentropy',
            optimizer=generator_optimizer
        )
        gan.summary()
        make_trainable(discriminator, True)
        discriminator.summary()
        return generator, discriminator, gan

    def pretrain_generator(self):
        vgg16 = VGG16(include_top=False, input_tensor=discriminator_input)


    def save_sample_grid(self, samples, filename=None):
        from PIL import Image

        if K.image_dim_ordering() == 'tf':
            num_samples, img_height, img_width, img_channels = samples.shape
        else:
            num_samples, img_channels, img_height, img_width = samples.shape
        num_wide = int(np.sqrt(num_samples))
        num_heigh = int(np.ceil(num_samples / num_wide))
        width = num_wide * img_width
        height = num_heigh * img_height
        img_mode = {1: 'L', 3: 'RGB'}[img_channels]
        output_img = Image.new(img_mode, (width, height), 'black')
        for i in range(num_samples):
            x = (i % num_wide) * img_width
            y = (i / num_wide) * img_height
            sample_arr = samples[i]
            #sample_arr = deprocess_input(samples[i])
            sample_arr = (samples[i] + 1.) * 128.
            #print sample_arr.min(), sample_arr.mean(), sample_arr.max()

            sample_img = array_to_img(sample_arr)
            output_img.paste(sample_img, (x, y))
        if filename is None:
            filename = self.config.sample_output_filename
        output_img.save(filename)


def batchnorm_tf(x):
    # work-around apparent theano/tf-dim_ordering bug
    if K.image_dim_ordering() == 'tf':
        x = Permute((2, 3, 1))(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    if K.image_dim_ordering() == 'tf':
        x = Permute((3, 1, 2))(x)
    return x


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
       l.trainable = val


# https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks/blob/master/layers.py
from keras.engine.topology import Layer
from keras import backend as K


class Normalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG and GAN networks.
    '''

    def __init__(self, type="vgg", value=120, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.type = type
        self.value = value

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if self.type == "gan":
            return x / self.value
        else:
            if K.backend() == "theano":
                import theano.tensor as T
                T.set_subtensor(x[:, 0, :, :], x[:, 0, :, :] - 103.939, inplace=True)
                T.set_subtensor(x[:, 1, :, :], x[:, 1, :, :] - 116.779, inplace=True)
                T.set_subtensor(x[:, 2, :, :], x[:, 2, :, :] - 123.680, inplace=True)
            else:
                # No exact substitute for set_subtensor in tensorflow
                # So we subtract an approximate value
                x = x - self.value
            return x


    def get_output_shape_for(self, input_shape):
        return input_shape


class Denormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG and GAN networks.
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return (x + 1) * 127.5

    def get_output_shape_for(self, input_shape):
        return input_shape


def deprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    if dim_ordering == 'th':
        # Zero-center by mean pixel
        x[0, :, :] += 103.939
        x[1, :, :] += 116.779
        x[2, :, :] += 123.68
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
    return x


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description='Bob Loss GAN model')
    GANModel.add_to_arg_parser(arg_parser)
    args = arg_parser.parse_args()
    args.generator_input_dim = 256

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
