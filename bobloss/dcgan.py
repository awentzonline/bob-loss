import os

import numpy as np
from PIL import Image
from keras.layers import (
    Activation, BatchNormalization, Convolution2D, Deconvolution2D, Dense,
    Dropout, Flatten, Input, LeakyReLU, Permute, Reshape, UpSampling2D
)
from keras import backend as K
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img
from tqdm import tqdm

from . import gan_training


class DCGAN(object):
    '''Deep Convolutional Generative Adversarial Network

    Got started with: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
    Reading:
        https://arxiv.org/pdf/1511.06434v2.pdf
        https://arxiv.org/pdf/1606.03498v1.pdf
    '''
    def __init__(self, config):
        self.config = config

    def load_dataset(self):
        self._load_dataset()

    def _load_dataset(self):
        '''Should assign `self.frame_data`'''
        raise NotImplementedError('Implement _load_dataset')

    @property
    def frame_shape(self):
        return self.frame_data[0].shape

    @property
    def frame_size(self):
        if K.image_dim_ordering() == 'tf':
            img_height, img_width, img_channels = self.frame_shape
        else:
            img_channels, img_height, img_width = self.frame_shape
        return img_height, img_width

    @property
    def num_frames_in_dataset(self):
        return self.frame_data.shape[0]

    def build_models(self):
        if not self.try_load_models():
            self.generator, self.discriminator, self.gan = self._build_models()

    def _build_models(self):
        raise NotImplementedError('Implement `_build_models`')

    def train(
            self, num_epochs=None, batch_size=None, train_discriminator=True,
            train_generator=True, discriminator_update_rate=None,
            batches_per_iteration=None):
        num_epochs = num_epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        discriminator_update_rate = discriminator_update_rate or self.config.discriminator_update_rate
        batches_per_iteration = batches_per_iteration or self.config.batches_per_iteration
        d_losses = []
        g_losses = []
        # discriminator_data = gan_training.gen_discriminator_training_data(batch_size)
        # generator_data = gan_training.gen_generator_training_data(batch_size)
        throttle_discriminator = throttle_generator = False
        for epoch in tqdm(range(num_epochs)):
            if epoch % self.config.generate_every_n_iters == 0:
                if train_generator:
                    self.save_sample_grid(self.generate_images(batch_size))
                    #self.save_sample_grid(self.sample_frames(batch_size), filename='real_images.png')
            if train_discriminator and not throttle_discriminator:
                d_loss = 9999.
                tries = 0
                while d_loss > 0.9 and tries < 5:
                    history = self.train_discriminator(
                        batch_size,
                        num_batches=discriminator_update_rate * batches_per_iteration,
                        verbose=True
                    )
                    d_loss = history.history['loss'][-1]
                    tries += 1
                d_losses.append(d_loss)
            if train_generator and not throttle_generator:
                g_loss = 9999.
                tries = 0
                while g_loss > 0.9 and tries < 5:
                    history = self.train_generator(
                        batch_size, num_batches=batches_per_iteration, verbose=True
                    )
                    g_loss = history.history['loss'][-1]
                    tries += 1
                g_losses.append(g_loss)
            if epoch % self.config.model_save_rate == 0:
                self.save_models()
            fake_accuracy, real_accuracy = self.check_accuracy()
            throttle_discriminator = (
                fake_accuracy > (1. - self.config.throttle_accuracy_p) and
                real_accuracy > (1. - self.config.throttle_accuracy_p)
            )
            throttle_generator = (
                fake_accuracy < self.config.throttle_accuracy_p or
                real_accuracy < self.config.throttle_accuracy_p
            )
            if throttle_discriminator and throttle_generator:
                throttle_generator = throttle_discriminator = False
            print fake_accuracy, real_accuracy, throttle_discriminator, throttle_generator
        self.save_sample_grid(self.generate_images(batch_size))
        return d_losses, g_losses

    def train_discriminator(self, batch_size, num_batches=1, verbose=False):
        # def gen_batches():
        #     while True:
        #         yield self.make_discriminator_batch(batch_size)
        # return self.discriminator.fit_generator(
        #     gen_batches(), samples_per_epoch=batch_size * num_batches,
        #     nb_epoch=1, verbose=verbose
        # )
        # Threading issue with fit_generator?
        X, Y = self.make_discriminator_batch(batch_size * num_batches)
        return self.discriminator.fit(
            X, Y, batch_size=batch_size, nb_epoch=1, verbose=verbose
        )

    def train_generator(self, batch_size, num_batches=1, verbose=False):
        def gen_batches():
            while True:
                num_samples = batch_size
                X = self.sample_noise(num_samples)
                Y = np.zeros((num_samples, 2))
                Y[:, 1] = 1.
                # Y = np.zeros((num_samples, 1))
                # Y[:] = self.config.positive_label
                yield X, Y
        return self.gan.fit_generator(
            gen_batches(), samples_per_epoch=batch_size * num_batches,
            nb_epoch=1, verbose=verbose
        )

    def check_accuracy(self):
        sample_size = 16
        X, Y = self.make_discriminator_batch(sample_size * 2, sample_memory=False)
        Y_hat = self.discriminator.predict(X)
        def accuracy(actual, prediction):
            y = np.argmax(Y, axis=1)
            y_hat = np.argmax(Y_hat, axis=1)
            errors = y - y_hat
            num_items = errors.shape[0]
            num_right = (errors == 0).sum()
            return num_right / float(num_items)
        fake_accuracy = accuracy(Y[:sample_size], Y_hat[:sample_size])
        real_accuracy = accuracy(Y[sample_size:], Y_hat[sample_size:])
        return fake_accuracy, real_accuracy

    def sample_noise(self, num_samples):
        return np.random.uniform(
            -1., 1., size=(num_samples, self.config.generator_input_dim)
        )

    def generate_images(self, num_images):
        noise = self.sample_noise(num_images)
        return self.generator.predict(noise)

    def sample_frames(self, num_samples):
        indexes = np.random.randint(
            0, self.num_frames_in_dataset, (num_samples,)
        )
        return self.frame_data[indexes]

    def make_discriminator_batch(self, batch_size, **kwargs):
        num_samples = batch_size // 2
        real_images = self.sample_frames(num_samples)
        generated_images = self.generate_images(num_samples)
        X = np.concatenate([generated_images, real_images])
        if np.random.random() < self.config.p_sample_batch:
            self.save_sample_grid(X, filename=self.config.sample_batch_name)
        Y = np.zeros((2 * num_samples, 2))
        Y[:num_samples, 0] = 1.
        Y[num_samples:, 1] = 1.
        # Y = np.zeros((2 * num_samples, 1))
        # Y[:num_samples] = self.config.negative_label
        # Y[num_samples:] = self.config.positive_label
        return X, Y

    def save_sample_grid(self, samples, filename=None):
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
            if self.config.generator_activation == 'relu':
                sample_arr = samples[i]
            elif self.config.generator_activation == 'tanh':
                sample_arr = (samples[i] + 1.) * 128.
            elif self.config.generator_activation == 'sigmoid':
                sample_arr = samples[i] * 256.
            sample_img = array_to_img(sample_arr, scale=False)
            output_img.paste(sample_img, (x, y))
        if filename is None:
            filename = self.config.sample_output_filename
        output_img.save(filename)

    def try_load_models(self, model_name=None):
        model_name = model_name or self.config.model_name
        if os.path.exists('{}_g.h5'.format(model_name)):
            if not self.config.ignore_existing_model:
                self.load_models(model_name=model_name)
                return True
        return False

    def save_models(self, model_name=None):
        model_name = model_name or self.config.model_name
        self.generator.save('{}_g.h5'.format(model_name))
        self.discriminator.save('{}_d.h5'.format(model_name))
        self.gan.save('{}_gan.h5'.format(model_name))  # KISS

    def load_models(self, model_name):
        model_name = model_name or self.config.model_name
        self.generator = load_model('{}_g.h5'.format(model_name))
        self.discriminator = load_model('{}_d.h5'.format(model_name))
        self.gan = load_model('{}_gan.h5'.format(model_name))

    @classmethod
    def load_model(cls, model_name):
        pass

    @classmethod
    def add_to_arg_parser(cls, arg_parser):
        arg_parser.add_argument(
            '--generator-input-dim', default=100, type=int,
            help='Input dimensionality of generator'
        )
        arg_parser.add_argument(
            '--dataset', default='bobloss_episodes.npz',
            help='Filename of Bob Loss dataset'
        )
        arg_parser.add_argument(
            '--batch-size', default=64, type=int,
            help='Training batch size'
        )
        arg_parser.add_argument(
            '--batches-per-iteration', default=4, type=int,
            help='Training batch size'
        )
        arg_parser.add_argument(
            '--discriminator-update-rate', default=1, type=int,
            help='How many more times is the disciminator updated vs generator'
        )
        arg_parser.add_argument(
            '--num-epochs', default=500000, type=int,
            help='Number of training epochs'
        )
        arg_parser.add_argument(
            '--num-pretrain-batches', default=0, type=int,
            help='Number of pretraining batches for the discriminator'
        )
        arg_parser.add_argument(
            '--generate-every-n-iters', default=2, type=int,
            help='Generate samples every N training iterations'
        )
        arg_parser.add_argument(
            '--sample-output-filename', default='bobloss_samples.png',
            help='Number of training epochs'
        )
        arg_parser.add_argument(
            '--dropout', default=0.25, type=float, help='Rate of dropout'
        )
        arg_parser.add_argument(
            '--positive-label', default=1.0, type=float,
            help='The value given for positive examples.'
        )
        arg_parser.add_argument(
            '--negative-label', default=0.0, type=float,
            help='The value given for negative examples.'
        )
        arg_parser.add_argument(
            '--model-name', default='bobloss_mdl',
            help='Filename prefix for serialized model data'
        )
        arg_parser.add_argument(
            '--p-sample-batch', default=0.01, type=float,
            help='Probability of saving an image of the current training batch'
        )
        arg_parser.add_argument(
            '--sample-batch-name', default='batch.png',
            help='Filename for batch image output'
        )
        arg_parser.add_argument(
            '--model', default='bobloss',
            help='Prefix of filename for model'
        )
        arg_parser.add_argument(
            '--ignore-existing-model', default=False, action='store_true',
            help='Ignore existing model files'
        )
        arg_parser.add_argument(
            '--model-save-rate', default=20, type=int,
            help='Save the model every n iterations'
        )
        arg_parser.add_argument(
            '--throttle-accuracy_p', default=0.3, type=float,
            help='Skip training parts of the model at this level of accuracy'
        )
        arg_parser.add_argument(
            '--generator-activation', default='tanh',
            help='Activation function for generator output (tanh, sigmoid, relu)'
        )
