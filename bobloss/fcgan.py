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
from keras.preprocessing.image import array_to_img, img_to_array
from tqdm import tqdm

from . import gan_training
from .dcgan import DCGAN


class FCGAN(DCGAN):
    '''Fully-Convolutional Deep Convolutional Generative Adversarial Network

    With constant frame shape.
    '''
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
                images = self.sample_frames(num_samples)
                X = self.get_scaled_images(images)
                Y = np.zeros((num_samples, 2))
                Y[:, 1] = 1.
                # Y = np.zeros((num_samples, 1))
                # Y[:] = self.config.positive_label
                yield X, Y
        return self.gan.fit_generator(
            gen_batches(), samples_per_epoch=batch_size * num_batches,
            nb_epoch=1, verbose=verbose
        )

    def make_discriminator_batch(self, batch_size, **kwargs):
        num_samples = batch_size // 2
        real_images = self.sample_frames(num_samples)
        scaled_real = self.get_scaled_images(real_images)

        sampled_images = self.sample_frames(num_samples)
        scaled_samples = self.get_scaled_images(sampled_images)
        generated_images = self.generator.predict(scaled_samples)
        #generated_images = self.generator.predict(scaled_real)

        X = np.concatenate([generated_images, real_images])
        Xi = np.concatenate([scaled_samples, scaled_real])
        #Xi = np.concatenate([scaled_real, scaled_real])
        X = np.concatenate([X, Xi], axis=1)
        if np.random.random() < self.config.p_sample_batch:
            # TODO: tf...
            self.save_sample_grid(X[:, :3, ...], filename=self.config.sample_batch_name)
        Y = np.zeros((2 * num_samples, 2))
        Y[:num_samples, 0] = 1.
        Y[num_samples:, 1] = 1.
        # Y = np.zeros((2 * num_samples, 1))
        # Y[:num_samples] = self.config.negative_label
        # Y[num_samples:] = self.config.positive_label
        return X, Y

    def predict(self, lowres_images):
        return self.generator.predict(lowres_images)

    def generate_images(self, num_images):
        real_images = self.sample_frames(num_images)
        scaled_images = self.get_scaled_images(real_images)
        return self.generator.predict(scaled_images)

    @property
    def scaled_frame_size(self):
        return map(lambda x: int(x * self.config.fc_scale), self.frame_size)

    @property
    def scaled_frame_shape(self):
        if K.image_dim_ordering() == 'tf':
            channel_index = 2
        else:
            channel_index = 0
        return map(
            lambda (i, x): x if i == channel_index else int(x * self.config.fc_scale),
            enumerate(self.frame_shape)
        )

    def get_scaled_images(self, images):
        results = []
        scaled_height, scaled_width = self.scaled_frame_size
        height, width = self.frame_size
        for image in images:
            img = array_to_img((image + 1.) * 127.5, scale=False)
            img = img.resize((scaled_width, scaled_height))
            img = img.resize((width, height)) # scale back up
            arr = img_to_array(img)
            arr = (arr / 127.5) - 1.
            results.append(arr)
        return np.stack(results)

    def sample_frames(self, num_samples):
        indexes = np.random.randint(
            0, self.num_frames_in_dataset, (num_samples,)
        )
        return self.frame_data[indexes]

    def try_load_models(self, model_name=None):
        model_name = model_name or self.config.model_name
        if os.path.exists('{}_fc_g.h5'.format(model_name)):
            if not self.config.ignore_existing_model:
                self.load_models(model_name=model_name)
                return True
        return False

    def save_models(self, model_name=None):
        model_name = model_name or self.config.model_name
        self.generator.save('{}_fc_g.h5'.format(model_name))
        self.discriminator.save('{}_fc_d.h5'.format(model_name))
        self.gan.save('{}_fc_gan.h5'.format(model_name))  # KISS

    def load_models(self, model_name):
        model_name = model_name or self.config.model_name
        self.generator = load_model('{}_fc_g.h5'.format(model_name))
        self.discriminator = load_model('{}_fc_d.h5'.format(model_name))
        self.gan = load_model('{}_fc_gan.h5'.format(model_name))

    @classmethod
    def add_to_arg_parser(cls, arg_parser):
        super(FCGAN, cls).add_to_arg_parser(arg_parser)
        arg_parser.add_argument(
            '--fc-scale', default=0.25, type=float,
            help='Scaling which is applied to dataset to create inputs'
        )
