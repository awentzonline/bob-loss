import numpy as np
from PIL import Image
from keras.layers import (
    Activation, BatchNormalization, Convolution2D, Deconvolution2D, Dense,
    Dropout, Flatten, Input, LeakyReLU, Permute, Reshape, UpSampling2D
)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img
from tqdm import tqdm

from . import gan_training
from .sample_memory import SampleMemory
from .dcgan import DCGAN


class DCGANSR(DCGAN):
    '''Deep Convolutional Generative Adversarial Network with Sample Replay

    Got started with: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
    Reading:
        https://arxiv.org/pdf/1511.06434v2.pdf
        https://arxiv.org/pdf/1606.03498v1.pdf
    '''
    def __init__(self, config):
        self.config = config

    def load_dataset(self):
        super(DCGANSR, self).load_dataset()
        self.reset_memory()

    def reset_memory(self):
        print('resetting sample memory')
        self.memory = SampleMemory(self.frame_shape, self.config.memory_size)

    @property
    def frame_shape(self):
        return self.frame_data[0].shape

    @property
    def num_frames_in_dataset(self):
        return self.frame_data.shape[0]

    def _build_models(self):
        raise NotImplementedError('Implement `_build_models`')

    def train_discriminator(self, batch_size, num_batches=1, verbose=False):
        def gen_batches():
            while True:
                num_samples = batch_size // 2
                num_recent = int(num_samples * 0.5)
                real_images = self.sample_frames(num_samples)
                self.build_memory(num_samples)  # add some more to the memory
                generated_images = self.sample_frames_from_memory(num_samples - num_recent)
                recent_images = self.last_n_frames_from_memory(num_recent)
                X = np.concatenate([generated_images, recent_images, real_images])
                Y = np.zeros((2 * num_samples, 2))
                Y[:num_samples, 0] = 1.
                Y[num_samples:, 1] = 1.
                # Y = np.zeros((2 * num_samples, 1))
                # Y[:num_samples] = self.config.negative_label
                # Y[num_samples:] = self.config.positive_label
                yield X, Y
        return self.discriminator.fit_generator(
            gen_batches(), samples_per_epoch=batch_size * num_batches,
            nb_epoch=1, verbose=verbose
        )

    def sample_frames_from_memory(self, num_samples):
        sample_deficit = num_samples - self.memory.num_stored
        if sample_deficit > 0:
            self.build_memory(sample_deficit)
        return self.memory.sample(num_samples)

    def last_n_frames_from_memory(self, n):
        return self.memory.last_n_frames(n)

    def build_memory(self, num_samples):
        samples = self.generate_images(num_samples)
        self.memory.append_batch(samples)

    @classmethod
    def add_to_arg_parser(cls, arg_parser):
        super(DCGANSR, cls).add_to_arg_parser(arg_parser)
        arg_parser.add_argument(
            '--memory-size', default=30000, type=int,
            help='Number of pretraining batches for the discriminator'
        )
        arg_parser.add_argument(
            '--memory-seed-size', default=10000, type=int,
            help='Seed the memory with this many items'
        )
