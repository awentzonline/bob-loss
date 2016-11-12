import numpy as np
from PIL import Image
from keras.layers import (
    Activation, Convolution2D, Dense, Dropout, Flatten, Input, Reshape,
    UpSampling2D
)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import array_to_img
from tqdm import tqdm


class GAN(object):
    '''Generative Adversarial Network

    Got started with: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
    '''
    def __init__(self, config):
        self.config = config

    def load_dataset(self):
        data = np.load(
            self.config.dataset
        )
        for name in ('frame_data', 'frame_ids', 'episode_ids'):
            setattr(self, name, data[name])
        #self.frame_data = np.transpose(self.frame_data, (0, 3, 2, 1))

    @property
    def frame_shape(self):
        return self.frame_data[0].shape

    @property
    def num_frames_in_dataset(self):
        return self.frame_data.shape[0]

    def build_models(self):
        # generator
        cnn_layers = ((128, 3, 3), (64, 3, 3))
        num_cnn_layers = len(cnn_layers)
        scale = 2 ** num_cnn_layers
        num_channels = 128
        img_height, img_width, img_channels = self.frame_shape
        self.generator_input = Input(shape=(self.config.generator_input_dim,))
        x = self.generator_input
        x = Dense(
            num_channels * (img_height // scale) * (img_width // scale), activation='relu'
        )(self.generator_input)
        x = Reshape((img_height // scale, img_width // scale, num_channels))(x)
        for num_channels, w, h in cnn_layers:
            x = Convolution2D(num_channels, w, h, border_mode='same')(x)
            x = Activation('relu')(x)
            x = UpSampling2D((2, 2))(x)
        x = Convolution2D(img_channels, 1, 1, activation='relu')(x)
        self.generator = Model(self.generator_input, x)
        self.generator_optimizer = Adam()
        self.generator.compile(
            loss='categorical_crossentropy',
            optimizer=self.generator_optimizer
        )

        # discriminator
        dense_dim = 64
        cnn_layers = ((32, 3, 3), (64, 3, 3))
        num_cnn_layers = len(cnn_layers)
        x = self.discriminator_input = Input(shape=self.frame_shape)
        for num_channels, w, h in cnn_layers:
            x = Convolution2D(num_channels, w, h, subsample=(2, 2))(x)
            x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(dense_dim, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(2, activation='softmax')(x)
        self.discriminator = Model(self.discriminator_input, x)
        self.discriminator_optimizer = Adam()
        self.discriminator.compile(
            loss='categorical_crossentropy',
            optimizer=self.discriminator_optimizer
        )

        def make_trainable(net, val):
            net.trainable = val
            for l in net.layers:
               l.trainable = val
        # end-to-end model for training generator
        make_trainable(self.discriminator, False)
        self.gan_input = Input(shape=(self.config.generator_input_dim,))
        g = self.generator(self.gan_input)
        d = self.discriminator(g)
        self.gan = Model(self.gan_input, d)
        self.gan.compile(
            loss='categorical_crossentropy',
            optimizer=self.generator_optimizer
        )
        make_trainable(self.discriminator, True)

    def train(
            self, num_epochs=None, batch_size=None, train_discriminator=True,
            train_generator=True):
        num_epochs = num_epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        d_losses = []
        g_losses = []
        for epoch in tqdm(range(num_epochs)):
            if epoch % self.config.generate_every_n_epochs == 0:
                if train_generator:
                    self.save_sample_grid(self.generate_images(batch_size))
                #self.save_sample_grid(real_images, filename='real_images.png')
            if train_discriminator:
                d_loss = self.train_discriminator(batch_size)
                d_losses.append(d_loss)
            if train_generator:
                g_loss = self.train_generator(batch_size)
                g_losses.append(g_loss)
        self.save_sample_grid(self.generate_images(batch_size))
        return d_losses, g_losses

    def train_discriminator(self, batch_size):
        real_images = self.sample_frames(batch_size)
        generated_images = self.generate_images(batch_size)
        X = np.concatenate([generated_images, real_images])
        Y = np.zeros((2 * batch_size, 2))
        Y[:batch_size, 0] = 1.
        Y[batch_size:, 1] = 1.
        return self.discriminator.train_on_batch(X, Y)

    def train_generator(self, batch_size):
        X = self.sample_noise(batch_size)
        Y = np.zeros((batch_size, 2))
        Y[:, 1] = 1.
        return self.gan.train_on_batch(X, Y)

    def sample_noise(self, num_samples):
        return np.random.uniform(
            0., 1., size=(num_samples, self.config.generator_input_dim)
        )

    def generate_images(self, num_images):
        noise = self.sample_noise(num_images)
        return self.generator.predict(noise)

    def sample_frames(self, num_samples):
        indexes = np.random.randint(
            0, self.num_frames_in_dataset, (num_samples,)
        )
        return self.frame_data[indexes]

    def save_sample_grid(self, samples, filename=None):
        num_samples, img_height, img_width, img_channels = samples.shape
        num_wide = int(np.sqrt(num_samples))
        num_heigh = int(np.ceil(num_samples / num_wide))
        width = num_wide * img_width
        height = num_heigh * img_height
        img_mode = {1: 'L', 3: 'RGB'}[img_channels]
        output_img = Image.new(img_mode, (width, height), 'black')
        for i in range(num_samples):
            x = (i % num_wide) * img_width
            y = (i / num_wide) * img_height
            sample_img = array_to_img(samples[i])
            output_img.paste(sample_img, (x, y))
        if filename is None:
            filename = self.config.sample_output_filename
        output_img.save(filename)

    @staticmethod
    def add_to_arg_parser(arg_parser):
        arg_parser.add_argument(
            '--generator-input-dim', default=100, type=int,
            help='Input dimensionality of generator'
        )
        arg_parser.add_argument(
            '--dataset', default='bobloss_episodes.npz',
            help='Filename of Bob Loss dataset'
        )
        arg_parser.add_argument(
            '--batch-size', default=32, type=int,
            help='Training batch size'
        )
        arg_parser.add_argument(
            '--num-epochs', default=1000, type=int,
            help='Number of training epochs'
        )
        arg_parser.add_argument(
            '--generate-every-n-epochs', default=20, type=int,
            help='Generate samples every N epochs during training'
        )
        arg_parser.add_argument(
            '--sample-output-filename', default='bobloss_samples.png',
            help='Number of training epochs'
        )
        arg_parser.add_argument(
            '--dropout', default=0.25, type=float,
            help='Rate of dropout'
        )
