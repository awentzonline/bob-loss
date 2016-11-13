import numpy as np


def gen_generator_training_data(model, batch_size):
    while True:
        num_samples = batch_size
        X = model.sample_noise(num_samples)
        #Y = np.zeros((num_samples, 2))
        #Y[:, 1] = 1.
        Y = np.zeros((num_samples, 1))
        Y[:] = model.config.positive_label
        yield X, Y


def gen_discriminator_training_data(model, batch_size):
    while True:
        num_samples = batch_size // 2
        real_images = model.sample_frames(num_samples)
        generated_images = model.generate_images(num_samples)
        X = np.concatenate([generated_images, real_images])
        #Y = np.zeros((2 * num_samples, 2))
        # Y[:num_samples, 0] = 1.
        # Y[num_samples:, 1] = 1.
        Y = np.zeros((batch_size, 1))
        Y[:num_samples] = model.config.negative_label
        Y[num_samples:] = model.config.positive_label
        yield X, Y


def gen_gan_training_batches(model, batch_size):
    '''Generates data for both models'''
    assert batch_size % 2 == 0, 'batch_size must be even'
    while True:
        # generator data
        num_samples = batch_size
        X_g = model.sample_noise(num_samples)
        Y_g = np.zeros((num_samples, 1))
        Y_g[:] = model.config.positive_label
        # discriminator data
        num_samples = batch_size // 2
        real_images = model.sample_frames(num_samples)
        generated_images = model.generate_images(num_samples)
        X_d = np.concatenate([generated_images, real_images])
        Y_d = np.zeros((batch_size, 1))
        Y_d[:num_samples] = model.config.negative_label
        Y_d[num_samples:] = model.config.positive_label
        yield X_g, Y_g, X_d, Y_d
