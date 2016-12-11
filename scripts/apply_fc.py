#!/usr/bin/env python
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model, Model
from keras.preprocessing.image import array_to_img, img_to_array
from PIL import Image

from bobloss.subpixel import SubPixelUpscaling


def save_sample_grid(samples, filename, activation='tanh'):
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
        if activation == 'relu':
            sample_arr = samples[i]
        elif activation == 'tanh':
            sample_arr = (samples[i] + 1.) * 127.5
        elif activation == 'sigmoid':
            sample_arr = samples[i] * 256.
        sample_img = array_to_img(sample_arr, scale=False)
        output_img.paste(sample_img, (x, y))
    if filename is None:
        filename = self.config.sample_output_filename
    output_img.save(filename)


def main(args):
    # load the model
    model = load_model(args.model_filename, custom_objects={
        'SubPixelUpscaling': SubPixelUpscaling
    })
    print model.layers
    # load the images and bucket them by shape
    images_by_size = defaultdict(list)
    for filename in glob.glob(args.image_glob):
        img = Image.open(filename)
        img = img.resize(map(lambda x: int(x * args.output_scale), img.size))  # scale up
        images_by_size[img.size].append(img)
    # apply the model to the images
    for size, imgs in images_by_size.items():
        images = map(img_to_array, imgs)
        images = (np.stack(images) / 127.5) - 1.
        # NOTE: :(
        x = input_layer = Input(shape=images.shape[1:])
        for layer in model.layers[1:]:
            x = layer(x)
        this_model = Model([input_layer], [x])
        this_model.compile(optimizer='sgd', loss='mse')
        # END :(
        new_images = images
        for _ in range(args.apply_n):
            new_images = this_model.predict(new_images, verbose=False)
        # save before/after images
        for i in range(new_images.shape[0]):
            new_image = new_images[i]
            image = images[i]
            samples = np.stack([image, new_image])
            filename = '{}_{}.png'.format(size, i)
            filename = os.path.join(args.output_path, filename)
            print('saving sample', samples.shape, filename)
            save_sample_grid(samples, filename)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Apply FCGAN to an image')
    arg_parser.add_argument(
        'model_filename', help='Name of the model file')
    arg_parser.add_argument(
        'image_glob', help='Glob representing images to be transformed')
    arg_parser.add_argument(
        '--output-path', default='./imgout', help='Output path')
    arg_parser.add_argument(
        '--output-scale', default=1.0, type=float, help='Scale size of output')
    arg_parser.add_argument(
        '--apply-n', default=1, type=int, help='Feedback n times')
    args = arg_parser.parse_args()
    main(args)
