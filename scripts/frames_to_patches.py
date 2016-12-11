#!/usr/bin/env python
import argparse
import glob
import os
import random

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image


def resize_img(img, width=None, height=None, filter=Image.ANTIALIAS):
    img_width, img_height = img.size
    if width and height:
        target_size = (width, height)
    elif width and not height:
        ratio = width / float(img_width)
        target_size = (width, int(img_height * ratio))
    elif not width and height:
        ratio = height / float(img_height)
        target_size = (int(img_width * ratio), height)
    else:
        target_size = None
    if target_size:
        img = img.resize(target_size, filter)
    return img


def episodes_to_numpy(
        episode_glob='', frame_glob='*.jpg', sample=None,
        patch_size=None, patches_per_frame=2,
        resize_range=1.0, **kwargs):
    frame_data = []
    frame_ids = []
    episode_ids = []
    for episode_id, episode_path in enumerate(glob.iglob(episode_glob)):
        print('Loading %s' % episode_path)
        for filename in glob.iglob(os.path.join(episode_path, frame_glob)):
            if sample:
                if random.random() > sample:
                    continue
            try:
                basename = os.path.basename(filename)
                frame_id = int(os.path.splitext(basename)[0])
            except ValueError:
                continue
            img = load_img(filename)
            # resize frame
            scale = np.random.uniform(resize_range, 1.0)
            img_width, img_height = img.size
            img = resize_img(
                img, width=int(scale * img_width), height=int(scale * img_height)
            )
            for patch_i in range(patches_per_frame):
                img_width, img_height = map(lambda x: x - patch_size, img.size)
                x = np.random.randint(0, img_width)
                y = np.random.randint(0, img_height)
                img_patch = img.crop((x, y, x + patch_size, y + patch_size))
                img_data = img_to_array(img_patch)
                frame_data.append(img_data)
                frame_ids.append(frame_id)
                episode_ids.append(episode_id)
    print('Numpyize...')
    frame_data = np.array(frame_data)
    frame_ids = np.array(frame_ids)
    episode_ids = np.array(episode_ids)
    print('shapes:')
    for d in (frame_data, frame_ids, episode_ids):
        print d.shape
    return dict(
        frame_data=frame_data, frame_ids=frame_ids, episode_ids=episode_ids
    )


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Episode frames to numpy arrays of patches'
    )
    arg_parser.add_argument(
        '--episode-glob', default='./frames/*/*/',
        help='Glob which matches episode directories.'
    )
    arg_parser.add_argument(
        '--frame-glob', default='*.jpg',
        help='Glob appended to episode directory which matches frame images.'
    )
    arg_parser.add_argument(
        '--sample', default=None, type=float,
        help='Probability of including a frame. Default is all frames.'
    )
    arg_parser.add_argument(
        '--patches-per-frame', default=8, type=int,
        help='Number of random patches taken per frame.'
    )
    arg_parser.add_argument(
        '--resize-range', default=0.8, type=float,
        help='Randomly resize frame between this ratio and full-sized before sampling'
    )
    arg_parser.add_argument(
        '--patch-size', type=int, default=16, help='Length of side of a patch'
    )
    arg_parser.add_argument(
        '--output', default='bobloss_episode_patches.npz', help='Filename for output.'
    )
    args = vars(arg_parser.parse_args())

    print('Converting episodes to numpy...')
    episodes_data = episodes_to_numpy(**args)
    print('Saving...')
    np.savez(args['output'], **episodes_data)
