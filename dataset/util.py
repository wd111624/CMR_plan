import numpy as np
import cv2
import os
import sys
import random
from scipy.ndimage import zoom, rotate
from torch.utils.data.dataloader import DataLoader
import time

sys.path.append("..")
from utils.util import EPS, arr_to_8bit_img


def est_data_loading_speed(dataset, batch_size, shuffle, drop_last, w_max=40, w_min=0, repeats=3):
    times = 1e3 * np.ones((w_max + 1, repeats))

    for w in range(w_min, w_max + 1)[::-1]:
        print("With %d workers.." % w)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=w, shuffle=shuffle, drop_last=drop_last,
                                 pin_memory=True)

        for r in range(repeats):
            since = time.time()
            for _, _, _ in data_loader:
                pass
            times[w, r] = time.time() - since

        print('\tAverage time spent on loading data: {:.0f}m {:.0f}s'.format(
            times[w].mean() // 60, times[w].mean() % 60))

    # determine the optimal w
    times = times.mean(axis=1)
    optw_train, optt_train = times.argmin(), times.min()

    print("Optimal number of workers for the data loader: %d (avg time %.0fm %.0fs)" %
          (optw_train, optt_train // 60, optt_train % 60))


def aug_image_heatmap_pair(config, image, htmap, height=None, width=None):
    if height is None or width is None:
        height, width = image.shape[0], image.shape[1]  # record for later use

    if config['SCALE']:
        factor = random.uniform(config['SCALE_RANGE'][0], config['SCALE_RANGE'][1])
        if np.abs(factor - 1) > EPS:
            image = zoom(image, factor)
            if len(htmap.shape) > 2:
                tmp = []
                for j in range(htmap.shape[0]):
                    tmp.append(zoom(htmap[j], factor))
                htmap = np.concatenate([ch[None, ...] for ch in tmp], axis=0)
            else:
                htmap = zoom(htmap, factor)

    if config['ROTATE']:
        angle = random.randint(config['ROTATE_RANGE'][0], config['ROTATE_RANGE'][1])
        # print(angle)
        if int(angle) != 0:
            image = rotate(image, float(angle), mode='constant', cval=image.min())
            if len(htmap.shape) > 2:
                tmp = []
                for j in range(htmap.shape[0]):
                    tmp.append(rotate(htmap[j], float(angle), mode='constant', cval=htmap[j].min()))
                htmap = np.concatenate([ch[None, ...] for ch in tmp], axis=0)
            else:
                htmap = rotate(htmap, float(angle), mode='constant', cval=htmap.min())

    # below implement random crop in effect
    dy = image.shape[0] - height
    if dy > 0:  # crop
        y_start = random.randint(0, dy)
        image = image[y_start:y_start + height, :]
        if len(htmap.shape) > 2:
            htmap = htmap[:, y_start:y_start + height, :]
        else:
            htmap = htmap[y_start:y_start + height, :]
    elif dy < 0:  # pad
        y_start = random.randint(0, -dy)
        pad_width = ((y_start, -dy - y_start), (0, 0))
        image = np.pad(image, pad_width, mode='constant', constant_values=image.min())
        if len(htmap.shape) > 2:
            tmp = []
            for j in range(htmap.shape[0]):
                tmp.append(np.pad(htmap[j], pad_width, mode='constant', constant_values=htmap[j].min()))
            htmap = np.concatenate([ch[None, ...] for ch in tmp], axis=0)
        else:
            htmap = np.pad(htmap, pad_width, mode='constant', constant_values=htmap.min())

    dx = image.shape[1] - width
    if dx > 0:  # crop
        x_start = random.randint(0, dx)
        image = image[:, x_start:x_start + width]
        if len(htmap.shape) > 2:
            htmap = htmap[:, :, x_start:x_start + width]
        else:
            htmap = htmap[:, x_start:x_start + width]
    elif dx < 0:  # pad
        x_start = random.randint(0, -dx)
        pad_width = ((0, 0), (x_start, -dx - x_start))
        image = np.pad(image, pad_width, mode='constant', constant_values=image.min())
        if len(htmap.shape) > 2:
            tmp = []
            for j in range(htmap.shape[0]):
                tmp.append(np.pad(htmap[j], pad_width, mode='constant', constant_values=htmap[j].min()))
            htmap = np.concatenate([ch[None, ...] for ch in tmp], axis=0)
        else:
            htmap = np.pad(htmap, pad_width, mode='constant', constant_values=htmap.min())

    if config['FLIP'] and random.choice((True, False)):  # flip at 50% chance
        if config['FLIP_WAY'] == 'LR':
            image = image[:, ::-1]
            if htmap.ndim > 2:
                htmap = htmap[:, :, ::-1]
            else:
                htmap = htmap[:, ::-1]
        elif config['FLIP_WAY'] == 'UD':
            image = image[::-1, :]
            if htmap.ndim > 2:
                htmap = htmap[:, ::-1, :]
            else:
                htmap = htmap[::-1, :]
        else:
            raise Exception("Unknown way of flipping augmentation.")
        image, htmap = np.ascontiguousarray(image), np.ascontiguousarray(htmap)

    return image, htmap


def z_score(arr):
    arr = arr.astype(np.float)
    return (arr - arr.mean()) / (arr.std() + EPS)


def convert_image_heatmap_pair_to_img_n_save(image, htmap, save_name):
    image = arr_to_8bit_img(image)
    if len(htmap.shape) > 2:
        htmap = htmap.max(axis=0)
    htmap = arr_to_8bit_img(htmap)
    htmap = cv2.applyColorMap(htmap, cv2.COLORMAP_JET)
    htmap = cv2.addWeighted(src1=np.tile(image[..., None], [1, 1, 3]), alpha=1., src2=htmap, beta=.2, gamma=0)
    cv2.imwrite(save_name, htmap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def convert_volume_label_pair_2_img_n_save(volume, label, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    volume = arr_to_8bit_img(volume)

    for i, (image, htmap) in enumerate(zip(volume, label)):
        if len(htmap.shape) > 2:
            htmap = htmap.max(axis=0)
        htmap = arr_to_8bit_img(htmap)
        htmap = cv2.applyColorMap(htmap, cv2.COLORMAP_JET)
        htmap = cv2.addWeighted(src1=np.tile(image[..., None], [1, 1, 3]), alpha=1., src2=htmap, beta=.2, gamma=0)
        cv2.imwrite(os.path.join(save_dir, '%02d.png' % i), htmap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
