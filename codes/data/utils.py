import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import os.path as osp

def _get_paths_from_images(path):
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            img_path = osp.join(dirpath, fname)
            images.append(img_path)

    return images


def _get_paths_from_lmdb(dataroot):
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    return paths, sizes


def get_image_paths(data_type, dataroot):
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
    return paths, sizes


def _read_img_lmdb(env, key, size):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    if env is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    return img


def read_img_seq(path):
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(osp.join(path, '*')))
    img_l = [read_img(None, v) for v in img_path_l]

    # stack to torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()

    return imgs

def augment(img_list, hflip=True, rot=True):
    '''horizontal flip OR rotate (0, 90, 180, 270 degrees)'''
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    augmented = []
    for img in img_list:
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        augmented.append(img)
    
    return augmented
