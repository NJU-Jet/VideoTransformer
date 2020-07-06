import os
import os.path as osp
import logging
import numpy as np
import cv2
import torch
import torch.utils.data as data
import lmdb
import random
import pickle
import data.utils as utils

logger = logging.getLogger('base')

class Vimeo90KDataset(data.Dataset):
    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = True        
        
        #### determine the LQ frames first
        self.LQ_frames_list = []
        for i in range(opt['N_frames']):
            self.LQ_frames_list.append(i + (9 - opt['N_frames']) // 2)

        #### directly load image keys what about val set?
        if self.data_type == 'lmdb':
            self.paths_GT, _ = utils.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        else:
            raise NotImplementedError()
            val_path = '/home/lj/dzc/vimeo/vimeo_septuplet/sep_testlist.txt'
            with open(val_path) as f:
                val_l = f.readlines()
                val_l = [v.strip() for v in val_l]
            self.paths_GT = []
            for line in val_l:
                folder, subfloder = line.split('/')
                self.paths_GT.append('{}_{}'.format(folder, subfloder))
            self.paths_GT = self.paths_GT[::50]


        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None

    def _init_lmdb(self):
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.paths_GT)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')     

        #### get GT image
        if self.data_type == 'lmdb':
            img_GT = utils.read_img(self.GT_env, key + '_4', (3, 256, 448))
        else:
            img_GT = utils.read_img(None, osp.join(self.GT_root, name_a, name_b, 'im4.png'))

        #### get LQ images
        LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)
        img_LQ_l = []
        for v in self.LQ_frames_list:
            if self.data_type == 'lmdb':
                img_LQ = utils.read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)
            else:
                img_LQ = utils.read_img(None, osp.join(self.LQ_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LQ_l.append(img_LQ)

        # augmentation - flip, rotate
        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple
            img_LQ_l.append(img_GT)
            rlt = utils.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        #BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, ::-1]
        img_LQs = img_LQs[:, :, :, ::-1]
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT.transpose(2, 0, 1))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(img_LQs.transpose(0, 3, 1, 2))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}
