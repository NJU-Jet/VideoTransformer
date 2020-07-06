import os.path as osp
import torch
import torch.utils.data as data
import data.utils as utils
import glob
import numpy as np
import cv2

class VideoTestDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']

        #### Generate data and cache data
        if opt['phase'] != 'val':
            raise NotImplementedError()
        val_txt = '/home/lj/dzc/vimeo/vimeo_septuplet/sep_testlist.txt'
        folder = []
        with open(val_txt) as f:
            folder = f.readlines()
            folder = [v.strip() for v in folder]
        folder = folder[::50]
        self.folder = folder

    def __getitem__(self, index):
        folder_name = self.folder[index]
        folder1, folder2 = folder_name.split('/')[0], folder_name.split('/')[1]
        img_paths_LQ = glob.glob(osp.join(self.LQ_root, folder1, folder2, '*'))
        dir = osp.join(self.LQ_root, folder1, folder2)
        img_path_GT = osp.join(self.GT_root, folder1, folder2, 'im4.png')
        #### read img
        imgs_LQ = utils.read_img_seq(img_paths_LQ)
        img_GT = utils.read_img(None, img_path_GT) #HWC BGR, numpy
        img_GT = img_GT[:, :, ::-1]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2,0,1)))).float()
        

        return {
                'LQs': imgs_LQ, # NCHW, float tensor
                'GT': img_GT,   # CHW, float tensor
                'folder': folder_name,
                }


    def __len__(self):
        return len(self.folder)
