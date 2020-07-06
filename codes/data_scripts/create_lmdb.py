import sys
import os
import os.path as osp
import numpy as np
import lmdb
import cv2
import glob
import pickle

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import ProgressBar


def main():
    dataset = 'vimeo90K'
    mode = 'LR'
    if dataset == 'vimeo90K':
        vimeo90K(mode)
    elif dataset == 'test':
        test_lmdb('/data/dzc/vimeo90K_train_GT.lmdb')
    elif dataset == 'pkl':
        generate_pkl('/data/dzc/vimeo90K_train_GT.lmdb', '/data/dzc/vimeo_septuplet/sep_trainlist.txt', 256, 448)

def vimeo90K(mode):
    BATCH = 5000
    if mode == 'GT':
        img_folder = '/data/dzc/vimeo_septuplet/sequences'
        lmdb_save_path = '/data/dzc/vimeo90K_train_GT.lmdb'
        txt_file = '/data/dzc/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 256, 448
    elif mode == 'LR':
        img_folder = '/data/dzc/vimeo_septuplet_matlabLRx4/sequences'
        lmdb_save_path = '/data/dzc/vimeo90K_train_LR.lmdb'
        txt_file = '/data/dzc/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 64, 112

    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the images paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        all_img_list.extend(glob.glob(osp.join(img_folder, folder, sub_folder, '*')))
        
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j+1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'GT':
        print('Only keep the 4th frame.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    #### write data to lmdb
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size*10)
    txn = env.begin(write=True)
    pbar = ProgressBar(len(all_img_list))
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        H, W, C = data.shape
        txn.put(key_byte, data)
        if idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb')

    #### create meta information
    meta_info = {}
    if mode == 'GT':
        meta_info['name'] = 'vimeo90K_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'vimeo90K_train_LR'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = list(key_set)
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')
    test_lmdb(lmdb_save_path)


def generate_pkl(lmdb_save_path, txt_file, H, W):
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    
    keys = []
    for line in train_l:
        a, b = line.split('/')
        keys.append('{}_{}'.format(a,b))
    meta_info = {}
    meta_info['name'] = 'vimeo90K_train_GT'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H, W)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), 'wb'))
    print('Finish creating lmdb meta info.')

def test_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), 'rb'))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))

    # read one image
    key = '00096_0936_7'
    print('Reading {} for test'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)


if __name__ == '__main__':
    main()
