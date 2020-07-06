import logging
import torch
import torch.utils.data
from .Vimeo90K import Vimeo90KDataset
from .VideoTestData import VideoTestDataset

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size = dataset_opt['batch_size']
        shuffle = True
        
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler, drop_last=True, pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'Vimeo90K':
        dataset = Vimeo90KDataset(dataset_opt)
    elif mode == 'video_test':
        dataset = VideoTestDataset(dataset_opt)
    else:
        print('Not Implemented yet!')
        exit()

    logger = logging.getLogger('base')
    logger.info('[{}]  Dataset [{:s} - {:s}] is created.'.format(dataset_opt['phase'], dataset.__class__.__name__, dataset_opt['name']))

    return dataset
