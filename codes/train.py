import argparse
import torch
import logging
import cv2
import math
import numpy as np
from options import parse, check_resume, NoneDict, dict_to_nonedict, dict2str
from utils import mkdir_and_rename, setup_logger, mkdirs, set_random_seed
import utils.utils as utils
from data import create_dataset, create_dataloader
from models import create_model
import os.path as osp

def main():
    #### options
    parser = argparse.ArgumentParser(description='EDVR options')
    parser.add_argument('-opt', type=str, help='Path to YAML file')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt)
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        mkdir_and_rename(opt['path']['experiments_root'])
        mkdirs((path for key, path in opt['path'].items() if not key=='experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, screen=True, tofile=True)  
    logger = logging.getLogger('base')
    #logger.info('\n'+dict2str(opt))

    # convert to NoneDict, return None for missing keys
    opt = dict_to_nonedict(opt)

    #### ramdom seed
    seed = opt['train']['manual_seed']
    logger.info('Random seed: {}'.format(seed))
    set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    #### create train and val dataloader
    dataset_ratio = 200
    for phase,  dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # create train dataset
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            
            # create train dataloader
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            #logging
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        else:
            # create val dataset
            val_set = create_dataset(dataset_opt)
            res = val_set[0]
            print('num of val_set: ', len(val_set))
            #create val dataloader
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            #logging
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        
    #create model
    model = create_model(opt) 

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0
    
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    psnr_max = 0.0
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.4f},'.format(v)
                message += ')]'
                for k,v in logs.items():
                    message += '{:s}: {:.4f} '.format(k, v)
                logger.info(message)

            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                pbar = utils.ProgressBar(len(val_loader))
                psnr_rlt = {} # store psnr of each folder
                psnr_total_avg = 0.
                for val_data in val_loader:
                    folder = val_data['folder'][0]
                    
                    #### forward
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    rlt_img = utils.tensor2img(visuals['rlt'])  #uint8
                    gt_img = utils.tensor2img(visuals['GT'])    #uint8
                
                    #### calculate PSNR
                    psnr = utils.calculate_psnr(rlt_img, gt_img)
                    psnr_rlt[folder] = psnr
                    pbar.update('Test {}'.format(folder))
                    psnr_total_avg += psnr

                cv2.imwrite(osp.join(opt['path']['val_images'], '{}.png'.format(current_step)), rlt_img)
                psnr_total_avg /= len(psnr_rlt)
                log_s = '# Validation # PSNR {:.4f}:'.format(psnr_total_avg)
                #psnr for each folder
                #for k,v in psnr_rlt.items():
                    #log_s += '\n{}: {:.4f}'.format(k, v)
                
                logger.info(log_s)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if psnr_total_avg > psnr_max:
                    logger.info('Saving temporal best models and training states.')
                    model.save('best')
                    model.save_training_state(epoch, current_step)
                    psnr_max = psnr_total_avg
                    
    logger.info('Saving the final model.')
    #model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()  
