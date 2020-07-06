import os
import os.path as osp
import logging
import yaml
import logging

def parse(opt_path, is_train=True):
    with open(opt_path, 'r') as f:
        opt = yaml.load(f.read())
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join([str(x) for x in opt['gpu_ids']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_list))
    
    opt['is_train'] = is_train
    if opt['distortion'] == 'sr':
        scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            if dataset['dataroot_GT'].endswith('.lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            if dataset['dataroot_LQ'].endswith('.lmdb'):
                is_lmdb = True

        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if opt['distortion'] == 'sr':
            dataset['scale'] = scale

    # path
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(opt['path']['experiments_root'], 'models')
        opt['path']['training_state'] = osp.join(opt['path']['experiments_root'], 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

        #debug
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['train']['niter'] = 20
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8

    else: #test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    if opt['distortion'] == 'sr':   
        opt['network_G']['scale'] = scale

    return opt


def dict2str(opt, indent=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + k + ':' + '[\n'
            msg += dict2str(v, indent+1)
            msg += ' ' * (indent * 2) + ']\n'
        else:
            msg += ' ' * (indent * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt):
    logger = logging.getLogger('base')
    if opt['path'].get('pretrain_model_G', None):
        logger.warning('pretrain_model path will be ignored when resuming training.')
    opt['path']['pretrain_model_G'] = opt['path']['resume_state']
    logger.info('Set [pretrain_model_G] to {}'.format(opt['path']['resume_state'])) 
