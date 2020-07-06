import logging
logger = logging.getLogger('base')
from .Video_base_model import VideoBaseModel as M

def create_model(opt):
    model = opt['model']
    if model == 'video_base':
        m = M(opt)
    else:
        raise NotImplementedError('Model [{:s)] not recognized.'.format(model))
    
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
