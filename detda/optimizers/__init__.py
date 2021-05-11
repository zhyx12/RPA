import copy
import logging
import functools

from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop

logger = logging.getLogger('detda')

key2opt = {'sgd': SGD,
           'adam': Adam,
           'asgd': ASGD,
           'adamax': Adamax,
           'adadelta': Adadelta,
           'adagrad': Adagrad,
           'rmsprop': RMSprop, }


def get_optimizer(model_params, optimizer_dict):
    param_dict = copy.deepcopy(optimizer_dict)
    param_dict.pop('name')
    if optimizer_dict is None:
        logger.info('Using SGD optimizer')
        return SGD(model_params, **param_dict)
    else:
        opt_name = optimizer_dict['name']
        if opt_name not in key2opt:
            raise NotImplementedError('Optimizer {} not implemented'.format(opt_name))
        logger.info('Using {} optimizer'.format(opt_name))
        return key2opt[opt_name](model_params, **param_dict)


def get_optim_param_by_name(base_model, layer_name=None, lr=None, logger=None):
    # 如果有optim_parameters()方法，则调用方法
    if hasattr(base_model, 'optim_parameters'):
        logger.info('Use optim_parameters within the model')
        return base_model.optim_parameters(lr=lr)

    optim_num = 0
    optim_param = []

    for name, param in base_model.named_parameters():
        if param.requires_grad:
            optim_param.append(param)
            optim_num += 1
            logger.info('{} will be optimized'.format(name))
        else:
            logger.info('{} will be ignored'.format(name))
    return optim_param
