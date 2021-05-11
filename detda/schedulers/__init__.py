import logging
from detda.schedulers.schedulers import *
from copy import deepcopy

logger = logging.getLogger('detda')

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'multi_step': MultiStepLR,
                 'cosine_annealing': CosineAnnealingLR,
                 'exp_lr': ExponentialLR,
                 "inv_lr": InvLR}


def get_scheduler(optimizer, scheduler_dict):
    temp_scheduler_dict = deepcopy(scheduler_dict)  # 保护一下
    if temp_scheduler_dict is None:
        logger.info('Using No LR Scheduling')
        return ConstantLR(optimizer)

    s_type = temp_scheduler_dict['name']
    temp_scheduler_dict.pop('name')

    logging.info('Using {} scheduler with {} params'.format(s_type,
                                                            temp_scheduler_dict))

    warmup_dict = {}
    if 'warmup_iters' in temp_scheduler_dict:
        # This can be done in a more pythonic way... 
        warmup_dict['warmup_iters'] = temp_scheduler_dict.get('warmup_iters', 100)
        warmup_dict['mode'] = temp_scheduler_dict.get('warmup_mode', 'linear')
        warmup_dict['gamma'] = temp_scheduler_dict.get('warmup_factor', 0.2)

        logger.info('Using Warmup with {} iters {} gamma and {} mode'.format(
            warmup_dict['warmup_iters'],
            warmup_dict['gamma'],
            warmup_dict['mode']))

        temp_scheduler_dict.pop('warmup_iters', None)
        temp_scheduler_dict.pop('warmup_mode', None)
        temp_scheduler_dict.pop('warmup_factor', None)

        base_scheduler = key2scheduler[s_type](optimizer, **temp_scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    return key2scheduler[s_type](optimizer, **temp_scheduler_dict)
