from copy import deepcopy
import torch.nn as nn
from detda.models.det_models import parse_args_for_one_det_model


def parse_args_for_models(model_args, n_classes, logger=None, task_type=None):
    # setup task type
    if task_type == 'det':
        parse_args_for_one_model = parse_args_for_one_det_model
    else:
        raise RuntimeError('wrong dataset task name {}'.format(task_type))

    shared_lr_scheduler_param = model_args['lr_scheduler']
    model_args.pop('lr_scheduler')
    model_dict = nn.ModuleDict()
    optimizer_dict = {}
    scheduler_dict = {}
    device_dict = {}
    for key in model_args:
        temp_res = parse_args_for_one_model(model_args[key], n_classes, shared_lr_scheduler_param, logger=logger)
        model_dict[key] = temp_res[0]
        optimizer_dict[key] = temp_res[1]
        if logger is not None:
            logger.info("Using optimizer {} for {} model parameters".format(temp_res[1], key))
        scheduler_dict[key] = temp_res[2]
        device_dict[key] = temp_res[3]
    # print('type are {}, {}, {}, {}'.format(type(model_dict), type(optimizer_dict), type(scheduler_dict),
    #                                        type(device_dict)))
    # print('device dict {}'.format(device_dict))
    # print('orig keys {}'.format(optimizer_dict.keys()))
    return model_dict, optimizer_dict, scheduler_dict, device_dict
