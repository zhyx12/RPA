# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import copy
import os
from .faster_rcnn import vgg16, resnet, Det_Dis_1, Det_Dis_2, \
    Det_Dis_3, Det_Dis_inst, BasicDetAdvVgg16, RPNClusterAlignVgg16, \
    RPNClusterAlignResnet,  \
     Det_Dis_2_Conv, Det_Dis_3_Conv


def get_model(model_dict, n_classes, version=None, deepcopy=True):
    name = model_dict['name']
    model_class = _get_model_instance(name)
    if deepcopy:
        param_dict = copy.deepcopy(model_dict)
    else:
        param_dict = model_dict
    param_dict.pop('name')

    if 'vgg16' in name:
        model = model_class(n_classes, pretrained=True, **param_dict)
        model.create_architecture()
    elif 'res50' in name or 'resnet50' in name:
        model = model_class(n_classes, num_layers=50, pretrained=True, **param_dict)
        model.create_architecture()
    elif 'res101' in name or 'resnet101' in name:
        model = model_class(n_classes, num_layers=101, pretrained=True, **param_dict)
        model.create_architecture()
    elif 'det_dis' in name:
        model = model_class(**param_dict)
    else:
        raise RuntimeError('wrong model name {} in detection task'.format(name))
    #

    return model


def _get_model_instance(name):
    try:
        return {
            "vgg16": vgg16,
            'res101': resnet,
            'res50': resnet,
            'basicdetadvvgg16': BasicDetAdvVgg16,
            "det_dis_1": Det_Dis_1,
            "det_dis_2": Det_Dis_2,
            "det_dis_3": Det_Dis_3,
            'det_dis_2_conv': Det_Dis_2_Conv,
            'det_dis_3_conv': Det_Dis_3_Conv,
            "det_dis_inst": Det_Dis_inst,
            "rpnclusteralignvgg16": RPNClusterAlignVgg16,
            "rpnclusteralignresnet50": RPNClusterAlignResnet,
            "rpnclusteralignresnet101": RPNClusterAlignResnet,
        }[name]
    except:
        raise RuntimeError("Model {} not available".format(name))


def parse_args_for_one_det_model(model_args, n_classes, scheduler_args, logger=None, deepcopy_for_model_args=True):
    """
    输入带名字的字典，
    :param model_args: 类型是字典，名字就是model，optimizer，scheduler的名字的前者
    :param n_classes:
    :param scheduler_args:
    :param logger:
    :return:
    """
    from detda.optimizers import get_optim_param_by_name
    from detda.optimizers import get_optimizer
    from detda.schedulers import get_scheduler

    assert len(model_args) == 2 or len(model_args) == 3, 'model args should have 2 or 3 keys, but there are {}'.format(
        model_args.keys())
    # 获取参数
    model_params = model_args['model']
    optimizer_params = model_args['optimizer']
    scheduler_params = model_args.get('scheduler', None)
    device_params = model_args.get('device', None)
    if isinstance(device_params, int):
        device_params = [device_params, ]
    if scheduler_params is None:
        scheduler_params = scheduler_args
    # 构造
    temp_model = get_model(model_params, n_classes, deepcopy=deepcopy_for_model_args)
    optim_model_param_name = optimizer_params.get('optim_param_name', None)
    # print('create optimizer for {}'.format(model_params['name']))
    optim_model_param = get_optim_param_by_name(temp_model, layer_name=optim_model_param_name,
                                                lr=optimizer_params['lr'],
                                                logger=logger)
    temp_optimizer = get_optimizer(optim_model_param, optimizer_params)
    temp_scheduler = get_scheduler(temp_optimizer, scheduler_params)

    return (temp_model, temp_optimizer, temp_scheduler, device_params)


if __name__ == "__main__":
    pass
