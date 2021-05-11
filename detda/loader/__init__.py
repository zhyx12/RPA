import json
import numpy as np
from detda.loader.det_loaders import process_one_det_dataset


def get_data_path(name, config_file="config.json"):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]["data_path"]


def parse_args_for_dataset(dataset_args, debug=False, logger=None, train_debug_sample_num=10,
                           test_debug_sample_num=10, random_seed=1234, data_root=None, task_type=None):
    """

    :param dataset_args:
    :param debug:
    :param logger:
    :param train_debug_sample_num:
    :param test_debug_sample_num:
    :return: 返回一个list,包含3个或4个loader，3个比4个少一个src_val_loader
    """
    if debug:
        print("YOU ARE IN DEBUG MODE!!!!!!!!!!!!!!!!!!!")

    # Setup task type
    if task_type == 'det':
        process_one_dataset = process_one_det_dataset

    else:
        raise RuntimeError('wrong dataset task name {}'.format(task_type))

    # Setup Augmentations
    trainset_args = dataset_args['train']
    testset_args = dataset_args['test']
    train_augmentations = trainset_args.get('augmentations', None)
    test_augmentations = testset_args.get('augmentations', None)
    # train_data_aug = augmentation_func(train_augmentations, logger)
    # test_data_aug = augmentation_func(test_augmentations, logger)

    # Setup Dataloader
    # 其它参数
    train_batchsize = trainset_args['batch_size']
    test_batchsize = testset_args['batch_size']
    n_workers = dataset_args['n_workers']
    drop_last = dataset_args.get('drop_last', True)

    # 训练集
    train_loaders = []
    for i in range(1, 100):
        if i in trainset_args.keys():
            temp_train_aug = trainset_args[i].get('augmentation', None)
            temp_train_aug = train_augmentations if temp_train_aug is None else temp_train_aug
            temp_train_loader = process_one_dataset(trainset_args[i], augmentations=temp_train_aug,
                                                    transfer_label=True,
                                                    batch_size=train_batchsize, n_workers=n_workers, shuffle=True,
                                                    debug=debug,
                                                    sample_num=train_debug_sample_num, drop_last=drop_last,
                                                    data_root=data_root, logger=logger, training=True,
                                                    random_seed=random_seed)
            train_loaders.append(temp_train_loader)
        else:
            break

    # 测试集
    test_loaders = []
    for i in range(1, 100):
        if i in testset_args.keys():
            temp_test_aug = testset_args[i].get('augmentation', None)
            temp_test_aug = test_augmentations if temp_test_aug is None else temp_test_aug
            temp_test_loader = process_one_dataset(testset_args[i], augmentations=temp_test_aug,
                                                   transfer_label=False, batch_size=test_batchsize,
                                                   n_workers=n_workers,
                                                   shuffle=False, debug=debug,
                                                   sample_num=test_debug_sample_num,
                                                   drop_last=False, data_root=data_root, logger=logger,
                                                   random_seed=random_seed, training=False,
                                                   )
            test_loaders.append(temp_test_loader)

    return train_loaders, test_loaders
