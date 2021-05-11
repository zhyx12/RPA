# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from copy import deepcopy
import torch
from torch.utils import data
import numpy as np
from torch.utils.data import RandomSampler
from detda.loader.det_loaders.roi_data_layer import combined_roidb, roibatchLoader
from torch.utils.data import Sampler


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def process_one_det_dataset(args, augmentations, transfer_label, batch_size, n_workers, shuffle, debug=False,
                            sample_num=10, drop_last=True, data_root=None, logger=None, training=None,
                            random_seed=1234):
    # def _init_fn(worker_id):
    #     np.random.seed(int(random_seed))

    dataset_params = deepcopy(args)
    dataset_params.pop('name')
    # print('name {}, param {}, training {}'.format(args['name'], dataset_params, training))
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args['name'], training=training,
                                                          dataset_params=dataset_params)
    train_size = len(roidb)
    if logger:
        logger.info('{:d} roidb entries'.format(train_size))

    if training:
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, imdb.num_classes, training=True)
        dataset.split = dataset_params['split'] if 'split' in dataset_params else 'train'
    else:
        # imdb.competition_mode(on=True)  # 主要用于清除检测结果的pkl文件
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, imdb.num_classes, training=False,
                                 normalize=False, path_return=True)
        dataset.split = dataset_params['split'] if 'split' in dataset_params else 'test'
    dataset.name = args['name']
    dataset.imdb = imdb
    #
    if debug:
        print('dataset has {} images'.format(len(dataset)))
        random_sampler = RandomSampler(dataset, replacement=True, num_samples=sample_num)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                                 sampler=random_sampler)
    else:
        if training:
            temp_sampler = sampler(train_size, batch_size)
            loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, sampler=temp_sampler,
                                     pin_memory=True)
        else:
            loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False,
                                     pin_memory=True)
    return loader
