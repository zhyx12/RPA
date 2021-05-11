# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import torch.nn as nn
import os
# import tensorboardX as tb
import torch.utils.tensorboard as tb
from detda.modules.sync_batchnorm import DataParallelWithCallback
from torch.nn import DataParallel
import glob
from detda.runner.hooks import _build_hook, _register_hook
from detda.runner.hooks import LrRecoder, TrainTimeRecoder
import time
from torch.cuda.amp import GradScaler


class BaseTrainer(object):
    def __init__(self, cuda, model_dict, optimizer_dict, scheduler_dict, device_dict=None, train_loaders=None,
                 test_loaders=None,
                 logger=None, logdir=None, max_iters=None, val_interval=5000, log_interval=100, save_interval=10000,
                 epoch_interval=10000,
                 update_iter=1, save_test_res=False, max_save_num=1, use_syncbn=False,
                 cudnn_deterministic=False, break_flag_in_val=False, use_amp=False):

        assert train_loaders is not None or test_loaders is not None, 'you must specify one dataloaders'
        self.cuda = cuda
        self.use_syncbn = use_syncbn
        self.logger = logger
        # 加载model
        self.model_dict = model_dict
        # 模型各部分部署的gpu
        self.device_dict = device_dict
        # model 移动到GPU上
        self.move_models_to_gpu()
        # 加载optimizer
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict
        self.break_flag_in_val = break_flag_in_val

        if isinstance(train_loaders, torch.utils.data.DataLoader):
            self.train_loaders = (train_loaders,)
        else:
            self.train_loaders = train_loaders
        if isinstance(test_loaders, torch.utils.data.DataLoader):
            self.test_loaders = (test_loaders)
        else:
            self.test_loaders = test_loaders
        self.train_loader_iterator = [item.__iter__() for item in self.train_loaders]
        #
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.writer = tb.SummaryWriter(log_dir=logdir)

        self.iteration = 1
        self.epoch_iter = 0
        self.max_iters = max_iters

        self.val_interval = self._deal_with_val_interval(val_interval, max_iters=max_iters)

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.epoch_interval = epoch_interval

        self.max_save_model = max_save_num
        self.update_iter = update_iter
        self.save_test_res = save_test_res
        self.cudnn_deterministic = cudnn_deterministic
        #
        self._hooks = []
        self.train_batch_output = {}
        self.use_amp = use_amp
        #
        lr_recoder = LrRecoder(self, log_interval)
        train_time_recoder = TrainTimeRecoder(self)
        self.register_hook(lr_recoder, priority='HIGH')
        self.register_hook(train_time_recoder)

    def train_iter(self, *args):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        torch.backends.cudnn.enabled = True

        if self.cudnn_deterministic:
            self.logger.info('Using cudnn deterministic model')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        # 设置训练标志
        self.set_train_state()
        # 设置scalar, 如果使用amp的话
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        # 根据scheduler的记录设置迭代次数
        self.iteration = None
        for name in self.scheduler_dict:
            temp_iteration = self.scheduler_dict[name].last_epoch
            if self.iteration is None:
                self.iteration = temp_iteration
            else:
                assert self.iteration == temp_iteration, 'iteration in {} is {}, different with others'.format(
                    name + '_scheduler', temp_iteration)
        self.iteration = 1 if self.iteration == -1 else self.iteration + 1  # 为了下面训练时从1开始
        #
        for i, loader in enumerate(self.train_loaders):
            self.logger.info('{} loader has {} training images'.format(i, len(loader.dataset)))
        #
        train_loader_num = len(self.train_loader_iterator)
        #
        while self.iteration <= self.max_iters:
            # if int(self.iteration % self.epoch_interval) == self.epoch_iter:
            #     self.call_hook('before_train_epoch')
            #
            all_data = []
            for ind in range(train_loader_num):
                try:
                    all_data.append(next(self.train_loader_iterator[ind]))
                except StopIteration:
                    self.logger.info("Iteration on a new Dataloader of ind {}".format(ind))
                    self.train_loader_iterator[ind] = self.train_loaders[ind].__iter__()
                    time.sleep(2)
                    all_data.append(next(self.train_loader_iterator[ind]))
            # 数据移动到GPU上
            relocated_data = all_data
            # TODO: 在rpn_cluster_align中需要手动move到gpu，以防止illegal memory access的错误
            if self.cuda:
                for ind1, item in enumerate(all_data):
                    for ind2, sub_item in enumerate(item):
                        if isinstance(sub_item, torch.Tensor):
                            relocated_data[ind1][ind2] = all_data[ind1][ind2].cuda()
            # train one batch and update running metrics
            self.call_hook('before_train_iter')
            # with torch.autograd.detect_anomaly():
            self.train_batch_output = self.train_iter(*relocated_data)
            self.call_hook('after_train_iter')
            # 保存模型
            if self.iteration % self.save_interval == 0 or self.iteration == self.max_iters:
                self.save_models()
            # 验证
            # 取当前的interval
            next_milestone, current_interval = list(self.val_interval.items())[0]
            if self.iteration % current_interval == 0 or self.iteration == self.max_iters:
                time.sleep(2)
                save_flag = self.validator(self.iteration)
                self.set_train_state()
                if save_flag:
                    save_path = os.path.join(self.logdir, "best_model.pth".format(self.iteration))
                    torch.save(self.state_dict(), save_path)
                # 检查是否应该进入下一个阶段的val_interval
                if self.iteration + current_interval > next_milestone and next_milestone < self.max_iters:
                    self.val_interval.pop(next_milestone)
                # 清理显存
                torch.cuda.empty_cache()
            if self.iteration == self.max_iters:
                break
            self.iteration += 1
            # if self.iteration % self.epoch_interval == 0:
            #     self.call_hook('after_train_epoch')
        self.writer.close()

    def state_dict(self):
        state_dict = {}
        for key in self.model_dict.keys():
            state_dict[key] = self.model_dict[key].state_dict()
            state_dict[key + '_optimizer'] = self.optimizer_dict[key].state_dict()
            state_dict[key + '_scheduler'] = self.scheduler_dict[key].state_dict()
        return state_dict

    def resume_training(self, file):
        if os.path.isfile(file):
            self.logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(file)
            )
            checkpoint = torch.load(file)
            for key in checkpoint:
                if key.endswith('optimizer'):
                    # print('keys {}'.format(list(self.optimizer_dict.keys())))
                    assert key[0:-10] in self.optimizer_dict.keys(), '{} not in base names'.format(key)
                    self.optimizer_dict[key[0:-10]].load_state_dict(checkpoint[key])
                elif key.endswith('scheduler'):
                    assert key[0:-10] in self.scheduler_dict.keys(), '{} not in base names'.format(key)
                    self.scheduler_dict[key[0:-10]].load_state_dict(checkpoint[key])
                elif key in self.model_dict.keys():
                    assert key in self.model_dict.keys(), '{} not in base names {}'.format(key, self.model_dict.keys())
                    self.model_dict[key].load_state_dict(checkpoint[key])
                else:
                    self.logger.info('Not loaded key {} in checkpoint file'.format(key))
        else:
            raise RuntimeError("No checkpoint found at '{}'".format(file))

    def register_hook(self, hook, priority='NORMAL'):
        _register_hook(self, hook, priority)

    def build_hook(self, args, hook_type=None):
        _build_hook(self, args, hook_type)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def print_attr_info(self):
        # 输出trainer的属性信息，即各个参数值
        for key, val in vars(self).items():
            if isinstance(val, DataParallel):
                val = type(val)
            self.logger.info('{}:\t\t\t\t{}'.format(key, val))

    def set_train_state(self):
        # 为模型设置训练标志
        for key in self.model_dict.keys():
            self.model_dict[key].train()

    def set_eval_state(self):
        # 为模型设置训练标志
        for name in self.model_dict.keys():
            self.model_dict[name].eval()

    def move_models_to_gpu(self):
        # 将模型分配到GPU上
        if self.cuda and torch.cuda.is_available():
            # 如果有模型是指定gpu的，那么默认的gpu_ids应该只有0，否则，取所有的gpu
            specified_gpu_flag = False
            for key, val in self.model_dict.items():
                if self.device_dict is not None and self.device_dict[key] is not None:
                    specified_gpu_flag = True
                    break
            devices_num = torch.cuda.device_count()
            # print('specific gpu flag is {}'.format(specified_gpu_flag))
            gpu_ids = list(range(0, devices_num)) if not specified_gpu_flag else [0, ]  # 还是假设从0开始依次排列
            self.logger.info('There are {} gpus'.format(devices_num))
            self.base_cuda = torch.device(gpu_ids[0])
            for key, val in self.model_dict.items():
                if isinstance(val, nn.Module):
                    if self.device_dict is not None and self.device_dict[key] is not None:
                        temp_gpu_ids = self.device_dict[key]
                    else:
                        temp_gpu_ids = gpu_ids
                    # print('gpu ids for {} is {}'.format(key, temp_gpu_ids))
                    self.model_dict[key].to('cuda:{}'.format(temp_gpu_ids[0]))
                    if self.use_syncbn:
                        self.model_dict[key] = DataParallelWithCallback(val, device_ids=temp_gpu_ids)
                    else:
                        self.model_dict[key] = DataParallel(val, device_ids=temp_gpu_ids)
        else:
            self.logger.info('Not Using CUDA')
            self.cuda = False

    def save_models(self):
        save_path = os.path.join(self.logdir, "iter_{}_model.pth".format(self.iteration))
        #
        search_template = self.logdir + '/' + 'iter_*_model.pth'
        saved_files = glob.glob(search_template)
        if len(saved_files) >= self.max_save_model:
            sorted_files_by_ctime = sorted(saved_files, key=lambda x: os.path.getctime(x))
            os.remove(sorted_files_by_ctime[0])
        torch.save(self.state_dict(), save_path)

    def _deal_with_val_interval(self, val_interval, max_iters):
        from collections import OrderedDict
        new_val_interval = OrderedDict()
        if isinstance(val_interval, (int, float)):
            new_val_interval[max_iters] = val_interval
            return new_val_interval
        elif isinstance(val_interval, dict):
            checkpoint_list = sorted(val_interval.keys())
            assert checkpoint_list[0] > 0 and checkpoint_list[-1] <= max_iters, 'check val interval keys'
            for val in checkpoint_list:
                new_val_interval[val] = val_interval[val]
            # 如果最后一个key不是max_iter，补充成最后一个的间隔
            if checkpoint_list[-1] < max_iters:
                new_val_interval[max_iters] = val_interval[checkpoint_list[-1]]
            return new_val_interval
        else:
            raise RuntimeError('only single value or dict is acceptable for val interval')

    def zero_grad_all(self):
        for name in self.optimizer_dict.keys():
            self.optimizer_dict[name].zero_grad()

    def step_grad_all(self):
        for name in self.optimizer_dict.keys():
            self.optimizer_dict[name].step()

    def scheduler_step_all(self):
        for name in self.scheduler_dict.keys():
            self.scheduler_dict[name].step()
