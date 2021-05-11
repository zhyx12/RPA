# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import os
from detda.runner.hooks import _build_hook, _register_hook
import time

class BaseValidator(object):
    def __init__(self, cuda, logdir, test_loaders, model_dict, logger=None, writer=None,
                 log_name=('',), comparison_key=None, img_show_num=0,
                 align_corners=False):
        self.cuda = cuda
        self.logdir = logdir
        self.test_loaders = {}
        for ind, loader in enumerate(test_loaders):
            self.test_loaders[loader.dataset.name + '_' + loader.dataset.split] = loader
        self.logger = logger
        self.writer = writer
        self.best_metrics = None
        self.batch_output = {}  # 每一次迭代产生的结果
        self.start_index = 0
        self.log_name = log_name
        self.class_num = test_loaders[0].dataset.n_classes
        self.align_corners = align_corners
        self.img_show_num = img_show_num
        # 设置网络
        self.model_dict = model_dict
        #
        self._hooks = []
        self.save_flag = False
        self.val_iter = None
        self.comparison_key = comparison_key
        # 注册基本的分割指标
        # seg_val_hook = SegValMetrics(runner=self, align_corners=self.align_corners, comparison_key=self.comparison_key)
        # self.register_hook(seg_val_hook)

    def eval_iter(self, val_batch_data):
        raise NotImplementedError

    def __call__(self, iteration):
        print('start validator')
        self.iteration = iteration
        self.save_flag = False
        for key in self.model_dict:
            self.model_dict[key].eval()
        #
        self.call_hook('before_val_epoch')
        # 测试
        for key, loader in self.test_loaders.items():
            self.val_dataset_key = key
            time.sleep(2)
            for val_iter, val_data in enumerate(loader):
                #
                # start_time = time.time()
                relocated_data = val_data
                if self.cuda:
                    for ind1, item in enumerate(val_data):
                        for ind2, sub_item in enumerate(item):
                            if isinstance(sub_item, torch.Tensor):
                                relocated_data[ind1][ind2] = val_data[ind1][ind2].cuda()
                #
                # print('alloc time {}'.format(time.time()-start_time))
                # start_time = time.time()
                self.val_iter = val_iter
                self.call_hook('before_val_iter')
                # print('before time {}'.format(time.time()-start_time))
                # start_time = time.time()
                self.batch_output = self.eval_iter(relocated_data)
                # print('forward time {}'.format(time.time()-start_time))
                self.batch_output.update({'dataset_name': key})
                # start_time = time.time()
                self.call_hook('after_val_iter')
                # print('after time {}'.format(time.time()-start_time))
        # 放在eval_on_dataloader里面，另一个dataloader上的metric没有update过，count=0
        self.call_hook('after_val_epoch')
        #
        torch.cuda.empty_cache()
        # 设置训练模式
        for key in self.model_dict:
            self.model_dict[key].train()
        return self.save_flag

    def make_save_dir(self):
        # 创建文件夹保存图像文件，每一次验证都根据当前的迭代次数创建一个文件夹
        self.label_save_path = os.path.join(self.logdir, 'iter_{}_results'.format(self.iteration))
        if not os.path.exists(self.label_save_path):
            os.makedirs(self.label_save_path)

    def register_hook(self, hook, priority='NORMAL'):
        _register_hook(self, hook, priority)

    def build_hook(self, args, hook_type=None):
        _build_hook(self, args, hook_type)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            # print(fn_name,type(hook))
            start_time = time.time()
            getattr(hook, fn_name)(self)
            # print('use tiem {}'.format(time.time()-start_time))
