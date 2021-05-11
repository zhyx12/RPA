# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models
from .faster_rcnn import _fasterRCNN
from detda.models.det_models.utils.config import cfg
import os.path as osp


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, fix_lower_layer=False):
        self.model_path = osp.join(cfg.ROOT_DIR, 'data', 'pretrained_model', 'vgg16_caffe.pth')
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.fix_lower_layer = fix_lower_layer

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.low_base1 = nn.Sequential(*list(vgg.features._modules.values())[:5])
        self.low_base2 = nn.Sequential(*list(vgg.features._modules.values())[5:10])
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[10:14])
        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
        self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])

        # Fix the layers before conv3:
        if self.fix_lower_layer:
            for layer in range(10):
                for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

    # def optim_parameters(self, lr):
    #     params = []
    #
    #     for key, value in dict(self.named_parameters()).items():
    #         if value.requires_grad:
    #             if 'bias' in key:
    #                 params += [{'params': [value], 'lr': lr * 2,
    #                             'weight_decay': 0}]
    #             else:
    #                 params += [{'params': [value], 'lr': lr}]
    #     return params
