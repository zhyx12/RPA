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
import torch.nn.functional as F
import math
import torchvision.models as models
from .faster_rcnn_basic_det_adv import _fasterRCNNBasicDetAdv
from detda.models.det_models.utils.config import cfg

import pdb


class BasicDetAdvVgg16(_fasterRCNNBasicDetAdv):
    def __init__(self, classes, pretrained=False, class_agnostic=False, gc1=False, gc2=False, gc3=False):
        self.model_path = cfg.VGG_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.gc1 = gc1
        self.gc2 = gc2
        self.gc3 = gc3

        _fasterRCNNBasicDetAdv.__init__(self, classes, class_agnostic, self.gc1, self.gc2, self.gc3)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # print(vgg.features)
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])
        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
        self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])
        feat_d = 4096

        for layer in range(10):
            for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7