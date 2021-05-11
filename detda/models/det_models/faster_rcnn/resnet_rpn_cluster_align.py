from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from .faster_rcnn_rpn_cluster_align import _fasterRCNNRPNClusterAlign
from detda.models.det_models.utils.config import cfg
from detda.models.det_models.faster_rcnn.resnet import resnet50, resnet101


class RPNClusterAlignResnet(_fasterRCNNRPNClusterAlign):
    def __init__(self, classes, cluster_num, num_layers, adain_layer=(), pretrained=False, class_agnostic=False,
                 init_cluster_path=None, src_metric_type='cos_similarity',
                 tgt_metric_type='cos_similarity', trainable_center=False, lambda_center_lr=1.0,
                 center_contrastive_loss=False, contrastive_margin=0,
                 src_ent_loss=False, tgt_ent_loss=False, detach_backbone=False, loss_type='contrastive',
                 ce_temperature=1.0
                 ):
        self.dout_base_model = 1024
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.layers = num_layers
        #
        self.cluster_num = cluster_num
        self.init_cluster_path = init_cluster_path
        self.src_metric_type = src_metric_type
        self.tgt_metric_type = tgt_metric_type
        self.trainable_center = trainable_center
        self.lambda_center_lr = lambda_center_lr
        self.center_contrastive_loss = center_contrastive_loss
        self.contrastive_margin = contrastive_margin
        self.src_ent_loss = src_ent_loss
        self.tgt_ent_loss = tgt_ent_loss
        self.detach_backbone = detach_backbone
        self.loss_type = loss_type
        self.ce_temperature = ce_temperature
        #
        _fasterRCNNRPNClusterAlign.__init__(self, classes, class_agnostic, adain_layer=adain_layer)
        self.register_buffer('rpn_cluster_cls', torch.zeros((cluster_num), dtype=torch.int64))
        self.register_parameter('rpn_cluster_center',
                                nn.Parameter(data=torch.zeros(cluster_num, 512), requires_grad=trainable_center))
        self.register_buffer('rpn_cluster_prob', torch.zeros(cluster_num, 2))
        self.register_buffer('rpn_cluster_src_sim', torch.zeros(cluster_num, dtype=torch.int64))

    def _init_modules(self):
        if self.layers == 101:
            resnet = resnet101()
            model_path = cfg.RESNET101_PATH
        elif self.layers == 50:
            resnet = resnet50()
            model_path = cfg.RESNET50_PATH
        else:
            raise RuntimeError('wrong num layers {}'.format(self.layers))
        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (model_path))
            state_dict = torch.load(model_path)
            resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})
        # Build resnet.
        self.low_base1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                       resnet.maxpool)
        self.low_base2 = resnet.layer1
        self.RCNN_base1 = nn.Identity()  # placeholder
        self.RCNN_base2 = resnet.layer2
        self.RCNN_base3 = resnet.layer3

        self.RCNN_top = resnet.layer4
        feat_d = 2048

        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)
        #
        for p in self.low_base1[0].parameters(): p.requires_grad = False
        for p in self.low_base1[1].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.low_base1.apply(set_bn_fix)
        self.low_base2.apply(set_bn_fix)
        self.RCNN_base1.apply(set_bn_fix)
        self.RCNN_base2.apply(set_bn_fix)
        self.RCNN_base3.apply(set_bn_fix)
        self.RCNN_top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        if mode:
            self.low_base1.eval()
            self.low_base2.eval()
            self.RCNN_base1.eval()
            self.RCNN_base2.train()
            self.RCNN_base3.train()
            #
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.low_base1.apply(set_bn_eval)
            self.low_base2.apply(set_bn_eval)
            self.RCNN_base1.apply(set_bn_eval)
            self.RCNN_base2.apply(set_bn_eval)
            self.RCNN_base3.apply(set_bn_eval)
            self.RCNN_top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)
        return fc7

    def optim_parameters(self, lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    # params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                    #             'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    params += [{'params': [value], 'lr': lr,
                                'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr}]
        return params
