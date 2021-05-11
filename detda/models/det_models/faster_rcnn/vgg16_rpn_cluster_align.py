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
from .faster_rcnn_rpn_cluster_align import _fasterRCNNRPNClusterAlign
from detda.models.det_models.utils.config import cfg
from detda.utils.gradcam import GradCam
import pickle
import pdb


class RPNClusterAlignVgg16(_fasterRCNNRPNClusterAlign):
    def __init__(self, classes, cluster_num, adain_layer=(), pretrained=False, class_agnostic=False,
                 fix_part=None, init_cluster_path=None, src_metric_type='cos_similarity',
                 tgt_metric_type='cos_similarity', trainable_center=False, lambda_center_lr=1.0,
                 center_contrastive_loss=False, contrastive_margin=0,
                 src_ent_loss=False, tgt_ent_loss=False, detach_backbone=False,
                 loss_type='contrastive', ce_temperature=1.0, lambda_center_contrastive=1.0, use_gt_for_tgt=True,
                 center_contrastive_margin=0.0, considered_num=2000, use_relu_for_align_feat=True,
                 use_transfer_net=False, center_init_type='normal', fg_thresh=0.7, bg_thresh=0.3,
                 use_cam_mask=False, lambda_cam=1.0, use_fg_mask=False, lambda_fg_mask=1.0, lambda_fg_temp=3.0,
                 use_fg_mask_for_rcnn=False, lambda_instance_fg=1.0, cam_prob_type='soft_fg',
                 rpn_assign_label_type='direct', debug=False, rois_dict=None, use_rpc_label=False,
                 rpc_label_type='soft'):
        self.model_path = cfg.VGG_PATH
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.fix_part = fix_part
        self.cluster_num = cluster_num
        if fix_part is not None:
            for item in fix_part:
                assert item in ['backbone', 'rpn', 'rcnn'], 'wrong fix part name {}'.format(self.fix_part)

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
        self.lambda_center_contrastive = lambda_center_contrastive
        self.use_gt_for_tgt = use_gt_for_tgt
        self.center_contrastive_margin = center_contrastive_margin
        self.considered_num = considered_num
        self.use_relu_for_align_feat = use_relu_for_align_feat
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.use_cam_mask = use_cam_mask
        self.lambda_cam = lambda_cam
        self.use_fg_mask = use_fg_mask
        self.lambda_fg_mask = lambda_fg_mask
        self.lambda_fg_temp = lambda_fg_temp
        self.use_fg_mask_for_rcnn = use_fg_mask_for_rcnn
        self.lambda_instance_fg = lambda_instance_fg
        self.cam_prob_type = cam_prob_type
        self.rpn_assign_label_type = rpn_assign_label_type
        self.debug = debug
        self.use_rpc_label = use_rpc_label
        self.rpc_label_type = rpc_label_type
        if rois_dict is not None:
            with open(rois_dict, 'rb') as f:
                self.rois_dict = pickle.load(f)
        #
        _fasterRCNNRPNClusterAlign.__init__(self, classes, class_agnostic, adain_layer=adain_layer,
                                            use_transfer_net=use_transfer_net)
        #
        center_cls = torch.arange(0, 2).to(torch.int64).view(2, -1)
        center_cls = center_cls.expand(2, int(cluster_num / 2)).reshape(-1)
        if use_transfer_net:
            align_feat_dim = 128
        else:
            align_feat_dim = 512
        #
        rpn_center = torch.zeros(cluster_num, align_feat_dim)
        if center_init_type == 'normal':
            rpn_center.data.normal_(0, 0.01)
        elif center_init_type == 'zero':
            pass
        else:
            raise RuntimeError('wrong type of center init type {}'.format(center_init_type))
        self.register_parameter('rpn_cluster_cls',
                                nn.Parameter(data=center_cls.clone(), requires_grad=False))
        self.register_parameter('rpn_cluster_center',
                                nn.Parameter(data=rpn_center,
                                             requires_grad=trainable_center))
        self.register_parameter('rpn_cluster_prob',
                                nn.Parameter(data=torch.zeros(cluster_num, 2), requires_grad=False))
        self.register_parameter('rpn_cluster_src_sim',
                                nn.Parameter(data=torch.zeros(cluster_num, dtype=torch.int64), requires_grad=False))

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # print(vgg.features)
        self.low_base1 = nn.Sequential(*list(vgg.features._modules.values())[:5])
        self.low_base2 = nn.Sequential(*list(vgg.features._modules.values())[5:10])
        self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[10:14])
        self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
        self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])
        feat_d = 4096

        self.RCNN_top = vgg.classifier
        self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)
        #
        if self.use_cam_mask:
            rcnn_classifier = nn.Sequential(self.RCNN_top, self.RCNN_cls_score)
            self.grad_cam = GradCam(rcnn_classifier, prob_type=self.cam_prob_type)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

    def optim_parameters(self, lr):
        def check_in(layer_name):
            if self.fix_part is not None:
                if 'backbone' in self.fix_part:
                    if 'RCNN_base' in layer_name or 'low_base' in layer_name:
                        return True
                if 'rpn' in self.fix_part:
                    if 'RCNN_rpn' in layer_name:
                        return True
                if 'rcnn' in self.fix_part:
                    if 'RCNN_top' in layer_name or 'RCNN_cls_score' in layer_name or 'RCNN_bbox_pred' in layer_name:
                        return True
                return False
            else:
                return False

        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad and not check_in(layer_name=name):
                if 'cluster_center' in name:
                    optim_param.append({'params': param, 'lr': lr * self.lambda_center_lr})
                    print('{} will be optimized, lr {}'.format(name, lr * self.lambda_center_lr))
                else:
                    optim_param.append({'params': param, 'lr': lr})
                    print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))
        return optim_param

    def load_pretrained_model(self, model_path):
        model = torch.load(model_path)
        self.load_state_dict(model)
