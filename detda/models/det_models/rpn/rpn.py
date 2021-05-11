from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from detda.models.det_models.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from detda.models.det_models.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time
from detda.utils import calc_mean_std, cal_feat_distance, find_corr_center
from torchvision.ops import RoIAlign
from detda.utils.det_utils import assign_labels_for_rpn_rois, assign_labels_for_rois
from detda.loss import contrastive_loss, contrastive_loss_for_euclidean


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din, use_transfer_net=False, fg_thresh=None, bg_thresh=None, use_cam_mask=False,
                 use_fg_mask=False, rpn_assign_label_type='direct',
                 ):
        super(_RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        #
        self.RCNN_roi_align = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        self.use_transfer_net = use_transfer_net
        if self.use_transfer_net:
            self.transfer_net = nn.Linear(in_features=512, out_features=128, bias=False)
        #
        if fg_thresh is not None:
            self.fg_thresh = fg_thresh
            self.bg_thresh = bg_thresh
        else:
            self.fg_thresh = cfg.TRAIN.RPN_POSITIVE_OVERLAP
            self.bg_thresh = cfg.TRAIN.RPN_NEGATIVE_OVERLAP
        self.use_cam_mask = use_cam_mask
        self.use_fg_mask = use_fg_mask
        self.rpn_assign_label_type = rpn_assign_label_type

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, target=False, loss_type='contrastive',
                ce_temperature=1.0, rpn_cluster_center=None,
                rpn_cluster_cls=None, start_align=False, metric_type='cos_similarity',
                center_contrastive_loss=False, contrastive_margin=0.0, ent_loss=False, detach_backbone=False,
                lambda_center_contrastive=1.0, use_gt_for_tgt=True, center_contrastive_margin=0.0, considered_num=None,
                use_relu=True, grad_cam=None, lambda_cam=1.0, lambda_fg_temp=3.0, lambda_fg_mask=1.0,
                cam_prob_type='soft_fg', use_rpc_label=False, rpc_module=None, rpc_label_type='soft'):
        """

        """
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.detach(), rpn_bbox_pred.detach(),
                                  im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        #
        align_loss = torch.tensor(0).to(base_feat.device)
        contrastive_loss_of_center = torch.tensor(0).to(base_feat.device)
        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.detach(), gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = rpn_label.view(-1).ne(-1).nonzero(as_tuple=True)[0].view(-1)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.detach())
            rpn_label = rpn_label.long()
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = rpn_bbox_inside_weights
            rpn_bbox_outside_weights = rpn_bbox_outside_weights
            rpn_bbox_targets = rpn_bbox_targets

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

        # 计算与聚类中心的对齐损失
        if start_align:
            # 从2000个rois获取，相同数量的正负样本
            fg_thresh = self.fg_thresh
            bg_thresh = self.bg_thresh
            #
            rois_for_align = rois.squeeze(0)
            if use_rpc_label:
                with torch.no_grad():
                    orig_pooled_feat = self.RCNN_roi_align(base_feat.detach(), rois.view(-1, 5))
                    pooled_feat = rpc_module[0](orig_pooled_feat)
                    cls_score = rpc_module[1](pooled_feat)
                    cls_prob = F.softmax(cls_score, dim=1)
                    max_prob, max_ind = torch.max(cls_prob, dim=1)
                    tmp_fg_ind = torch.nonzero(max_ind > 0, as_tuple=True)[0]
                    max_ind[tmp_fg_ind] = 1
                    rois_label = max_ind
                    if rpc_label_type == 'soft':
                        proposal_weights = max_prob
                    elif rpc_label_type == 'hard':
                        proposal_weights = torch.ones_like(rois_label)
                    else:
                        raise RuntimeError('wrong type of rpc rois label')
            else:
                rois_label = assign_labels_for_rpn_rois(rois_for_align[:, 1:5], gt_boxes.squeeze(0),
                                                        fg_thresh, bg_thresh, mode=self.rpn_assign_label_type)
                proposal_weights = torch.ones_like(rois_label)

            fg_ind = torch.nonzero(rois_label == 1, as_tuple=True)[0]
            fg_num = fg_ind.numel()
            bg_ind = torch.nonzero(rois_label == 0, as_tuple=True)[0]
            bg_num = bg_ind.numel()
            if fg_num > 0 and bg_num > 0:
                tmp_considered_num = bg_num if considered_num is None else min(considered_num + fg_num, bg_num)
                if bg_num >= fg_num:
                    select_ind = np.random.choice(tmp_considered_num, size=fg_num, replace=False)
                else:
                    select_ind = np.random.choice(tmp_considered_num, size=fg_num, replace=True)
                bg_ind = bg_ind[torch.from_numpy(select_ind).to(bg_ind.device)]
                keep_ind = torch.cat((fg_ind, bg_ind), dim=0)
                align_rois = torch.index_select(rois_for_align.view(-1, 5), 0, keep_ind)
                align_rois_label = torch.index_select(rois_label, 0, keep_ind).to(torch.long)
                proposal_weights = torch.index_select(proposal_weights, 0, keep_ind)
                # 改成从base_feat中提取对应框的特征，然后经过RPN_conv
                rpn_align_base_feat = self.RCNN_roi_align(base_feat, align_rois.view(-1, 5))
                rpn_align_feat = F.relu(self.RPN_Conv(rpn_align_base_feat), inplace=True)
                #
                if self.use_cam_mask:
                    if cam_prob_type == 'soft_fg':
                        fg_rpn_feat = rpn_align_base_feat[0:fg_num, :, :].clone().detach()
                        fg_rpn_feat = fg_rpn_feat.view(fg_num, -1)
                        fg_rpn_feat.requires_grad = True
                        cam_mask = grad_cam(fg_rpn_feat).detach().unsqueeze(1)
                        final_cam_mask = torch.ones_like(rpn_align_feat)
                        final_cam_mask[0:fg_num, :] += cam_mask * lambda_cam
                    elif cam_prob_type == 'gt':
                        _, _, rpc_labels = assign_labels_for_rois(align_rois[:, 1:5], gt_boxes.squeeze(0),
                                                                  thresh_high=0.5, thresh_low=0.0)
                        all_rpn_feat = rpn_align_base_feat.clone().detach()
                        all_rpn_feat = all_rpn_feat.view(fg_num * 2, -1)
                        all_rpn_feat.requires_grad = True
                        cam_mask = grad_cam(all_rpn_feat, rpc_labels.to(torch.long)).detach().unsqueeze(1)
                        final_cam_mask = torch.ones_like(rpn_align_feat)
                        final_cam_mask += cam_mask * lambda_cam
                    else:
                        raise RuntimeError('wrong type of cam prob')
                    rpn_align_feat = torch.mean((final_cam_mask * rpn_align_feat).view(*rpn_align_feat.shape[0:2], -1),
                                                dim=2)
                elif self.use_fg_mask:
                    fg_rpn_feat = rpn_align_base_feat[0:fg_num, :, :].detach()
                    fg_mask = self.get_fg_mask(fg_rpn_feat, rpn_cluster_center, mask_temp=lambda_fg_temp).unsqueeze(1)
                    final_cam_mask = torch.ones_like(rpn_align_feat)
                    final_cam_mask[0:fg_num, :] += fg_mask * lambda_fg_mask
                    rpn_align_feat = torch.mean((final_cam_mask * rpn_align_feat).view(*rpn_align_feat.shape[0:2], -1),
                                                dim=2)
                else:
                    rpn_align_feat = torch.mean(
                        rpn_align_feat.view(rpn_align_feat.shape[0], rpn_align_feat.shape[1], -1),
                        dim=2)
                # 过滤
                rpn_align_feat, align_rois_label, proposal_weights = self.filter_feat(rpn_align_feat, align_rois_label,
                                                                                      proposal_weights)
                # 对proposal weights做归一化
                proposal_weights = proposal_weights / float(proposal_weights.shape[0])
                # 特征维度转换
                if self.use_transfer_net:
                    rpn_align_feat = self.transfer_net(rpn_align_feat)
                #
                if rpn_align_feat.numel() > 0:
                    if metric_type == 'cos_similarity':
                        tmp_score = cal_feat_distance(rpn_align_feat, rpn_cluster_center, metric_type=metric_type)
                        align_loss = contrastive_loss(align_rois_label, rpn_cluster_cls, tmp_score, class_num=2,
                                                      margin=contrastive_margin, instance_weight=proposal_weights)
                    elif metric_type == 'euclidean':
                        tmp_feat_1 = rpn_align_feat.unsqueeze(0)
                        tmp_feat_2 = rpn_cluster_center.unsqueeze(0)
                        tmp_dist = torch.cdist(tmp_feat_1, tmp_feat_2, p=2).squeeze(0)
                        align_loss = contrastive_loss_for_euclidean(align_rois_label, rpn_cluster_cls, tmp_dist,
                                                                    class_num=2,
                                                                    margin=contrastive_margin)
                    else:
                        raise RuntimeError('wrong metric type')
                    if ent_loss:
                        pred_score, pred_cluster_ind = find_corr_center(rpn_cluster_cls, tmp_score, class_num=2)
                        pred_prob = torch.softmax(pred_score, dim=1)
                        log_pred_prob = torch.log_softmax(pred_score, dim=1)
                        align_loss += - (pred_prob * log_pred_prob).sum(1).mean()
                    if center_contrastive_loss and target:
                        center_score = cal_feat_distance(rpn_cluster_center, rpn_cluster_center.detach(),
                                                         metric_type=metric_type)
                        contrastive_loss_of_center = contrastive_loss(rpn_cluster_cls, rpn_cluster_cls, center_score,
                                                                      class_num=2,
                                                                      margin=center_contrastive_margin) * lambda_center_contrastive
        return rois, self.rpn_loss_cls, self.rpn_loss_box, rpn_cls_score, rpn_conv1, align_loss, contrastive_loss_of_center

    def filter_feat(self, feat, label, weights):
        max_val, _ = torch.max(feat, dim=1)
        selected_ind = torch.nonzero(torch.mul(max_val > 1e-5, max_val < 30), as_tuple=True)[0]
        return feat[selected_ind, :], label[selected_ind], weights[selected_ind]

    def get_class_balanced_mask(self, rois_label):
        rois_cls_hist = torch.histc(rois_label, bins=2, min=0, max=2)
        existed_class_mask = torch.zeros(rois_cls_hist.shape[0])
        existed_class_ind = torch.nonzero(rois_cls_hist > 0, as_tuple=True)[0]
        existed_class_mask[existed_class_ind] = 1
        existed_class_mask = existed_class_mask.to(rois_cls_hist.device)
        rois_cls_mask = torch.reciprocal(rois_cls_hist + 10e-8) * existed_class_mask
        rois_mask = rois_cls_mask[rois_label]
        return rois_mask

    def get_fg_mask(self, rpn_align_feat, rpn_center, mask_temp=3.0):
        rpn_align_feat = rpn_align_feat.detach()
        num_batch, num_dim, feat_shape_1, feat_shape_2 = rpn_align_feat.shape
        rpn_align_feat = torch.transpose(torch.transpose(rpn_align_feat, 1, 2), 2, 3)
        sim = cal_feat_distance(rpn_align_feat.reshape(-1, num_dim), rpn_center.detach())
        mask = torch.softmax(sim * mask_temp, dim=1)[:, 1].view(num_batch, feat_shape_1, feat_shape_2)
        return mask
