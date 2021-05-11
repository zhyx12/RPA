import torch
import torch.nn as nn
import torch.nn.functional as F
from detda.models.det_models.utils.config import cfg
from ..rpn import _RPN
from torchvision.ops import RoIAlign, RoIPool
from ..rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from ..utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse
import numpy as np
from detda.utils import calc_mean_std, cal_feat_distance, find_corr_center
import pickle
import os


class _fasterRCNNRPNClusterAlign(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, adain_layer, use_transfer_net=False):
        super(_fasterRCNNRPNClusterAlign, self).__init__()
        self.classes = classes
        self.n_classes = classes
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, use_transfer_net, fg_thresh=self.fg_thresh, bg_thresh=self.bg_thresh,
                             use_cam_mask=self.use_cam_mask, use_fg_mask=self.use_fg_mask,
                             rpn_assign_label_type=self.rpn_assign_label_type)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.adain_layer = adain_layer

    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False,
                target_sample_strategy='random', tgt_mean_std=(), src_start_align=False, tgt_start_align=False,
                img_id=None,
                ):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        #
        return_mean_std = []
        # feed image data to base model to obtain base feature map
        low_feat_1 = self.low_base1(im_data)
        if 1 in self.adain_layer:
            low_feat_1, tmp_mean_std = self.normalize_feat(low_feat_1, tgt_mean_std, detach_mean_std=True)
            return_mean_std.append(tmp_mean_std)
        low_feat_2 = self.low_base2(low_feat_1)
        if 2 in self.adain_layer:
            low_feat_2, tmp_mean_std = self.normalize_feat(low_feat_2, tgt_mean_std, detach_mean_std=True)
            return_mean_std.append(tmp_mean_std)
        base_feat1 = self.RCNN_base1(low_feat_2)
        if 3 in self.adain_layer:
            base_feat1, tmp_mean_std = self.normalize_feat(base_feat1, tgt_mean_std, detach_mean_std=True)
            return_mean_std.append(tmp_mean_std)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat = self.RCNN_base3(base_feat2)

        tgt_structure_loss = torch.tensor(0).to(im_data.device)
        src_align_loss = torch.tensor(0).to(im_data.device)
        if not target:
            start_align = src_start_align
            metric_type = self.src_metric_type
            ent_loss = self.src_ent_loss
        else:
            start_align = tgt_start_align
            metric_type = self.tgt_metric_type
            ent_loss = self.tgt_ent_loss
        if self.use_cam_mask:
            grad_cam = self.grad_cam
            for param in self.RCNN_top.parameters():
                param.requires_grad = False
            for param in self.RCNN_cls_score.parameters():
                param.requires_grad = False
        else:
            grad_cam = None
        #
        if self.use_rpc_label:
            rpc_module = [self._head_to_tail, self.RCNN_cls_score]
        else:
            rpc_module = None
        rpn_res = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, target=target, loss_type=self.loss_type,
                                rpn_cluster_center=self.rpn_cluster_center, ce_temperature=self.ce_temperature,
                                rpn_cluster_cls=self.rpn_cluster_cls, start_align=start_align,
                                metric_type=metric_type, center_contrastive_loss=self.center_contrastive_loss,
                                contrastive_margin=self.contrastive_margin, ent_loss=ent_loss,
                                detach_backbone=self.detach_backbone,
                                lambda_center_contrastive=self.lambda_center_contrastive,
                                use_gt_for_tgt=self.use_gt_for_tgt,
                                center_contrastive_margin=self.center_contrastive_margin,
                                considered_num=self.considered_num, use_relu=self.use_relu_for_align_feat,
                                grad_cam=grad_cam, lambda_cam=self.lambda_cam, cam_prob_type=self.cam_prob_type,
                                use_rpc_label=self.use_rpc_label, rpc_module=rpc_module,
                                rpc_label_type=self.rpc_label_type,
                                )
        if self.use_cam_mask:
            for param in self.RCNN_top.parameters():
                param.requires_grad = True
            for param in self.RCNN_cls_score.parameters():
                param.requires_grad = True
        rois, rpn_loss_cls, rpn_loss_bbox, rpn_cls_score, rpn_conv1, align_loss, center_contrastive_loss = rpn_res
        if self.debug:
            rois_from_gt = torch.ones_like(gt_boxes)
            rois_from_gt[:, :, 1:5] = gt_boxes[:, :, 0:4]
            rois = torch.cat((rois, rois_from_gt), dim=1)
            # base_name = os.path.basename(img_id[0])
            # rois = torch.from_numpy(self.rois_dict[base_name]).to(rois.device)
            # if rois_from_gt.shape[1]>0:
            #     rois = rois_from_gt
            # else:
            #     rois = rois
        if not target:
            src_align_loss = align_loss
        else:
            tgt_structure_loss = align_loss
        #
        # if it is training phrase, then use ground truth bboxes for refining
        if self.training and (not target):
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2))
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))
        else:
            # TODO: 在rpn的对齐中，目标域后面的计算尽量少
            if target:
                rois = rois[:, 0:2, :]
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            # rpn_loss_cls = 0
            # rpn_loss_bbox = 0
        #
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            raise NotImplementedError
        elif cfg.POOLING_MODE == 'align':
            orig_pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            orig_pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        else:
            raise NotImplementedError
        # 使用instance前景mask进行加权
        if self.use_fg_mask_for_rcnn and start_align:
            global_mask = self.get_attention_mask(base_feat).unsqueeze(0).unsqueeze(0)
            instance_fg_mask = self.RCNN_roi_align(global_mask, rois.view(-1, 5))
            orig_pooled_feat = orig_pooled_feat * (instance_fg_mask * self.lambda_instance_fg + 1)
        #
        if not self.training:
            if self.use_relu_for_align_feat:
                rpn_align_feat = F.relu(self.RCNN_rpn.RPN_Conv(orig_pooled_feat), inplace=True)
            else:
                rpn_align_feat = self.RCNN_rpn.RPN_Conv(orig_pooled_feat)
            rpn_align_feat = rpn_align_feat.view((rpn_align_feat.shape[0], rpn_align_feat.shape[1], -1))
            rpn_align_feat = torch.mean(rpn_align_feat, dim=2)
        else:
            rpn_align_feat = None
        rpn_related = [rpn_cls_score, rpn_align_feat]
        #
        backbone_related = [base_feat1, base_feat2, base_feat, orig_pooled_feat, return_mean_std]
        #
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(orig_pooled_feat)
        #
        RCNN_related = [pooled_feat, ]
        #
        orig_bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        #
        if self.training and not self.class_agnostic and rois_label is not None:
            bbox_pred_view = orig_bbox_pred.view(orig_bbox_pred.size(0), int(orig_bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
        else:
            bbox_pred = orig_bbox_pred
        #
        if target:
            bbox_pred = orig_bbox_pred.view(batch_size, rois.size(1), -1)
            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)

            final_predict = [rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred]
            return final_predict, [tgt_structure_loss,
                                   center_contrastive_loss, rpn_loss_cls,
                                   rpn_loss_bbox], backbone_related, rpn_related, RCNN_related
        # RCNN Loss
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        #
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        #
        final_predict = [rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred]
        losses = [rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, src_align_loss]
        #
        return final_predict, losses, backbone_related, rpn_related, RCNN_related

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
        #

    def normalize_feat(self, feat, tgt_mean_std=(), detach_mean_std=True):
        feat_size = feat.size()

        feat_mean, feat_std = calc_mean_std(feat, detach_mean_std=detach_mean_std)
        if len(tgt_mean_std) > 0:
            # print('normalize')
            temp_tgt_mean_std = tgt_mean_std[0]
            normalized_feat = (feat - feat_mean.expand(feat_size)) / (feat_std.expand(feat_size) + 10e-8)
            feat = normalized_feat * temp_tgt_mean_std[1].expand(feat_size) + temp_tgt_mean_std[0].expand(feat_size)
            tgt_mean_std.pop(0)
        return feat, (feat_mean, feat_std)

    def get_attention_mask(self, global_feat, attention_mask_lam=3.0):
        # global feat经过rpn
        global_rpn_feat = F.relu(self.RCNN_rpn.RPN_Conv(global_feat.detach()),
                                 inplace=True).squeeze(0).detach()
        num_dim, feat_shape_1, feat_shape_2 = global_rpn_feat.shape
        global_rpn_feat = torch.transpose(torch.transpose(global_rpn_feat, 0, 1), 1, 2)
        sim = cal_feat_distance(global_rpn_feat.view(-1, num_dim),
                                self.rpn_cluster_center.detach())
        mask = torch.softmax(sim * attention_mask_lam, dim=1)[:, 1].view(feat_shape_1, feat_shape_2)
        return mask
