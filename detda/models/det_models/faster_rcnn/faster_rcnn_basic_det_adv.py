import torch
import torch.nn as nn
import torch.nn.functional as F
from detda.models.det_models.utils.config import cfg
from ..rpn import _RPN
from torchvision.ops import RoIAlign, RoIPool
from ..rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from ..utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse
import numpy as np


def flip_data(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


def loc_flip_consistency_loss(orig_loc, flip_loc):
    # print('shape {} {}'.format(orig_loc.shape,flip_loc.shape))
    consistency_loc_loss_x = torch.mean(torch.pow(orig_loc[:, 0] + flip_loc[:, 0], exponent=2))
    consistency_loc_loss_y = torch.mean(torch.pow(orig_loc[:, 1] - flip_loc[:, 1], exponent=2))
    consistency_loc_loss_w = torch.mean(torch.pow(orig_loc[:, 2] - flip_loc[:, 2], exponent=2))
    consistency_loc_loss_h = torch.mean(torch.pow(orig_loc[:, 3] - flip_loc[:, 3], exponent=2))
    #
    consistency_loc_loss = torch.div(
        consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
        4)
    return consistency_loc_loss


def cls_flip_consistency_loss(orig_prob, flip_prob):
    conf_sampled_flip = flip_prob + 1e-7
    conf_sampled = orig_prob + 1e-7
    consistency_conf_loss_a = F.kl_div(conf_sampled.log(), conf_sampled_flip.detach(), size_average=False,
                                       reduce=False).sum(-1).mean()
    consistency_conf_loss_b = F.kl_div(conf_sampled_flip.log(), conf_sampled.detach(),
                                       size_average=False, reduce=False).sum(-1).mean()
    consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b
    return consistency_conf_loss


def sample_foreground_index(prob):
    cls_index = torch.argmax(prob, dim=1)
    mask = torch.nonzero(cls_index > 0, as_tuple=False).squeeze()
    return mask


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cross_domain_mixup_data(x_1, x_2, y_1, y_2, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x_1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x_1[index, :] + (1 - lam) * x_2
    y_a, y_b = y_1[index], y_2
    return mixed_x, y_a, y_b, lam


class _fasterRCNNBasicDetAdv(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, gc1, gc2, gc3):
        super(_fasterRCNNBasicDetAdv, self).__init__()
        self.classes = classes
        self.n_classes = classes
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.gc1 = gc1
        self.gc2 = gc2
        self.gc3 = gc3
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False, eta=1.0,
                use_pseudo=False, target_sample_strategy='random', outer_rois=(), outer_gt=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat = self.RCNN_base3(base_feat2)

        if len(outer_rois) == 0:
            rois, rpn_loss_cls, rpn_loss_bbox, rpn_cls_score, rpn_conv1 = self.RCNN_rpn(base_feat, im_info, gt_boxes,
                                                                                        num_boxes)
            # if it is training phrase, then use ground truth bboxes for refining
            if self.training and (not target or use_pseudo):
                roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                rois_label = rois_label.view(-1).long()
                rois_target = rois_target.view(-1, rois_target.size(2))
                rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
                rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))
            else:
                if target:
                    if target_sample_strategy == 'random':
                        select_index = np.random.permutation(rois.shape[1])
                        rois = rois[:, select_index[0:cfg.TRAIN.RPN_POST_NMS_TOP_N_TARGET], :]
                    elif target_sample_strategy == 'top':
                        rois = rois[:, 0:cfg.TRAIN.RPN_POST_NMS_TOP_N_TARGET:]
                    elif target_sample_strategy == 'all':
                        rois = rois
                    else:
                        raise RuntimeError('unknown target sample strategy {}'.format(target_sample_strategy))
                rois_label = None
                rois_target = None
                rois_inside_ws = None
                rois_outside_ws = None
                rpn_loss_cls = 0
                rpn_loss_bbox = 0
        else:
            rois, rois_label = outer_rois
            rpn_loss_cls, rpn_loss_bbox, rpn_cls_score, rpn_conv1 = None, None, None, None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        #
        rpn_related = [rpn_cls_score, rpn_conv1]

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            raise NotImplementedError
        elif cfg.POOLING_MODE == 'align':
            orig_pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            orig_pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        else:
            raise NotImplementedError
        #
        backbone_related = [base_feat1, base_feat2, base_feat, orig_pooled_feat]

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(orig_pooled_feat)
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
            return final_predict, [], backbone_related, rpn_related, RCNN_related
        # RCNN Loss
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            # 计算loss的时候，不能使用view之后的结果
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        #
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        final_predict = [rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred]
        losses = [rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox]
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

    def cross_domain_rcnn_mixup(self, src_data, tgt_data, mixup_alpha):
        src_pooled_feat, src_rois_label = src_data
        tgt_pooled_feat, tgt_rois_label = tgt_data
        #
        mix_pooled_feat, targets_a, targets_b, lam = cross_domain_mixup_data(src_pooled_feat, tgt_pooled_feat,
                                                                             src_rois_label, tgt_rois_label,
                                                                             alpha=mixup_alpha)
        pooled_feat = self._head_to_tail(mix_pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)
        criterion = nn.CrossEntropyLoss()
        loss_func = mixup_criterion(targets_a, targets_b, lam)
        mixup_rcnn_loss_cls = loss_func(criterion, cls_score)
        return mixup_rcnn_loss_cls
