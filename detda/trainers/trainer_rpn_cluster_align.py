# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import numpy as np
import torch.nn.functional as F
from detda.runner.validator import BaseValidator
from detda.runner.trainer import BaseTrainer
from detda.runner.hooks import LossMetrics, BackwardUpdate, GradientClipper
from detda.models.det_models.utils.config import cfg_from_dict
from detda.runner.hooks import DetValMetrics, RCNNConfusionMetrics, RPNRecall, RPNClusterAlign, \
    DetOnlinePseudoLabel, DetCAMMask
from detda.loss import FocalLoss, EFocalLoss
from detda.runner.hooks import DetOnlinePseudoLabel
from detda.models.det_models.utils.net_utils import grad_reverse
from detda.utils.det_utils import flip_data, cls_flip_consistency_loss, loc_flip_consistency_loss
from detda.utils.utils import cal_feat_distance
import math


def sample_foreground_index(prob):
    cls_index = torch.argmax(prob, dim=-1)
    mask = torch.nonzero(cls_index > 0, as_tuple=False).squeeze()
    return mask


class ValidatorBasicAdvDet(BaseValidator):
    def __init__(self, cuda, logdir, test_loaders, model_dict, thresh, class_agnostic, max_per_image,
                 align_start_iteration,
                 logger=None, writer=None, trainer=None, kmeans_dist='cos',
                 init_center=False, align_prob_type='hard', cluster_sample_ratio=1.0,
                 num_cluster_per_class=1, pseudo_filter_ratio=0.5, use_cluster_align=True, use_pseudo_label=True,
                 save_align_features=False, use_ensemble=False, assign_center_feat=True, assign_once=False,
                 use_val_as_train=False, view_global_mask=False, dynamic_ratio=True, view_cam_mask=False,
                 cluster_fg_thresh=0.7, cluster_bg_thresh=0.3, save_rois=False, filter_low_thresh=0.1,
                 save_vis=False):
        super(ValidatorBasicAdvDet, self).__init__(cuda=cuda, logdir=logdir, test_loaders=test_loaders,
                                                   logger=logger, writer=writer, log_name=('det',),
                                                   model_dict=model_dict)
        self.class_agnostic = class_agnostic
        self.trainer = trainer
        for ind, (key, _) in enumerate(self.test_loaders.items()):
            det_val_metrics = DetValMetrics(self, thresh, class_agnostic, max_per_image, dataset_name=key,
                                            save_vis=save_vis)
            self.register_hook(det_val_metrics)
            rpn_recall = RPNRecall(self, dataset_name=key)
            self.register_hook(rpn_recall)
            if view_cam_mask:
                cam_mask = DetCAMMask(self, dataset_name=key)
                self.register_hook(cam_mask)
            # 为目标域训练集增加聚类
            if ind == 1 and use_cluster_align:
                rcnn_cluster_align = RPNClusterAlign(self, dataset_name=key, proposal_sample_num=100, dim=512,
                                                     sample_ratio_per_class=cluster_sample_ratio,
                                                     num_cluster_per_class=num_cluster_per_class,
                                                     start_iteration=align_start_iteration,
                                                     kmeans_dist=kmeans_dist, save_features=save_align_features,
                                                     init_center=init_center, prob_type=align_prob_type,
                                                     assign_center_feat=assign_center_feat, assign_once=assign_once,
                                                     fg_thresh=cluster_fg_thresh, bg_thresh=cluster_bg_thresh)
                self.register_hook(rcnn_cluster_align, priority=100)
            if ((ind == 1 or (ind == 0 and use_val_as_train)) and use_pseudo_label):
                online_pseudo_label = DetOnlinePseudoLabel(runner=self, dataset_name=key, high_thresh=1.0,
                                                           low_thresh=filter_low_thresh, ratio=pseudo_filter_ratio,
                                                           filter_by_ratio=True,
                                                           dynamic_ratio=dynamic_ratio, use_ensemble=False,
                                                           max_iter=100000, min_low_thresh=0.1)
                self.register_hook(online_pseudo_label, priority=90)


    def eval_iter(self, val_batch_data):
        im_data, im_info, gt_boxes, num_boxes, src_id = val_batch_data
        faster_rcnn = self.model_dict['base_model']
        #
        if self.cuda:
            im_data = im_data.to('cuda:0')
            gt_boxes = gt_boxes.to('cuda:0')
        #
        if self.val_iter % 100 == 0:
            print('iter {}'.format(self.val_iter))
        with torch.no_grad():
            final_predict, losses, backbone_related, \
            rpn_related, rcnn_related = faster_rcnn(im_data, im_info, gt_boxes, num_boxes, img_id=src_id)
            #
            rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred = final_predict
            # rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox = losses
            feat1, feat2, feat3, orig_pooled_feat, _ = backbone_related
            rpn_cls_score, rpn_align_feat = rpn_related
            pooled_feat, = rcnn_related
            #
            return {"rois": rois,
                    "rois_label": rois_label,
                    # "rois_target": rois_target,
                    "img_id": src_id,
                    'base_feat': feat3,
                    'rpn_align_feat': rpn_align_feat,
                    "orig_pooled_feat": orig_pooled_feat,
                    'cls_prob': cls_prob,
                    "bbox_pred": bbox_pred,
                    'im_data': im_data,
                    'im_info': im_info,
                    'gt_boxes': gt_boxes,
                    'num_boxes': num_boxes,
                    }


class TrainerRPNClusterAlign(BaseTrainer):
    def __init__(self, cuda, model_dict, optimizer_dict, scheduler_dict, device_dict,
                 train_loaders=None, test_loaders=None,
                 logger=None, logdir=None, max_iters=None,
                 val_interval=5000, log_interval=50, save_interval=5000,
                 update_iter=1, save_test_res=False,
                 use_syncbn=False, max_save_num=3, cudnn_deterministic=False,
                 # new parameters for detection
                 class_agnostic=False, val_thresh=0, max_per_image=100, ef=False, lambda_adv=1.0,
                 gamma=5, lambda_src_det=1.0, lambda_tgt_det=1.0,
                 clip_gradient=False,
                 lambda_adv_1=1.0, lambda_adv_2=1.0, lambda_adv_3=1.0, lambda_adv_inst=1.0,
                 lambda_grad_reverse=1.0, target_sample_strategy='top',
                 flip_consistency=False, sample_for_flip=False, lambda_flip_cls=1.0, lambda_flip_loc=1.0,
                 lambda_tgt_structure=1.0, lambda_src_align=1.0, src_align_start_iteration=1000000,
                 tgt_align_start_iteration=1000000, kmeans_dist='cos',
                 init_center_for_cluster=False, align_prob_type='hard', cluster_sample_ratio=1.0,
                 num_cluster_per_class=1, pseudo_filter_ratio=0.5, use_cluster_align=True,
                 use_pseudo_label=True, save_align_features=False, use_ensemble=False, assign_center_feat=True,
                 assign_once=False, use_attention_mask=False, lambda_att_mask=5.0, use_val_as_train=False,
                 view_global_mask=False, dynamic_ratio=True, ignore_adv3=False, cluster_fg_thresh=0.7,
                 cluster_bg_thresh=0.3, use_focal_loss=False, view_cam_mask=False, save_rois=False,
                 use_tgt_rpn_cls_loss=False, use_tgt_rpn_box_loss=False, filter_low_thresh=0.1, save_vis=False,
                 ):
        super(TrainerRPNClusterAlign, self).__init__(cuda=cuda, model_dict=model_dict, optimizer_dict=optimizer_dict,
                                                     scheduler_dict=scheduler_dict, device_dict=device_dict,
                                                     train_loaders=train_loaders,
                                                     test_loaders=test_loaders, max_iters=max_iters, logger=logger,
                                                     logdir=logdir,
                                                     val_interval=val_interval, log_interval=log_interval,
                                                     save_interval=save_interval, update_iter=update_iter,
                                                     cudnn_deterministic=cudnn_deterministic,
                                                     save_test_res=save_test_res,
                                                     use_syncbn=use_syncbn,
                                                     max_save_num=max_save_num)
        self.class_agnostic = class_agnostic
        self.lambda_src_det = lambda_src_det
        self.lambda_tgt_det = lambda_tgt_det
        self.lambda_grad_reverse = lambda_grad_reverse
        self.lambda_adv_1 = lambda_adv_1
        self.lambda_adv_2 = lambda_adv_2
        self.lambda_adv_3 = lambda_adv_3
        self.lambda_adv_inst = lambda_adv_inst
        self.target_sample_strategy = target_sample_strategy
        self.flip_consistency = flip_consistency
        self.sample_for_flip = sample_for_flip
        self.lambda_flip_cls = lambda_flip_cls
        self.lambda_flip_loc = lambda_flip_loc
        self.lambda_tgt_structure = lambda_tgt_structure
        self.lambda_src_align = lambda_src_align
        self.src_align_start_iteration = src_align_start_iteration
        self.tgt_align_start_iteration = tgt_align_start_iteration
        self.kmeans_dist = kmeans_dist
        self.align_prob_type = align_prob_type
        self.use_attention_mask = use_attention_mask
        self.lambda_att_mask = lambda_att_mask
        self.ignore_adv3 = ignore_adv3
        self.use_tgt_rpn_cls_loss = use_tgt_rpn_cls_loss
        self.use_tgt_rpn_box_loss = use_tgt_rpn_box_loss
        if ef:
            self.focal_loss = EFocalLoss(class_num=2, gamma=gamma)
        else:
            self.focal_loss = FocalLoss(class_num=2, gamma=gamma, sigmoid=True)
        self.lambda_adv = lambda_adv
        self.use_focal_loss = use_focal_loss
        # 验证
        self.validator = ValidatorBasicAdvDet(cuda=cuda, logdir=self.logdir, test_loaders=self.test_loaders,
                                              logger=self.logger, writer=self.writer, class_agnostic=class_agnostic,
                                              model_dict=self.model_dict, thresh=val_thresh,
                                              max_per_image=max_per_image,
                                              trainer=self,
                                              align_start_iteration=min(src_align_start_iteration,
                                                                        tgt_align_start_iteration),
                                              kmeans_dist=self.kmeans_dist,
                                              init_center=init_center_for_cluster, align_prob_type=align_prob_type,
                                              cluster_sample_ratio=cluster_sample_ratio,
                                              num_cluster_per_class=num_cluster_per_class,
                                              pseudo_filter_ratio=pseudo_filter_ratio,
                                              use_cluster_align=use_cluster_align, use_pseudo_label=use_pseudo_label,
                                              save_align_features=save_align_features, use_ensemble=use_ensemble,
                                              assign_center_feat=assign_center_feat, assign_once=assign_once,
                                              use_val_as_train=use_val_as_train, view_global_mask=view_global_mask,
                                              dynamic_ratio=dynamic_ratio, cluster_fg_thresh=cluster_fg_thresh,
                                              cluster_bg_thresh=cluster_bg_thresh, view_cam_mask=view_cam_mask,
                                              save_rois=save_rois, filter_low_thresh=filter_low_thresh,
                                              save_vis=save_vis,
                                              )

        # 增加记录项
        log_names = ['rpn_cls', 'rpn_box', 'rcnn_cls', 'rcnn_box', 'fg_count', 'bg_count',
                     'src_domain_1', 'src_domain_2', 'src_domain_3',
                     'tgt_domain_1', 'tgt_domain_2', 'tgt_domain_3',
                     # 'tgt_rpn_cls', 'tgt_rpn_box', 'tgt_rcnn_cls', 'tgt_rcnn_box',
                     'tgt_fg_count', 'tgt_structure', 'src_align', 'center_contrastive',
                     ]
        # 不同聚类中心之间的距离
        # 类别之间的距离
        cluster_num = model_dict['base_model'].module.rpn_cluster_center.shape[0]
        self.cluster_num = cluster_num
        dist_log_names = []
        for i in range(cluster_num):
            for j in range(i + 1, cluster_num):
                dist_log_names.append('dist_{}_and_{}'.format(i, j))
        dist_metrics = LossMetrics(self, log_names=dist_log_names, group_name='center_dist', log_interval=log_interval)
        self.register_hook(dist_metrics)

        loss_metrics = LossMetrics(self, log_names=log_names, group_name='loss', log_interval=log_interval)
        self.register_hook(loss_metrics)
        #
        update_param = BackwardUpdate(self)
        self.register_hook(update_param)
        #
        if clip_gradient:
            gradient_clipper = GradientClipper(max_num=20.0)
            self.register_hook(gradient_clipper, priority=0)

    def train_iter(self, *args):
        src_im_data, src_im_info, src_gt_boxes, src_num_boxes, src_id = args[0]
        tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_id = args[1]
        # src_im_data = src_im_data.cuda(non_blocking=True)
        # tgt_im_data = tgt_im_data.cuda(non_blocking=True)
        batch_metrics = {}
        batch_metrics['loss'] = {}
        batch_metrics['center_dist'] = {}

        faster_rcnn = self.model_dict['base_model']
        dis_1 = self.model_dict['dis_1']
        dis_2 = self.model_dict['dis_2']
        dis_3 = self.model_dict['dis_3']
        # 构造一个lam备用
        # lam = 2 / (1 + math.exp(-1 * 10 * self.iteration / self.max_iters)) - 1
        #
        if self.iteration > self.tgt_align_start_iteration:  # 大于号，取等于号的话会在计算聚类中心之前使用
            tgt_start_align_flag = True
        else:
            tgt_start_align_flag = False
        if self.iteration > self.src_align_start_iteration:
            src_start_align_flag = True
        else:
            src_start_align_flag = False
        #
        # print('target')
        tgt_loss = 0
        # 目标域前向传播
        tgt_final_predict, tgt_losses, tgt_backbone_related, \
        tgt_rpn_related, tgt_rcnn_related = faster_rcnn(tgt_im_data, tgt_im_info,
                                                        tgt_gt_boxes,
                                                        tgt_num_boxes, target=True,
                                                        src_start_align=src_start_align_flag,
                                                        tgt_start_align=tgt_start_align_flag,
                                                        target_sample_strategy=self.target_sample_strategy,
                                                        )
        #
        tgt_rois, tgt_cls_prob, tgt_bbox_pred, tgt_rois_label, tgt_orig_bbox_pred = tgt_final_predict
        tgt_feat1, tgt_feat2, tgt_feat3, tgt_orig_pooled_feat, tgt_mean_std = tgt_backbone_related
        tgt_pool_feat, = tgt_rcnn_related
        tgt_structure_loss, center_contrastive_loss, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = tgt_losses
        tgt_loss += (tgt_structure_loss + center_contrastive_loss) * self.lambda_tgt_structure
        if self.use_attention_mask:
            att_mask = self.get_attention_mask(tgt_feat3, self.lambda_att_mask)
        else:
            att_mask = torch.ones_like(tgt_feat3)
        # 目标域对抗损失
        tgt_domain_p1 = dis_1(grad_reverse(tgt_feat1, lambd=self.lambda_grad_reverse))
        tgt_domain_p2 = dis_2(grad_reverse(tgt_feat2, lambd=self.lambda_grad_reverse))
        #
        tgt_label_2 = torch.zeros(tgt_domain_p2.size(), dtype=torch.float).cuda()
        tgt_domain_loss_1 = 0.5 * torch.mean((1 - tgt_domain_p1) ** 2) * self.lambda_adv_1
        tgt_domain_loss_2 = 0.5 * F.binary_cross_entropy_with_logits(tgt_domain_p2,
                                                                     tgt_label_2, ) * 0.15 * self.lambda_adv_2

        #
        if self.ignore_adv3:
            tgt_domain_loss_3 = torch.tensor(0.0)
            tgt_loss += self.lambda_adv * (tgt_domain_loss_1 + tgt_domain_loss_2 + tgt_domain_loss_3)
        else:
            tgt_domain_p3 = dis_3(grad_reverse(tgt_feat3, lambd=self.lambda_grad_reverse))
            tgt_label_3 = torch.zeros(tgt_domain_p3.size(), dtype=torch.float).cuda()
            if self.use_focal_loss:
                tgt_domain_loss_3 = 0.5 * self.focal_loss(tgt_domain_p3, tgt_label_3) * self.lambda_adv_3
            else:
                tgt_domain_loss_3 = 0.5 * F.binary_cross_entropy_with_logits(tgt_domain_p3, tgt_label_3,
                                                                             reduction='none') * self.lambda_adv_3
                tgt_domain_loss_3 = torch.mean(tgt_domain_loss_3 * att_mask)
            tgt_loss += self.lambda_adv * (tgt_domain_loss_1 + tgt_domain_loss_2 + tgt_domain_loss_3)
        #
        if self.use_tgt_rpn_cls_loss and tgt_start_align_flag:
            tgt_loss += tgt_rpn_loss_cls
        if self.use_tgt_rpn_box_loss and tgt_start_align_flag:
            tgt_loss += tgt_rpn_loss_bbox
        tgt_loss.backward()
        ############################################################################################
        # print('source')
        # 源域前向传播
        final_predict, losses, backbone_related, \
        rpn_related, rcnn_related = faster_rcnn(src_im_data, src_im_info, src_gt_boxes,
                                                src_num_boxes, tgt_mean_std=tgt_mean_std,
                                                src_start_align=src_start_align_flag,
                                                tgt_start_align=tgt_start_align_flag, )
        #
        rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred = final_predict
        rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, src_align_loss = losses
        feat1, feat2, feat3, orig_pooled_feat, src_mean_std = backbone_related
        rpn_cls_score, rpn_conv1 = rpn_related
        pooled_feat, = rcnn_related
        # 源域检测损失
        src_det_loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        if self.use_attention_mask:
            att_mask = self.get_attention_mask(feat3, self.lambda_att_mask)
        else:
            att_mask = torch.ones_like(feat3)
        # 源域对抗损失
        domain_p1 = dis_1(grad_reverse(feat1, lambd=self.lambda_grad_reverse))
        domain_p2 = dis_2(grad_reverse(feat2, lambd=self.lambda_grad_reverse))
        #
        src_label_2 = torch.zeros(domain_p2.size(), dtype=torch.float).cuda()

        src_domain_loss_1 = 0.5 * torch.mean(domain_p1 ** 2) * self.lambda_adv_1
        src_domain_loss_2 = 0.5 * F.binary_cross_entropy_with_logits(domain_p2, src_label_2) * 0.15 * self.lambda_adv_2

        src_loss = src_det_loss * self.lambda_src_det + src_align_loss * self.lambda_src_align
        if self.ignore_adv3:
            src_domain_loss_3 = torch.tensor(0.0)
            src_loss += self.lambda_adv * (src_domain_loss_1 + src_domain_loss_2)
        else:
            domain_p3 = dis_3(grad_reverse(feat3, lambd=self.lambda_grad_reverse))
            src_label_3 = torch.zeros(domain_p3.size(), dtype=torch.float).cuda()
            if self.use_focal_loss:
                src_domain_loss_3 = 0.5 * self.focal_loss(domain_p3, src_label_3) * self.lambda_adv_3
            else:
                src_domain_loss_3 = 0.5 * F.binary_cross_entropy_with_logits(domain_p3, src_label_3,
                                                                             reduction='none') * self.lambda_adv_3
                src_domain_loss_3 = torch.mean(att_mask * src_domain_loss_3)
            src_loss += self.lambda_adv * (src_domain_loss_1 + src_domain_loss_2 + src_domain_loss_3)
        src_loss.backward()
        #

        # print('rois label require grad new {}'.format(rois_label.requires_grad))
        #
        batch_metrics['loss']['rpn_cls'] = rpn_loss_cls.mean().item()
        batch_metrics['loss']['rpn_box'] = rpn_loss_bbox.mean().item()
        batch_metrics['loss']['rcnn_cls'] = RCNN_loss_cls.mean().item()
        batch_metrics['loss']['rcnn_box'] = RCNN_loss_bbox.mean().item()
        fg_cnt = torch.sum(rois_label.detach().ne(0))
        batch_metrics['loss']['fg_count'] = fg_cnt
        batch_metrics['loss']['bg_count'] = rois_label.detach().numel() - fg_cnt
        batch_metrics['loss']['src_domain_1'] = src_domain_loss_1.mean().item()
        batch_metrics['loss']['src_domain_2'] = src_domain_loss_2.mean().item()
        batch_metrics['loss']['src_domain_3'] = src_domain_loss_3.mean().item()
        #
        batch_metrics['loss']['tgt_domain_1'] = tgt_domain_loss_1.mean().item()
        batch_metrics['loss']['tgt_domain_2'] = tgt_domain_loss_2.mean().item()
        batch_metrics['loss']['tgt_domain_3'] = tgt_domain_loss_3.mean().item()

        # batch_metrics['loss']['tgt_rpn_cls'] = tgt_rpn_loss_cls.mean().item()
        # batch_metrics['loss']['tgt_rpn_box'] = tgt_rpn_loss_box.mean().item()
        # batch_metrics['loss']['tgt_rcnn_cls'] = tgt_RCNN_loss_cls.mean().item()
        # batch_metrics['loss']['tgt_rcnn_box'] = tgt_RCNN_loss_bbox.mean().item()
        batch_metrics['loss']['tgt_structure'] = tgt_structure_loss.item()
        batch_metrics['loss']['src_align'] = src_align_loss.item()
        batch_metrics['loss']['center_contrastive'] = center_contrastive_loss.item()
        # 聚类中心之间的距离
        tmp_center = faster_rcnn.module.rpn_cluster_center.detach()
        center_dist = cal_feat_distance(tmp_center, tmp_center)
        for i in range(self.cluster_num):
            for j in range(i + 1, self.cluster_num):
                batch_metrics['center_dist']['dist_{}_and_{}'.format(i, j)] = center_dist[i][j].item()

        tgt_fg_cnt = 0 if tgt_rois_label is None else torch.sum(tgt_rois_label.detach().ne(0))
        batch_metrics['loss']['tgt_fg_count'] = tgt_fg_cnt

        return batch_metrics

    def load_pretrained_model(self, weights_path):
        model = torch.load(weights_path)
        self.model_dict['base_model'].load_state_dict(model, strict=False)

    def get_attention_mask(self, global_feat, attention_mask_lam=10.0):
        # global feat经过rpn
        global_rpn_feat = F.relu(self.model_dict['base_model'].module.RCNN_rpn.RPN_Conv(global_feat.detach()),
                                 inplace=True).squeeze(0).detach()
        num_dim, feat_shape_1, feat_shape_2 = global_rpn_feat.shape
        global_rpn_feat = torch.transpose(torch.transpose(global_rpn_feat, 0, 1), 1, 2)
        sim = cal_feat_distance(global_rpn_feat.view(-1, num_dim),
                                self.model_dict['base_model'].module.rpn_cluster_center.detach())
        mask = torch.softmax(sim * attention_mask_lam, dim=1)[:, 1].view(feat_shape_1, feat_shape_2) + 1
        return mask
