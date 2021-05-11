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
from detda.runner.hooks import DetValMetrics, RCNNConfusionMetrics, RPNRecall
from detda.loss import FocalLoss, EFocalLoss
from detda.runner.hooks import DetOnlinePseudoLabel
from detda.models.det_models.utils.net_utils import grad_reverse
from detda.utils.det_utils import flip_data, cls_flip_consistency_loss, loc_flip_consistency_loss


def sample_foreground_index(prob):
    cls_index = torch.argmax(prob, dim=-1)
    mask = torch.nonzero(cls_index > 0, as_tuple=False).squeeze()
    return mask


class ValidatorBasicAdvDet(BaseValidator):
    def __init__(self, cuda, logdir, test_loaders, model_dict, thresh, class_agnostic, max_per_image, logger=None,
                 writer=None, pseudo_label=False, high_thresh=1.0, low_thresh=0.0, ratio=0.5, pseudo_loader_num=1,
                 trainer=None, filter_by_ratio=False, dynamic_ratio=False, max_iter=30000, min_low_thresh=0.1):
        super(ValidatorBasicAdvDet, self).__init__(cuda=cuda, logdir=logdir, test_loaders=test_loaders,
                                                   logger=logger, writer=writer, log_name=('det',),
                                                   model_dict=model_dict)
        self.class_agnostic = class_agnostic
        self.trainer = trainer
        for key, _ in self.test_loaders.items():
            det_val_metrics = DetValMetrics(self, thresh, class_agnostic, max_per_image, dataset_name=key)
            # rcnn_confusion_matrix = RCNNConfusionMetrics(self, dataset_name=key)
            rpn_recall = RPNRecall(self, dataset_name=key)
            self.register_hook(det_val_metrics)
            # self.register_hook(rcnn_confusion_matrix)
            self.register_hook(rpn_recall)
            rcnn_feature = RCNNFeature(runner=self, dataset_name=key, dim1=4096, dim2=4096)
            self.register_hook(rcnn_feature)
        # 创建第二个测试集的伪标签
        if pseudo_label:
            for iter, (key, dataloader) in enumerate(self.test_loaders.items()):
                if iter == pseudo_loader_num:
                    online_pseudo_label = DetOnlinePseudoLabel(runner=self, dataset_name=key, high_thresh=high_thresh,
                                                               low_thresh=low_thresh, ratio=ratio,
                                                               filter_by_ratio=filter_by_ratio,
                                                               dynamic_ratio=dynamic_ratio,
                                                               max_iter=max_iter, min_low_thresh=min_low_thresh)
                    self.register_hook(online_pseudo_label, priority=100)
                    break

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
            rpn_related, rcnn_related = faster_rcnn(im_data, im_info, gt_boxes, num_boxes)
            #
            rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred = final_predict
            # rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox = losses
            # feat1, feat2, feat3, orig_pooled_feat = backbone_related
            # rpn_cls_score, rpn_conv1 = rpn_related
            pooled_feat, = rcnn_related

            # self.logger.info('concat prob {}'.format(concat_prob.cpu()))
            # print('val bbox pred shape {}'.format(bbox_pred.shape))
            #
            return {"rois": rois,
                    "rois_label": rois_label,
                    # "rois_target": rois_target,
                    'cls_prob': cls_prob,
                    "bbox_pred": bbox_pred,
                    'im_data': im_data,
                    'im_info': im_info,
                    'gt_boxes': gt_boxes,
                    'num_boxes': num_boxes,
                    }


class TrainerBasicAdvDetection(BaseTrainer):
    def __init__(self, cuda, model_dict, optimizer_dict, scheduler_dict, device_dict,
                 train_loaders=None, test_loaders=None,
                 logger=None, logdir=None, max_iters=None,
                 val_interval=5000, log_interval=50, save_interval=5000,
                 update_iter=1, save_test_res=False,
                 use_syncbn=False, max_save_num=3, cudnn_deterministic=False,
                 # new parameters for detection
                 class_agnostic=False, val_thresh=0, max_per_image=100, ef=False, lambda_adv=1.0,
                 gamma=5, use_target_label=False, lambda_tgt_det=1.0,
                 tgt_loss_type=0, detach_context=False, src_rcnn_cls_mixup=False, rcnn_mixup_alpha=0.2,
                 tgt_rcnn_cls_mixup=False,
                 online_pseudo_label=False, ratio=0.5, high_threshold=1.0, low_threshold=0.0, pseudo_loader_num=1,
                 pseudo_start_iter=0, clip_gradient=False, filter_by_ratio=False,
                 dynamic_ratio=False, max_iter=30000, min_low_thresh=0.1,
                 cross_domain_mixup=False, mixup_detach_context=False, context_lambda=0.5,
                 lambda_adv_1=1.0, lambda_adv_2=1.0, lambda_adv_3=1.0, lambda_adv_inst=1.0,
                 lambda_grad_reverse=1.0, target_sample_strategy='top',
                 flip_consistency=False, sample_for_flip=False, lambda_flip_cls=1.0, lambda_flip_loc=1.0,
                 ):
        super(TrainerBasicAdvDetection, self).__init__(cuda=cuda, model_dict=model_dict, optimizer_dict=optimizer_dict,
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
        self.use_target_label = use_target_label
        self.lambda_tgt_det = lambda_tgt_det
        self.tgt_loss_type = tgt_loss_type
        self.detach_context = detach_context
        self.rcnn_cls_mixup = src_rcnn_cls_mixup
        self.rcnn_mixup_alpha = rcnn_mixup_alpha
        self.tgt_rcnn_cls_mixup = tgt_rcnn_cls_mixup
        self.pseudo_start_iter = pseudo_start_iter
        self.cross_domain_mixup = cross_domain_mixup
        self.mixup_detach_context = mixup_detach_context
        self.context_lambda = context_lambda
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

        if ef:
            self.focal_loss = EFocalLoss(class_num=2, gamma=gamma)
        else:
            self.focal_loss = FocalLoss(class_num=2, gamma=gamma, sigmoid=True)
        self.lambda_adv = lambda_adv
        # 验证
        self.validator = ValidatorBasicAdvDet(cuda=cuda, logdir=self.logdir, test_loaders=self.test_loaders,
                                              logger=self.logger, writer=self.writer, class_agnostic=class_agnostic,
                                              model_dict=self.model_dict, thresh=val_thresh,
                                              max_per_image=max_per_image,
                                              ratio=ratio, low_thresh=low_threshold, high_thresh=high_threshold,
                                              pseudo_label=online_pseudo_label, pseudo_loader_num=pseudo_loader_num,
                                              trainer=self, filter_by_ratio=filter_by_ratio,
                                              dynamic_ratio=dynamic_ratio, max_iter=max_iter,
                                              min_low_thresh=min_low_thresh)

        # 增加记录项
        log_names = ['rpn_cls', 'rpn_box', 'rcnn_cls', 'rcnn_box', 'fg_count', 'bg_count',
                     'src_domain_1', 'src_domain_2', 'src_domain_3', 'src_domain_inst',
                     'tgt_domain_1', 'tgt_domain_2', 'tgt_domain_3', 'tgt_domain_inst',
                     'tgt_rpn_cls', 'tgt_rpn_box', 'tgt_rcnn_cls', 'tgt_rcnn_box',
                     'flip_cls', 'flip_loc', 'tgt_fg_count',
                     ]
        if self.cross_domain_mixup:
            log_names.append('cross_domain_mixup')
        loss_metrics = LossMetrics(self, log_names=log_names, group_name='loss', log_interval=log_interval)
        self.register_hook(loss_metrics)
        #
        update_param = BackwardUpdate(self)
        self.register_hook(update_param)

        # gl原文中没有clip_gradient
        # model_class_name = self.model_dict['base_model'].module.__class__.__name__
        # # 如果是vgg16，需要注册一个clip gradient的hook,紧跟在backward后面，需要高优先级
        # if 'vgg16' in model_class_name:
        #     gradient_clipper = GradientClipper(max_num=10.0)
        #     self.register_hook(gradient_clipper, 30)
        if clip_gradient:
            gradient_clipper = GradientClipper(max_num=10.0)
            self.register_hook(gradient_clipper, 0)

    def train_iter(self, *args):
        src_im_data, src_im_info, src_gt_boxes, src_num_boxes, src_id = args[0]
        tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_id = args[1]

        # src_im_data = src_im_data.cuda(non_blocking=True)
        # tgt_im_data = tgt_im_data.cuda(non_blocking=True)
        batch_metrics = {}
        batch_metrics['loss'] = {}

        faster_rcnn = self.model_dict['base_model']
        dis_1 = self.model_dict['dis_1']
        dis_2 = self.model_dict['dis_2']
        dis_3 = self.model_dict['dis_3']
        dis_inst = self.model_dict['dis_inst']
        #
        # print('source')
        # 源域前向传播
        final_predict, losses, backbone_related, \
        rpn_related, rcnn_related = faster_rcnn(src_im_data, src_im_info, src_gt_boxes,
                                                src_num_boxes)
        #
        rois, cls_prob, bbox_pred, rois_label, orig_bbox_pred = final_predict
        rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox = losses
        feat1, feat2, feat3, orig_pooled_feat = backbone_related
        rpn_cls_score, rpn_conv1 = rpn_related
        pooled_feat, = rcnn_related
        # print('rois label require grad {}'.format(rois_label.requires_grad))
        # 源域检测损失
        src_det_loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        # 源域对抗损失
        domain_p1 = dis_1(grad_reverse(feat1, lambd=self.lambda_grad_reverse))
        domain_p2 = dis_2(grad_reverse(feat2, lambd=self.lambda_grad_reverse))
        domain_p3 = dis_3(grad_reverse(feat3, lambd=self.lambda_grad_reverse))
        domain_pinst = dis_inst(grad_reverse(pooled_feat, lambd=self.lambda_grad_reverse))
        #
        src_label_2 = torch.zeros(domain_p2.size(), dtype=torch.float).cuda()
        src_label_3 = torch.zeros(domain_p3.size(0), dtype=torch.long).cuda()
        src_label_inst = torch.zeros(domain_pinst.size(0), dtype=torch.long).cuda()
        src_domain_loss_1 = 0.5 * torch.mean(domain_p1 ** 2) * self.lambda_adv_1
        src_domain_loss_2 = 0.5 * F.binary_cross_entropy_with_logits(domain_p2, src_label_2) * 0.15 * self.lambda_adv_2
        src_domain_loss_3 = 0.5 * self.focal_loss(domain_p3, src_label_3) * self.lambda_adv_3
        src_domain_loss_inst = 0.5 * self.focal_loss(domain_pinst, src_label_inst) * self.lambda_adv_inst
        src_loss = src_det_loss + self.lambda_adv * (
                src_domain_loss_1 + src_domain_loss_2 + src_domain_loss_3 + src_domain_loss_inst)

        if self.cross_domain_mixup:
            src_loss.backward(retain_graph=True)
        else:
            src_loss.backward()
        #
        # print('target')
        tgt_loss = 0
        # 目标域前向传播
        if self.use_target_label and tgt_num_boxes > 0 and self.iteration > self.pseudo_start_iter:
            tgt_final_predict, tgt_losses, tgt_backbone_related, \
            tgt_rpn_related, tgt_rcnn_related = faster_rcnn(tgt_im_data, tgt_im_info,
                                                            tgt_gt_boxes,
                                                            tgt_num_boxes,
                                                            use_pseudo=True,
                                                            )
            tgt_rpn_loss_cls, tgt_rpn_loss_box, tgt_RCNN_loss_cls, tgt_RCNN_loss_bbox = tgt_losses
            tgt_loss += (tgt_rpn_loss_cls.mean() + tgt_rpn_loss_box.mean() +
                         tgt_RCNN_loss_cls.mean() + tgt_RCNN_loss_bbox.mean()) * self.lambda_tgt_det
            # cross domain
            if self.cross_domain_mixup:
                cross_rcnn_loss = faster_rcnn.module.cross_domain_rcnn_mixup((orig_pooled_feat, rois_label),
                                                                             (tgt_orig_pooled_feat, tgt_rois_label),
                                                                             self.rcnn_mixup_alpha)
                tgt_loss += cross_rcnn_loss
            tgt_fg_count = torch.tensor(0.0)
        else:
            tgt_final_predict, tgt_losses, tgt_backbone_related, \
            tgt_rpn_related, tgt_rcnn_related = faster_rcnn(tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes,
                                                            target=True,
                                                            target_sample_strategy=self.target_sample_strategy,
                                                            )
            tgt_rpn_loss_cls = torch.tensor(0.0)
            tgt_rpn_loss_box = torch.tensor(0.0)
            tgt_RCNN_loss_cls = torch.tensor(0.0)
            tgt_RCNN_loss_bbox = torch.tensor(0.0)
            cross_rcnn_loss = torch.tensor(0.0)
        #
        tgt_rois, tgt_cls_prob, tgt_bbox_pred, tgt_rois_label, tgt_orig_bbox_pred = tgt_final_predict
        tgt_feat1, tgt_feat2, tgt_feat3, tgt_orig_pooled_feat = tgt_backbone_related
        tgt_pool_feat, = tgt_rcnn_related
        # 翻转一致性
        if self.flip_consistency:
            flip_tgt_data = flip_data(tgt_im_data, 3)
            tgt_rois_flip = tgt_rois.clone().detach()
            tgt_rois_flip[:, :, 1] = tgt_im_data.shape[3] - tgt_rois_flip[:, :, 1] - 1
            tgt_rois_flip[:, :, 3] = tgt_im_data.shape[3] - tgt_rois_flip[:, :, 3] - 1
            #
            flip_pred, _, _, _, _ = faster_rcnn(flip_tgt_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, target=True,
                                                outer_rois=(tgt_rois, tgt_rois_label))
            _, tgt_cls_prob_flip, tgt_bbox_pred_flip, _ = flip_pred
            #
            foreground_ind = sample_foreground_index(tgt_cls_prob)
            foreground_count = torch.numel(foreground_ind)
            #

            if self.sample_for_flip:
                if foreground_count > 0:
                    tgt_cls_prob = tgt_cls_prob[foreground_ind, :]
                    tgt_cls_prob_flip = tgt_cls_prob_flip[foreground_ind, :]
                    tgt_bbox_pred = tgt_bbox_pred[foreground_ind, :]
                    tgt_bbox_pred_flip = tgt_bbox_pred_flip[foreground_ind, :]
                    # TODO:验证维数
                    flip_loss_cls = cls_flip_consistency_loss(tgt_cls_prob, tgt_cls_prob_flip)
                    flip_loss_loc = loc_flip_consistency_loss(tgt_bbox_pred, tgt_bbox_pred_flip)
                else:
                    flip_loss_cls = torch.tensor(0.0)
                    flip_loss_loc = torch.tensor(0.0)
            else:
                flip_loss_cls = cls_flip_consistency_loss(tgt_cls_prob, tgt_cls_prob_flip)
                flip_loss_loc = loc_flip_consistency_loss(tgt_bbox_pred, tgt_bbox_pred_flip)
        else:
            flip_loss_cls = torch.tensor(0.0)
            flip_loss_loc = torch.tensor(0.0)

        # 目标域对抗损失
        tgt_domain_p1 = dis_1(grad_reverse(tgt_feat1, lambd=self.lambda_grad_reverse))
        tgt_domain_p2 = dis_2(grad_reverse(tgt_feat2, lambd=self.lambda_grad_reverse))
        tgt_domain_p3 = dis_3(grad_reverse(tgt_feat3, lambd=self.lambda_grad_reverse))
        # tgt_domain_pinst = dis_inst(grad_reverse(tgt_pool_feat, lambd=self.lambda_grad_reverse))
        #
        tgt_label_2 = torch.zeros(tgt_domain_p2.size(), dtype=torch.float).cuda()
        tgt_label_3 = torch.zeros(tgt_domain_p3.size(0), dtype=torch.long).cuda()
        tgt_label_inst = torch.zeros(tgt_domain_pinst.size(0), dtype=torch.long).cuda()
        tgt_domain_loss_1 = 0.5 * torch.mean((1 - tgt_domain_p1) ** 2) * self.lambda_adv_1
        tgt_domain_loss_2 = 0.5 * F.binary_cross_entropy_with_logits(tgt_domain_p2,
                                                                     tgt_label_2) * 0.15 * self.lambda_adv_2
        tgt_domain_loss_3 = 0.5 * self.focal_loss(tgt_domain_p3, tgt_label_3) * self.lambda_adv_3
        # tgt_domain_loss_inst = 0.5 * self.focal_loss(tgt_domain_pinst, tgt_label_inst) * self.lambda_adv_inst
        #
        tgt_loss += self.lambda_adv * (tgt_domain_loss_1 + tgt_domain_loss_2 + tgt_domain_loss_3)
        tgt_loss += self.lambda_flip_cls * flip_loss_cls + self.lambda_flip_loc * flip_loss_loc
        tgt_loss.backward()
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
        batch_metrics['loss']['src_domain_inst'] = src_domain_loss_inst.mean().item()
        #
        batch_metrics['loss']['tgt_domain_1'] = tgt_domain_loss_1.mean().item()
        batch_metrics['loss']['tgt_domain_2'] = tgt_domain_loss_2.mean().item()
        batch_metrics['loss']['tgt_domain_3'] = tgt_domain_loss_3.mean().item()
        batch_metrics['loss']['tgt_domain_inst'] = tgt_domain_loss_inst.mean().item()

        batch_metrics['loss']['tgt_rpn_cls'] = tgt_rpn_loss_cls.mean().item()
        batch_metrics['loss']['tgt_rpn_box'] = tgt_rpn_loss_box.mean().item()
        batch_metrics['loss']['tgt_rcnn_cls'] = tgt_RCNN_loss_cls.mean().item()
        batch_metrics['loss']['tgt_rcnn_box'] = tgt_RCNN_loss_bbox.mean().item()
        #
        batch_metrics['loss']['flip_cls'] = flip_loss_cls.item()
        batch_metrics['loss']['flip_loc'] = flip_loss_loc.item()
        tgt_fg_cnt = 0 if tgt_rois_label is None else torch.sum(tgt_rois_label.detach().ne(0))
        batch_metrics['loss']['tgt_fg_count'] = tgt_fg_cnt
        if self.cross_domain_mixup:
            batch_metrics['loss']['cross_domain_mixup'] = cross_rcnn_loss.item()

        return batch_metrics

    def load_pretrained_model(self, weights_path):
        model = torch.load(weights_path)
        self.model_dict['base_model'].load_state_dict(model)
