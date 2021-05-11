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
from detda.runner.hooks import DetValMetrics, RPNRecall


class ValidatorAdaptSet(BaseValidator):
    def __init__(self, cuda, logdir, test_loaders, model_dict, thresh, class_agnostic, max_per_image, logger=None,
                 writer=None):
        super(ValidatorAdaptSet, self).__init__(cuda=cuda, logdir=logdir, test_loaders=test_loaders,
                                                logger=logger, writer=writer, log_name=('det',),
                                                model_dict=model_dict)
        self.class_agnostic = class_agnostic
        for key, _ in self.test_loaders.items():
            det_val_metrics = DetValMetrics(self, thresh, class_agnostic, max_per_image, dataset_name=key)
            self.register_hook(det_val_metrics)
            rpn_recall = RPNRecall(self, dataset_name=key)
            # self.register_hook(rcnn_confusion_matrix)
            self.register_hook(rpn_recall)

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
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = faster_rcnn(im_data, im_info, gt_boxes, num_boxes)
            # print('val bbox pred shape {}'.format(bbox_pred.shape))
            #
            return {"rois": rois.detach(),
                    'cls_prob': cls_prob.detach(),
                    "bbox_pred": bbox_pred.detach(),
                    'im_data': im_data.detach(),
                    'im_info': im_info.detach(),
                    'gt_boxes': gt_boxes.detach(),
                    'num_boxes': num_boxes.detach(),
                    }


class TrainerBasicDetection(BaseTrainer):
    def __init__(self, cuda, model_dict, optimizer_dict, scheduler_dict, device_dict,
                 train_loaders=None, test_loaders=None,
                 logger=None, logdir=None, max_iters=None,
                 val_interval=5000, log_interval=50, save_interval=5000,
                 update_iter=1, save_test_res=False,
                 use_syncbn=False, max_save_num=3,
                 # new parameters for detection
                 class_agnostic=False, val_thresh=0, max_per_image=100,
                 ):
        super(TrainerBasicDetection, self).__init__(cuda=cuda, model_dict=model_dict, optimizer_dict=optimizer_dict,
                                                    scheduler_dict=scheduler_dict, device_dict=device_dict,
                                                    train_loaders=train_loaders,
                                                    test_loaders=test_loaders, max_iters=max_iters, logger=logger,
                                                    logdir=logdir,
                                                    val_interval=val_interval, log_interval=log_interval,
                                                    save_interval=save_interval, update_iter=update_iter,
                                                    save_test_res=save_test_res,
                                                    use_syncbn=use_syncbn,
                                                    max_save_num=max_save_num)
        self.class_agnostic = class_agnostic
        # 验证
        self.validator = ValidatorAdaptSet(cuda=cuda, logdir=self.logdir, test_loaders=self.test_loaders,
                                           logger=self.logger, writer=self.writer, class_agnostic=class_agnostic,
                                           model_dict=self.model_dict, thresh=val_thresh, max_per_image=max_per_image)

        # 增加记录项
        log_names = ['rpn_cls', 'rpn_box', 'rcnn_cls', 'rcnn_box', 'fg_count', 'bg_count']
        loss_metrics = LossMetrics(self, log_names=log_names, group_name='loss', log_interval=log_interval)
        self.register_hook(loss_metrics)
        #
        update_param = BackwardUpdate(self)
        self.register_hook(update_param)
        #
        model_class_name = self.model_dict['base_model'].module.__class__.__name__
        # 如果是vgg16，需要注册一个clip gradient的hook,跟在backward后面
        if 'vgg16' in model_class_name:
            gradient_clipper = GradientClipper(max_num=10.0)
            self.register_hook(gradient_clipper, 30)

    def train_iter(self, *args):
        im_data, im_info, gt_boxes, num_boxes, src_id = args[0]

        batch_metrics = {}
        batch_metrics['loss'] = {}

        faster_rcnn = self.model_dict['base_model']
        #
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = faster_rcnn(im_data, im_info, gt_boxes, num_boxes)
        # print('bbox pred shape {}'.format(bbox_pred.shape))
        #
        overall_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        overall_loss.backward()
        #
        batch_metrics['loss']['rpn_cls'] = rpn_loss_cls.mean().item()
        batch_metrics['loss']['rpn_box'] = rpn_loss_box.mean().item()
        batch_metrics['loss']['rcnn_cls'] = RCNN_loss_cls.mean().item()
        batch_metrics['loss']['rcnn_box'] = RCNN_loss_bbox.mean().item()
        fg_cnt = torch.sum(rois_label.detach().ne(0))
        batch_metrics['loss']['fg_count'] = fg_cnt
        batch_metrics['loss']['bg_count'] = rois_label.detach().numel() - fg_cnt

        return batch_metrics

    def load_pretrained_model(self, weights_path):
        model = torch.load(weights_path)
        self.model_dict['base_model'].load_state_dict(model)
