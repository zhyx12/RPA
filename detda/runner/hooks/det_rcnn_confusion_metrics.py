# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import torch.nn.functional as F
from .hook import Hook
from detda.loss import cross_entropy2d
from detda.utils.metrics import runningMetric
import os
import numpy as np
from detda.models.det_models.utils.config import cfg
from detda.models.det_models.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from torchvision.ops import nms
import pickle
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import seaborn as sn


class RCNNConfusionMetrics(Hook):
    def __init__(self, runner, dataset_name):
        self.dataset_name = dataset_name
        self.imdb = runner.test_loaders[dataset_name].dataset.imdb
        self.num_classes = runner.test_loaders[dataset_name].dataset.n_classes
        num_images = len(self.imdb.image_index)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        cls_prob = batch_output['cls_prob']
        rois_label = batch_output['rois_label']
        rois = batch_output['rois']
        bbox_pred = batch_output['bbox_pred']
        im_info = batch_output['im_info']
        dataset_name = batch_output['dataset_name']
        val_iter = runner.val_iter
        #
        if dataset_name == self.dataset_name:
            pred_label = np.argmax(cls_prob.cpu().numpy(), axis=2).reshape(-1)
            gt_label = rois_label.cpu().numpy().reshape(-1)
            self.confusion_matrix[gt_label, pred_label] += 1

    def after_val_epoch(self, runner):
        val_dir = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                               'iter_{}_val_result'.format(runner.iteration))
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        confusion_mat_file = os.path.join(val_dir, 'confusion_mat.jpg')
        fig = plt.figure()
        sn.heatmap(self.confusion_matrix)
        plt.savefig(confusion_mat_file)
        runner.writer.add_figure('rcnn_confusion_matrix', figure=fig, global_step=runner.iteration)
        # rcnn准确率
        rcnn_acc = np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)
        runner.writer.add_scalar('rcnn_acc', rcnn_acc, global_step=runner.iteration)
        runner.logger.info('RCNN accuracy is {}'.format(rcnn_acc))
        # 将confusion matrix中的所有非对焦元素按照大小排序,并打印占据前90%的值
        confusion_matrix = self.confusion_matrix
        index = np.arange(self.num_classes)
        confusion_matrix[index, index] = 0
        sorted_ind = np.argsort(-confusion_matrix, None)
        overall_false_num = np.sum(confusion_matrix)
        accumulate_val = 0
        for temp_ind in range(confusion_matrix.size - self.num_classes):
            gt_class_ind = sorted_ind[temp_ind] // self.num_classes
            pred_class_ind = sorted_ind[temp_ind] % self.num_classes
            gt_class_name = self.imdb._classes[gt_class_ind]
            pred_class_name = self.imdb._classes[pred_class_ind]
            ratio = confusion_matrix[gt_class_ind, pred_class_ind] / overall_false_num
            accumulate_val += ratio
            if accumulate_val > 0.9:
                break
            runner.logger.info('false pred from {}---->{}, ratio {}'.format(gt_class_name, pred_class_name, ratio))
