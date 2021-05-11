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


class RPNRecall(Hook):
    def __init__(self, runner, dataset_name, thresh=0.5, bins=10):
        self.dataset_name = dataset_name
        self.imdb = runner.test_loaders[dataset_name].dataset.imdb
        self.num_classes = runner.test_loaders[dataset_name].dataset.n_classes
        num_images = len(self.imdb.image_index)
        self.thresh = thresh
        self.bins = bins
        # 二维矩阵，表示每个类别的recall样本的概率分布（rpn网络输出的概率）
        self.recall_prob_hist = np.zeros((self.num_classes, self.bins))
        # 每个类别的gt后选框数量
        self.class_gt_num = np.zeros(self.num_classes)
        self.class_recall_num = np.zeros(self.num_classes)  # TODO:当前计算过程中没有考虑重叠的情况，即一个rpn box可能会被算两次

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        rois = batch_output['rois']
        dataset_name = batch_output['dataset_name']
        gt_boxes = batch_output['gt_boxes']
        #
        if dataset_name == self.dataset_name:
            gt_boxes = gt_boxes.cpu().numpy().squeeze(0)  # batch size 为1
            rois = rois.cpu().numpy().squeeze(0)  # shape一般是300x5
            # print('rois shape {}'.format(rois.shape))
            rois_prob = rois[:, 0]
            for class_ind in range(1, self.num_classes):
                gt_inds = np.where(gt_boxes[:, 4] == class_ind)[0]  #
                if gt_inds.size > 0:
                    self.class_gt_num[class_ind] += gt_inds.size
                    for temp_gt_ind in gt_inds.tolist():
                        temp_gt = gt_boxes[temp_gt_ind, 0:4]
                        ixmin = np.maximum(rois[:, 1], temp_gt[0])
                        iymin = np.maximum(rois[:, 2], temp_gt[1])
                        ixmax = np.minimum(rois[:, 3], temp_gt[2])
                        iymax = np.minimum(rois[:, 4], temp_gt[3])
                        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                        ih = np.maximum(iymax - iymin + 1.0, 0.0)
                        inters = iw * ih

                        # union
                        uni = ((temp_gt[2] - temp_gt[0] + 1.) * (temp_gt[3] - temp_gt[1] + 1.) +
                               (rois[:, 3] - rois[:, 1] + 1.) *
                               (rois[:, 4] - rois[:, 2] + 1.) - inters)
                        overlaps = inters / uni
                        # 只要有 proposal与该gt重叠，则认为被检测到了
                        real_overlap_ind = np.where(overlaps > self.thresh)[0]
                        if real_overlap_ind.size > 0:
                            self.class_recall_num[class_ind] += 1
                        # 将重叠的proposal的概率记录，放到直方图里
                        rpn_overlap_prob = rois_prob[real_overlap_ind]
                        temp_hist, _ = np.histogram(rpn_overlap_prob, bins=self.bins, range=(0.0, 1.0))
                        self.recall_prob_hist[class_ind, :] += temp_hist

    def after_val_epoch(self, runner):
        rpn_recall_path = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                                       'iter_{}_val_result'.format(runner.iteration), 'recall_matrix')
        if not os.path.exists(rpn_recall_path):
            os.makedirs(rpn_recall_path)

        for class_ind in range(1, self.num_classes):
            temp_class_name = self.imdb._classes[class_ind]
            temp_path = os.path.join(rpn_recall_path, '{}_hist.jpg'.format(temp_class_name))
            class_hist = self.recall_prob_hist[class_ind, :] / (self.class_gt_num[class_ind] + 1e-8)
            # fig = plt.figure()
            # plt.bar(range(self.bins), class_hist)
            # plt.savefig(temp_path)
            # fig.clear()
            # 输出每一类的recallrate
            class_recall = self.class_recall_num[class_ind] / (self.class_gt_num[class_ind] + 1e-8)
            runner.logger.info('{} recall rate of RPN is {}'.format(temp_class_name, class_recall))
            runner.writer.add_scalar('rpn_recall_{}/{}_{}'.format(self.imdb._name, class_ind, temp_class_name),
                                     class_recall, global_step=runner.iteration)
        overall_recall = np.sum(self.class_recall_num) / (np.sum(self.class_gt_num) + 1e-8)
        runner.logger.info("Overall recall rate of RPN is {}".format(overall_recall))
        runner.writer.add_scalar('rpn_recall_{}/{}_{}'.format(self.imdb._name, 0, 'overall'), overall_recall,
                                 global_step=runner.iteration)
