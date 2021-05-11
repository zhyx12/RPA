# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from .hook import Hook
import os
import numpy as np
from detda.models.det_models.utils.config import cfg
import pickle
from detda.loader.det_loaders import process_one_det_dataset
from detda.utils.det_utils import boxes_ensemble, box_filter, generate_pseudo_from_all_boxes, gt_num_per_image, \
    cal_acc, filter_by_ratio
import time


class DetOnlinePseudoLabel(Hook):
    def __init__(self, runner, dataset_name, high_thresh, low_thresh, ratio, filter_by_ratio=False, dynamic_ratio=False,
                 max_iter=100000, min_low_thresh=0.1, use_ensemble=False):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.ratio = ratio
        self.dynamic_ratio = dynamic_ratio
        self.max_iter = max_iter
        self.min_low_thresh = min_low_thresh
        self.dataset_name = dataset_name
        self.imdb = runner.test_loaders[dataset_name].dataset.imdb
        self.num_classes = runner.test_loaders[dataset_name].dataset.n_classes
        num_images = len(self.imdb.image_index)
        self.ensemble_res_list = []
        self.ensemble_length = 3
        self.ensemble_iou_threshold = 0.5
        self.use_ensemble = use_ensemble
        # self.filter_by_ratio = filter_by_ratio
        # if filter_by_ratio:
        #     max_ratio_path = os.path.join(self.imdb._data_path, 'max_ratio.pkl')
        #     min_ratio_path = os.path.join(self.imdb._data_path, 'min_ratio.pkl')
        #     with open(max_ratio_path, 'rb') as f:
        #         self.max_ratio = pickle.load(f)
        #     with open(min_ratio_path, 'rb') as f:
        #         self.min_ratio = pickle.load(f)

    def after_val_iter(self, runner):
        pass
        # batch_output = runner.batch_output
        # cls_prob = batch_output['cls_prob']
        # rois = batch_output['rois']
        # bbox_pred = batch_output['bbox_pred']
        # im_info = batch_output['im_info']
        # dataset_name = batch_output['dataset_name']
        # val_iter = runner.val_iter
        # #
        # if dataset_name == self.dataset_name:
        #     pass

    def after_val_epoch(self, runner):
        val_dir = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                               'iter_{}_val_result'.format(runner.iteration))
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        det_file = os.path.join(val_dir, 'detections.pkl')
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)
        # 创建保存新标签的文件夹
        dataset_root = os.path.join(val_dir, 'dataset_root')
        dataset_path = os.path.join(dataset_root, self.imdb.name_for_path)
        # 根据filter过滤
        # if self.filter_by_ratio:
        #     all_boxes, filter_num = filter_by_ratio(all_boxes, self.max_ratio, self.min_ratio)
        #     runner.logger.info('filter {} boxes by ratio'.format(filter_num))
        # 过滤标签，生产新的检测结果
        if self.dynamic_ratio:
            temp_ratio = self.ratio - (runner.iteration / self.max_iter) * self.ratio
            if temp_ratio < self.min_low_thresh:
                temp_ratio = self.min_low_thresh
        else:
            temp_ratio = self.ratio
        cls_prob_thresh, new_all_prob, available_img_num, available_box_num = box_filter(all_boxes, self.imdb,
                                                                                         primary_low_thresh=self.low_thresh,
                                                                                         primary_high_thresh=self.high_thresh,
                                                                                         ratio_thresh=temp_ratio)
        # # box 融合
        # if self.use_ensemble:
        #     if len(self.ensemble_res_list) < self.ensemble_length - 1:
        #         self.ensemble_res_list.append(new_all_prob)
        #     elif len(self.ensemble_res_list) == self.ensemble_length - 1:
        #         self.ensemble_res_list.append(new_all_prob)
        #         new_all_prob = boxes_ensemble(self.ensemble_res_list, cluster_iou_thresh=self.ensemble_iou_threshold)
        #         self.ensemble_res_list[self.ensemble_length - 1] = new_all_prob
        #     else:
        #         self.ensemble_res_list.append(new_all_prob)
        #         self.ensemble_res_list.pop(0)
        #         new_all_prob = boxes_ensemble(self.ensemble_res_list, cluster_iou_thresh=self.ensemble_iou_threshold)
        #         self.ensemble_res_list[self.ensemble_length - 1] = new_all_prob
        # 保存过滤后的检测结果
        filtered_det_path = os.path.join(val_dir, 'filtered_detections.pkl')
        with open(filtered_det_path, 'wb') as f:
            pickle.dump(new_all_prob, f)
        # 保存类别的概率阈值
        prob_thresh_path = os.path.join(val_dir, 'class_threshold.pkl')
        with open(prob_thresh_path, 'wb') as f:
            pickle.dump(cls_prob_thresh, f)
        runner.logger.info('class prob thresholds are {}'.format(cls_prob_thresh))
        # 统计新结果的mAP和acc
        aps = self.imdb.evaluate_detections(new_all_prob, dataset_path)
        class_acc = cal_acc(dataset_path, self.imdb)
        runner.logger.info('Mean ap of filtered boxes is {}, class acc is {}'.format(np.mean(aps[0]), class_acc[0]))
        runner.logger.info('img num {}, min box num {}, avgbox num {}'.format(available_img_num,
                                                                              np.min(available_box_num[1:]),
                                                                              np.average(available_box_num[1:])))
        for ind, (class_name, ap) in enumerate(zip(self.imdb._classes[1:], aps[0])):
            runner.writer.add_scalar('filtered/{}_{}'.format(self.imdb._name, ind + 1, class_name), ap,
                                     global_step=runner.iteration)
        runner.writer.add_scalar('filtered/0_map', np.mean(aps[0]), global_step=runner.iteration)
        # 在val_dir下建立数据集文件夹，链接图像文件夹
        orig_img_path = os.path.join(self.imdb._data_path, 'JPEGImages')
        if os.path.islink(orig_img_path):
            orig_img_path = os.readlink(orig_img_path)
        new_img_path = os.path.join(dataset_path, 'JPEGImages')
        os.symlink(orig_img_path, new_img_path)
        # 将过滤之后的检测结果写入到数据集文件夹中
        generate_pseudo_from_all_boxes(new_all_prob, dataset_path, self.imdb, split=self.imdb._image_set)
        # 根据新的文件夹创建 train_loader，替换trainer原来的
        dataset_name = '_'.join(self.imdb.name.split('_')[0:-1])
        dataset_param = {'name': dataset_name, 'split': self.imdb._image_set, 'keep_empty_image': True,
                         'devkit_path': dataset_root}
        new_trainer_loader = process_one_det_dataset(dataset_param, batch_size=1, n_workers=4, logger=runner.logger,
                                                     training=True, augmentations=None, shuffle=True,
                                                     transfer_label=False)
        if runner.trainer.train_loaders is not None and len(runner.trainer.train_loaders) > 1:
            print('assign new train loader')
            runner.logger.info('assign new train loader')
            del runner.trainer.train_loader_iterator[1]
            del runner.trainer.train_loaders[1]
            runner.trainer.train_loaders.append(new_trainer_loader)
            runner.logger.info('Change target dataloader')
            time.sleep(2)
            runner.trainer.train_loader_iterator.append(runner.trainer.train_loaders[1].__iter__())
            # runner.trainer.train_loaders[1] = new_trainer_loader
