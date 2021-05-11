# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
from .hook import Hook
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from detda.utils.det_utils import assign_labels_for_rois, cal_class_mean_feat
import time
from detda.models.det_models.utils.config import cfg
from detda.utils.utils import cluster_within_per_class
import torch
import shutil
from detda.models.det_models.rpn.bbox_transform import bbox_overlaps
from detda.utils.det_utils import assign_labels_for_rpn_rois
from detda.utils.spkmeans import Clustering
from detda.utils.utils import cal_feat_distance

plt.switch_backend('agg')


class RPNClusterAlign(Hook):
    """
    提取RCNN网络输出的特征
    """

    def __init__(self, runner, dataset_name, proposal_sample_num, dim, sample_ratio_per_class,
                 num_cluster_per_class, start_iteration, save_features=False, prob_type='hard',
                 kmeans_dist='normal', init_center=False, assign_center_feat=True, assign_once=False,
                 fg_thresh=0.7, bg_thresh=0.3,
                 ):
        self.dataset_name = dataset_name
        self.imdb = runner.test_loaders[dataset_name].dataset.imdb
        self.num_classes = runner.test_loaders[dataset_name].dataset.n_classes
        #
        self.sample_num = proposal_sample_num
        self.proposal_num = proposal_sample_num * self.imdb.num_images
        self.dim = dim
        self.sample_ratio_per_class = sample_ratio_per_class
        self.num_cluster_per_class = num_cluster_per_class
        self.start_iteration = start_iteration
        self.save_features = save_features
        self.prob_type = prob_type
        self.kmeans_dist = kmeans_dist
        self.init_center = init_center
        self.assign_center_feat = assign_center_feat
        self.assign_once = assign_once
        #
        assert prob_type in ['soft', 'hard'], 'prob type should be soft or hard, not {}'.format(prob_type)
        #
        bbox_std = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        bbox_mean = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        if runner.cuda:
            bbox_std = bbox_std.to('cuda:0')
            bbox_mean = bbox_mean.to('cuda:0')
        self.bbox_std = bbox_std
        self.bbox_mean = bbox_mean
        #
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh

    def before_val_epoch(self, runner):
        # 应该像现在这样提前把空间分配好，而不是使用concat逐个添加
        # 二维矩阵，记录每一个proposal的FC1,FC2的特征向量
        self.file_num = 0
        self.feature_save_root = os.path.join(runner.logdir,
                                              self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                                              'iter_{}_val_result'.format(runner.iteration), 'rpn_feature')
        if not os.path.exists(self.feature_save_root):
            os.makedirs(self.feature_save_root)
        self.rpn_feature_save_path = os.path.join(self.feature_save_root, 'rpn_features_{}.pkl')
        self.cluster_res_save_path = os.path.join(runner.logdir,
                                                  self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                                                  'iter_{}_val_result'.format(runner.iteration), 'rpn_cluster_res.pkl')
        self.rpn_feat = np.zeros((50300, self.dim), dtype=np.float32)
        # 记录原图名称
        self.proposal_img = []
        # 记录proposal坐标
        self.bbox = np.zeros((50300, 5), dtype=np.float32)
        self.prob = np.zeros((50300, self.num_classes), dtype=np.float32)
        self.gt = np.zeros((50300), dtype=np.float32)
        self.scale = np.zeros((50300), dtype=np.float32)
        self.current_ind = 0

    def after_val_iter(self, runner):
        if runner.iteration >= self.start_iteration:
            batch_output = runner.batch_output
            rois = batch_output['rois'].squeeze(0)
            rpn_feat = batch_output['rpn_align_feat'].detach().cpu().numpy()
            # print('rpn feat dtype {}'.format(rpn_feat.dtype))
            cls_prob = batch_output['cls_prob'].cpu().numpy().squeeze(0)
            im_path = batch_output['img_id']
            dataset_name = batch_output['dataset_name']
            gt_boxes = batch_output['gt_boxes'].squeeze(0)
            im_info = batch_output['im_info']
            #
            if dataset_name == self.dataset_name:
                if rpn_feat.size > 0:
                    if gt_boxes.numel() > 0:
                        rois_labels = assign_labels_for_rpn_rois(rois[:, 1:5], gt_boxes, self.fg_thresh,
                                                                 self.bg_thresh)
                        rois_labels = rois_labels.cpu()
                        # keep_ind = np.concatenate((fg_ind, bg_ind), axis=0)
                        # rois = rois[keep_ind,:].cpu()
                    else:
                        rois_labels = np.ones(rpn_feat.shape[0], dtype=np.float32) * (-1)
                        rois = rois.cpu()
                    #
                    rois = rois.cpu()
                    # 固定的采样
                    start_ind = self.current_ind
                    end_ind = self.current_ind + self.sample_num
                    # self.feat_1[start_id:end_id, :] = feat_fc1
                    self.rpn_feat[start_ind:end_ind, :] = rpn_feat[0:self.sample_num, :]
                    self.bbox[start_ind:end_ind, :] = rois[0:self.sample_num, :]
                    # self.prob[start_ind:end_ind, :] = cls_prob[0:self.sample_num, :]
                    # 替换成rpn输出的预测结果
                    self.prob[start_ind:end_ind, :] = rois[0:self.sample_num, [0]]
                    self.gt[start_ind:end_ind] = rois_labels[0:self.sample_num]
                    self.scale[start_ind:end_ind] = im_info[0][2].item()
                    for i in range(self.sample_num):
                        self.proposal_img.append(im_path)
                    #
                    self.current_ind = end_ind
                    if self.current_ind >= 50000:
                        self.save_res()
                        self.current_ind = 0
                        self.file_num += 1

    def after_val_epoch(self, runner):
        if runner.iteration >= self.start_iteration:
            if self.current_ind > 0:
                self.save_res()
            #
            runner.logger.info('Start RPN cluster for iteration {}'.format(runner.iteration))
            res = self.read_extracted_rcnn_features(self.feature_save_root)
            feat = res['rpn_feat']
            prob = res['prob']
            bbox = res['bbox']
            scale = res['scale']
            #
            num_images = len(self.imdb.image_index)
            # 提取目标域的伪标签：filtered_all_boxes
            val_dir = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                                   'iter_{}_val_result'.format(runner.iteration))
            filtered_det_path = os.path.join(val_dir, 'filtered_detections.pkl')
            if os.path.exists(filtered_det_path):
                with open(filtered_det_path, 'rb') as f:
                    all_prob = pickle.load(f)
                # 获取每一张图像的标注
                gt_label = []
                for img_ind in range(num_images):
                    tmp_gt = np.empty((0, 5), dtype=np.float32)
                    for cls_ind in range(self.num_classes):
                        tmp_cls_gt = all_prob[cls_ind][img_ind]
                        if tmp_cls_gt == []:
                            continue
                        tmp_gt = np.concatenate((tmp_gt, tmp_cls_gt), axis=0).astype(np.float32)
                    gt_label.append(tmp_gt)
            else:
                # gt_label = self.imdb.create_annotation_file()
                # gt_label = None
                raise RuntimeError('You should not use gt information, unless you knew it')
            # 通过
            # 构造标签，
            proposal_rpn_labels = np.ones(feat.shape[0]) * (-1)
            # 按顺序对每一张图像进行处理
            for img_ind in range(num_images):
                #
                start_proposal_ind = self.sample_num * img_ind
                end_proposal_ind = self.sample_num * (img_ind + 1)
                # 预测结果和伪标签
                tmp_bbox = bbox[start_proposal_ind:end_proposal_ind, :]
                tmp_scale = scale[start_proposal_ind:end_proposal_ind][:, np.newaxis]
                tmp_gt = gt_label[img_ind]
                if tmp_gt.size == 0:
                    continue
                #
                if tmp_gt.size > 0 and tmp_bbox.size > 0:
                    # 计算overlap
                    tmp_rpn_labels = assign_labels_for_rpn_rois(torch.from_numpy(tmp_bbox[:, 1:5] / tmp_scale),
                                                                torch.from_numpy(tmp_gt),
                                                                fg_thresh=self.fg_thresh, bg_thresh=self.bg_thresh)
                    #
                    proposal_rpn_labels[start_proposal_ind:end_proposal_ind] = tmp_rpn_labels.cpu().numpy()
            # # TODO: 去掉这个操作
            # proposal_rpn_labels = res['gt']
            #
            rpn_label_save_path = os.path.join(val_dir, 'rpn_label_save.pkl')
            with open(rpn_label_save_path, 'wb') as f:
                pickle.dump(proposal_rpn_labels, f)
            print('before num {}'.format(np.where(proposal_rpn_labels == -1)[0].size))
            # 对feat做检查，如果有值特别大的情况，就删除掉
            feat, prob, proposal_rpn_labels = self.filtered_feat(feat, prob, proposal_rpn_labels, runner.logger)
            print('after num {}'.format(np.where(proposal_rpn_labels == -1)[0].size))
            # 根据proposal_rpn_label,如果等于-1，就去掉
            feat, prob, proposal_rpn_labels = self.filtered_by_rpn_label(feat, prob, proposal_rpn_labels, runner.logger)
            print('after2 num {}'.format(np.where(proposal_rpn_labels == -1)[0].size))
            # 获取两类的概率
            if self.prob_type == 'hard':
                fg_prob = proposal_rpn_labels
                bg_prob = 1 - proposal_rpn_labels
            else:
                # 对prob做处理
                # bg_prob = prob[:, 0]
                # fg_prob = 1 - bg_prob
                fg_prob = prob[:, 0]
                bg_prob = 1 - fg_prob
                # 根据rpn_label_置零
                fg_ind = np.where(proposal_rpn_labels == 1)[0]
                bg_ind = np.where(proposal_rpn_labels == 0)[0]
                assert fg_ind.size + bg_ind.size == proposal_rpn_labels.size, 'wrong rpn label'
                fg_prob[bg_ind] = 0
                bg_prob[fg_ind] = 0
            prob = np.concatenate((bg_prob[:, np.newaxis], fg_prob[:, np.newaxis]), axis=1)
            if self.init_center and runner.iteration > self.start_iteration:
                init_center = runner.model_dict['base_model'].module.rpn_cluster_center.detach().cpu().numpy()
            else:
                init_center = None
            if self.kmeans_dist == 'normal':
                cluster_res = cluster_within_per_class(feat, prob, 2,
                                                       sample_ratio=self.sample_ratio_per_class,
                                                       num_per_class=self.num_cluster_per_class,
                                                       init_centers=init_center)
            else:
                sp_kmeans = Clustering(eps=0.0001, num_classes=2,
                                       num_per_class=self.num_cluster_per_class,
                                       sample_ratio=self.sample_ratio_per_class)
                cluster_res = sp_kmeans.cluster_all_features(feat, prob, init_centers=init_center)
            #
            cluster_cls, cluster_feat, cluster_prob = cluster_res
            assert np.max(cluster_feat) < 30, 'wrong cluster result which has a larger value'
            cluster_res = {'cluster_cls': cluster_cls,
                           'cluster_center': cluster_feat,
                           'cluster_prob': cluster_prob,
                           }
            # 记录距离
            center_tensor = torch.from_numpy(cluster_feat)
            center_dist = cal_feat_distance(center_tensor, center_tensor)[0, 1].cpu().numpy().item()
            runner.writer.add_scalar('offshell_center_dist_{}'.format(self.imdb._name, ), center_dist,
                                     global_step=runner.iteration)
            with open(self.cluster_res_save_path, 'wb') as f:
                pickle.dump(cluster_res, f)
            if self.assign_center_feat:
                if self.assign_once:
                    if runner.iteration == self.start_iteration:
                        self.assign_center(runner, cls=cluster_cls, feat=cluster_feat, prob=cluster_prob)
                else:
                    self.assign_center(runner, cls=cluster_cls, feat=cluster_feat, prob=cluster_prob)
            #
            del self.rpn_feat
            del self.proposal_img
            del self.bbox
            del self.prob
            del self.gt
            # 删除中间结果
            self.delete_internal_res()

    def save_res(self):
        #
        save_res = {'rpn_feat': self.rpn_feat[0:self.current_ind, :],
                    'img_path': self.proposal_img,
                    'bbox': self.bbox[0:self.current_ind, :],
                    'prob': self.prob[0:self.current_ind, :],
                    'gt': self.gt[0:self.current_ind],
                    'scale': self.scale[0:self.current_ind]}
        with open(self.rpn_feature_save_path.format(self.file_num), 'wb') as f:
            pickle.dump(save_res, f, protocol=4)
        #
        self.rpn_feat = np.zeros((50300, self.dim), dtype=np.float32)
        # 记录原图名称
        self.proposal_img = []
        # 记录proposal坐标
        self.bbox = np.zeros((50300, 5), dtype=np.float32)
        self.prob = np.zeros((50300, self.num_classes), dtype=np.float32)
        self.gt = np.zeros((50300), dtype=np.float32)

    def read_extracted_rcnn_features(self, res_path):
        file_len = len(os.listdir(res_path))
        file_list = []
        for i in range(file_len):
            file_list.append('rpn_features_{}.pkl'.format(i))
        print('feature save files are {}'.format(file_list))
        final_res = {}
        for file in file_list:
            tmp_path = os.path.join(res_path, file)
            with open(tmp_path, 'rb') as f:
                tmp_res = pickle.load(f)
            for key, item in tmp_res.items():
                if key not in final_res:
                    final_res[key] = item
                else:
                    if isinstance(item, np.ndarray):
                        final_res[key] = np.concatenate((final_res[key], item), axis=0)
                    elif isinstance(item, list):
                        final_res[key].extend(item)
                    else:
                        raise RuntimeError('can not handle {}'.format(type(item)))
        return final_res

    def delete_internal_res(self):
        if not self.save_features:
            for name in os.listdir(self.feature_save_root):
                tmp_path = os.path.join(self.feature_save_root, name)
                os.remove(tmp_path)
            shutil.move(self.cluster_res_save_path, self.feature_save_root)

    def filtered_feat(self, feat, prob, proposal_rpn_labels, logger):
        max_val = np.max(feat, axis=1)
        selected_ind = np.where((max_val < 30) & (max_val > 1e-5))[0]
        ignored_ind = np.where((max_val >= 30) + (max_val <= 1e-5))[0]
        if ignored_ind.size > 0:
            logger.info("{} feat are ignored, max val {}".format(ignored_ind.size, np.max(max_val[ignored_ind])))
        #
        fg_bg_ind = np.where(proposal_rpn_labels >= 0)[0]
        final_ind = np.intersect1d(selected_ind, fg_bg_ind)
        feat = feat[final_ind, :]
        prob = prob[final_ind, :]
        proposal_rpn_labels = proposal_rpn_labels[final_ind]
        return feat, prob, proposal_rpn_labels

    def filtered_by_rpn_label(self, feat, prob, proposal_rpn_labels, logger):
        keep_ind = np.where(proposal_rpn_labels >= 0)[0]
        logger.info('orig shape {}, after filtered by rpn label {}'.format(feat.shape[0], keep_ind.size))
        feat = feat[keep_ind, :]
        prob = prob[keep_ind, :]
        proposal_rpn_labels = proposal_rpn_labels[keep_ind]
        return feat, prob, proposal_rpn_labels

    def assign_center(self, runner, cls, feat, prob):
        with torch.no_grad():
            runner.model_dict['base_model'].module.rpn_cluster_cls.data.copy_(torch.from_numpy(cls))
            runner.model_dict['base_model'].module.rpn_cluster_center.data.copy_(torch.from_numpy(feat))
            print('assign new rpn center')
