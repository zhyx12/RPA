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
from detda.utils.det_utils import transform_back_to_img
from PIL import Image, ImageDraw, ImageFont


class DetValMetrics(Hook):
    def __init__(self, runner, thresh, class_agnostic, max_per_image, dataset_name, save_vis=False, set_save_flag=True):
        self.thresh = thresh  # 置信度阈值
        self.max_per_image = max_per_image
        self.dataset_name = dataset_name
        self.imdb = runner.test_loaders[dataset_name].dataset.imdb
        self.num_classes = runner.test_loaders[dataset_name].dataset.n_classes
        self.class_agnostic = class_agnostic
        self.set_save_flag = set_save_flag
        num_images = len(self.imdb.image_index)
        print('val class number is {}'.format(self.num_classes))
        self.all_boxes = [[[] for _ in range(num_images)]
                          for _ in range(self.imdb.num_classes)]
        self.empty_array = np.transpose(np.array([[], [], [], [], []], dtype=np.float32), (1, 0))
        bbox_std = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        bbox_mean = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        if runner.cuda:
            bbox_std = bbox_std.to('cuda:0')
            bbox_mean = bbox_mean.to('cuda:0')
        self.bbox_std = bbox_std
        self.bbox_mean = bbox_mean
        self.best_map = 0
        # 保存类别预测概率分布和预测box长宽比分布
        self.pred_prob_hist = np.zeros((self.num_classes, 10))
        self.pred_box_ration_hist = np.zeros((self.num_classes, 100))
        self.save_vis = save_vis

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        #
        if dataset_name == self.dataset_name:
            #
            cls_prob = batch_output['cls_prob']
            rois = batch_output['rois']
            bbox_pred = batch_output['bbox_pred']
            im_info = batch_output['im_info']
            val_iter = runner.val_iter
            #
            scores = cls_prob
            boxes = rois[:, :, 1:5]
            #
            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if self.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * self.bbox_std + self.bbox_mean
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * self.bbox_std + self.bbox_mean
                        box_deltas = box_deltas.view(1, -1, 4 * self.num_classes)
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                # pred_boxes = np.tile(boxes, (1, scores.shape[1]))
                # print('boxes shape {}'.format(boxes.shape))
                # print('score shape {}'.format(scores.shape))
                pred_boxes = boxes.repeat(1, 1, scores.shape[2])
                # print('pred boxes shape {}'.format(pred_boxes.shape))
                # exit(0)

            pred_boxes /= im_info[0][2].item()

            scores = scores.squeeze(0)
            pred_boxes = pred_boxes.squeeze(0)
            #
            for j in range(1, self.num_classes):
                condition = scores[:, j] > self.thresh
                inds = torch.nonzero(condition, as_tuple=False).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if self.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    self.all_boxes[j][val_iter] = cls_dets.cpu().numpy()
                else:
                    self.all_boxes[j][val_iter] = self.empty_array
            # Limit to max_per_image detections *over all classes*
            if self.max_per_image > 0:
                new_list = []
                for j in range(1, self.num_classes):
                    temp1 = self.all_boxes[j][val_iter]
                    temp2 = temp1[:, -1]
                    new_list.append(temp2)
                image_scores = np.hstack([self.all_boxes[j][val_iter][:, -1]
                                          for j in range(1, self.num_classes)])
                if len(image_scores) > self.max_per_image:
                    image_thresh = np.sort(image_scores)[-self.max_per_image]
                    for j in range(1, self.num_classes):
                        keep = np.where(self.all_boxes[j][val_iter][:, -1] >= image_thresh)[0]
                        self.all_boxes[j][val_iter] = self.all_boxes[j][val_iter][keep, :]

            #
            if self.save_vis:
                #
                val_dir = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                                       'iter_{}_val_result'.format(runner.iteration))
                self.vis_save_path = os.path.join(val_dir, 'result_vis')
                if self.save_vis:
                    if not os.path.exists(self.vis_save_path):
                        os.makedirs(self.vis_save_path)
                #
                im_data = batch_output['im_data']
                orig_img = transform_back_to_img(im_data)
                orig_size_img = Image.open(batch_output['img_id'][0])
                x_ratio = im_data.shape[3] / float(orig_size_img.size[0])
                y_ratio = im_data.shape[2] / float(orig_size_img.size[1])
                draw = ImageDraw.Draw(orig_img)
                thick = 3  # 标注框的宽度
                tmp_name = os.path.basename(batch_output['img_id'][0])
                tmp_path = os.path.join(self.vis_save_path, tmp_name)
                class_name = self.imdb._classes
                color_palette = self.imdb.cls_palette
                for j in range(1, self.num_classes):
                    dets = self.all_boxes[j][val_iter]
                    tmp_dets_num = dets.shape[0]
                    if tmp_dets_num > 0:
                        for box_ind in range(tmp_dets_num):
                            x1, y1, x2, y2, tmp_prob = dets[box_ind, :].tolist()
                            x1, y1, x2, y2 = int(x1 * x_ratio), int(y1 * y_ratio), int(x2 * x_ratio), int(y2 * y_ratio)
                            if tmp_prob > 0.3:
                                for t1 in range(thick):
                                    draw.polygon(
                                        [(x1 + t1, y1 + t1), (x1 + t1, y2 - t1), (x2 - t1, y2 - t1),
                                         (x2 - t1, y1 + t1)],
                                        outline=color_palette[j-1])
                orig_img.save(tmp_path)

    def after_val_epoch(self, runner):
        # 创建保存检测结果的文件夹
        val_dir = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                               'iter_{}_val_result'.format(runner.iteration))
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        # 将all_boxes写入detections.pkl
        det_file = os.path.join(val_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)
        # 调用imdb的测试函数，计算各类别ap
        aps, recs = self.imdb.evaluate_detections(self.all_boxes, val_dir, logger=runner.logger)
        # 分析检测结果中的fp,输出每一种错误类型的数量
        fp_type_count = self.imdb.analyze_fp(val_dir, logger=runner.logger)
        for i, val in enumerate(fp_type_count):
            runner.writer.add_scalar('fp_count_{}/type_{}'.format(self.imdb._name, i + 1), val,
                                     global_step=runner.iteration)
        # 记录每一类的检测AP,以及总的mAP
        for ind, (class_name, ap) in enumerate(zip(self.imdb._classes[1:], aps)):
            runner.writer.add_scalar('ap_{}/{}_{}'.format(self.imdb._name, ind + 1, class_name), ap,
                                     global_step=runner.iteration)
        current_map = np.mean(aps)
        runner.writer.add_scalar('ap_{}/0_map'.format(self.imdb._name), current_map, global_step=runner.iteration)
        # 记录每一类的recall
        for ind, (class_name, rec) in enumerate(zip(self.imdb._classes[1:], recs)):
            tmp_rec = 0 if len(rec) == 0 else rec[-1]
            runner.writer.add_scalar('rcnn_recall_{}/{}_{}'.format(self.imdb._name, ind + 1, class_name), tmp_rec,
                                     global_step=runner.iteration)
        # 如果结果更好，设置保存flag=True
        if (current_map > self.best_map) and self.set_save_flag:
            self.best_map = current_map
            runner.save_flag = True
        runner.logger.info('Best mAP for now is {}'.format(self.best_map))
