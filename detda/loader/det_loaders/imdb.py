# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
from detda.models.det_models.rpn.bbox_transform import bbox_overlaps
import numpy as np
import scipy.sparse
from detda.models.det_models.utils.config import cfg
import pickle
from .voc_eval import voc_eval
import xml.etree.ElementTree as ET
import seaborn as sns

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')


def get_my_palette(cls_number):
    current_palette = sns.color_palette()
    my_palette = []
    if cls_number == 8:
        for i in range(8):
            my_palette.append(current_palette[i])
    elif cls_number == 5:
        tmp_table = [4, 5, 2, 7, 6]
        for ind in tmp_table:
            my_palette.append(current_palette[ind])
    elif cls_number == 1:
        my_palette.append(current_palette[2])
    elif cls_number == 2:
        for i in range(2):
            my_palette.append(current_palette[i])
    else:
        raise RuntimeError('wrong cls number {}'.format(cls_number))
    #
    for i in range(cls_number):
        color_int = []
        for j in range(3):
            color_int.append(int(my_palette[i][j]*255))
        my_palette[i] = tuple(color_int)
    return my_palette


class imdb(object):
    """Image database."""

    def __init__(self, name, image_set, classes=None, devkit_path=None, data_path=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._image_set = image_set
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._devkit_path = devkit_path
        self._data_path = data_path
        self._image_index = self._load_image_set_index()
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.gt_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}
        #
        self._image_ext = '.jpg'
        self._year = '2007'
        self.cls_palette = get_my_palette(len(self._classes) - 1)

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(self._data_path, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        """
                Return the absolute path to image i in the image sequence.
                """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
                Return the absolute path to image i in the image sequence.
                """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR)

    def default_roidb(self):
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]

    def append_flipped_images(self):
        num_images = self.num_images
        # widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            # assert self.roidb[i]['image_shape'][0] == widths[i],'wrong shape'
            tmp_width = self.roidb[i]['width']
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = tmp_width - oldx2 - 1
            boxes[:, 2] = tmp_width - oldx1 - 1
            ##############################################
            if boxes.size is not 0:
                for j in range(boxes.shape[0]):
                    if boxes[j, 0] == 65535:
                        boxes[j, 0] = 0
            ##############################################
            try:
                assert (boxes[:, 2] >= boxes[:, 0]).all()
            except:
                # print('error')
                # print(boxes[:, 2] >= boxes[:, 0])
                # print(boxes)
                # print(widths[i])
                # print('old x1 {}, old x2 {}'.format(oldx1, oldx2))
                print('index {}, path {}'.format(i, self.image_path_at(i)))
                # print('box {}'.format(boxes))
                # print('width {}'.format(widths[i]))

            if 'seg_map' in self.roidb[i].keys():
                seg_map = self.roidb[i]['seg_map'][::-1, :]
                entry = {'boxes': boxes,
                         'gt_overlaps': self.roidb[i]['gt_overlaps'],
                         'gt_classes': self.roidb[i]['gt_classes'],
                         'flipped': True,
                         'seg_map': seg_map,
                         'height': self.roidb[i]['height'],
                         'width': self.roidb[i]['width'],
                         'img_id': self.roidb[i]['img_id'] + len(self._image_index),
                         'image': self.roidb[i]['image'],
                         }
            else:
                entry = {'boxes': boxes,
                         'gt_overlaps': self.roidb[i]['gt_overlaps'],
                         'gt_classes': self.roidb[i]['gt_classes'],
                         'flipped': True,
                         'height': self.roidb[i]['height'],
                         'width': self.roidb[i]['width'],
                         'img_id': self.roidb[i]['img_id'] + len(self._image_index),
                         'image': self.roidb[i]['image'],
                         }
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.

        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                       [0 ** 2, 32 ** 2],  # small
                       [32 ** 2, 96 ** 2],  # medium
                       [96 ** 2, 1e5 ** 2],  # large
                       [96 ** 2, 128 ** 2],  # 96-128
                       [128 ** 2, 256 ** 2],  # 128-256
                       [256 ** 2, 512 ** 2],  # 256-512
                       [512 ** 2, 1e5 ** 2],  # 512-inf
                       ]
        assert area in areas, 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert (gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert (_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # print(cache_file)
        if os.path.exists(cache_file):
            if os.path.getsize(cache_file) > 0:
                with open(cache_file, 'rb') as fid:
                    roidb = pickle.load(fid)
                print('{} gt roidb loaded from {}'.format(self.name, cache_file))
                return roidb
        gt_roidb = []
        for ind, index in enumerate(self._image_index):
            gt_roidb.append(self._load_pascal_annotation(index, ind))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pascal_annotation(self, index, ind):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        imagename = os.path.join(self._data_path, 'JPEGImages', index + self._image_ext)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)
        tree = ET.parse(filename)
        img_size = tree.find('size')  # [0]
        width = int(img_size.find('width').text)
        height = int(img_size.find('height').text)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # ##############################################################
            # if bbox == None:
            #     continue
            if bbox.find('xmin') is None:
                print(filename)
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            assert x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0, 'wrong box {}, {}, {}, {}'.format(x1, y1, x2, y2)
            # assert x1 < x2 and y1<y2,'wrong small {},{},{},{},{}'.format(index,x1,x2,y1,y2)

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult
            # 针对某些图像中可能没有包含任何关注类别
            try:
                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            except:
                continue
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        # img = PIL.Image.open(imagename)
        # width, height = img.size
        # img.close()
        current_ind = ind
        # print('ind is {}'.format(current_ind))
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                # 'seg_map':seg_map
                'img_id': self.image_id_at(current_ind),
                'image': self.image_path_at(current_ind),
                'width': width,
                'height': height,
                }

    def _get_voc_results_file_template(self, output_dir):
        filename = 'det_' + self._image_set + '_{:s}.txt'
        filedir = output_dir
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, output_dir=None):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output', logger=None):
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, '{}_annotations_cache'.format(self._image_set))
        aps = []
        recs = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        if logger is not None:
            logger.info('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        else:
            print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(output_dir)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            recs += [rec]
            aps += [ap]
            if logger is not None:
                logger.info('AP for {} = {:.4f}'.format(cls, ap))
            else:
                print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, 'eval_result.txt'), 'a') as result_f:
                result_f.write('AP for {} = {:.4f}'.format(cls, ap) + '\n')
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        with open(os.path.join(output_dir, 'eval_result.txt'), 'a') as result_f:
            result_f.write('Mean AP = {:.4f}'.format(np.mean(aps)) + '\n')
        if logger is not None:
            logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
            logger.info('--------------------------------------------------------------')
        else:
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('--------------------------------------------------------------')
        return aps, recs

    def evaluate_detections(self, all_boxes, output_dir, logger=None):
        """
        评价检测结果，
        :param all_boxes: 两层list,第一层表示类别，第二层表示图像（按测试集顺序），最里面是ndarray,nx5,n大于等于0
        :param output_dir: 用来暂存中间输出的文件夹，会产生每一类的检测结果，pr曲线
        :param logger: logger，输出到日志文件
        :return:
        """
        self._write_voc_results_file(all_boxes, output_dir)
        aps = self._do_python_eval(output_dir, logger=logger)
        return aps

    def analyze_fp(self, output_dir, thresh_low=0.3, thresh_high=0.5, logger=None):
        # 统计每一类FP的数量
        fp_type_count = [0, 0, 0, 0, 0]
        # 处理gt,按照图像分开
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        cachedir = os.path.join(self._data_path, '{}_annotations_cache'.format(self._image_set))
        cachefile = os.path.join(cachedir, 'val_annots.pkl')
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')
        class_recs = {}
        for imagename in imagenames:
            R = [obj for obj in recs[imagename]]
            # 为了解决cache file中会保存不相关类别的问题，采用下面的code
            # TODO: 应该统一一下cache file的创建，从voc_eval中放到imdb里面来
            class_ind_list = []
            bbox_list = []
            for x in R:
                if x['name'] in self._classes:
                    class_ind_list.append(self._classes.index(x['name']))
                    bbox_list.append(x['bbox'])
            class_ind = np.array(class_ind_list)[:, np.newaxis]
            # class_ind = np.array([self._classes.index(x['name']) for x in R])[:, np.newaxis]
            bbox = np.array(bbox_list)
            if bbox.size > 0:
                if bbox.ndim == 1:
                    bbox = bbox[np.newaxis, :]
                bbox = np.concatenate((bbox, class_ind), axis=1)
            else:
                bbox = np.zeros((1, 5))
            class_recs[imagename] = {'bbox': bbox}
        # 依次处理每一类中的每一个FP bbox
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # 获取fp文件名
            filename = self._get_voc_results_file_template(output_dir)
            fp_file = filename.format('FP_{}'.format(cls))
            # 输出fp文件名
            new_fp_file = filename.format("FP_NEW_{}".format(cls))
            new_fp = open(new_fp_file, 'wb')
            # 处理每一个bbox
            with open(fp_file, 'rb') as f:
                lines = f.readlines()
            splitlines = [x.decode().strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            nd = len(image_ids)
            class_fp_box = np.array([[float(z) for z in x[2:]] for x in splitlines])
            if class_fp_box.size == 0:  # 某一类可能没有fp
                continue
            class_fp_box = np.concatenate((class_fp_box, np.ones((nd, 1))), axis=1)  # 最后一列添加类别标志
            for item_ind in range(nd):
                gt_boxes = class_recs[image_ids[item_ind]]['bbox'].astype(float)
                fp_type, corrsp_gt = self.compare_gt_fp(class_fp_box[item_ind, :], gt_boxes, thresh_low=thresh_low,
                                                        thresh_high=thresh_high)
                fp_type_count[fp_type - 1] += 1
                new_str = " ".join(splitlines[item_ind])
                if corrsp_gt is not None:
                    new_str += ' {} {} {} {} {}\n'.format(fp_type, corrsp_gt[0], corrsp_gt[1], corrsp_gt[2],
                                                          corrsp_gt[3])
                else:
                    new_str += ' {} {}\n'.format(fp_type, 'None')
                new_fp.write(new_str.encode())
            new_fp.close()
        #
        if logger is not None:
            logger.info('FP type count are as follows'.format(fp_type_count[0]) + '-' * 40)
            logger.info('background {}'.format(fp_type_count[0]))
            logger.info('only mis localization {}'.format(fp_type_count[1]))
            logger.info('mis localization + mis classification {}'.format(fp_type_count[2]))
            logger.info('duplicated {}'.format(fp_type_count[3]))
            logger.info('mis classification {}'.format(fp_type_count[4]))
        return fp_type_count

    def compare_gt_fp(self, fp_box, gt_box, thresh_low, thresh_high):
        """
        :param gt_box: gt，
        :param fp_box:
        :return:
        """
        tmp_fp_box = fp_box
        fp_class = fp_box[4]
        gt_class = gt_box[:, 4]
        overlaps = self.cal_iou(tmp_fp_box, gt_box)
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax <= thresh_low:
            fp_type = 1
            corresponding_gt = None
        elif ovmax >= thresh_high:
            if fp_class == gt_class[jmax]:  # TODO: 测试格式
                fp_type = 4
            else:
                fp_type = 5
            corresponding_gt = gt_box[jmax, :]
        else:
            if fp_class == gt_class[jmax]:  # TODO:测试格式
                fp_type = 2
            else:
                fp_type = 3
            corresponding_gt = gt_box[jmax, :]
        return fp_type, corresponding_gt

    def cal_iou(self, bb, BBGT):
        assert bb.size > 0 and BBGT.size > 0, 'box should not be none for iou'
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        # ovmax = np.max(overlaps)
        # jmax = np.argmax(overlaps)
        return overlaps

    def create_annotation_file(self):
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        gt_annotations_path = os.path.join(self._data_path,
                                           '{}_annotations_cache'.format(self._image_set), 'val_annots.pkl')
        assert os.path.exists(gt_annotations_path), 'gt annotation path {} not exists'.format(
            gt_annotations_path)
        with open(gt_annotations_path, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')
        gt_label = []
        for imagename in imagenames:
            # print('name {}'.format(imagename))
            tmp_gt_box = np.empty((0, 5), dtype=np.float32)
            for cls_ind, classname in enumerate(self._classes):
                R = [obj for obj in recs[imagename] if obj['name'] == classname]
                tmp_cls = np.ones((len(R), 1), dtype=np.float32) * cls_ind
                bbox = np.array([x['bbox'] for x in R], dtype=np.float32).reshape(-1, 4)
                bbox = np.concatenate((bbox, tmp_cls), axis=1)
                tmp_gt_box = np.concatenate((tmp_gt_box, bbox), axis=0)
            gt_label.append(tmp_gt_box)
        return gt_label
