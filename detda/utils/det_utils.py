import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from PIL import Image
import os
import numpy as np
from xml.dom import minidom
import pickle
import torch
from detda.models.det_models.utils.config import cfg
import cv2
import torch.nn.functional as F
from detda.models.det_models.rpn.bbox_transform import bbox_overlaps
from collections import OrderedDict
from torchvision.ops import nms


# from detda.loader.det_loaders import combined_roidb  #TODO: 这个import会报错 import no module named combined_roidb


def generate_pseudo_from_all_boxes(all_boxes, root_path, imdb, split='train'):
    """
    从all_boxes生成伪标签，
    :param all_boxes: 格式是 all_boxes[类别][图像index(固定的)][x1,x2,y1,y2,prob]
    :param root_path: 生成的伪标签的目录
    :return:
    """
    # 创建文件夹
    anno_path = os.path.join(root_path, 'Annotations')
    imageset_path = os.path.join(root_path, 'ImageSets', 'Main')
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)
    if not os.path.exists(imageset_path):
        os.makedirs(imageset_path)
    split_file = os.path.join(imageset_path, '{}.txt'.format(split))
    # 一次处理所有图像
    with open(split_file, 'wb') as f:
        for im_ind, index in enumerate(imdb.image_index):
            anno = Element('annotation')
            folder = SubElement(anno, 'folder')
            folder.text = 'VOC2007'
            filename = SubElement(anno, 'filename')
            filename.text = index + '.jpg'
            segmented = SubElement(anno, 'segmented')
            segmented.text = '0'
            temp_img_path = imdb.image_path_at(im_ind)
            temp_img = Image.open(temp_img_path)
            img_size = SubElement(anno, 'size')
            width = SubElement(img_size, 'width')
            width.text = str(temp_img.size[0])
            height = SubElement(img_size, 'height')
            height.text = str(temp_img.size[1])
            depth = SubElement(img_size, 'depth')
            depth.text = '3'
            for cls_ind, cls in enumerate(imdb.classes):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    temp_object = SubElement(anno, 'object')
                    temp_box = SubElement(temp_object, 'bndbox')
                    xmin = SubElement(temp_box, 'xmin')
                    tmp = int(np.round(dets[k, 0]))
                    tmp = 1 if tmp == 0 else tmp
                    xmin.text = str(tmp)
                    ymin = SubElement(temp_box, 'ymin')
                    tmp = int(np.round(dets[k, 1]))
                    tmp = 1 if tmp == 0 else tmp
                    ymin.text = str(tmp)
                    xmax = SubElement(temp_box, 'xmax')
                    tmp = int(np.round(dets[k, 2]))
                    tmp = 1 if tmp == 0 else tmp
                    xmax.text = str(tmp)
                    ymax = SubElement(temp_box, 'ymax')
                    tmp = int(np.round(dets[k, 3]))
                    tmp = 1 if tmp == 0 else tmp
                    ymax.text = str(tmp)
                    class_name = SubElement(temp_object, 'name')
                    class_name.text = cls
                    difficult = SubElement(temp_object, 'difficult')
                    difficult.text = '0'
                    truncated = SubElement(temp_object, 'truncated')
                    truncated.text = '0'
            tmp_anno_path = os.path.join(anno_path, '{}.xml'.format(index))
            xml_string = ET.tostring(anno)
            tree = minidom.parseString(xml_string)
            pretty_xml_as_string = tree.toprettyxml()
            with open(tmp_anno_path, 'wb') as new_f:
                new_f.write(pretty_xml_as_string.encode('utf-8'))
            #
            f.write((index + '\n').encode('utf-8'))


def box_filter(all_boxes, imdb, primary_low_thresh=0.1, primary_high_thresh=1.0, ratio_thresh=0.5, min_length=5):
    available_img_exist = np.zeros(imdb.num_images)
    available_box_num = np.zeros(len(imdb.classes))
    # calculate thresh for each classes
    cls_prob_thresh = []
    for cls_ind, cls in enumerate(imdb.classes):
        dets = all_boxes[cls_ind]
        all_probs = []
        for ind, tmp_dets in enumerate(dets):
            if tmp_dets == []:
                continue
            else:
                all_probs.extend(
                    [a for a in dets[ind][:, 4].tolist() if a > primary_low_thresh and a < primary_high_thresh])
        if len(all_probs) == 0:
            cls_prob_thresh.append(0)
        else:
            sorted_all_probs = np.sort(all_probs)
            cls_prob_thresh.append(sorted_all_probs[np.int(np.floor(len(all_probs) * ratio_thresh))])
    cls_prob_thresh = np.array(cls_prob_thresh)
    # filter all_prob
    new_all_prob = []
    for cls_ind, cls in enumerate(imdb.classes):
        temp_cls_pred = []
        dets = all_boxes[cls_ind]
        for img_ind, tmp_dets in enumerate(dets):
            if tmp_dets == []:
                temp_cls_pred.append([])
            else:
                orig_img_pred = all_boxes[cls_ind][img_ind]
                temp_img_pred = np.empty((0, 5))
                for box_ind in range(orig_img_pred.shape[0]):
                    tmp_box = orig_img_pred[box_ind, :]
                    tmp_prob = tmp_box[4]
                    tmp_length = min(tmp_box[2] - tmp_box[0], tmp_box[3] - tmp_box[1])
                    if tmp_prob > cls_prob_thresh[cls_ind] and tmp_length > min_length:
                        temp_img_pred = np.concatenate((temp_img_pred, orig_img_pred[[box_ind], :]), axis=0)
                if temp_img_pred.size == 0:
                    temp_cls_pred.append([])
                else:
                    temp_cls_pred.append(temp_img_pred)
                    available_img_exist[img_ind] = 1
                    available_box_num[cls_ind] += temp_img_pred.shape[0]
        new_all_prob.append(temp_cls_pred)

    return cls_prob_thresh, new_all_prob, np.sum(available_img_exist), available_box_num


def filter_by_ratio(all_boxes, max_ratio, min_ratio):
    class_num = len(all_boxes)
    # print('class num {}'.format(class_num))
    assert max_ratio.size == class_num and min_ratio.size == class_num, 'wrong shape'
    new_boxes = []
    filtered_num = 0
    remain_num = 0
    filtered_prob = []
    for cls_ind in range(class_num):
        dets = all_boxes[cls_ind]
        temp_cls_pred = []
        for img_ind, tmp_dets in enumerate(dets):
            if tmp_dets == []:
                temp_cls_pred.append([])
            else:
                orig_img_pred = all_boxes[cls_ind][img_ind]
                temp_img_pred = np.empty((0, 5))
                for box_ind in range(orig_img_pred.shape[0]):
                    tmp_ratio = (orig_img_pred[box_ind, 2] - orig_img_pred[box_ind, 0]) / (
                            orig_img_pred[box_ind, 3] - orig_img_pred[box_ind, 1] + 1e-8)
                    if tmp_ratio <= max_ratio[cls_ind] and tmp_ratio >= min_ratio[cls_ind]:
                        temp_img_pred = np.concatenate((temp_img_pred, orig_img_pred[[box_ind], :]), axis=0)
                        remain_num += 1
                    else:
                        filtered_num += 1
                        filtered_prob.append(orig_img_pred[[box_ind], 4])
                if temp_img_pred.size == 0:
                    temp_cls_pred.append([])
                else:
                    temp_cls_pred.append(temp_img_pred)
        new_boxes.append(temp_cls_pred)
    print('remain {}'.format(remain_num))
    print('remain {}'.format(np.max(np.array(filtered_prob))))
    return new_boxes, filtered_num


def gt_num_per_image(all_boxes):
    img_num = len(all_boxes[0])
    gt_num_list = [0 for ind in range(img_num)]
    for img_ind in range(img_num):
        for cls_ind, cls_boxes in enumerate(all_boxes):
            tmp_boxes = cls_boxes[img_ind]
            if tmp_boxes == []:
                continue
            else:
                gt_num_list[img_ind] += cls_boxes[img_ind].shape[0]
    return sorted(gt_num_list, reverse=True)


def cal_acc(output_dir, imdb):
    overall_sample_num = 0
    right_sample_num = 0
    class_acc = [0]
    for ind, class_name in enumerate(imdb.classes):
        if ind > 0:
            tmp_path = os.path.join(output_dir, '{}_pr.pkl'.format(class_name))
            with open(tmp_path, 'rb') as f:
                tmp_class_pr = pickle.load(f)
            prec = tmp_class_pr['prec']
            sample_num = prec.size
            if sample_num > 0:
                acc = prec[sample_num - 1]
            else:
                acc = 0
            class_acc.append(acc)
            overall_sample_num += sample_num
            right_sample_num += sample_num * acc
    class_acc[0] = right_sample_num / overall_sample_num
    return class_acc


def get_overlap_matrix(gt_bbox, res_bbox):
    overlap_matrix = np.zeros((gt_bbox.shape[0], res_bbox.shape[0]))
    gt_num = gt_bbox.shape[0]
    for i in range(gt_num):
        ixmin = np.maximum(res_bbox[:, 0], gt_bbox[i, 0])
        iymin = np.maximum(res_bbox[:, 1], gt_bbox[i, 1])
        ixmax = np.minimum(res_bbox[:, 2], gt_bbox[i, 2])
        iymax = np.minimum(res_bbox[:, 3], gt_bbox[i, 3])
        iw = np.maximum(ixmax - ixmin + 1.0, 0.)
        ih = np.maximum(iymax - iymin + 1.0, 0.)
        inters = iw * ih
        uni = ((gt_bbox[i, 2] - gt_bbox[i, 0] + 1.0) * (gt_bbox[i, 3] - gt_bbox[i, 1] + 1.0) +
               (res_bbox[:, 2] - res_bbox[:, 0] + 1.0) *
               (res_bbox[:, 3] - res_bbox[:, 1] + 1.0) - inters)
        overlap_matrix[i, :] = np.transpose(inters / uni)
    return overlap_matrix


def cluster_boxes_by_miou(overlap_matrix):
    """
    根据overlap矩阵计算不同的box簇（考虑第0维）, 并且返回每一个簇在overlap_matrix中的index
    :param overlap_matrix:
    :return:
    """
    box_num = overlap_matrix.shape[0]
    assigned_flag = np.ones((box_num,), dtype=np.long) * (-1)
    max_cluster_index = -1
    for box_ind in range(box_num):
        associated_index = np.where(overlap_matrix[box_ind, :] == 1)[0]
        #
        if assigned_flag[box_ind] == -1:
            max_cluster_index += 1
            assigned_flag[box_ind] = max_cluster_index
            assigned_flag[associated_index] = max_cluster_index
        else:
            current_cluster_ind = assigned_flag[box_ind]
            associated_cluster_ind = np.unique(assigned_flag[associated_index])
            associated_cluster_ind = np.setdiff1d(associated_cluster_ind, np.array([-1, current_cluster_ind]))
            if associated_cluster_ind.size > 0:
                # 最小的簇编号
                min_cluster_ind = np.min(associated_cluster_ind)
                # 处理其他的类别
                for ind in associated_cluster_ind.tolist():
                    assigned_flag[ind] = min_cluster_ind
                # 处理该样本相关样本
                assigned_flag[associated_index] = min_cluster_ind
            else:
                assigned_flag[associated_index] = current_cluster_ind

    boxes_cluster = []
    unique_num = np.unique(assigned_flag).tolist()
    for ind in unique_num:
        boxes_cluster.append(np.where(assigned_flag == ind)[0].tolist())
    return boxes_cluster


def get_boxes_clusters(all_boxes, cluster_iou_thresh):
    # 计算overlap矩阵
    overlap_matrix = get_overlap_matrix(all_boxes, all_boxes)
    # 根据thresh,确定是否overlap
    overlap_matrix[overlap_matrix >= cluster_iou_thresh] = 1
    overlap_matrix[overlap_matrix < cluster_iou_thresh] = 0
    # 将box聚类，获取簇的index
    cluster_index = cluster_boxes_by_miou(overlap_matrix)
    #  根据簇，重新组织all_boxes
    cluster_boxes = []
    for tmp_index in cluster_index:
        cluster_boxes.append(all_boxes[tmp_index, :])
    return cluster_boxes, cluster_index


def refine_box_prob(box_cluster, outside_weight, thresh):
    """
    融合不同box的概率，如果小于阈值，则设置keep_flag为0，否则为1
    :param box_cluster:
    :param outside_weight:
    :param thresh:
    :return:
    """
    prob = box_cluster[:, 4]
    mean_prob = np.mean(prob * outside_weight, keepdims=True)
    if mean_prob < thresh:
        keep_flag = 0
    else:
        keep_flag = 1
    return mean_prob, keep_flag


def refine_box_location(box_cluster, outside_weights):
    """
    根据outsize_weights，融合边的位置
    :param box_cluster:
    :param outside_weights:
    :return:
    """
    # 根据prob和outsize_weight计算新的weights
    new_weights = outside_weights * box_cluster[:, 4]
    new_weights = (new_weights / np.sum(new_weights))
    new_boxes = np.sum(box_cluster * np.expand_dims(new_weights, axis=1), axis=0, keepdims=True)
    return new_boxes


def boxes_ensemble(boxes_list, cluster_iou_thresh=0.5, weight_list=None, cls_thresh=None):
    # 创建新的all_boxes,一个
    num_detectors = len(boxes_list)
    num_classes = len(boxes_list[0])
    num_images = len(boxes_list[0][0])
    final_all_boxes = [[[] for _ in range(num_images)]
                       for _ in range(num_classes)]
    if weight_list is not None:
        assert len(boxes_list) == weight_list.shape[0], 'wrong match between box and weight'
    else:
        weight_list = np.array([1.0 / num_detectors for i in range(num_detectors)])
    if cls_thresh is not None:
        assert len(boxes_list) == cls_thresh.shape[0], 'wrong match between box and cls thresh'
    else:
        cls_thresh = np.zeros((num_classes,))
    for cls_ind in range(num_classes):
        print('class {}'.format(cls_ind))
        for img_ind in range(num_images):
            # 合并所有的boxes
            all_detector_res = np.empty((0, 5))
            res_weight = np.empty((0,))
            for detector_ind in range(num_detectors):
                tmp_res = boxes_list[detector_ind][cls_ind][img_ind]
                if tmp_res == []:
                    continue
                all_detector_res = np.concatenate((all_detector_res, tmp_res), axis=0)
                res_weight = np.concatenate((res_weight, np.repeat(np.array([weight_list[detector_ind]]),
                                                                   repeats=all_detector_res.shape[0])), axis=0)
            if all_detector_res.size == 0:
                continue
            # 根据overlap矩阵得到聚类的结果
            boxes_cluster, boxes_cluster_index = get_boxes_clusters(all_detector_res, cluster_iou_thresh)
            outside_weights = []
            for i, weight_ind in enumerate(boxes_cluster_index):
                outside_weights.append(res_weight[weight_ind])
            # 对概率进行调整
            final_prob = np.empty((0,))
            keep_flag = np.empty((0,))
            for cluster_ind in range(len(boxes_cluster)):
                tmp_prob, tmp_flag = refine_box_prob(boxes_cluster[cluster_ind],
                                                     outside_weight=outside_weights[cluster_ind],
                                                     thresh=cls_thresh[cls_ind])
                final_prob = np.concatenate((final_prob, tmp_prob), axis=0)
                keep_flag = np.concatenate((keep_flag, np.array([tmp_flag])), axis=0)
            final_box = np.empty((0, 5))
            # 对boxes边的位置进行调整
            for cluster_ind, flag in enumerate(keep_flag.tolist()):
                if flag > 0:
                    tmp_box = refine_box_location(boxes_cluster[cluster_ind], outside_weights[cluster_ind])
                    final_box = np.concatenate((final_box, tmp_box), axis=0)
            if final_box.size > 0:
                final_box[:, 4] = final_prob.squeeze()
                final_all_boxes[cls_ind][img_ind] = final_box
    return final_all_boxes


def cal_class_ratio(dataset_name, split):
    from detda.loader.det_loaders import combined_roidb
    dataset_param = {'split': split, }
    imdb, roidb, _, ratio_index = combined_roidb(dataset_name, training=False, dataset_params=dataset_param)
    class_num = len(imdb.classes)
    max_ratio = np.zeros((class_num,))
    min_ratio = np.ones((class_num,)) * 100
    for box_info in roidb:
        gt_class = box_info['gt_classes']
        box = box_info['boxes']
        box_ratio = (box[:, 2] - box[:, 0]) / (box[:, 3] - box[:, 1] + 1e-8)
        box_num = box.shape[0]
        for box_ind in range(box_num):
            tmp_ratio = box_ratio[box_ind]
            if tmp_ratio < 15 and tmp_ratio > 0.1:
                tmp_class = gt_class[box_ind]
                if tmp_ratio > max_ratio[tmp_class]:
                    max_ratio[tmp_class] = tmp_ratio
                if tmp_ratio < min_ratio[tmp_class]:
                    min_ratio[tmp_class] = tmp_ratio
            else:
                print(tmp_ratio)
                pass
    return max_ratio, min_ratio


def class_sample_extraction(dataloader, img_save_path, res_save_path, min_size=10):
    """
    对于给定的数据集，提取每一类别所有的框
    :param dataloader:
    :return:
    """
    ratio_index = []
    box_sample = []
    ratio_list = []
    num_classes = dataloader.dataset.imdb.num_classes
    print('num class {}'.format(num_classes))
    for i in range(num_classes):
        ratio_index.append([])
        box_sample.append([])
        ratio_list.append([])

    for img_ind, item in enumerate(dataloader):
        data, im_info, gt_boxes, num_boxes, img_id = item
        base_name = os.path.basename(img_id[0])[0:-4]
        gt_boxes = gt_boxes.numpy().squeeze(0)
        if gt_boxes.size > 0:
            num_boxes = gt_boxes.shape[0]
            for box_ind in range(num_boxes):
                tmp_box = gt_boxes[box_ind]
                cls_ind = int(tmp_box[4])
                if (tmp_box[3] - tmp_box[1]) >= min_size and (tmp_box[2] - tmp_box[0]) >= min_size:
                    tmp_array = data[:, :, int(tmp_box[1]):int(tmp_box[3]), int(tmp_box[0]):int(tmp_box[2])]
                    tmp_array = tmp_array.numpy().copy().squeeze(0).transpose(1, 2, 0)
                    tmp_array += cfg.PIXEL_MEANS
                    tmp_array = tmp_array[:, :, ::-1].astype(np.uint8)
                    tmp_img = Image.fromarray(tmp_array)
                    tmp_name = '{}_{}.jpg'.format(base_name, box_ind)
                    tmp_path = os.path.join(img_save_path, tmp_name)
                    tmp_img.save(tmp_path)
                    tmp_img.close()
                    box_sample[cls_ind].append(tmp_path)
                    ratio_list[cls_ind].append((tmp_box[2] - tmp_box[0]) / (tmp_box[3] - tmp_box[1]))
        if img_ind % 100 == 0 and img_ind > 0:
            print('{} processed'.format(img_ind))

    for cls_ind in range(num_classes):
        if len(box_sample[cls_ind]) > 0:
            #
            cls_save_path = os.path.join(img_save_path, 'cls_{}'.format(cls_ind))
            if not os.path.exists(cls_save_path):
                os.makedirs(cls_save_path)
            #
            ratio = np.array(ratio_list[cls_ind])
            sorted_ind = np.argsort(ratio)
            sorted_ratio = ratio[sorted_ind]
            old_box = box_sample[cls_ind]
            tmp_box = []
            for box_ind in range(len(old_box)):
                old_path = old_box[sorted_ind[box_ind]]
                new_path = os.path.join(cls_save_path, os.path.basename(old_path))
                shutil.move(old_path, new_path)
                tmp_box.append(new_path)
            box_sample[cls_ind] = tmp_box
            for ratio_point in range(100):
                tmp_ratio = 0.1 * ratio_point
                ratio_index[cls_ind].append(np.searchsorted(sorted_ratio, tmp_ratio))

    final_output = {'sample_path': box_sample,
                    'ratio_index': ratio_index
                    }
    with open(res_save_path, 'wb') as f:
        pickle.dump(final_output, f)


def paste_data(im_data, gt_boxes, sample_file, sample_range_percent=0.05):
    """
    根据gt_boxes,将每一个box的对应的im_data区域，替换成sample_file里的文件
    :param im_data:
    :param gt_boxes:
    :param sample_file:
    :return:
    """
    # TODO:处理 gt_box中补了0的情况
    im_number = im_data.shape[0]
    # print('gt shape {}'.format(gt_boxes.shape))
    # break
    for im_ind in range(im_number):
        im_gt = gt_boxes[im_ind, :, :]
        im_gt = get_paste_order(im_gt)
        box_num = im_gt.shape[0]
        for box_ind in range(box_num):
            tmp_gt = im_gt[box_ind, :]
            im_gt_size = (tmp_gt[3] - tmp_gt[1]) * (tmp_gt[2] - tmp_gt[0])
            if im_gt_size == 0:
                continue
            tmp_ratio = ((tmp_gt[2] - tmp_gt[0]) / (tmp_gt[3] - tmp_gt[1])).item()
            tmp_ratio = int(round(tmp_ratio * 10, 0))
            tmp_ratio = tmp_ratio if tmp_ratio <= 99 else 99
            tmp_cls = int(tmp_gt[4])
            sample = sample_file['sample_path'][tmp_cls]
            sample_ratio = sample_file['ratio_index'][tmp_cls]
            insert_ind = sample_ratio[tmp_ratio]
            sample_range_num = len(sample_ratio) * sample_range_percent
            sample_ind = int(insert_ind + (np.random.sample() - 0.5) * sample_range_num)
            sample_ind = sample_ind if sample_ind < len(sample) else len(sample) - 1
            sample_ind = sample_ind if sample_ind >= 0 else 0
            tmp_path = sample[sample_ind]
            # print('tmp path {}'.format(tmp_path))
            sample_img = Image.open(tmp_path)
            im = np.asarray(sample_img)
            if len(im.shape) == 2:
                im = im[:, :, np.newaxis]
                im = np.concatenate((im, im, im), axis=2)
            im = im[:, :, ::-1]
            if np.random.sample() >= 0.5:
                im = im[:, ::-1, :]
            im = im.astype(np.float32, copy=False)
            im -= cfg.PIXEL_MEANS
            x_scale = float(tmp_gt[2] - tmp_gt[0]) / im.shape[1]
            y_scale = float(tmp_gt[3] - tmp_gt[1]) / im.shape[0]
            im = cv2.resize(im, None, None, fx=x_scale, fy=y_scale,
                            interpolation=cv2.INTER_LINEAR)
            im = torch.from_numpy(im).to(im_data.device).transpose(1, 2).transpose(0, 1)
            # print('im shape {}, data shape {}'.format(im.shape, im_data.shape))
            # print('gt shape {},{}'.format(float(tmp_gt[3] - tmp_gt[1]), float(tmp_gt[2] - tmp_gt[0])))
            im_data[im_ind, :, int(tmp_gt[1]):int(tmp_gt[1]) + im.shape[1],
            int(tmp_gt[0]):int(tmp_gt[0]) + im.shape[2]] = im
    return im_data


def get_paste_order(gt_boxes, larger_box_percent=0.6):
    """
    根据gt_boxes对gt_boxes进行排序
    这一部分是可以离线计算好的
    :param gt_boxes:
    :return:
    """
    box_num = gt_boxes.shape[0]
    final_ind = torch.zeros(box_num).to(torch.long)
    # 按照面积取前面的box
    box_size = (gt_boxes[:, 3] - gt_boxes[:, 1]) * (gt_boxes[:, 2] - gt_boxes[:, 0])
    box_size_sum = torch.sum(box_size)
    ind_by_size = torch.argsort(box_size, descending=True)
    box_size_cumsum = torch.cumsum(box_size[ind_by_size], dim=0) / box_size_sum
    insert_ind = 0
    for i in range(box_num):
        if box_size_cumsum[i] > larger_box_percent:
            insert_ind = i
            break
    final_ind[0:insert_ind] = ind_by_size[0:insert_ind]
    orig_remain_ind = ind_by_size[insert_ind:]
    remain_box = gt_boxes[orig_remain_ind, :]
    # 剩下的按照中点的竖直方向从上到下排列
    height = remain_box[:, 1]
    new_remain_ind = torch.argsort(height)
    final_ind[insert_ind:] = orig_remain_ind[new_remain_ind]  # 转换成原来的index
    new_gt_boxes = gt_boxes[final_ind, :]
    return new_gt_boxes


def get_complete_sample_file(sample_root, sample_file_path):
    with open(sample_file_path, 'rb') as f:
        res = pickle.load(f)

    sample_path = res['sample_path']
    ratio_index = res['ratio_index']
    for i in range(len(sample_path)):
        tmp_ratio_index = ratio_index[i]
        if isinstance(tmp_ratio_index, dict):
            new_ratio_index = []
            ratio_keys = sorted(tmp_ratio_index.keys())
            for key in ratio_keys:
                new_ratio_index.append(tmp_ratio_index[key])
            res['ratio_index'][i] = new_ratio_index
        for j in range(len(sample_path[i])):
            sample_path[i][j] = os.path.join(sample_root, sample_path[i][j])
    return res


def transform_back_to_img(img_tensor):
    if isinstance(img_tensor, torch.Tensor):
        tmp_array = img_tensor.cpu().numpy()
    else:
        tmp_array = img_tensor
    tmp_array = tmp_array.copy().squeeze(0).transpose(1, 2, 0)
    tmp_array += cfg.PIXEL_MEANS
    tmp_array = tmp_array[:, :, ::-1].astype(np.uint8)
    tmp_img = Image.fromarray(tmp_array, mode='RGB')
    return tmp_img


# 翻转一致性
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


def assign_labels_for_rois(rois, gt_boxes, thresh_high, thresh_low):
    """"
    根据gt_boxes，赋予rois类别
    :param rois: tensor格式
    :param gt_boxes: tensor格式
    """
    rois_num = rois.shape[0]

    overlaps = bbox_overlaps(rois, gt_boxes[:, 0:4]).cpu().numpy()

    max_overlaps = np.max(overlaps, 1)
    gt_assignment = np.argmax(overlaps, 1)
    labels_to_gt = gt_boxes[:, 4].view(-1)[gt_assignment]

    fg_inds = np.where(max_overlaps >= thresh_high)[0]
    fg_num = len(fg_inds)

    bg_inds = np.where((thresh_low <= max_overlaps) & (max_overlaps < thresh_high))[0]

    keep_inds = np.concatenate((fg_inds, bg_inds), axis=0)
    final_labels = labels_to_gt[keep_inds]

    if fg_num < rois_num:
        final_labels[fg_num:] = 0

    return fg_inds, bg_inds, final_labels


def assign_labels_for_rpn_rois(rois, gt_boxes, fg_thresh, bg_thresh, mode='direct'):
    tmp_rpn_labels = (torch.ones(rois.shape[0]) * (-1)).to(rois.device)
    overlaps = bbox_overlaps(rois, gt_boxes[:, 0:4])
    #
    max_overlaps, argmax_overlaps = torch.max(overlaps, 1)
    gt_max_overlaps, _ = torch.max(overlaps, 0)
    if mode == 'direct':
        tmp_rpn_labels[max_overlaps >= fg_thresh] = 1
        tmp_rpn_labels[max_overlaps < bg_thresh] = 0
    elif mode == 'original':
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            tmp_rpn_labels[max_overlaps < bg_thresh] = 0

        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(1, -1).expand_as(overlaps)), 1)
        #
        if torch.sum(keep) > 0:
            tmp_rpn_labels[keep > 0] = 1
        # fg label: above threshold IOU
        tmp_rpn_labels[max_overlaps >= fg_thresh] = 1
        #
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            tmp_rpn_labels[max_overlaps < bg_thresh] = 0
    else:
        raise RuntimeError('wrong type of assign rpn label mode')
    return tmp_rpn_labels


def cal_class_mean_feat(train_feat, train_prob, class_num, ratio=0.5):
    max_prob = np.max(train_prob, axis=1)
    max_prob_ind = np.argmax(train_prob, axis=1)
    cls_mean_feat = np.zeros((class_num, train_feat.shape[1]), dtype=np.float32)
    # 对每一类聚类
    for cls_ind in range(class_num):
        # 使用网络输出的伪标签确认类别
        tmp_cls_ind = np.where(max_prob_ind == cls_ind)[0]
        cls_feat = train_feat[tmp_cls_ind]
        cls_max_prob = max_prob[tmp_cls_ind]
        # cls_prob = train_prob[tmp_cls_ind]
        sorted_cls_prob_ind = np.argsort(-cls_max_prob)
        filtered_ind = sorted_cls_prob_ind[0:int(sorted_cls_prob_ind.shape[0] * ratio)]
        filtered_cls_feat = cls_feat[filtered_ind]
        cls_mean_feat[cls_ind, :] = np.mean(filtered_cls_feat, axis=0)
    return np.mean(cls_mean_feat, axis=0, keepdims=True)


def extract_det_res(res_root, save_path=None, res_name='filtered_detections.pkl'):
    # 对于一个给定的实验结果的目录，提取所有测试阶段的测试结果，并按照大小排序
    # 返回一个ordereddict, key是迭代次数，val是检测结果
    res_dict = {}
    val_dir_list = os.listdir(res_root)
    for val_item in val_dir_list:
        tmp_path = os.path.join(res_root, val_item, res_name)
        with open(tmp_path, 'rb') as f:
            tmp_res = pickle.load(f)
        tmp_name = int(val_item.split('_')[1])
        res_dict[tmp_name] = tmp_res
    sorted_keys = sorted(res_dict.keys())
    new_res_dict = OrderedDict()
    for key in sorted_keys:
        new_res_dict[key] = res_dict[key]
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(new_res_dict, f)
    return new_res_dict


def boxes_ensemble_with_nms(boxes_list, cluster_iou_thresh=0.5, weight_list=None, cls_thresh=None):
    # 创建新的all_boxes,一个
    num_detectors = len(boxes_list)
    num_classes = len(boxes_list[0])
    num_images = len(boxes_list[0][0])
    final_all_boxes = [[[] for _ in range(num_images)]
                       for _ in range(num_classes)]
    if weight_list is not None:
        assert len(boxes_list) == weight_list.shape[0], 'wrong match between box and weight'
    else:
        weight_list = np.array([1.0 / num_detectors for i in range(num_detectors)])
    if cls_thresh is not None:
        assert len(boxes_list) == cls_thresh.shape[0], 'wrong match between box and cls thresh'
    else:
        cls_thresh = np.zeros((num_classes,))
    for cls_ind in range(num_classes):
        print('class {}'.format(cls_ind))
        for img_ind in range(num_images):
            # 合并所有的boxes
            all_detector_res = np.empty((0, 5))
            res_weight = np.empty((0,))
            for detector_ind in range(num_detectors):
                tmp_res = boxes_list[detector_ind][cls_ind][img_ind]
                if tmp_res == []:
                    continue
                all_detector_res = np.concatenate((all_detector_res, tmp_res), axis=0)
                res_weight = np.concatenate((res_weight, np.repeat(np.array([weight_list[detector_ind]]),
                                                                   repeats=all_detector_res.shape[0])), axis=0)
            if all_detector_res.size == 0:
                continue
            #
            cls_scores = torch.from_numpy(all_detector_res[:, 4])  #
            cls_boxes = torch.from_numpy(all_detector_res[:, 0:4])
            _, order = torch.sort(cls_scores, 0, True)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = all_detector_res[keep.view(-1).long().cpu().numpy(), :]
            #
            final_all_boxes[cls_ind][img_ind] = cls_dets

    return final_all_boxes
