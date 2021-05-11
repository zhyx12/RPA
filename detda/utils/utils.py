"""
Misc Utility functions
"""
import datetime
import functools
import logging
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.cluster import MiniBatchKMeans


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger('detda')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elasped = time.time() - t0
        name = func.__name__
        arg_list = []
        if args:
            arg_list.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['{}={}'.format(k, w) for k, w in sorted(kwargs.items())]
            arg_list.append(', '.join(pairs))
        arg_str = ', '.join(arg_list)
        print('runing {} with args {} use [{:0.4f}]s'.format(name, arg_str, elasped))
        return result

    return clocked


def find_lower_threshold(classwise_hist, proportion):
    """
    根据每一类的概率分布直方图，计算前proportion对应的threshold
    :param classwise_hist: 维度是c x 100,没有经过归一化
    :return:
    """
    # 归一化，使行和为1
    classwise_hist = classwise_hist.float()
    classwise_hist = classwise_hist / torch.sum(classwise_hist, dim=1, keepdim=True)
    # 计算累计的占比
    accumulate_proportion = torch.cumsum(classwise_hist, dim=1)
    # 计算最接近proportion的位置
    threshold_ind = torch.argmin(torch.abs(accumulate_proportion - proportion), dim=1).float()
    return threshold_ind / classwise_hist.shape[1]


def create_mask(pred, classwise_upper_threshold, cuda, label=None):
    """
    :param classwise_upper_threshold: 这个的维度是c
    :param pred: 维度是c x w x h
    :return:
    """
    batch_size = pred.shape[0]
    classwise_upper_threshold = classwise_upper_threshold.unsqueeze(1).unsqueeze(2).unsqueeze(0)
    classwise_upper_threshold = classwise_upper_threshold.expand(pred.shape)
    # print('pred shape is {}'.format(pred.shape))
    w, h = pred.shape[2:]
    max_val, max_index = torch.max(F.softmax(pred, dim=1), dim=1)  # 这里需要做softmax，才表示概率!!!!!!!!!!!!!!!!!!!!!!!
    assert torch.max(max_val).item() <= 1.0, 'pred prob should be lower than 1?'
    temp_x = torch.ones(w, h)
    temp_y = torch.zeros(w, h)
    # numpy版本，pytorch0.4.0没有meshgrid这个函数，但是下面的实现是有问题的
    # w_ind, h_ind, batch_ind = np.meshgrid(np.arange(h), np.arange(w), np.arange(pred.shape[0]))
    # batch_ind = torch.from_numpy(batch_ind.transpose(2, 0, 1)).long()
    # w_ind = torch.from_numpy(w_ind.transpose(2, 0, 1)).long()
    # h_ind = torch.from_numpy(h_ind.transpose(2, 0, 1)).long()
    batch_ind, w_ind, h_ind = torch.meshgrid([torch.arange(pred.shape[0]), torch.arange(w), torch.arange(h)])

    select_threshold = classwise_upper_threshold[batch_ind, max_index, w_ind, h_ind]
    if cuda:
        select_threshold = select_threshold.cuda()
        temp_x = temp_x.cuda()
        temp_y = temp_y.cuda()
    # 这里应该是取小于阈值的点反传
    mask = torch.where(max_val < select_threshold, temp_x, temp_y)
    if label is not None:
        # print('use label mask')
        ignore_mask = torch.where(label < 255, temp_x, temp_y)
    else:
        ignore_mask = torch.ones((batch_size, w, h))
        if cuda:
            ignore_mask = ignore_mask.cuda()

    new_mask = torch.where((mask + ignore_mask) > 1, temp_x, temp_y)
    return new_mask.detach()


def update_classwise_hist(hist, pred):
    max_val, max_index = torch.max(pred.cpu(), dim=0)
    for i in range(19):
        print('update {}'.format(i))
        temp_ind = (max_index == i).nonzero()
        if torch.numel(temp_ind) > 0:
            hist[i, :] += torch.histc(max_val[temp_ind[:, 0], temp_ind[:, 1]], bins=100, min=0,
                                      max=1)
    return hist


def calc_mean_std(feat, eps=1e-5, detach_mean_std=True):
    # eps is a small value added to the variance to avoid divide-by-zero.
    # print('detach flat {}'.format(detach_mean_std))
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # var will cause inf in amp mode, use std instead
    # feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_std = torch.std(feat.view(N, C, -1), dim=2).view(N, C, 1, 1) + eps
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    if detach_mean_std:
        # print('detach mean std back')
        return feat_mean.detach(), feat_std.detach()
    else:
        # print('with mean std back')
        return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def show_imgs_in_grid(dir_list, res_save_path, img_size=(100, 200), row_num=5, img_gap=10, img_name_list=None):
    if img_name_list == None:
        img_name_list = os.listdir(dir_list[0])
    img_num = len(img_name_list)
    column_num = len(dir_list)  # 最后一列表示图像的名字，方便查找
    # 根据行列数，以及图像大小，gap大小确定整个图像的大小
    total_height = img_size[0] * row_num + img_gap * (row_num - 1)
    total_width = img_size[1] * column_num + img_gap * (column_num - 1)
    total_img_array = np.zeros((total_height, total_width, 4), dtype=np.uint8)
    transparency_mask = np.ones((*img_size, 1), dtype=np.uint8) * 255
    # 用作
    temp_name_list = []
    for img_ind in range(img_num):
        if img_ind % row_num == 0:
            total_img_array = np.zeros((total_height, total_width, 4), dtype=np.uint8)
            temp_name_list = []
        start_height = (img_ind % row_num) * img_size[0] + img_gap * (img_ind % row_num)
        for img_path_ind in range(column_num):
            start_width = img_path_ind * img_size[1] + img_gap * img_path_ind
            temp_img_path = os.path.join(dir_list[img_path_ind], img_name_list[img_ind])
            if os.path.isfile(temp_img_path):
                temp_img_path = temp_img_path
            else:
                temp_img_path = temp_img_path[0:-10] + '.jpg'
                assert os.path.isfile(temp_img_path), 'wrong img path {}'.format(temp_img_path)
            # print('start height {}, start width {}'.format(start_height, start_width))
            temp_img = Image.open(temp_img_path)
            temp_resized_img = temp_img.resize((img_size[1], img_size[0]), Image.ANTIALIAS)
            temp_img_array = np.asarray(temp_resized_img)
            if temp_img_array.ndim == 3:
                if temp_img_array.shape[2] == 3:
                    temp_img_array = np.concatenate((temp_img_array, transparency_mask), axis=2)
            elif temp_img_array.ndim == 2:
                temp_img = Image.open(temp_img_path)
                rgbimg = Image.new('RGBA', temp_img.size)
                rgbimg.paste(temp_img)
                temp_resized_img = rgbimg.resize((img_size[1], img_size[0]), Image.ANTIALIAS)
                temp_img_array = np.asarray(temp_resized_img)

            total_img_array[start_height:start_height + img_size[0],
            start_width:start_width + img_size[1]] = temp_img_array
        temp_name_list.append(img_name_list[img_ind])
        if (img_ind + 1) % row_num == 0:
            total_img = Image.fromarray(total_img_array, mode='RGBA')

            temp_save_path = os.path.join(res_save_path, 'num_{}_res.PNG'.format(img_ind // row_num))
            total_img.save(temp_save_path)
            print('Here is {}'.format(img_ind // row_num))
            for i in range(row_num):
                print(temp_name_list[i])

            #
            # new_img = Image.open(temp_save_path)
            # text_overlay = Image.new("RGBA",new_img.size,(255,255,255,0))
            # image_draw = ImageDraw.Draw(text_overlay)
            #
            # assert len(temp_name_list) == row_num, 'wrong match of img name list and row number'
            # start_width = column_num * img_size[1] + column_num * img_gap
            # for i in range(row_num):
            #     start_height = row_num * img_size[0] + row_num * img_gap
            #     print(temp_name_list[i])
            #     image_draw.text((start_width,start_height), temp_name_list[i], fill=(76, 234, 124, 180))
            # image_with_text = Image.alpha_composite(new_img,text_overlay)
            # image_with_text.save(temp_save_path)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            #
            # for ind in range(row_num):
            #     start_height = row_num * img_size[0] + row_num * img_gap
            #     new_img = cv2.putText(img=total_img_array, text=temp_name_list[ind], org=(start_width, start_height),
            #                           fontFace=font, fontScale=5, color=(0, 255, 0))
            # temp_save_path = os.path.join(res_save_path, 'num_{}_res.PNG'.format(img_ind // row_num))
            # cv2.imwrite(temp_save_path, total_img_array)


def cal_mean_domain_output(domain_output, gt, pred, choose_type='pred'):
    """
    根据gt 和 pred，计算每一类别的domain output的包含的像素数量和概率之和，用于提供给avgmeter计算均值

    :param domain_output: h x w
    :param gt: h x w
    :param pred: h x w
    :prarm choose_type: gt pred intersect
    :return: 类别的均值，和是否有该类别
    """
    assert domain_output.shape == gt.shape and gt.shape == pred.shape, 'wrong shape'
    temp_domain_output_cls_sum = np.zeros(19)
    temp_exist_cls = np.zeros(19)
    # 先reshape成一列
    domain_output = domain_output.reshape(-1)
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    for cls_ind in range(19):
        gt_cls_index = np.where(gt == cls_ind)[0]
        pred_cls_index = np.where(pred == cls_ind)[0]
        if choose_type == 'gt':
            cls_index = gt_cls_index
        elif choose_type == 'pred':
            cls_index = pred_cls_index
        else:
            cls_index = np.intersect1d(gt_cls_index, pred_cls_index)
        if cls_index.size > 0:
            temp_domain_output_cls_sum[cls_ind] = np.sum(domain_output[cls_index])
            temp_exist_cls[cls_ind] = cls_index.size
    return temp_domain_output_cls_sum, temp_exist_cls


def cal_pixel_acc_by_domain_prob(domain_output, gt, pred, seg_score=None):
    """
    计算每一个domain output概率区间（宽度是0.1）的总像素数量，正确分类像素数量
    :param domain_output:
    :param gt:
    :param pred:
    :param choose_type:
    :return:
    """
    assert domain_output.shape == gt.shape and gt.shape == pred.shape, 'wrong shape'
    pixel_all = np.zeros(10)
    pixel_right = np.zeros(10)
    domain_output = domain_output.reshape(-1)
    gt = gt.reshape(-1)
    pred = pred.reshape(-1)
    # 获取domain output的区间index
    domain_output = np.floor(domain_output * 10)
    temp_size = 0
    for i in range(10):
        section_index = np.where(domain_output == i)[0]
        # print('{} mean {}'.format(i, np.mean(domain_output[section_index])))
        pixel_all[i] = section_index.size
        temp_size += pixel_all[i]
        temp_gt = gt[section_index]
        temp_pred = pred[section_index]
        pixel_right[i] = np.where(temp_gt == temp_pred)[0].size
        if seg_score is not None:
            temp_gt = gt.copy()
            ignore_index = np.where(domain_output != i)[0]
            temp_gt[ignore_index] = 255
            temp_gt = temp_gt.reshape(1, 1024, 2048)
            temp_pred = pred.reshape(1, 1024, 2048)
            seg_score[i].update((temp_gt, temp_pred))
        # print('right {}, all {}'.format(pixel_right[i], pixel_all[i]))
        # assert pixel_all[i]>pixel_right[i],'wrong num'
    # print('domain size {}, sum {}'.format(domain_output.size, temp_size))
    # print('right {}, all {}'.format(pixel_all, pixel_right))
    # print('cal done')
    return pixel_right, pixel_all


def cal_feat_distance(feat_1, feat_2, metric_type='cos_similarity', alpha=1.0):
    """
    计算特征之间的相似度，值越大，相似度越高
    feat_1和feat_2位于不同的设备时，不会报错，但是计算结果会有问题
    :param feat_1: MxD
    :param feat_2: NxD
    :return: MxN
    """
    assert metric_type in ['student_t', 'inner_product', 'cos_similarity'], "wrong metric type {}".format(
        metric_type)
    # print('metric type {}'.format(metric_type))
    if metric_type == 'student_t':
        # score表示 NxM的距离矩阵
        score = (1 + (feat_1.unsqueeze(1) - feat_2.unsqueeze(0)).pow(2).sum(2) / alpha).pow(-(alpha + 1) / 2.0)
    elif metric_type == 'inner_product':
        score = feat_1.mm(feat_2.t())
    else:
        feat_1_norm = torch.sqrt(torch.sum(feat_1 * feat_1, dim=1, keepdim=True) + 1e-8)
        feat_2_norm = torch.sqrt(torch.sum(feat_2 * feat_2, dim=1) + 1e-8).unsqueeze(1).t()
        score = feat_1.mm(feat_2.t()) / ((feat_1_norm.mm(feat_2_norm)) + 1e-8)
    return score


def find_corr_center(cluster_cls, score_matrix, class_num):
    """
    根据簇的类别，和score矩阵，为每一个query特征找到对应每一类的聚类中心
    :param cluster_cls:
    :param score_matrix:
    :param class_num: 类别数量
    :return:
    """
    new_score_matrix = torch.zeros((score_matrix.shape[0], class_num)).to(score_matrix.device)
    new_score_ind = torch.zeros((score_matrix.shape[0], class_num), dtype=torch.int64).to(score_matrix.device)
    for i in range(class_num):
        tmp_cluster_ind = torch.nonzero(cluster_cls == i, as_tuple=True)[0]
        if tmp_cluster_ind.numel() == 0:
            continue
        tmp_score = score_matrix[:, tmp_cluster_ind]
        # argmax选取最近的类别
        tmp_closest_ind = np.argmax(tmp_score.detach().cpu().numpy(), axis=1)[:, np.newaxis]  # 需要求导的不能随便转numpy
        tmp_closest_ind = torch.from_numpy(tmp_closest_ind).to(tmp_score.device)
        # print('tmp score shape {}'.format(tmp_score.shape))
        # print('tmp min ind shape {}'.format(tmp_closest_ind.shape))
        new_score_matrix[:, i] = torch.gather(tmp_score, dim=1, index=tmp_closest_ind).t()
        # print('tmp closest ind {}'.format(tmp_closest_ind.shape))
        # print('new_score ind shape {}'.format(new_score_ind.shape))
        new_score_ind[:, i] = tmp_cluster_ind[tmp_closest_ind].squeeze()
    return new_score_matrix, new_score_ind


def cluster_within_per_class(train_feat, train_prob, class_num, sample_ratio, num_per_class, batch_size=10000,
                             random_seed=None, logger=None, init_centers=None):
    if init_centers is not None:
        assert init_centers.shape[0] == num_per_class * class_num, 'wrong shape of init centers {}, {} * {}'.format(
            init_centers.shape[0], num_per_class, class_num)
    # 按类别统计各类的样本ind, 然后针对各类做聚类，获取初始化的类别中心
    max_prob = np.max(train_prob, axis=1)
    max_prob_ind = np.argmax(train_prob, axis=1)
    # 初始化聚类结果
    init_cluster_feat = np.zeros((0, train_feat.shape[1]))
    init_cluster_cls = np.zeros((0), dtype=np.int64)
    init_cluster_prob = np.zeros((0, class_num))
    # 对每一类聚类
    for cls_ind in range(class_num):
        # 使用网络输出的伪标签确认类别
        tmp_cls_ind = np.where(max_prob_ind == cls_ind)[0]
        if logger is not None:
            logger.info('cls {} has {} samples'.format(cls_ind, tmp_cls_ind.shape[0]))
        print('cls {} has {} samples'.format(cls_ind, tmp_cls_ind.shape[0]))
        #
        cls_feat = train_feat[tmp_cls_ind]
        cls_max_prob = max_prob[tmp_cls_ind]
        cls_prob = train_prob[tmp_cls_ind]
        sorted_cls_prob_ind = np.argsort(-cls_max_prob)
        filtered_ind = sorted_cls_prob_ind[0:int(sorted_cls_prob_ind.shape[0] * sample_ratio) + 1]
        filtered_cls_feat = cls_feat[filtered_ind]
        filtered_cls_prob = cls_prob[filtered_ind]
        # 初始化聚类中心
        if init_centers is None:
            tmp_init_centers = 'k-means++'
        else:
            tmp_init_centers = init_centers[cls_ind * num_per_class:(cls_ind + 1) * num_per_class, :]
        #
        tmp_batch_kmeans = MiniBatchKMeans(n_clusters=num_per_class, random_state=random_seed,
                                           batch_size=batch_size, init=tmp_init_centers)
        iter_num = math.ceil(filtered_cls_feat.shape[0] / batch_size)
        for i in range(iter_num):
            tmp_batch_kmeans = tmp_batch_kmeans.partial_fit(filtered_cls_feat[batch_size * i:batch_size * (i + 1), :])
        # 添加簇的类别
        tmp_cls = np.array([cls_ind] * num_per_class)
        init_cluster_cls = np.concatenate((init_cluster_cls, tmp_cls), axis=0)
        # 添加簇的特征向量
        init_cluster_feat = np.concatenate((init_cluster_feat, tmp_batch_kmeans.cluster_centers_), axis=0)
        # 添加簇的概率表示
        for i in range(num_per_class):
            tmp_cluster_label = tmp_batch_kmeans.labels_
            tmp_cluster_ind = np.where(tmp_cluster_label == i)[0]
            tmp_prob = filtered_cls_prob[tmp_cluster_ind]
            tmp_mean_prob = np.mean(tmp_prob, axis=0, keepdims=True)
            init_cluster_prob = np.concatenate((init_cluster_prob, tmp_mean_prob), axis=0)
    return init_cluster_cls, init_cluster_feat, init_cluster_prob


def get_acc_by_prob(test_prob, test_gt):
    test_pred = np.argmax(test_prob, axis=1)
    test_acc = np.sum(test_pred == test_gt) / test_gt.size
    confusion_mat = confusion_matrix(test_gt, test_pred)
    class_acc = np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)
    return test_acc, class_acc
