import torch
from torch.nn import functional as F
from math import ceil
import numpy as np


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def to_onehot(label, num_classes):
    identity = torch.eye(num_classes)
    onehot = torch.index_select(identity, 0, label)
    return onehot


class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA + 1e-8, dim=1)
        pointB = F.normalize(pointB + 1e-8, dim=1)
        a = torch.min(torch.sum(torch.abs(pointA), dim=1))
        b = torch.min(torch.sum(torch.abs(pointB), dim=1))
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    def __init__(self, eps, num_classes, num_per_class, max_len=5000, dist_type='cos', sample_ratio=0.5, logger=None):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.max_len = max_len
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.centers = [[] * self.num_classes]
        self.sample_ratio = sample_ratio
        self.logger = logger

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def set_random_init_centers(self, features):
        select_ind = np.random.choice(features.shape[0], size=self.num_per_class, replace=False)
        return features[torch.from_numpy(select_ind), :].clone()

    def clustering_stop(self, centers, last_centers):
        if centers is None:
            return False
        dist = self.Dist.get_dist(centers, last_centers)
        dist = torch.mean(dist, dim=0)
        print('dist %.4f' % dist.item())
        return dist.item() < self.eps

    def assign_labels(self, feats, centers):
        dists = self.Dist.get_dist(feats, centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def feature_clustering(self, feature, init_centers=None):
        refs = torch.LongTensor(range(self.num_per_class)).unsqueeze(1)
        num_samples = feature.shape[0]
        num_split = ceil(1.0 * num_samples / self.max_len)
        #
        if init_centers is None:
            init_centers = self.set_random_init_centers(feature)
            print('center None')
        last_centers = init_centers
        centers = None

        if num_split > 0:
            while True:
                stop = self.clustering_stop(centers, last_centers)
                if stop:
                    start = 0
                    all_labels = torch.zeros(feature.shape[0])
                    for N in range(num_split):
                        cur_len = min(self.max_len, num_samples - start)
                        cur_feature = feature.narrow(0, start, cur_len)
                        dist2center, labels = self.assign_labels(cur_feature, last_centers)
                        all_labels[start:cur_len] = labels
                    # dist2center, labels = self.assign_labels(feature, centers)
                    return F.normalize(centers, dim=1), all_labels
                if centers is not None:
                    last_centers = centers

                centers = 0
                count = 0
                start = 0

                for N in range(num_split):
                    cur_len = min(self.max_len, num_samples - start)
                    cur_feature = feature.narrow(0, start, cur_len)
                    dist2center, labels = self.assign_labels(cur_feature, last_centers)
                    labels_onehot = to_onehot(labels, self.num_per_class)
                    count += torch.sum(labels_onehot, dim=0)
                    labels = labels.unsqueeze(0)
                    mask = (labels == refs).unsqueeze(2).type(torch.float)
                    reshaped_feature = cur_feature.unsqueeze(0)
                    # update centers
                    centers += torch.sum(reshaped_feature * mask, dim=1)
                    start += cur_len

                mask = (count.unsqueeze(1) > 0).type(torch.float)
                centers = mask * centers + (1 - mask) * last_centers
        else:
            return last_centers, torch.zeros(feature.shape[0])

    def cluster_all_features(self, train_feat, train_prob, init_centers=None):
        max_prob = np.max(train_prob, axis=1)
        max_prob_ind = np.argmax(train_prob, axis=1)
        #
        # 初始化聚类结果
        init_cluster_feat = np.zeros((0, train_feat.shape[1]), dtype=np.float32)
        init_cluster_cls = np.zeros((0), dtype=np.int64)
        init_cluster_prob = np.zeros((0, self.num_classes), dtype=np.float32)
        #
        if init_centers is not None:
            assert init_centers.shape[0] == self.num_per_class * self.num_classes, \
                'wrong shape of init center {}, {}, {}'.format(init_centers.shape[0], self.num_per_class,
                                                               self.num_classes)
            if isinstance(init_centers, np.ndarray):
                init_centers = torch.from_numpy(init_centers)
        for cls_ind in range(self.num_classes):
            # 使用网络输出的伪标签确认类别
            tmp_cls_ind = np.where(max_prob_ind == cls_ind)[0]
            cls_feat = train_feat[tmp_cls_ind]
            cls_max_prob = max_prob[tmp_cls_ind]
            cls_prob = train_prob[tmp_cls_ind]
            sorted_cls_prob_ind = np.argsort(-cls_max_prob)
            filtered_ind = sorted_cls_prob_ind[0:int(sorted_cls_prob_ind.shape[0] * self.sample_ratio) + 1]
            filtered_cls_feat = cls_feat[filtered_ind]
            filtered_cls_prob = cls_prob[filtered_ind]
            print('cls {} has {} samples'.format(cls_ind, filtered_cls_feat.shape[0]))
            if self.logger is not None:
                self.logger.info('cls {} has {} samples'.format(cls_ind, filtered_cls_feat.shape[0]))
            # 初始化聚类中心
            if init_centers is not None:
                tmp_init_centers = init_centers[cls_ind * self.num_per_class:(cls_ind + 1) * self.num_per_class, :]
            else:
                tmp_init_centers = None
            # 这里转化成了float32进行计算
            tmp_cluster, tmp_labels = self.feature_clustering(torch.from_numpy(filtered_cls_feat).to(torch.float32),
                                                              init_centers=tmp_init_centers)
            #
            tmp_cluster = tmp_cluster.cpu().numpy()
            tmp_labels = tmp_labels.cpu().numpy()
            init_cluster_feat = np.concatenate((init_cluster_feat, tmp_cluster), axis=0)
            tmp_cls = np.array([cls_ind] * self.num_per_class)
            init_cluster_cls = np.concatenate((init_cluster_cls, tmp_cls), axis=0)
            #
            # 添加簇的概率表示
            for i in range(self.num_per_class):
                tmp_cluster_ind = np.where(tmp_labels == i)[0]
                tmp_prob = filtered_cls_prob[tmp_cluster_ind]
                tmp_mean_prob = np.mean(tmp_prob, axis=0, keepdims=True)
                init_cluster_prob = np.concatenate((init_cluster_prob, tmp_mean_prob), axis=0)
        return init_cluster_cls, init_cluster_feat, init_cluster_prob
