from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from detda.models.det_models.utils.config import cfg
# imdb的第一个参数, 在名字和split之间，需要加入下划线


class pascal_voc_2007(imdb):
    def __init__(self, split, devkit_path=None):
        # print('year is {}'.format(year))
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'VOC2007')
        self.name_for_path = 'VOC2007'
        classes = ('__background__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
        imdb.__init__(self, 'voc_2007' + '_' + split, split, classes, devkit_path, data_path)


class pascal_voc_2012(imdb):
    def __init__(self, split, devkit_path=None):
        # print('year is {}'.format(year))
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'VOC2012')
        self.name_for_path = 'VOC2012'
        classes = ('__background__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
        imdb.__init__(self, 'voc_2012' + '_' + split, split, classes, devkit_path, data_path)


class clipart(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'clipart')
        self.name_for_path = 'clipart'
        classes = ('__background__',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
        imdb.__init__(self, 'clipart_' + split, split, classes, devkit_path, data_path)


class cityscapes(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'cityscapes_' + split, split, classes, devkit_path, data_path)


class cityscapes_car(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'cityscapes_car_' + split, split, classes, devkit_path, data_path)


class cityscapes_less1(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'car', 'person')
        imdb.__init__(self, 'cityscapes_less1_' + split, split, classes, devkit_path, data_path)


class cityscapes_less2(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'bicycle', 'car', 'person')
        imdb.__init__(self, 'cityscapes_less2_' + split, split, classes, devkit_path, data_path)


class cityscapes_less3(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'car', 'person', 'train')
        imdb.__init__(self, 'cityscapes_less3_' + split, split, classes, devkit_path, data_path)


class cityscapes_less4(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'person')
        imdb.__init__(self, 'cityscapes_less4_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy')
        self.name_for_path = 'Cityscapes_foggy'
        classes = ('__background__',
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'foggy_cityscapes_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_car(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy')
        self.name_for_path = 'Cityscapes_foggy'
        classes = ('__background__',
                   'car',)
        imdb.__init__(self, 'foggy_cityscapes_car_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_less1(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy')
        self.name_for_path = 'Cityscapes_foggy'
        classes = ('__background__',
                   'car', 'person')
        imdb.__init__(self, 'foggy_cityscapes_less1_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_less2(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy')
        self.name_for_path = 'Cityscapes_foggy'
        classes = ('__background__',
                   'bicycle', 'car', 'person')
        imdb.__init__(self, 'foggy_cityscapes_less2_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_less3(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy')
        self.name_for_path = 'Cityscapes_foggy'
        classes = ('__background__',
                   'car', 'person', 'train')
        imdb.__init__(self, 'foggy_cityscapes_less3_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_less4(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy')
        self.name_for_path = 'Cityscapes_foggy'
        classes = ('__background__',
                   'person')
        imdb.__init__(self, 'foggy_cityscapes_less4_' + split, split, classes, devkit_path, data_path)


class kitti_car(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, "KITTI")
        self.name_for_path = 'KITTI'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'kitti_car_' + split, split, classes, devkit_path, data_path)
        self._image_ext = '.png'


class sim10k(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Sim10k')
        self.name_for_path = 'Sim10k'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'sim10k_' + split, split, classes, devkit_path, data_path)


class pascal_voc_2007_water(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'VOC' + self._year)
        self.name_for_path = 'VOC' + self._year
        classes = ('__background__',  # always index 0
                   'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        imdb.__init__(self, 'voc_water_2007' + '_' + split, split, classes, devkit_path, data_path)


class pascal_voc_2012_water(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'VOC' + self._year)
        self.name_for_path = 'VOC' + self._year
        classes = ('__background__',  # always index 0
                   'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        imdb.__init__(self, 'voc_water_2012' + '_' + split, split, classes, devkit_path, data_path)


class water(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'watercolor')
        self.name_for_path = 'watercolor'
        classes = ('__background__',  # always index 0
                   'bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        imdb.__init__(self, 'watercolor_' + '_' + split, split, classes, devkit_path, data_path)


class defeat_synthetic(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'defeat_synthetic')
        self.name_for_path = 'defeat_synthetic'
        classes = ('__background__',  # always index 0
                   'bjmh', 'bjbmyw', 'jsxs', 'gkxfw', 'gjbs')
        imdb.__init__(self, 'defeat_synthetic_' + split, split, classes, devkit_path, data_path)


class defeat_real(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'defeat_real')
        self.name_for_path = 'defeat_real'
        classes = ('__background__',  # always index 0
                   'bjmh', 'bjbmyw', 'jsxs', 'gkxfw', 'gjbs')
        imdb.__init__(self, 'defeat_real_' + split, split, classes, devkit_path, data_path)


# 针对kitti和cityscapes的多类别迁移
class cityscapes_5c(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'person', 'rider', 'car', 'truck', 'train')
        imdb.__init__(self, 'cityscapes_5c_' + split, split, classes, devkit_path, data_path)


class kitti_5c(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, "KITTI")
        self.name_for_path = 'KITTI'
        classes = ('__background__',  # always index 0
                   'person', 'rider', 'car', 'truck', 'train')
        imdb.__init__(self, 'kitti_5c_' + split, split, classes, devkit_path, data_path)
        self._image_ext = '.png'


# cityscapes到bkk100k的迁移
class cityscapes_7c(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes')
        self.name_for_path = 'Cityscapes'
        classes = ('__background__',  # always index 0
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'truck')
        imdb.__init__(self, 'cityscapes_7c_' + split, split, classes, devkit_path, data_path)


class bdd100k_7c(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'BDD100k')
        self.name_for_path = 'BDD100k'
        classes = ('__background__',  # always index 0
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'truck')
        imdb.__init__(self, 'bdd100k_7c_' + split, split, classes, devkit_path, data_path)


# 原始标签系列
class cityscapes_from_png(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_from_png')
        self.name_for_path = 'Cityscapes_from_png'
        classes = ('__background__',
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'cityscapes_from_png_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_from_png(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy_from_png')
        self.name_for_path = 'Cityscapes_foggy_from_png'
        classes = ('__background__',
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'foggy_cityscapes_from_png_' + split, split, classes, devkit_path, data_path)


class cityscapes_car_from_png(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_from_png')
        self.name_for_path = 'Cityscapes_from_png'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'cityscapes_car_from_png_' + split, split, classes, devkit_path, data_path)


class cityscapes_from_json(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_from_json')
        self.name_for_path = 'Cityscapes_from_json'
        classes = ('__background__',
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'cityscapes_from_json_' + split, split, classes, devkit_path, data_path)


class foggy_cityscapes_from_json(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_foggy_from_json')
        self.name_for_path = 'Cityscapes_foggy_from_json'
        classes = ('__background__',
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'foggy_cityscapes_from_json_' + split, split, classes, devkit_path, data_path)


class cityscapes_car_from_json(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_from_json')
        self.name_for_path = 'Cityscapes_from_json'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'cityscapes_car_from_json_' + split, split, classes, devkit_path, data_path)


#

class cityscapes_5c_from_png(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_from_png')
        self.name_for_path = 'Cityscapes_from_png'
        classes = ('__background__',  # always index 0
                   'person', 'rider', 'car', 'truck', 'train')
        imdb.__init__(self, 'cityscapes_5c_from_png_' + split, split, classes, devkit_path, data_path)


class cityscapes_5c_from_json(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_from_json')
        self.name_for_path = 'Cityscapes_from_json'
        classes = ('__background__',  # always index 0
                   'person', 'rider', 'car', 'truck', 'train')
        imdb.__init__(self, 'cityscapes_5c_from_json_' + split, split, classes, devkit_path, data_path)


# cyclegan转换
class cityscapes_cyclegan(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_cyclegan')
        self.name_for_path = 'Cityscapes_cyclegan'
        classes = ('__background__',  # always index 0
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'cityscapes_cyclegan_' + split, split, classes, devkit_path, data_path)


class sim10k_cyclegan(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Sim10k_cyclegan')
        self.name_for_path = 'Sim10k_cyclegan'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'sim10k_cyclegan_' + split, split, classes, devkit_path, data_path)


class cityscapes_cyclegan_from_json(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Cityscapes_cyclegan_from_json')
        self.name_for_path = 'Cityscapes_cyclegan_from_json'
        classes = ('__background__',  # always index 0
                   'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        imdb.__init__(self, 'cityscapes_cyclegan_from_json_' + split, split, classes, devkit_path, data_path)


class sim10k_munit(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Sim10k_munit')
        self.name_for_path = 'Sim10k_munit'
        classes = ('__background__',  # always index 0
                   'car')
        imdb.__init__(self, 'sim10k_munit_' + split, split, classes, devkit_path, data_path)


class Defeat(imdb):
    def __init__(self, split, devkit_path=None):
        devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        data_path = os.path.join(devkit_path, 'Defeat')
        self.name_for_path = 'Defeat'
        classes = ('__background__',  # always index 0
                   'jyz','jyzpl')
        imdb.__init__(self, 'Defeat_' + split, split, classes, devkit_path, data_path)