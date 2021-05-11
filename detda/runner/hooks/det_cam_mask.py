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
from detda.utils.utils import cal_feat_distance
from PIL import Image
from detda.utils.det_utils import transform_back_to_img
import cv2
import cmapy


class DetCAMMask(Hook):
    def __init__(self, runner, dataset_name, fg_attention_lam=1.0, cam_attention_lam=1.0, save_number_per_image=20):
        self.dataset_name = dataset_name
        self.imdb = runner.test_loaders[dataset_name].dataset.imdb
        self.num_classes = runner.test_loaders[dataset_name].dataset.n_classes
        self.fg_attention_lam = fg_attention_lam
        self.cam_attention_lam = cam_attention_lam
        self.rpn_conv = runner.model_dict['base_model'].module.RCNN_rpn.RPN_Conv
        self.rpn_cluster_center = runner.model_dict['base_model'].module.rpn_cluster_center.detach()
        self.save_number_per_image = save_number_per_image
        self.grad_cam = runner.model_dict['base_model'].module.grad_cam

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        rois = batch_output['rois']
        rois_label = batch_output['rois_label']
        base_feat = batch_output['base_feat']
        pooled_feat = batch_output['orig_pooled_feat']
        im_data = batch_output['im_data']
        gt_boxes = batch_output['gt_boxes']
        #
        if dataset_name == self.dataset_name:
            #
            val_dir = os.path.join(runner.logdir, self.imdb.name_for_path + '_{}'.format(self.imdb._image_set),
                                   'iter_{}_val_result'.format(runner.iteration))
            img_save_root = os.path.join(val_dir, 'cam_mask_images')
            if not os.path.exists(img_save_root):
                os.makedirs(img_save_root)
            img_id = os.path.basename(batch_output['img_id'][0])
            #
            tmp_rois = rois.view(-1, 5)[0:self.save_number_per_image, :]
            accu_num = 0
            for ind in range(self.save_number_per_image):
                tmp_img_path = os.path.join(img_save_root, img_id[0:-4] + '_mask_{}.jpg'.format(ind))
                #
                x1 = int(tmp_rois[ind:ind + 1, 1])
                x2 = int(tmp_rois[ind:ind + 1, 3])
                y1 = int(tmp_rois[ind:ind + 1, 2])
                y2 = int(tmp_rois[ind:ind + 1, 4])
                w = x2 - x1
                h = y2 - y1
                if accu_num > self.save_number_per_image:
                    break
                if (w > 60 and h > 60) and h * w > 0:
                    #
                    final_array = np.zeros((h, w * 5, 3), dtype=np.uint8)
                    # 获取原图
                    rois_orig_img = transform_back_to_img(im_data[:, :, y1:y2, x1:x2])
                    orig_img_array = np.asarray(rois_orig_img)
                    final_array[:, 0:w, :] = orig_img_array
                    tmp_rois_feat = pooled_feat[ind:ind + 1, :, :, :].clone()
                    tmp_rois_feat.requires_grad = True
                    # cam mask
                    cam_mask = self.grad_cam(tmp_rois_feat.view(tmp_rois_feat.shape[0], -1)).squeeze()
                    cam_mask = (cam_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                    cam_mask_array = np.repeat(cam_mask[:, :, np.newaxis], 3, axis=2)
                    cam_img = Image.fromarray(cam_mask_array)
                    resized_cam_mask = cam_img.resize((w, h))
                    resized_cam_array = np.asarray(resized_cam_mask)
                    colored_cam_img = self.show_cam_on_image(orig_img_array / 255.0, resized_cam_array[:, :, 0] / 255.0)
                    final_array[:, w:2 * w, :] = resized_cam_array
                    final_array[:, w * 2:w * 3, :] = colored_cam_img
                    # foreground mask
                    rois_rpn_feat = F.relu(self.rpn_conv(tmp_rois_feat.detach()), inplace=True).squeeze().detach()
                    num_dim, feat_shape_1, feat_shape_2 = rois_rpn_feat.shape
                    rois_rpn_feat = torch.transpose(torch.transpose(rois_rpn_feat, 0, 1), 1, 2)
                    sim = cal_feat_distance(rois_rpn_feat.view(-1, num_dim), self.rpn_cluster_center)
                    fg_mask = torch.softmax(sim * self.fg_attention_lam, dim=1)[:, 1].view(feat_shape_1, feat_shape_2)
                    mask_array = (fg_mask.cpu().numpy() * 255).astype(np.uint8)
                    mask_array = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)
                    mask_img = Image.fromarray(mask_array)
                    mask_img = mask_img.resize((w, h))
                    resize_mask_array = np.asarray(mask_img)
                    colored_img = self.show_cam_on_image(orig_img_array / 255.0, resize_mask_array[:, :, 0] / 255.0)
                    final_array[:, w * 3:w * 4, :] = resize_mask_array
                    final_array[:, w * 4:w * 5, :] = colored_img
                    #
                    final_img = Image.fromarray(final_array)
                    final_img.save(tmp_img_path)
                    #
                    rois_orig_img.close()
                    cam_img.close()
                    resized_cam_mask.close()
                    mask_img.close()
                    final_img.close()
                    #
                    accu_num += 1
                else:
                    continue
                #

    def after_val_epoch(self, runner):
        pass

    def show_cam_on_image(self, img, mask):
        color_palette = cmapy.cmap('PuBu')
        # color_palette = cmapy.cmap('YlOrRd')
        # color_palette = cv2.COLORMAP_JET
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), color_palette)
        heatmap = heatmap[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
