from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.image import square_padding_resize, square_padding_transform_coords
import math


class SinglePoseDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        in_gt_keypoints = []
        for ann in anns[:self.max_objs]:
            in_gt_keypoints.append(np.array(ann['keypoints'], np.float32).reshape(17, 3))
        in_gt_keypoints = np.array(in_gt_keypoints)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        height, width = img.shape[0], img.shape[1]
        
        # Store original dimensions for coordinate transformation
        original_dims = (height, width)
        
        flipped = False
        if self.split == 'train':
            # Apply flip augmentation if needed
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                width = img.shape[1]  # Update width after flip

        # Apply square padding and resize (matching inference pipeline)
        inp, original_dims = square_padding_resize(img, (self.opt.input_res, self.opt.input_res))
        
        # Normalize image
        inp = (inp.astype(np.float32) / 127.5)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_res = self.opt.output_res
        num_joints = self.num_joints

        hm = np.zeros((self.num_classes, output_res,
                      output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res),
                         dtype=np.float32)
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        kps_mask = np.zeros(
            (self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
            
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
            
            # Transform bounding box coordinates using square padding approach
            bbox_coords = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
            bbox_coords = square_padding_transform_coords(
                bbox_coords, original_dims, (output_res, output_res))
            bbox = np.array([bbox_coords[0, 0], bbox_coords[0, 1], 
                           bbox_coords[1, 0], bbox_coords[1, 1]])
            
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(
                    0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                ind[k] = ct_int[1] * output_res + ct_int[0]
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(num_joints):
                    if pts[j, 2] > 0:
                        # Transform keypoint coordinates using square padding approach
                        kp_coords = np.array([pts[j, :2]])
                        kp_coords = square_padding_transform_coords(
                            kp_coords, original_dims, (output_res, output_res))
                        pts[j, :2] = kp_coords[0]
                        
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                           pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            # TODO: Check the ordering of y,x here.
                            # kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps[k, j * 2] = pts[j, 1:2] - ct_int[1]
                            kps[k, j * 2 + 1] = pts[j, 0:1] - ct_int[0]

                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            # hp_offset[k * num_joints + j] = pts[j, :2] - pt_int
                            # TODO: Check the ordering of y,x here.
                            hp_offset[k * num_joints + j][0] = pts[j, 1:2] - pt_int[1]
                            hp_offset[k * num_joints + j][1] = pts[j, 0:1] - pt_int[0]
                            hp_ind[k * num_joints + j] = pt_int[1] * \
                                output_res + pt_int[0]
                            hp_mask[k * num_joints + j] = 1
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        
        ret = {'input': inp, 'hm': hm, 'ind': ind,
               'hps': kps, 'hps_mask': kps_mask,
               'hm_hp': hm_hp, 'hp_offset': hp_offset,
               'hp_ind': hp_ind, 'hp_mask': hp_mask}
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'in_height': original_dims[0], 'in_width': original_dims[1], 
                   'gt_det': gt_det, 'img_id': img_id, 'in_gt_keypoints': in_gt_keypoints}
            ret['meta'] = meta    
        return ret
