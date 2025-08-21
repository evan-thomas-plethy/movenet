from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.decode import single_pose_decode, multi_pose_decode
from models.utils import _sigmoid
from utils.post_process import multi_pose_post_process, single_pose_post_process
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


class SinglePoseValidationLoss(torch.nn.Module):
    """
    Validation loss class for single pose estimation that includes mAP calculation.
    Forward method works exactly like SinglePoseLoss with addition of mAP accumulation.
    """
    
    def __init__(self, opt):
        super(SinglePoseValidationLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt
        self.predictions = []
        self.ground_truth = []
        
    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, hp_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
        
        # Calculate regular loss (exactly like SinglePoseLoss)
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            output['hm_hp'] = _sigmoid(output['hm_hp'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                    batch['ind'], batch['hps']) / opt.num_stacks
            hp_offset_loss += self.crit_reg(
                output['hp_offset'], batch['hp_mask'],
                batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
            hm_hp_loss += self.crit_hm_hp(
                output['hm_hp'], batch['hm_hp']) / opt.num_stacks    
        
        loss = opt.hm_weight * hm_loss + \
            opt.hp_weight * hp_loss + \
            opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss

        # Accumulate predictions and ground truth for mAP calculation
        self._accumulate_predictions_and_ground_truth(outputs, batch)
        
        # Return placeholder mAP (will be computed at epoch end)
        mAP = torch.tensor(0.0, device=outputs[-1]['hm'].device, dtype=torch.float32)
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'mAP': mAP}
        return loss, loss_stats

    def _accumulate_predictions_and_ground_truth(self, outputs, batch):
        try:            
            img_ids = batch['meta']['img_id'].cpu().numpy()
            batch_size = len(img_ids)
            
            for i in range(batch_size):
                # Store predictions
                dets = single_pose_decode(
                    heat=outputs[i]['hm'],
                    wh=torch.zeros_like(outputs[i]['hm']),
                    kps=outputs[i]['hps'],
                    reg=None,
                    hm_hp=outputs[i]['hm_hp'],
                    hp_offset=outputs[i]['hp_offset'],
                    K=1
                )
                
                # Extract keypoints from detections
                keypoints = dets[0, 0, 5:39].reshape(17, 2).cpu().numpy()
                scores = dets[0, 0, 4].cpu().numpy()  # Overall score
                
                # Convert to your format
                keypoints_with_conf = []
                for j in range(17):
                    keypoints_with_conf.append([
                        float(keypoints[j, 0]),
                        float(keypoints[j, 1]), 
                        float(scores)
                    ])
                
                self.predictions.append({
                    'image_id': int(img_ids[i]),
                    'keypoints': keypoints_with_conf
                })
                
                # Store ground truth
                gt_dets = batch['meta']['gt_det'][i][0].numpy()
                kpts = gt_dets[5:39].reshape(17, 2)
                kpts_with_vis = np.concatenate([
                    kpts,
                    np.ones((17, 1), dtype=kpts.dtype)
                ], axis=1)
                
                self.ground_truth.append({
                    'image_id': int(img_ids[i]),
                    'ground_truth': kpts_with_vis.tolist()
                })
                
        except Exception as e:
            print(f"Error accumulating predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_final_map(self):
        """Compute final mAP across all accumulated validation batches"""
        if not self.predictions or not self.ground_truth:
            return 0.0
        
        try:
            # Convert to COCO format
            coco_predictions = []
            coco_ground_truth = []
            
            # Convert predictions to COCO format
            for pred in self.predictions:
                img_id = pred['image_id']
                keypoints = pred['keypoints']  # List of [x, y, confidence] for 17 keypoints
                
                if len(keypoints) == 0:
                    continue
                
                # Convert keypoints to COCO format: [x1, y1, v1, x2, y2, v2, ...]
                coco_keypoints = []
                for kp in keypoints:
                    x, y, conf = kp
                    # Convert confidence to visibility: 0=invisible, 1=occluded, 2=visible
                    if conf > 0.5:
                        visibility = 2  # visible
                    elif conf > 0.1:
                        visibility = 1  # occluded
                    else:
                        visibility = 0  # invisible
                    coco_keypoints.extend([float(x), float(y), visibility])
                
                # Create bbox from keypoints (min/max of visible keypoints)
                visible_kps = [kp for kp in keypoints if kp[2] > 0.1]
                if len(visible_kps) > 0:
                    xs = [kp[0] for kp in visible_kps]
                    ys = [kp[1] for kp in visible_kps]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
                else:
                    bbox = [0, 0, 1, 1]  # fallback bbox
                
                # Calculate average confidence as score
                score = float(np.mean([kp[2] for kp in keypoints]))
                
                # Calculate area from bbox
                area = bbox[2] * bbox[3]  # width * height
                
                coco_predictions.append({
                    'image_id': img_id,
                    'category_id': 1,  # person class
                    'keypoints': coco_keypoints,
                    'bbox': bbox,
                    'score': float(score),
                    'area': float(area),
                    'num_keypoints': 17
                })
                
            # Convert ground truth to COCO format
            ann_id = 1  # Start with annotation ID 1
            for gt in self.ground_truth:
                img_id = gt['image_id']
                gt_keypoints = gt['ground_truth']  # List of [x, y, visibility] for 17 keypoints
                
                if len(gt_keypoints) == 0:
                    continue
                
                # Convert keypoints to COCO format
                coco_keypoints = []
                for kp in gt_keypoints:
                    x, y, vis = kp
                    coco_keypoints.extend([float(x), float(y), int(vis)])
                
                # Create bbox from keypoints (min/max of visible keypoints)
                visible_kps = [kp for kp in gt_keypoints if kp[2] > 0]
                if len(visible_kps) > 0:
                    xs = [kp[0] for kp in visible_kps]
                    ys = [kp[1] for kp in visible_kps]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
                else:
                    bbox = [0, 0, 1, 1]  # fallback bbox
                
                # Calculate area from bbox
                area = bbox[2] * bbox[3]  # width * height
                
                coco_ground_truth.append({
                    'id': ann_id,  # Unique annotation ID
                    'image_id': img_id,
                    'category_id': 1,  # person class
                    'keypoints': coco_keypoints,
                    'bbox': [float(x) for x in bbox],
                    'area': float(area),
                    'iscrowd': 0,
                    'num_keypoints': 17
                })
                ann_id += 1  # Increment annotation ID
            
            if not coco_predictions or not coco_ground_truth:
                return 0.0
            
            # Create proper COCO dataset structure
            # Get unique image IDs from ground truth
            unique_img_ids = set(gt['image_id'] for gt in coco_ground_truth)
            
            # Create images array with required fields
            images = []
            for img_id in unique_img_ids:
                images.append({
                    'id': img_id,
                })
            
            coco_gt_dataset = {
                'images': images,
                'annotations': coco_ground_truth,
                'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]
            }
            
            # Create temporary files for COCO evaluation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(coco_gt_dataset, f, cls=NumpyEncoder)
                gt_file = f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(coco_predictions, f, cls=NumpyEncoder)
                pred_file = f.name
            
            try:
                # Load COCO ground truth and predictions
                coco_gt = COCO(gt_file)
                coco_dt = coco_gt.loadRes(pred_file)
                
                # Evaluate keypoints
                coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                # Get mAP (AP at IoU=0.50:0.95)
                mAP = coco_eval.stats[0]
                
                return mAP
                
            finally:
                # Clean up temporary files
                os.unlink(gt_file)
                os.unlink(pred_file)
                
        except Exception as e:
            print(f"Error computing final mAP: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def reset_metrics(self):
        """Reset accumulated metrics for a new validation epoch"""
        self.predictions = []
        self.ground_truth = []
