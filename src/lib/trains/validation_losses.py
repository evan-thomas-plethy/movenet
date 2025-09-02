from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from models.decode import single_pose_decode
from utils.image import transform_preds, inverse_square_padding_transform_coords
from models.utils import _sigmoid
from utils.post_process import single_pose_post_process
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
        self.model = None
        self.gt_annotation_path = None

    def set_model(self, model):
        self.model = model

    def forward(self, outputs, batch):
        opt = self.opt

        # Accumulate predictions for mAP calculation
        self._accumulate_predictions(outputs, batch)

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
        
        # Return placeholder mAP (will be computed at epoch end)
        mAP = torch.tensor([0.0, 0.0, 0.0], device=outputs[-1]['hm'].device, dtype=torch.float32)
        
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'mAP0.50:0.95': mAP[0], 'AP0.50': mAP[1], 'AP0.75': mAP[2]}
        return loss, loss_stats

    def _accumulate_predictions(self, outputs, batch):
        try:            
            img_ids = batch['meta']['img_id'].cpu().numpy()
            
            # Store predictions
            output = outputs[0]
            dets = self.model.decode(output)

            if int(batch['meta']['img_id']) == 3364:
                pred_dir = '/home/ubuntu/visionAI/movenet/images/train_pipeline_dets'
                os.makedirs(pred_dir, exist_ok=True)
                pred_path = os.path.join(pred_dir, f'{batch["meta"]["img_id"]}.json')
                with open(pred_path, 'w') as f:
                    json.dump(dets.tolist(), f)
                print(f"Saved dets to {pred_path}")

            dets = dets[0, 0, :, :]
            dets = dets.cpu().numpy()
            dets[:, [0, 1]] = dets[:, [1, 0]] 
            dets[:, :2] = dets[:, :2] * self.opt.output_res

            if self.opt.preserve_aspect_ratio:  
                dets = inverse_square_padding_transform_coords(
                    dets.copy(),
                    (batch['meta']['in_height'].cpu().numpy(), batch['meta']['in_width'].cpu().numpy()),
                    (self.opt.output_res, self.opt.output_res)
                )
            else:                
                in_height = float(batch['meta']['in_height'].cpu().numpy())
                in_width = float(batch['meta']['in_width'].cpu().numpy())
                c = np.array([in_width / 2., in_height / 2.], dtype=np.float32)
                s = max(in_height, in_width) * 1.0
                
                dets = transform_preds(
                    dets.copy(),
                    c, s,
                    (self.opt.output_res, self.opt.output_res)
                )
           
            if int(batch['meta']['img_id']) == 3364:
                pred_dir = '/home/ubuntu/visionAI/movenet/images/train_pipeline_dets_post_process'
                os.makedirs(pred_dir, exist_ok=True)
                pred_path = os.path.join(pred_dir, f'{batch["meta"]["img_id"]}.json')
                with open(pred_path, 'w') as f:
                    json.dump(dets.tolist(), f)
                print(f"Saved dets to {pred_path}")
            
            self.predictions.append({
                'image_id': int(img_ids[0]), 
                'keypoints': dets.tolist()
            })
                
        except Exception as e:
            print(f"Error accumulating predictions: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_final_map(self):
        """Compute final mAP across all accumulated validation batches"""
        if not self.predictions:
            return [0.0, 0.0, 0.0]  # Return array of 3 zeros instead of single 0.0
        
        try:
            # Load ground truth annotations from file
            if self.gt_annotation_path is None:
                # Construct the path to validation annotations
                data_dir = '../data/active'
                self.gt_annotation_path = os.path.join(data_dir, 'annotations', 'active_val.json')
            
            if not os.path.exists(self.gt_annotation_path):
                print(f"Error: Ground truth annotation file not found: {self.gt_annotation_path}")
                return 0.0
            
            print(f"Loading ground truth from: {self.gt_annotation_path}")
            
            # Load and fix COCO annotations if needed (same as mAP_debug.py)
            with open(self.gt_annotation_path, 'r') as f:
                coco_data = json.load(f)
            
            # Ensure all annotations have num_keypoints field
            for ann in coco_data['annotations']:
                if 'num_keypoints' not in ann:
                    # Count visible keypoints (visibility > 0)
                    keypoints = ann['keypoints']
                    visible_count = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0)
                    ann['num_keypoints'] = visible_count
            
            # Convert predictions to COCO format
            coco_predictions = []
            
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
            
            if not coco_predictions:
                return [0.0, 0.0, 0.0]  # Return array of 3 zeros instead of single 0.0
            
            # Create temporary files for COCO evaluation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(coco_data, f, cls=NumpyEncoder)
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
                mAP = coco_eval.stats[:3]
                
                return mAP
                
            finally:
                # Clean up temporary files
                os.unlink(gt_file)
                os.unlink(pred_file)
                
        except Exception as e:
            print(f"Error computing final mAP: {e}")
            import traceback
            traceback.print_exc()
            return [0.0, 0.0, 0.0]  # Return array of 3 zeros instead of single 0.0
    
    def reset_metrics(self):
        """Reset accumulated metrics for a new validation epoch"""
        self.predictions = []
