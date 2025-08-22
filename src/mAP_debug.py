from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from opts import opts
from detectors.detector_factory import detector_factory

# Debug configuration - change these to match your specific experiment and run
EXPERIMENT_ID = "heel-slides-partial-unfreeze-movenet-thunder-finetune"
RUN_NAME = "HEEL_SLIDES_run1_mid_unfreeze_with_fpn"  # Change this to your specific run name

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def evaluate_single_model(model_path, opt, coco_gt, temp_gt_path):
    """Evaluate mAP for a single model"""
    
    # Update the model path
    opt.load_model = model_path
    
    # Create detector
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    
    # Get all image IDs from COCO annotations
    img_ids = coco_gt.getImgIds()
    
    # Run detection on all images
    all_detections = []
    bar = Bar(f'Evaluating {Path(model_path).name}', max=len(img_ids))
    
    for img_id in img_ids:
        try:
            # Get image info from COCO
            img_info = coco_gt.loadImgs(ids=[img_id])[0]
            img_path = os.path.join(opt.mAP, 'val', img_info['file_name'])
            
            if not os.path.exists(img_path):
                bar.next()
                continue
            
            # Run detection
            ret = detector.run(img_path)
            
            # Convert single pose detection to COCO format
            if isinstance(ret['results'], np.ndarray):
                # Single pose: (17, 3) array with [x, y, confidence] for each keypoint
                det = ret['results']  # Shape: (17, 3)
                
                # Convert keypoints to COCO format
                coco_keypoints = []
                for j in range(17):
                    y, x, conf = det[j]  # MoveNet returns (y, x, conf)
                    # Convert confidence to visibility
                    if conf > 0.5:
                        visibility = 2  # visible
                    elif conf > 0.1:
                        visibility = 1  # occluded
                    else:
                        visibility = 0  # invisible
                    coco_keypoints.extend([float(x), float(y), visibility])  # COCO format: (x, y, visibility)
                
                # Create bbox from keypoints
                visible_kps = [kp for kp in det if kp[2] > 0.1]
                if len(visible_kps) > 0:
                    xs = [kp[1] for kp in visible_kps]  # x coordinates (index 1)
                    ys = [kp[0] for kp in visible_kps]  # y coordinates (index 0)
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
                else:
                    bbox = [0, 0, 1, 1]  # fallback bbox
                
                # Calculate average confidence as score
                score = float(np.mean([kp[2] for kp in det]))
                
                detection = {
                    "image_id": img_id,
                    "category_id": 1,  # person class
                    "bbox": bbox,
                    "score": score,
                    "keypoints": coco_keypoints,
                    "num_keypoints": 17
                }
                all_detections.append(detection)
            
            bar.next()
            
        except Exception as e:
            print(f"Error processing image_id {img_id}: {e}")
            bar.next()
            continue
    
    bar.finish()
    
    if not all_detections:
        print(f"Warning: No detections generated for {model_path}")
        return None
    
    # Save predictions to temporary file
    temp_pred_file = os.path.join(opt.save_dir, f'temp_pred_{Path(model_path).stem}.json')
    with open(temp_pred_file, 'w') as f:
        json.dump(all_detections, f, cls=NumpyEncoder)
    
    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(temp_pred_file)
    
    # Evaluate keypoints
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Get keypoint mAP
    keypoint_map = coco_eval.stats[0]  # AP at IoU=0.50:0.95
    
    # Clean up temporary prediction file
    if os.path.exists(temp_pred_file):
        os.remove(temp_pred_file)
    
    return keypoint_map


def evaluate_all_checkpoints(opt):
    """Evaluate mAP for all model checkpoints in a specific run"""
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = 0  # Disable debug output for evaluation
    
    # Define experiment and run directory
    experiment_dir = Path(f"../exp/single_pose/{EXPERIMENT_ID}")
    run_dir = experiment_dir / RUN_NAME
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    print(f"Evaluating checkpoints in: {run_dir}")
    
    # Load existing COCO ground truth
    gt_annotation_path = os.path.join(opt.mAP, 'annotations', 'active_val.json')
    if not os.path.exists(gt_annotation_path):
        print(f"Error: Ground truth annotation file not found: {gt_annotation_path}")
        return
    
    print(f"Loading ground truth from: {gt_annotation_path}")
    
    # Load and fix COCO annotations if needed
    with open(gt_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    # Ensure all annotations have num_keypoints field
    for ann in coco_data['annotations']:
        if 'num_keypoints' not in ann:
            # Count visible keypoints (visibility > 0)
            keypoints = ann['keypoints']
            visible_count = sum(1 for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0)
            ann['num_keypoints'] = visible_count
    
    # Save fixed annotations to temporary file
    temp_gt_path = os.path.join(opt.save_dir, 'temp_gt.json')
    os.makedirs(opt.save_dir, exist_ok=True)
    with open(temp_gt_path, 'w') as f:
        json.dump(coco_data, f)
    
    coco_gt = COCO(temp_gt_path)
    
    # Find all model checkpoint files (model_*.pth)
    model_files = []
    for model_file in run_dir.glob("model_*.pth"):
        if model_file.is_file():
            model_files.append(model_file)
    
    # Sort model files by name for consistent ordering
    model_files.sort(key=lambda x: x.name)
    
    if not model_files:
        print(f"No model checkpoint files found in {run_dir}")
        print("Looking for files matching pattern: model_*.pth, model_best.pth, model_last.pth")
        return
    
    print(f"Found {len(model_files)} model checkpoints to evaluate:")
    for model_file in model_files:
        print(f"  - {model_file.name}")
    
    # Evaluate each model checkpoint
    results = {}
    
    for i, model_path in enumerate(model_files):
        model_name = model_path.name
        
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint {i+1}/{len(model_files)}: {model_name}")
        print(f"Model: {model_path}")
        print(f"{'='*60}")
        
        try:
            map_score = evaluate_single_model(str(model_path), opt, coco_gt, temp_gt_path)
            if map_score is not None:
                results[model_name] = {
                    'model_path': str(model_path),
                    'mAP_50_95': float(map_score),
                    'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"✓ {model_name}: mAP = {map_score:.4f}")
            else:
                print(f"✗ {model_name}: Failed to evaluate")
        except Exception as e:
            print(f"✗ {model_name}: Error - {e}")
            continue
    
    # Print final results summary
    if results:
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Experiment: {EXPERIMENT_ID}")
        print(f"Run: {RUN_NAME}")
        print(f"Total checkpoints evaluated: {len(results)}")
        print(f"{'='*60}")
        
        # Sort results by mAP score (descending)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mAP_50_95'], reverse=True)
        
        print(f"{'Checkpoint':<20} {'mAP_50_95':<10} {'Time':<20}")
        print("-" * 50)
        for model_name, result in sorted_results:
            print(f"{model_name:<20} {result['mAP_50_95']:<10.4f} {result['evaluation_time']:<20}")
        
        # Find best checkpoint
        best_checkpoint = sorted_results[0]
        print(f"\n🏆 BEST CHECKPOINT: {best_checkpoint[0]} (mAP: {best_checkpoint[1]['mAP_50_95']:.4f})")
        
        # Save results to JSON file for reference
        results_file = os.path.join(opt.save_dir, f'{EXPERIMENT_ID}_{RUN_NAME}_debug_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
    else:
        print("No successful evaluations to report")
    
    # Clean up temporary ground truth file
    if os.path.exists(temp_gt_path):
        os.remove(temp_gt_path)

if __name__ == '__main__':
    opt = opts().init()
    
    opt.mAP = "../data/active"

    print(f"DEBUG MODE: Evaluating mAP for task: {opt.task}")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Run Name: {RUN_NAME}")
    print(f"Validation directory: {opt.mAP}")
    print(f"Save directory: {opt.save_dir}")
    print(f"Note: No MLflow logging in debug mode")
    
    evaluate_all_checkpoints(opt)
