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
import mlflow
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from opts import opts
from detectors.detector_factory import detector_factory

# Global experiment ID - change this to match your experiment
EXPERIMENT_ID = "robust-augs-heel-slides-movenet-thunder-finetune"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def evaluate_single_model(model_path, opt, coco_gt, temp_gt_path, run_id=None):
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
    processed_images = 0
    skipped_images = 0
    bar = Bar(f'Evaluating {Path(model_path).parent.name}', max=len(img_ids))
    
    for img_id in img_ids:
        try:
            # Get image info from COCO
            img_info = coco_gt.loadImgs(ids=[img_id])[0]
            # For general_val_1500, images are in the same directory as annotations.json
            img_path = os.path.join(opt.mAP, img_info['file_name'])
            
            if not os.path.exists(img_path):
                skipped_images += 1
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
                    x, y, conf = det[j]
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
                    xs = [kp[0] for kp in visible_kps]
                    ys = [kp[1] for kp in visible_kps]
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
                processed_images += 1
            
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
    temp_pred_file = os.path.join(opt.save_dir, f'temp_pred_{Path(model_path).parent.name}.json')
    with open(temp_pred_file, 'w') as f:
        json.dump(all_detections, f, cls=NumpyEncoder)
    
    # Load predictions into COCO format
    coco_dt = coco_gt.loadRes(temp_pred_file)
    
    # Evaluate keypoints
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    coco_eval.summarize()
    
    # Get keypoint mAP with proper error handling
    if hasattr(coco_eval.stats, '__len__') and len(coco_eval.stats) > 0:
        keypoint_map = coco_eval.stats[0]  # AP at IoU=0.50:0.95
    else:
        print("Warning: COCO evaluation stats is empty or invalid")
        print("This usually indicates a mismatch between GT and DT annotations")
        keypoint_map = 0.0
    
    # Clean up temporary prediction file
    if os.path.exists(temp_pred_file):
        os.remove(temp_pred_file)
    
    # Log to MLflow if run_id is provided
    if run_id is not None:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("test/mAP0.50:0.95", float(keypoint_map))
                print(f"✓ Logged mAP_50_95 = {keypoint_map:.4f} to MLflow run {run_id}")
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")
    
    return keypoint_map


def evaluate_all_models(opt):
    """Evaluate mAP for all models in the experiment directory"""
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = 0  # Disable debug output for evaluation
    
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri("http://35.165.139.156:5000")
    
    # Define experiment directory
    experiment_dir = Path(f"../exp/single_pose/{EXPERIMENT_ID}")
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return
    
    print(f"Evaluating models in: {experiment_dir}")
    
    # Get the experiment
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_ID)
        if experiment is None:
            print(f"Experiment '{EXPERIMENT_ID}' not found in MLflow!")
            return
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error accessing experiment '{EXPERIMENT_ID}' in MLflow: {e}")
        return
    
    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    if runs.empty:
        print("No runs found in the experiment.")
        return
    
    print(f"Found {len(runs)} runs in MLflow experiment")
    
    # Create mapping from run names to run IDs
    run_name_to_id = {}
    for idx, run in runs.iterrows():
        run_id = run['run_id']
        run_name = run.get('tags.mlflow.runName', f"run_{run_id[:8]}")
        run_name_to_id[run_name] = run_id
    
    # Load existing COCO ground truth
    gt_annotation_path = os.path.join(opt.mAP, 'annotations.json')
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
    
    # Find all model files (handle both files and directories)
    model_files = []
    run_ids = []
    for run_dir in experiment_dir.iterdir():
        if run_dir.is_dir():
            run_name = run_dir.name
            model_path = run_dir / "model_best.pth"
            if model_path.exists():
                if model_path.is_file():
                    model_files.append(model_path)
                    run_ids.append(run_name_to_id.get(run_name))
                elif model_path.is_dir():
                    # If model_best.pth is a directory, look for actual .pth files inside
                    pth_files = list(model_path.glob("*.pth"))
                    if pth_files:
                        model_files.append(pth_files[0])  # Use the first .pth file found
                        run_ids.append(run_name_to_id.get(run_name))
                        print(f"Found model file in directory: {pth_files[0]}")
                    else:
                        print(f"Warning: model_best.pth is a directory but contains no .pth files: {model_path}")
    
    if not model_files:
        print(f"No model files found in {experiment_dir}")
        return
    
    print(f"Found {len(model_files)} models to evaluate")
    
    # Evaluate each model
    results = {}
    
    for i, model_path in enumerate(model_files):
        run_name = model_path.parent.name
        run_id = run_ids[i] if i < len(run_ids) else None
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_name}")
        print(f"Model: {model_path}")
        if run_id:
            print(f"MLflow Run ID: {run_id}")
        print(f"{'='*60}")
        
        try:
            map_score = evaluate_single_model(str(model_path), opt, coco_gt, temp_gt_path, run_id)
            if map_score is not None:
                results[run_name] = {
                    'model_path': str(model_path),
                    'mAP_50_95': float(map_score),
                    'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'mlflow_run_id': run_id
                }
                print(f"✓ {run_name}: mAP = {map_score:.4f}")
            else:
                print(f"✗ {run_name}: Failed to evaluate")
        except Exception as e:
            print(f"✗ {run_name}: Error - {e}")
            continue
    
    # Save all results to a single JSON file
    if results:
        results_file = os.path.join(opt.save_dir, f'{EXPERIMENT_ID}_mAP_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        for run_name, result in results.items():
            print(f"{run_name}: {result['mAP_50_95']:.4f}")
        
        print(f"\nResults saved to: {results_file}")
    else:
        print("No successful evaluations to save")
    
    # Clean up temporary ground truth file
    if os.path.exists(temp_gt_path):
        os.remove(temp_gt_path)

if __name__ == '__main__':
    opt = opts().init()
    
    opt.mAP = "../data/general_val_1500"

    print(f"Evaluating mAP for task: {opt.task}")
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Validation directory: {opt.mAP}")
    print(f"Save directory: {opt.save_dir}")
    
    evaluate_all_models(opt)
