from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import shutil
import cv2
import numpy as np

import torch

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def calculate_bounding_box_from_keypoints(keypoints, margin_ratio=0.2):
    """
    Calculate bounding box from keypoints with margin.
    
    Args:
        keypoints: numpy array of shape (17, 3) with [x, y, confidence]
        margin_ratio: ratio of margin to add around the bounding box
    
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box
    """
    # Filter keypoints with confidence > 0.1
    valid_keypoints = keypoints[keypoints[:, 2] > 0.1]
    
    if len(valid_keypoints) == 0:
        return None
    
    # Get min and max coordinates
    x_coords = valid_keypoints[:, 0]
    y_coords = valid_keypoints[:, 1]
    
    x1, x2 = np.min(x_coords), np.max(x_coords)
    y1, y2 = np.min(y_coords), np.max(y_coords)
    
    # Add margin
    width = x2 - x1
    height = y2 - y1
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = x2 + margin_x
    y2 = y2 + margin_y
    
    return int(x1), int(y1), int(x2), int(y2)

def crop_image_with_bbox(image, bbox):
    """
    Crop image using bounding box coordinates.
    
    Args:
        image: input image
        bbox: tuple (x1, y1, x2, y2)
    
    Returns:
        cropped image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def demo_smart_cropping(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    
    # Create cropped images directory
    cropped_dir = f"{opt.demo}_cropped"
    os.makedirs(cropped_dir, exist_ok=True)
    print(f"Created cropped images directory: {cropped_dir}")
    
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]
    
    cropped_image_names = []
    
    # First pass: detect and crop all images
    print("\n" + "="*60)
    print("FIRST PASS: Detecting and cropping images")
    print("="*60)
    
    for (image_name) in image_names:
        print(f"\nProcessing: {image_name}")
        
        # Load image
        img = cv2.imread(image_name)
        if img is None:
            print(f"Could not load image: {image_name}")
            continue
        
        # First inference - normal detection
        print("=== First Inference (Full Image) ===")
        ret1 = detector.run(image_name)
        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret1[stat])
        print(time_str)
        
        # Calculate bounding box from first detection
        keypoints = ret1['results']
        bbox = calculate_bounding_box_from_keypoints(keypoints)
        
        if bbox is not None:
            # Crop image based on bounding box
            cropped_img = crop_image_with_bbox(img, bbox)
            
            # Save cropped image to dedicated directory
            base_name = os.path.basename(image_name)
            name_without_ext, ext = os.path.splitext(base_name)
            cropped_path = os.path.join(cropped_dir, f"{name_without_ext}_cropped{ext}")
            cv2.imwrite(cropped_path, cropped_img)
            cropped_image_names.append(cropped_path)
            print(f"Saved cropped image to: {cropped_path}")
            print(f"Bounding box: {bbox}")
        else:
            print("No valid keypoints detected in first inference")
    
    # Second pass: run inference on all cropped images
    if cropped_image_names:
        debug_dir = '../exp/cache/debug/'
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
        print("\n" + "="*60)
        print("SECOND PASS: Running inference on cropped images")
        print("="*60)
        
        for cropped_image_name in cropped_image_names:
            print(f"\nProcessing cropped image: {cropped_image_name}")
            
            # Second inference - on cropped image
            print("=== Second Inference (Cropped Image) ===")
            ret2 = detector.run(cropped_image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret2[stat])
            print(time_str)
            print("-" * 50)
    else:
        print("\nNo cropped images were created. Skipping second pass.")

if __name__ == '__main__':
    opt = opts().init()
    demo_smart_cropping(opt)
