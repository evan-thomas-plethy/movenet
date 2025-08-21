import json
import random
import os
from shutil import copy2

# === CONFIG ===
coco_path = 'train/_annotations.coco.json'  # Input COCO annotation file from Roboflow
images_dir = 'train'                        # Directory containing the images
output_dir = 'active'                       # Where to save train/val splits
split = 0.8                                 # Fraction of data to use for training

# === Roboflow to COCO Keypoint Index Mapping ===
# Roboflow: ["nose", "neck", "left-shoulder", "left-elbow", "left-wrist",
#            "right-shoulder", "right-elbow", "right-wrist", "left-hip", 
#            "left-knee", "left-ankle", "right-hip", "right-knee", 
#            "right-ankle", "left-eye", "right-eye", "right-ear", "left-ear"]
# COCO:     ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
#            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
#            "left_wrist", "right_wrist", "left_hip", "right_hip", 
#            "left_knee", "right_knee", "left_ankle", "right_ankle"]

roboflow_to_coco_index = [
    0,   # nose
    14,  # left_eye
    15,  # right_eye
    17,  # left_ear
    16,  # right_ear
    2,   # left_shoulder
    5,   # right_shoulder
    3,   # left_elbow
    6,   # right_elbow
    4,   # left_wrist
    7,   # right_wrist
    8,   # left_hip
    11,  # right_hip
    9,   # left_knee
    12,  # right_knee
    10,  # left_ankle
    13,  # right_ankle
]

# === FUNCTIONS ===

def reorder_keypoints(ann):
    old_kps = ann['keypoints']
    # Reshape to (18, 3)
    old_kps = [old_kps[i:i+3] for i in range(0, len(old_kps), 3)]
    
    new_kps = [[0, 0, 0] for _ in range(17)]
    for new_idx, old_idx in enumerate(roboflow_to_coco_index):
        if old_idx < len(old_kps):
            new_kps[new_idx] = old_kps[old_idx]

    ann['keypoints'] = [coord for kp in new_kps for coord in kp]
    ann['num_keypoints'] = sum(1 for i in range(0, len(ann['keypoints']), 3) if ann['keypoints'][i+2] > 0)
    return ann

def split_coco(coco, image_ids):
    images = [img for img in coco['images'] if img['id'] in image_ids]
    annots = [reorder_keypoints(ann) for ann in coco['annotations'] if ann['image_id'] in image_ids]
    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],
                [12,13],[6,12],[7,13],[6,8],[7,9],
                [8,10],[9,11],[2,3],[1,2],[1,3],
                [2,4],[3,5],[4,6],[5,7]
            ]
        }],
        'images': images,
        'annotations': annots
    }

# === LOAD COCO ===
with open(coco_path) as f:
    coco = json.load(f)

# === SHUFFLE + SPLIT ===
image_ids = [img['id'] for img in coco['images']]
random.shuffle(image_ids)
split_idx = int(split * len(image_ids))
train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

# === SPLIT AND SAVE ===
train_coco = split_coco(coco, train_ids)
val_coco = split_coco(coco, val_ids)

os.makedirs(f'{output_dir}/annotations', exist_ok=True)
with open(f'{output_dir}/annotations/active_train.json', 'w') as f:
    json.dump(train_coco, f)
with open(f'{output_dir}/annotations/active_val.json', 'w') as f:
    json.dump(val_coco, f)

# === COPY IMAGES ===
for split, ids in [('train', train_ids), ('val', val_ids)]:
    split_dir = f'{output_dir}/{split}'
    os.makedirs(split_dir, exist_ok=True)
    for img in coco['images']:
        if img['id'] in ids:
            src = os.path.join(images_dir, img['file_name'])
            dst = os.path.join(split_dir, img['file_name'])
            if os.path.exists(src):
                copy2(src, dst)