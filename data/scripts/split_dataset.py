import json
import os
import random
import shutil

# Set paths
json_path = "heel/input_keypoints.json"
merged_dir = "heel/input_frames/"
train_dir = "active/train/"
val_dir = "active/val/"

if os.path.exists(train_dir):
    shutil.rmtree(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load COCO JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Shuffle images randomly
random.shuffle(data["images"])

split_idx = int(0.80 * len(data["images"]))
train_images = data["images"][:split_idx]
val_images = data["images"][split_idx:]

# Get image IDs for filtering annotations
train_ids = {img["id"] for img in train_images}
val_ids = {img["id"] for img in val_images}

# Split annotations based on image IDs
train_annotations = [ann for ann in data["annotations"] if ann["image_id"] in train_ids]
val_annotations = [ann for ann in data["annotations"] if ann["image_id"] in val_ids]

# Save new COCO JSON files
train_data = {"images": train_images, "annotations": train_annotations}
val_data = {"images": val_images, "annotations": val_annotations}

with open("active/annotations/active_train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open("active/annotations/active_val.json", "w") as f:
    json.dump(val_data, f, indent=4)

for img in train_images:
    src = os.path.join(merged_dir, img["file_name"])
    dst = os.path.join(train_dir, img["file_name"])
    if os.path.exists(src):
        shutil.copy(src, dst)  # Using copy instead of move

for img in val_images:
    src = os.path.join(merged_dir, img["file_name"])
    dst = os.path.join(val_dir, img["file_name"])
    if os.path.exists(src):
        shutil.copy(src, dst)  # Using copy instead of move

print(f"✅ Split completed: {len(train_images)} train images, {len(val_images)} val images")