import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# List of dataset directories to merge
DATASET_DIRS = [
    "../dataset_1.0",
    "../dataset_2.0",
    "../dataset_3.0",
    # "general_val_500",
]

MERGED_DIR = "../merged_dataset"
INPUT_FRAMES_DIR = os.path.join(MERGED_DIR, "input_frames")

# Remove merged directory if it exists
if os.path.exists(MERGED_DIR):
    shutil.rmtree(MERGED_DIR)

# Create merged directory structure
os.makedirs(INPUT_FRAMES_DIR, exist_ok=True)

# Get all aug_name directories from the first dataset
aug_names = [
    d for d in os.listdir(DATASET_DIRS[0])
    if os.path.isdir(os.path.join(DATASET_DIRS[0], d)) and d != "input_frames"
]

for aug_name in aug_names:
    os.makedirs(os.path.join(MERGED_DIR, aug_name), exist_ok=True)

# Initialize JSON structures
merged_annotations = defaultdict(lambda: {
    "info": {},
    "images": [],
    "annotations": [],
    "categories": []
})

id_counters = {
    "image_id": 1,
    "annotation_id": 1
}

def merge_json(json_path, image_dir, json_key, dataset_key):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if dataset_key not in merged_annotations[json_key]["info"]:
        merged_annotations[json_key]["info"][dataset_key] = data.get("info", {})

    if not merged_annotations[json_key]["categories"]:
        merged_annotations[json_key]["categories"] = data.get("categories", [])

    image_id_mapping = {}

    for image in data.get("images", []):
        old_image_id = image["id"]
        original_file_name = image["file_name"]  # Keep this unchanged
        new_image_id = id_counters["image_id"]
        image_id_mapping[old_image_id] = new_image_id

        # Copy image using original filename
        src_path = os.path.join(image_dir, os.path.basename(original_file_name))
        dst_dir = os.path.join(MERGED_DIR, os.path.basename(image_dir))
        dst_path = os.path.join(dst_dir, os.path.basename(original_file_name))

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

        new_image = dict(image)
        new_image["id"] = new_image_id
        new_image["file_name"] = original_file_name
        merged_annotations[json_key]["images"].append(new_image)

        id_counters["image_id"] += 1

    for ann in data.get("annotations", []):
        new_ann = dict(ann)
        new_ann["id"] = id_counters["annotation_id"]
        new_ann["image_id"] = image_id_mapping.get(ann["image_id"], ann["image_id"])
        merged_annotations[json_key]["annotations"].append(new_ann)
        id_counters["annotation_id"] += 1

# Merge all datasets
for dataset_dir in tqdm(DATASET_DIRS, desc="Merging datasets"):
    dataset_key = os.path.basename(dataset_dir)

    # Copy input_frames images and annotations
    input_frames_path = os.path.join(dataset_dir, "input_frames")
    input_json_path = os.path.join(dataset_dir, "person_keypoints.json")

    if os.path.exists(input_json_path):
        merge_json(input_json_path, input_frames_path, "person_keypoints", dataset_key)

    # Copy aug_* directories and json
    for aug_name in aug_names:
        aug_dir = os.path.join(dataset_dir, aug_name)
        aug_json_path = os.path.join(dataset_dir, f"aug_{aug_name}.json")

        if os.path.exists(aug_json_path):
            merge_json(aug_json_path, aug_dir, f"aug_{aug_name}", dataset_key)

# Save merged annotation files
for key, json_data in merged_annotations.items():
    output_path = os.path.join(MERGED_DIR, f"{key}.json")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

print(f"✅ Merged dataset created at: {MERGED_DIR}")