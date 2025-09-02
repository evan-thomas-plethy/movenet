import json
import os

val_json_path = "../merged_dataset/person_keypoints.json"

# Load COCO-style JSON
with open(val_json_path, 'r') as f:
    val_data = json.load(f)

# Get number of images
num_images = len(val_data.get("images", []))
print(f"Number of images in validation set: {num_images}")