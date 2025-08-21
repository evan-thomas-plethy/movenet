import json
import os

json_path = os.path.join("active", "annotations", "active_train.json")

with open(json_path, "r") as f:
    data = json.load(f)

# Create a mapping from image_id to file_name
image_map = {img["id"]: img["file_name"] for img in data.get("images", [])}

invalid_annotations = []

for ann in data.get("annotations", []):
    keypoints = ann.get("keypoints")
    image_name = image_map.get(ann.get("image_id"), "unknown")
    if keypoints is None:
        invalid_annotations.append((ann.get("id", "unknown"), "missing keypoints", image_name))
    elif len(keypoints) != 51:
        invalid_annotations.append((ann.get("id", "unknown"), len(keypoints), image_name))

if invalid_annotations:
    print(f"Found {len(invalid_annotations)} invalid annotations:")
    for ann_id, kp_len, image_name in invalid_annotations:
        print(f" - Annotation ID: {ann_id}, keypoints length: {kp_len}, Image: {image_name}")
else:
    print("All annotations have valid keypoints.")