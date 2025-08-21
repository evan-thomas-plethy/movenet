import os
import json
import shutil

from PIL import Image  # just to verify image integrity

# === Paths ===
images_dir = "active/train"
annotations_path = "active/annotations/active_train.json"
output_annotations_path = annotations_path  # overwrite or change if desired

# === Load original annotations ===
with open(annotations_path, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# === Get max image and annotation IDs to start from ===
next_image_id = max(img["id"] for img in images) + 1
next_annotation_id = max(ann["id"] for ann in annotations) + 1

# === Track mapping from original image ID to new ID ===
image_id_map = {}

# === Duplicate images and create new image entries ===
new_images = []
new_annotations = []

for img in images:
    original_file_name = img["file_name"]
    original_image_id = img["id"]

    # Build new filename
    name, ext = os.path.splitext(original_file_name)
    new_file_name = f"{name}_2{ext}"

    # Copy image file
    src_path = os.path.join(images_dir, original_file_name)
    dst_path = os.path.join(images_dir, new_file_name)
    shutil.copy2(src_path, dst_path)

    # Verify image shape
    with Image.open(dst_path) as im:
        width, height = im.size

    # Create new image entry
    new_img_entry = {
        "id": next_image_id,
        "file_name": new_file_name,
        "width": width,
        "height": height
    }

    image_id_map[original_image_id] = next_image_id
    new_images.append(new_img_entry)
    next_image_id += 1

# === Duplicate annotations ===
for ann in annotations:
    original_image_id = ann["image_id"]
    if original_image_id not in image_id_map:
        continue

    new_ann = ann.copy()
    new_ann["id"] = next_annotation_id
    new_ann["image_id"] = image_id_map[original_image_id]
    new_annotations.append(new_ann)
    next_annotation_id += 1

# === Combine original and new ===
coco["images"].extend(new_images)
coco["annotations"].extend(new_annotations)

# === Save updated annotations ===
with open(output_annotations_path, "w") as f:
    json.dump(coco, f, indent=4)

print(f"Duplicated {len(new_images)} images and {len(new_annotations)} annotations.")