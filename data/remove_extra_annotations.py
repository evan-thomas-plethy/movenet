import os
import json

val_dir = "active/val/"
json_in = "active/annotations/active_val.json"
json_out = "active/annotations/active_val_filtered.json"

val_filenames = set(os.listdir(val_dir))

with open(json_in, "r") as f:
    coco_data = json.load(f)

# Filter and deduplicate images
seen_filenames = set()
filtered_images = []
for img in coco_data["images"]:
    if img["file_name"] in val_filenames and img["file_name"] not in seen_filenames:
        filtered_images.append(img)
        seen_filenames.add(img["file_name"])

valid_image_ids = {img["id"] for img in filtered_images}

# Filter and deduplicate annotations by image_id + annotation id
seen_ann_ids = set()
filtered_annotations = []
for ann in coco_data["annotations"]:
    if ann["image_id"] in valid_image_ids and ann["id"] not in seen_ann_ids:
        filtered_annotations.append(ann)
        seen_ann_ids.add(ann["id"])

# Save filtered JSON
with open(json_out, "w") as f:
    json.dump({
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": filtered_images,
        "annotations": filtered_annotations
    }, f, indent=4)

print(f"✅ Filtered COCO JSON saved to {json_out}")
print(f"Number of images: {len(filtered_images)}")
print(f"Number of annotations: {len(filtered_annotations)}")