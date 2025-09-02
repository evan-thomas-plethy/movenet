import os
import json
import shutil
import random

# === Parameters ===
SOURCE_IMAGES_DIR = "../general-val/val"
SOURCE_ANNOTATION_FILE = "../general-val/annotations/annotations.json"

# === Roboflow → COCO keypoint index mapping ===
indices_map = [0, 14, 15, 17, 16, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13]

def reorder_keypoints(annotation):
    """Reorder keypoints from Roboflow order to COCO order."""
    if "keypoints" not in annotation or not annotation["keypoints"]:
        return annotation

    kpts = annotation["keypoints"]
    if len(kpts) % 3 != 0:
        print(f"⚠️ Skipping annotation {annotation['id']} — invalid keypoints length.")
        return annotation

    # kpts is a flat list [x1, y1, v1, x2, y2, v2, ...]
    coco_kpts = []
    for i in indices_map:
        coco_kpts.extend(kpts[i*3:(i+1)*3])
    annotation["keypoints"] = coco_kpts

    annotation["num_keypoints"] = sum(
        1 for i in range(0, len(annotation["keypoints"]), 3)
        if annotation["keypoints"][i+2] > 0.1
    )
    return annotation

def reorder_all_annotations(coco):
    """Apply keypoint reordering to all annotations in the COCO dict."""
    coco["annotations"] = [reorder_keypoints(ann) for ann in coco.get("annotations", [])]
    return coco

def sample_val_subset(num_samples):
    # Load and reorder full annotations
    with open(SOURCE_ANNOTATION_FILE, 'r') as f:
        coco = json.load(f)

    coco = reorder_all_annotations(coco)

    all_images = coco.get("images", [])
    all_annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    categories = [cat for cat in categories if cat["id"] == 1]
    categories[0]["keypoints"] = [
        "nose",
        "left-eye",
        "right-eye", 
        "left-ear",
        "right-ear",
        "left-shoulder",
        "right-shoulder",
        "left-elbow",
        "right-elbow", 
        "left-wrist",
        "right-wrist",
        "left-hip",
        "right-hip",
        "left-knee",
        "right-knee",
        "left-ankle",
        "right-ankle"
    ]
    categories[0]["skeleton"] = []

    if num_samples > len(all_images):
        raise ValueError(f"Requested {num_samples} samples, but only {len(all_images)} available.")

    # Randomly sample images
    sampled_images = random.sample(all_images, num_samples)
    sampled_image_ids = {img["id"] for img in sampled_images}

    # Filter annotations for sampled images
    filtered_annotations = [ann for ann in all_annotations if ann["image_id"] in sampled_image_ids]

    # Create output dir
    os.makedirs(DEST_IMAGE_DIR, exist_ok=True)

    # Copy sampled image files
    for img in sampled_images:
        src_path = os.path.join(SOURCE_IMAGES_DIR, img["file_name"])
        dst_path = os.path.join(DEST_IMAGE_DIR, img["file_name"])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"⚠️ Warning: Image file not found: {src_path}")

    # Save new annotations.json
    filtered_coco = {
        "info": {"description": f"General validation set of size {num_samples}.", "version": "1.0", "year": 2025, "contributor": "recupevision"},
        "licenses": coco.get("licenses", []),
        "images": sampled_images,
        "annotations": filtered_annotations,
        "categories": categories,
    }

    with open(DEST_JSON_PATH, 'w') as f:
        json.dump(filtered_coco, f, indent=2)

    print(f"✅ Sampled {num_samples} images to: {DEST_IMAGE_DIR}")
    print(f"✅ Saved filtered annotations to: {DEST_JSON_PATH}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reorder keypoints in annotations.json and sample COCO val set")
    parser.add_argument("num_samples", type=int, help="Number of images to sample")

    args = parser.parse_args()
    DEST_DIR = f"../general_val_{args.num_samples}"
    DEST_IMAGE_DIR = DEST_DIR
    DEST_JSON_PATH = os.path.join(DEST_DIR, "annotations.json")
    sample_val_subset(args.num_samples)