import os
import json
import cv2
import random
from math import sqrt
from tqdm import tqdm
import albumentations as A
import yaml

# ---------- Config ----------
def load_cfg(path="data_processing/augmentations.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)["augmentations"]

CFG = load_cfg()

# ---------- IO ----------
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    categories = data['categories']
    info = data.get("info", {})
    licenses = data.get("licenses", [])
    return data, images, annotations, categories, info, licenses

# ---------- Core augmentation helper ----------
def apply_augmentation(name, aug, p, coco_data, images, annotations, img_dir, original_image_ids):
    next_img_id = max(images.keys()) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1

    original_annotations = [ann for ann in annotations if ann['image_id'] in original_image_ids]

    added_count = 0
    for ann in tqdm(original_annotations, desc=f"Applying {name}", unit="img"):
        if random.random() > p:
            continue

        img = images[ann['image_id']]
        image_path = os.path.join(img_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        kps = ann["keypoints"]
        # Split into (x, y) coordinates and visibility
        keypoints = [(kps[i], kps[i+1]) for i in range(0, len(kps), 3)]
        vis = kps[2::3]

        try:
            augmented = aug(image=image, keypoints=keypoints)
        except Exception as e:
            print(f"⚠️ Augmentation failed for {img['file_name']}: {e}")
            continue

        aug_img = augmented["image"]
        aug_kps = augmented["keypoints"]

        # Reconstruct flat keypoints safely, always length = 51
        aug_kps_flat = []
        for i in range(len(vis)):  # 17 keypoints
            if i < len(aug_kps):
                x, y = aug_kps[i]
            else:
                # If Albumentations dropped a keypoint, fallback to original coords
                x, y = keypoints[i]

            # Determine visibility
            # Keep original v=0 if not present, else if occluded/visible can remain 1/2
            v = vis[i]
            aug_kps_flat.extend([x, y, v])

        base_name, ext = os.path.splitext(img["file_name"])
        new_fname = f"{base_name}_{name}{ext}"
        out_path = os.path.join(img_dir, new_fname)
        cv2.imwrite(out_path, aug_img)

        coco_data["images"].append({
            "id": next_img_id,
            "file_name": new_fname,
            "height": aug_img.shape[0],
            "width": aug_img.shape[1]
        })

        coco_data["annotations"].append({
            "id": next_ann_id,
            "image_id": next_img_id,
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": ann["iscrowd"],
            "keypoints": aug_kps_flat
        })

        images[next_img_id] = coco_data["images"][-1]
        annotations.append(coco_data["annotations"][-1])

        next_img_id += 1
        next_ann_id += 1
        added_count += 1

    return added_count

# ---------- Build augmentations from YAML ----------
def build_all_augs():
    aug_list = []

    # Rotation
    if "rotation" in CFG:
        r0, r1 = CFG["rotation"]["degrees"]
        p = CFG["rotation"].get("probability", 1.0)
        aug_list.append(("rotate", A.Compose([A.Rotate(limit=(r0, r1), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Scale
    if "scale" in CFG:
        s0, s1 = CFG["scale"]["ratio"]
        p = CFG["scale"].get("probability", 1.0)
        aug_list.append(("scale", A.Compose([A.Affine(scale=(s0, s1), fit_output=True, mode=cv2.BORDER_CONSTANT, cval=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Translation
    if "translation" in CFG:
        x0, x1 = CFG["translation"]["x_pct"]
        y0, y1 = CFG["translation"]["y_pct"]
        p = CFG["translation"].get("probability", 1.0)
        aug_list.append(("translation", A.Compose([A.Affine(translate_percent=(x0, y0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Brightness
    if "brightness" in CFG:
        low, high = CFG["brightness"]["delta"]
        p = CFG["brightness"].get("probability", 1.0)
        aug_list.append(("brightness", A.Compose([A.RandomBrightnessContrast(brightness_limit=(low, high), contrast_limit=0, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Contrast
    if "contrast" in CFG:
        c0, c1 = CFG["contrast"]["ratio"]
        p = CFG["contrast"].get("probability", 1.0)
        low = c0 - 1.0
        high = c1 - 1.0
        aug_list.append(("contrast", A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(low, high), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Gaussian blur
    if "gaussian_blur" in CFG:
        s0, s1 = CFG["gaussian_blur"]["sigma"]
        p = CFG["gaussian_blur"].get("probability", 1.0)
        aug_list.append(("blur", A.Compose([A.GaussianBlur(blur_limit=(3,7), sigma_limit=(s0, s1), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Crop
    if "crop" in CFG:
        s0, s1 = CFG["crop"]["ratio"]
        p = CFG["crop"].get("probability", 1.0)
        aug_list.append(("crop", A.Compose([A.RandomResizedCrop(height=720, width=1280, scale=(s0,s1), ratio=(1.33,1.77), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    # Occlusion
    if "occlusion" in CFG:
        a0, a1 = CFG["occlusion"]["area_pct"]
        p = CFG["occlusion"].get("probability", 1.0)
        aug_list.append(("occlusion", A.Compose([A.CoarseDropout(min_holes=1, max_holes=1, min_height=int(a0*100), max_height=int(a1*100),
                                                                    min_width=int(a0*100), max_width=int(a1*100), fill_value=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)), p))

    return aug_list

# ---------- Public callable ----------
def run_all_augmentations():
    json_path = "../data/active/annotations/active_train.json"
    img_dir = "../data/active/train/"

    coco_data, images, annotations, categories, info, licenses = load_data(json_path)
    original_image_ids = set(images.keys())

    aug_list = build_all_augs()
    for name, aug, p in aug_list:
        added = apply_augmentation(name, aug, p, coco_data, images, annotations, img_dir, original_image_ids)
        print(f"Augmentation '{name}' added {added} new images to {img_dir}")

    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"✅ All augmentations applied and JSON saved to {json_path}")
