import os
import json
import cv2
import numpy as np
import random
import shutil
import yaml
from tqdm import tqdm
import albumentations as A
from albumentations.core.transforms_interface import BasicTransform

# ---------- Custom Transforms ----------
class ColorOverlay(A.DualTransform):
    """Apply a random color overlay to the image."""

    def __init__(self, colors, alpha_range=(0.2, 0.5), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.colors = colors
        self.alpha_range = alpha_range

    def apply(self, img, **params):
        color = random.choice(self.colors)
        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])

        # Convert RGB to BGR (OpenCV format)
        color_bgr = np.array([color[2], color[1], color[0]], dtype=np.uint8)

        overlay = np.full(img.shape, color_bgr, dtype=np.uint8)
        return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    def apply_to_keypoints(self, keypoints, **params):
        return keypoints  # Keypoints unchanged by color overlay

    def get_transform_init_args_names(self):
        return ("colors", "alpha_range")

# ---------- Config ----------
def load_cfg(path="data_processing/augmentations.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
def apply_augmentation(name, aug, p, coco_data, images, annotations, input_img_dir, output_img_dir, original_image_ids):
    next_img_id = max(images.keys()) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1

    original_annotations = [ann for ann in annotations if ann['image_id'] in original_image_ids]

    added_count = 0
    for ann in tqdm(original_annotations, desc=f"Applying {name}", unit="img"):
        if random.random() > p:
            continue

        img = images[ann['image_id']]
        # Read from input directory
        image_path = os.path.join(input_img_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        kps = ann["keypoints"]
        # Split into (x, y) coordinates and visibility
        keypoints = [(kps[i], kps[i+1]) for i in range(0, len(kps), 3)]
        vis = kps[2::3]

        try:
            # For area-based augmentations, use actual image dimensions
            if "occlusion" in name:
                img_height = img["height"]
                img_width = img["width"]
                area_pct = aug.transforms[0].area_pct
                
                dynamic_occlusion = A.Compose([A.CoarseDropout(
                    min_holes=1, max_holes=1, 
                    min_height=int(area_pct[0] * img_height), max_height=int(area_pct[1] * img_height),
                    min_width=int(area_pct[0] * img_width), max_width=int(area_pct[1] * img_width), 
                    fill_value=(0,0,0), p=1.0
                )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                augmented = dynamic_occlusion(image=image, keypoints=keypoints)
            elif "crop" in name:
                img_height = img["height"]
                img_width = img["width"]
                # Get scale from the transform
                scale = aug.transforms[0].scale
                dynamic_crop = A.Compose([A.RandomResizedCrop(
                    height=img_height, width=img_width, scale=scale, ratio=(1.33,1.77), p=1.0
                )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                augmented = dynamic_crop(image=image, keypoints=keypoints)
            else:
                augmented = aug(image=image, keypoints=keypoints)
        except Exception as e:
            print(f"⚠️ Augmentation failed for {img['file_name']}: {e}")
            continue

        aug_img = augmented["image"]
        aug_kps = augmented["keypoints"]

        # Get image dimensions
        img_height, img_width = aug_img.shape[:2]

        # Reconstruct flat keypoints safely, always length = 51
        aug_kps_flat = []
        for i in range(len(vis)):  # 17 keypoints
            if i < len(aug_kps):
                x, y = aug_kps[i]
            else:
                # If Albumentations dropped a keypoint, fallback to original coords
                x, y = keypoints[i]

            # Determine visibility - set to 0 if keypoint is outside image bounds
            v = vis[i]
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                v = 0  # Keypoint is outside image bounds
            
            aug_kps_flat.extend([x, y, v])

        # Update bbox for geometric augmentations
        updated_bbox = ann["bbox"]
        if any(geom_aug in name for geom_aug in ["crop", "rotation", "scale", "translation"]):
            # Calculate new bbox from keypoints
            visible_keypoints = [(aug_kps_flat[i], aug_kps_flat[i+1]) for i in range(0, len(aug_kps_flat), 3) if aug_kps_flat[i+2] > 0]
            if visible_keypoints:
                x_coords = [kp[0] for kp in visible_keypoints]
                y_coords = [kp[1] for kp in visible_keypoints]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add 5% margin around the bbox (with safety check)
                if x_max > x_min and y_max > y_min:
                    margin_x = (x_max - x_min) * 0.05
                    margin_y = (y_max - y_min) * 0.05
                    x_min -= margin_x
                    x_max += margin_x
                    y_min -= margin_y
                    y_max += margin_y
                    
                    # Clamp bbox to image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)
                    
                    updated_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    # If all keypoints are at same position, use original bbox
                    updated_bbox = ann["bbox"]
            else:
                # If no visible keypoints, use original bbox
                updated_bbox = ann["bbox"]

        base_name, ext = os.path.splitext(img["file_name"])
        new_fname = f"{base_name}_{name}{ext}"
        # Write to output directory
        out_path = os.path.join(output_img_dir, new_fname)
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
            "bbox": updated_bbox,
            "area": updated_bbox[2] * updated_bbox[3],  # width * height
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
def build_augmentation(aug_name, config):
    """Build a single augmentation transform from config."""
    
    if aug_name == "brightness":
        delta = config["delta"]
        return A.Compose([A.RandomBrightnessContrast(brightness_limit=(delta[0], delta[1]), contrast_limit=0, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "color_overlay":
        colors = config["colors"]
        alpha_range = config["alpha"]
        return A.Compose([ColorOverlay(colors=colors, alpha_range=alpha_range, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "contrast":
        ratio = config["ratio"]
        low = ratio[0] - 1.0
        high = ratio[1] - 1.0
        return A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(low, high), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "crop":
        ratio = config["ratio"]
        # Create a placeholder transform - actual dimensions will be set at runtime
        return A.Compose([A.RandomResizedCrop(height=1, width=1, scale=(ratio[0], ratio[1]), ratio=(1.33,1.77), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "gamma":
        gamma_range = config["gamma_range"]
        return A.Compose([A.RandomGamma(gamma_limit=(gamma_range[0], gamma_range[1]), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "gaussian_blur":
        sigma = config["sigma"]
        return A.Compose([A.GaussianBlur(blur_limit=(3,7), sigma_limit=(sigma[0], sigma[1]), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "hue_shift":
        shift_limit = config["shift_limit"]
        return A.Compose([A.ColorJitter(brightness=0, contrast=0, saturation=1.0, hue=(shift_limit[0], shift_limit[1]), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "motion_blur":
        kernel_size = config["kernel_size"]
        return A.Compose([A.MotionBlur(blur_limit=(kernel_size[0], kernel_size[1]), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "occlusion":
        area_pct = config["area_pct"]
        # Store area_pct in the transform for later dynamic calculation
        transform = A.CoarseDropout(min_holes=1, max_holes=1, min_height=1, max_height=1,
                                  min_width=1, max_width=1, fill_value=(0,0,0), p=1.0)
        transform.area_pct = area_pct  # Store the area_pct for dynamic calculation
        return A.Compose([transform], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "rotation":
        degrees = config["degrees"]
        return A.Compose([A.Rotate(limit=(degrees[0], degrees[1]), border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "saturation":
        ratio = config["ratio"]
        return A.Compose([A.ColorJitter(brightness=0, contrast=0, saturation=(ratio[0], ratio[1]), hue=0, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "scale":
        ratio = config["ratio"]
        return A.Compose([A.Affine(scale=(ratio[0], ratio[1]), fit_output=False, mode=cv2.BORDER_CONSTANT, cval=(0,0,0), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "shadow":
        intensity = config["intensity"]
        return A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=1, 
                                       shadow_dimension=5, p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_name == "translation":
        x_pct = config["x_pct"]
        y_pct = config["y_pct"]
        return A.Compose([A.Affine(translate_percent=(x_pct[0], x_pct[1]), p=1.0)],
                        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    else:
        return None

def create_min_max_configs():
    """Create min and max configurations for each augmentation."""
    # Load the config
    cfg = load_cfg("data_processing/augmentations.yaml")
    augmentations = cfg["augmentations"]
    min_configs = {}
    max_configs = {}
    
    for aug_name, aug_config in augmentations.items():
        if aug_name == "brightness":
            min_configs[aug_name] = {"delta": [aug_config["delta"][0], aug_config["delta"][0]], "probability": 1.0}
            max_configs[aug_name] = {"delta": [aug_config["delta"][1], aug_config["delta"][1]], "probability": 1.0}
        
        elif aug_name == "color_overlay":
            min_configs[aug_name] = {"colors": aug_config["colors"], "alpha": [aug_config["alpha"][0], aug_config["alpha"][0]], "probability": 1.0}
            max_configs[aug_name] = {"colors": aug_config["colors"], "alpha": [aug_config["alpha"][1], aug_config["alpha"][1]], "probability": 1.0}
        
        elif aug_name == "contrast":
            min_configs[aug_name] = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
            max_configs[aug_name] = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
        
        elif aug_name == "crop":
            min_configs[aug_name] = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
            max_configs[aug_name] = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
        
        elif aug_name == "gamma":
            min_configs[aug_name] = {"gamma_range": [aug_config["gamma_range"][0], aug_config["gamma_range"][0]], "probability": 1.0}
            max_configs[aug_name] = {"gamma_range": [aug_config["gamma_range"][1], aug_config["gamma_range"][1]], "probability": 1.0}
        
        elif aug_name == "gaussian_blur":
            min_configs[aug_name] = {"sigma": [aug_config["sigma"][0], aug_config["sigma"][0]], "probability": 1.0}
            max_configs[aug_name] = {"sigma": [aug_config["sigma"][1], aug_config["sigma"][1]], "probability": 1.0}
        
        elif aug_name == "hue_shift":
            min_configs[aug_name] = {"shift_limit": [aug_config["shift_limit"][0], aug_config["shift_limit"][0]], "probability": 1.0}
            max_configs[aug_name] = {"shift_limit": [aug_config["shift_limit"][1], aug_config["shift_limit"][1]], "probability": 1.0}
        
        elif aug_name == "motion_blur":
            min_configs[aug_name] = {"kernel_size": [aug_config["kernel_size"][0], aug_config["kernel_size"][0]], "probability": 1.0}
            max_configs[aug_name] = {"kernel_size": [aug_config["kernel_size"][1], aug_config["kernel_size"][1]], "probability": 1.0}
        
        elif aug_name == "occlusion":
            min_configs[aug_name] = {"area_pct": [aug_config["area_pct"][0], aug_config["area_pct"][0]], "probability": 1.0}
            max_configs[aug_name] = {"area_pct": [aug_config["area_pct"][1], aug_config["area_pct"][1]], "probability": 1.0}
        
        elif aug_name == "rotation":
            min_configs[aug_name] = {"degrees": [aug_config["degrees"][0], aug_config["degrees"][0]], "probability": 1.0}
            max_configs[aug_name] = {"degrees": [aug_config["degrees"][1], aug_config["degrees"][1]], "probability": 1.0}
        
        elif aug_name == "saturation":
            min_configs[aug_name] = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
            max_configs[aug_name] = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
        
        elif aug_name == "scale":
            min_configs[aug_name] = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
            max_configs[aug_name] = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
        
        elif aug_name == "shadow":
            min_configs[aug_name] = {"intensity": [aug_config["intensity"][0], aug_config["intensity"][0]], "probability": 1.0}
            max_configs[aug_name] = {"intensity": [aug_config["intensity"][1], aug_config["intensity"][1]], "probability": 1.0}
        
        elif aug_name == "translation":
            min_configs[aug_name] = {
                "x_pct": [aug_config["x_pct"][0], aug_config["x_pct"][0]], 
                "y_pct": [aug_config["y_pct"][0], aug_config["y_pct"][0]], 
                "probability": 1.0
            }
            max_configs[aug_name] = {
                "x_pct": [aug_config["x_pct"][1], aug_config["x_pct"][1]], 
                "y_pct": [aug_config["y_pct"][1], aug_config["y_pct"][1]], 
                "probability": 1.0
            }
    
    return min_configs, max_configs

# ---------- Dependent augmentation helper for range testing ----------
def apply_dependent_augmentation_ranges(coco_data, images, annotations, input_img_dir, output_img_dir, original_image_ids):
    """Apply dependent augmentations with min/max ranges for testing."""
    cfg = load_cfg("data_processing/augmentations.yaml")
    augmentations = cfg["augmentations"]
    
    # Separate color and geometric augmentations
    color_augs = []
    geometric_augs = []
    
    for aug_name, aug_config in augmentations.items():
        if aug_config.get("is_geometric", False):
            geometric_augs.append(aug_name)
        else:
            color_augs.append(aug_name)
    
    print(f"Color augmentations: {color_augs}")
    print(f"Geometric augmentations: {geometric_augs}")
    
    next_img_id = max(images.keys()) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1
    
    original_annotations = [ann for ann in annotations if ann['image_id'] in original_image_ids]
    
    added_count = 0
    for ann in tqdm(original_annotations, desc="Applying dependent augmentation ranges", unit="img"):
        img = images[ann['image_id']]
        # Read from input directory
        image_path = os.path.join(input_img_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        kps = ann["keypoints"]
        keypoints = [(kps[i], kps[i+1]) for i in range(0, len(kps), 3)]
        vis = kps[2::3]
        
        # Test min and max for each color augmentation
        for color_aug_name in color_augs:
            aug_config = augmentations[color_aug_name]
            
            # Create min and max configs for this color augmentation
            if color_aug_name == "brightness":
                min_config = {"delta": [aug_config["delta"][0], aug_config["delta"][0]], "probability": 1.0}
                max_config = {"delta": [aug_config["delta"][1], aug_config["delta"][1]], "probability": 1.0}
            elif color_aug_name == "color_overlay":
                min_config = {"colors": aug_config["colors"], "alpha": [aug_config["alpha"][0], aug_config["alpha"][0]], "probability": 1.0}
                max_config = {"colors": aug_config["colors"], "alpha": [aug_config["alpha"][1], aug_config["alpha"][1]], "probability": 1.0}
            elif color_aug_name == "contrast":
                min_config = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
                max_config = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
            elif color_aug_name == "gamma":
                min_config = {"gamma_range": [aug_config["gamma_range"][0], aug_config["gamma_range"][0]], "probability": 1.0}
                max_config = {"gamma_range": [aug_config["gamma_range"][1], aug_config["gamma_range"][1]], "probability": 1.0}
            elif color_aug_name == "gaussian_blur":
                min_config = {"sigma": [aug_config["sigma"][0], aug_config["sigma"][0]], "probability": 1.0}
                max_config = {"sigma": [aug_config["sigma"][1], aug_config["sigma"][1]], "probability": 1.0}
            elif color_aug_name == "hue_shift":
                min_config = {"shift_limit": [aug_config["shift_limit"][0], aug_config["shift_limit"][0]], "probability": 1.0}
                max_config = {"shift_limit": [aug_config["shift_limit"][1], aug_config["shift_limit"][1]], "probability": 1.0}
            elif color_aug_name == "motion_blur":
                min_config = {"kernel_size": [aug_config["kernel_size"][0], aug_config["kernel_size"][0]], "probability": 1.0}
                max_config = {"kernel_size": [aug_config["kernel_size"][1], aug_config["kernel_size"][1]], "probability": 1.0}
            elif color_aug_name == "occlusion":
                min_config = {"area_pct": [aug_config["area_pct"][0], aug_config["area_pct"][0]], "probability": 1.0}
                max_config = {"area_pct": [aug_config["area_pct"][1], aug_config["area_pct"][1]], "probability": 1.0}
            elif color_aug_name == "saturation":
                min_config = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
                max_config = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
            elif color_aug_name == "shadow":
                min_config = {"intensity": [aug_config["intensity"][0], aug_config["intensity"][0]], "probability": 1.0}
                max_config = {"intensity": [aug_config["intensity"][1], aug_config["intensity"][1]], "probability": 1.0}
            else:
                continue
            
            # Test min and max for this color augmentation
            for test_type, test_config in [("min", min_config), ("max", max_config)]:
                # Build the color augmentation
                color_aug = build_augmentation(color_aug_name, test_config)
                if color_aug is None:
                    continue
                
                try:
                    # Apply color augmentation
                    if "occlusion" in color_aug_name:
                        img_height = img["height"]
                        img_width = img["width"]
                        area_pct = test_config["area_pct"]
                        
                        dynamic_occlusion = A.Compose([A.CoarseDropout(
                            min_holes=1, max_holes=1, 
                            min_height=int(area_pct[0] * img_height), max_height=int(area_pct[1] * img_height),
                            min_width=int(area_pct[0] * img_width), max_width=int(area_pct[1] * img_width), 
                            fill_value=(0,0,0), p=1.0
                        )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                        augmented = dynamic_occlusion(image=image, keypoints=keypoints)
                    elif "crop" in color_aug_name:
                        img_height = img["height"]
                        img_width = img["width"]
                        scale = test_config["ratio"]
                        dynamic_crop = A.Compose([A.RandomResizedCrop(
                            height=img_height, width=img_width, scale=scale, ratio=(1.33,1.77), p=1.0
                        )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                        augmented = dynamic_crop(image=image, keypoints=keypoints)
                    else:
                        augmented = color_aug(image=image, keypoints=keypoints)
                    
                    current_image = augmented["image"]
                    current_keypoints = augmented["keypoints"]
                    
                    # Apply one random geometric augmentation
                    if geometric_augs:
                        chosen_geometric = random.choice(geometric_augs)
                        geom_config = augmentations[chosen_geometric]
                        
                        # Randomly choose min or max value for geometric augmentation
                        use_min = random.choice([True, False])
                        
                        if chosen_geometric == "crop":
                            if use_min:
                                geom_config_to_use = {"ratio": [geom_config["ratio"][0], geom_config["ratio"][0]], "probability": 1.0}
                            else:
                                geom_config_to_use = {"ratio": [geom_config["ratio"][1], geom_config["ratio"][1]], "probability": 1.0}
                        elif chosen_geometric == "rotation":
                            if use_min:
                                geom_config_to_use = {"degrees": [geom_config["degrees"][0], geom_config["degrees"][0]], "probability": 1.0}
                            else:
                                geom_config_to_use = {"degrees": [geom_config["degrees"][1], geom_config["degrees"][1]], "probability": 1.0}
                        elif chosen_geometric == "scale":
                            if use_min:
                                geom_config_to_use = {"ratio": [geom_config["ratio"][0], geom_config["ratio"][0]], "probability": 1.0}
                            else:
                                geom_config_to_use = {"ratio": [geom_config["ratio"][1], geom_config["ratio"][1]], "probability": 1.0}
                        elif chosen_geometric == "translation":
                            if use_min:
                                geom_config_to_use = {
                                    "x_pct": [geom_config["x_pct"][0], geom_config["x_pct"][0]], 
                                    "y_pct": [geom_config["y_pct"][0], geom_config["y_pct"][0]], 
                                    "probability": 1.0
                                }
                            else:
                                geom_config_to_use = {
                                    "x_pct": [geom_config["x_pct"][1], geom_config["x_pct"][1]], 
                                    "y_pct": [geom_config["y_pct"][1], geom_config["y_pct"][1]], 
                                    "probability": 1.0
                                }
                        else:
                            continue
                        
                        geom_aug = build_augmentation(chosen_geometric, geom_config_to_use)
                        if geom_aug is not None:
                            try:
                                # Handle dynamic crop for geometric augmentations
                                if "crop" in chosen_geometric:
                                    img_height = current_image.shape[0]
                                    img_width = current_image.shape[1]
                                    scale = geom_config_to_use["ratio"]
                                    dynamic_crop = A.Compose([A.RandomResizedCrop(
                                        height=img_height, width=img_width, scale=scale, ratio=(1.33,1.77), p=1.0
                                    )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                                    geom_augmented = dynamic_crop(image=current_image, keypoints=current_keypoints)
                                else:
                                    geom_augmented = geom_aug(image=current_image, keypoints=current_keypoints)
                                current_image = geom_augmented["image"]
                                current_keypoints = geom_augmented["keypoints"]
                            except Exception as e:
                                print(f"⚠️ Geometric augmentation {chosen_geometric} failed: {e}")
                                continue
                    
                    # Get image dimensions
                    img_height, img_width = current_image.shape[:2]
                    
                    # Reconstruct flat keypoints safely, always length = 51
                    aug_kps_flat = []
                    for i in range(len(vis)):  # 17 keypoints
                        if i < len(current_keypoints):
                            x, y = current_keypoints[i]
                        else:
                            # If Albumentations dropped a keypoint, fallback to original coords
                            x, y = keypoints[i]
                        
                        # Determine visibility - set to 0 if keypoint is outside image bounds
                        v = vis[i]
                        if x < 0 or x >= img_width or y < 0 or y >= img_height:
                            v = 0  # Keypoint is outside image bounds
                        
                        aug_kps_flat.extend([x, y, v])
                    
                    # Update bbox for geometric augmentations
                    updated_bbox = ann["bbox"]
                    applied_augs = [color_aug_name]
                    if geometric_augs:
                        applied_augs.append(chosen_geometric)
                    if any(geom_aug in applied_augs for geom_aug in ["crop", "rotation", "scale", "translation"]):
                        # Calculate new bbox from keypoints
                        visible_keypoints = [(aug_kps_flat[i], aug_kps_flat[i+1]) for i in range(0, len(aug_kps_flat), 3) if aug_kps_flat[i+2] > 0]
                        if visible_keypoints:
                            x_coords = [kp[0] for kp in visible_keypoints]
                            y_coords = [kp[1] for kp in visible_keypoints]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            # Add 5% margin around the bbox (with safety check)
                            if x_max > x_min and y_max > y_min:
                                margin_x = (x_max - x_min) * 0.05
                                margin_y = (y_max - y_min) * 0.05
                                x_min -= margin_x
                                x_max += margin_x
                                y_min -= margin_y
                                y_max += margin_y
                                
                                # Clamp bbox to image bounds
                                x_min = max(0, x_min)
                                y_min = max(0, y_min)
                                x_max = min(img_width, x_max)
                                y_max = min(img_height, y_max)
                                
                                updated_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                            else:
                                # If all keypoints are at same position, use original bbox
                                updated_bbox = ann["bbox"]
                        else:
                            # If no visible keypoints, use original bbox
                            updated_bbox = ann["bbox"]
                    
                    # Create filename with applied augmentations
                    base_name, ext = os.path.splitext(img["file_name"])
                    aug_suffix = f"{color_aug_name}_{test_type}"
                    if geometric_augs:
                        geom_value = "min" if use_min else "max"
                        aug_suffix += f"_{chosen_geometric}_{geom_value}"
                    new_fname = f"{base_name}_{aug_suffix}{ext}"
                    
                    # Write to output directory
                    out_path = os.path.join(output_img_dir, new_fname)
                    cv2.imwrite(out_path, current_image)
                    
                    coco_data["images"].append({
                        "id": next_img_id,
                        "file_name": new_fname,
                        "height": current_image.shape[0],
                        "width": current_image.shape[1]
                    })
                    
                    coco_data["annotations"].append({
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": ann["category_id"],
                        "bbox": updated_bbox,
                        "area": updated_bbox[2] * updated_bbox[3],  # width * height
                        "iscrowd": ann["iscrowd"],
                        "keypoints": aug_kps_flat
                    })
                    
                    images[next_img_id] = coco_data["images"][-1]
                    annotations.append(coco_data["annotations"][-1])
                    
                    next_img_id += 1
                    next_ann_id += 1
                    added_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Dependent augmentation {color_aug_name}_{test_type} failed: {e}")
                    continue
    
    return added_count

# ---------- Single color dependent augmentation helper for range testing ----------
def apply_dependent_augmentation_ranges_single_color(color_aug_name, test_config, test_type, coco_data, images, annotations, input_img_dir, output_img_dir, original_image_ids):
    """Apply a single color augmentation with min/max range and one geometric augmentation."""
    cfg = load_cfg("data_processing/augmentations.yaml")
    augmentations = cfg["augmentations"]
    
    # Get geometric augmentations
    geometric_augs = []
    for aug_name, aug_config in augmentations.items():
        if aug_config.get("is_geometric", False):
            geometric_augs.append(aug_name)
    
    next_img_id = max(images.keys()) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1
    
    original_annotations = [ann for ann in annotations if ann['image_id'] in original_image_ids]
    
    added_count = 0
    for ann in tqdm(original_annotations, desc=f"Applying {color_aug_name}_{test_type}", unit="img"):
        img = images[ann['image_id']]
        # Read from input directory
        image_path = os.path.join(input_img_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        kps = ann["keypoints"]
        keypoints = [(kps[i], kps[i+1]) for i in range(0, len(kps), 3)]
        vis = kps[2::3]
        
        # Build the color augmentation
        color_aug = build_augmentation(color_aug_name, test_config)
        if color_aug is None:
            continue
        
        try:
            # Apply color augmentation
            if "occlusion" in color_aug_name:
                img_height = img["height"]
                img_width = img["width"]
                area_pct = test_config["area_pct"]
                
                dynamic_occlusion = A.Compose([A.CoarseDropout(
                    min_holes=1, max_holes=1, 
                    min_height=int(area_pct[0] * img_height), max_height=int(area_pct[1] * img_height),
                    min_width=int(area_pct[0] * img_width), max_width=int(area_pct[1] * img_width), 
                    fill_value=(0,0,0), p=1.0
                )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                augmented = dynamic_occlusion(image=image, keypoints=keypoints)
            elif "crop" in color_aug_name:
                img_height = img["height"]
                img_width = img["width"]
                scale = test_config["ratio"]
                dynamic_crop = A.Compose([A.RandomResizedCrop(
                    height=img_height, width=img_width, scale=scale, ratio=(1.33,1.77), p=1.0
                )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                augmented = dynamic_crop(image=image, keypoints=keypoints)
            else:
                augmented = color_aug(image=image, keypoints=keypoints)
            
            current_image = augmented["image"]
            current_keypoints = augmented["keypoints"]
            
            # Apply one random geometric augmentation
            if geometric_augs:
                chosen_geometric = random.choice(geometric_augs)
                geom_config = augmentations[chosen_geometric]
                
                # Randomly choose min or max value for geometric augmentation
                use_min = random.choice([True, False])
                
                if chosen_geometric == "crop":
                    if use_min:
                        geom_config_to_use = {"ratio": [geom_config["ratio"][0], geom_config["ratio"][0]], "probability": 1.0}
                    else:
                        geom_config_to_use = {"ratio": [geom_config["ratio"][1], geom_config["ratio"][1]], "probability": 1.0}
                elif chosen_geometric == "rotation":
                    if use_min:
                        geom_config_to_use = {"degrees": [geom_config["degrees"][0], geom_config["degrees"][0]], "probability": 1.0}
                    else:
                        geom_config_to_use = {"degrees": [geom_config["degrees"][1], geom_config["degrees"][1]], "probability": 1.0}
                elif chosen_geometric == "scale":
                    if use_min:
                        geom_config_to_use = {"ratio": [geom_config["ratio"][0], geom_config["ratio"][0]], "probability": 1.0}
                    else:
                        geom_config_to_use = {"ratio": [geom_config["ratio"][1], geom_config["ratio"][1]], "probability": 1.0}
                elif chosen_geometric == "translation":
                    if use_min:
                        geom_config_to_use = {
                            "x_pct": [geom_config["x_pct"][0], geom_config["x_pct"][0]], 
                            "y_pct": [geom_config["y_pct"][0], geom_config["y_pct"][0]], 
                            "probability": 1.0
                        }
                    else:
                        geom_config_to_use = {
                            "x_pct": [geom_config["x_pct"][1], geom_config["x_pct"][1]], 
                            "y_pct": [geom_config["y_pct"][1], geom_config["y_pct"][1]], 
                            "probability": 1.0
                        }
                else:
                    continue
                
                geom_aug = build_augmentation(chosen_geometric, geom_config_to_use)
                if geom_aug is not None:
                    try:
                        # Handle dynamic crop for geometric augmentations
                        if "crop" in chosen_geometric:
                            img_height = current_image.shape[0]
                            img_width = current_image.shape[1]
                            scale = geom_config_to_use["ratio"]
                            dynamic_crop = A.Compose([A.RandomResizedCrop(
                                height=img_height, width=img_width, scale=scale, ratio=(1.33,1.77), p=1.0
                            )], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
                            geom_augmented = dynamic_crop(image=current_image, keypoints=current_keypoints)
                        else:
                            geom_augmented = geom_aug(image=current_image, keypoints=current_keypoints)
                        current_image = geom_augmented["image"]
                        current_keypoints = geom_augmented["keypoints"]
                    except Exception as e:
                        print(f"⚠️ Geometric augmentation {chosen_geometric} failed: {e}")
                        continue
            
            # Get image dimensions
            img_height, img_width = current_image.shape[:2]
            
            # Reconstruct flat keypoints safely, always length = 51
            aug_kps_flat = []
            for i in range(len(vis)):  # 17 keypoints
                if i < len(current_keypoints):
                    x, y = current_keypoints[i]
                else:
                    # If Albumentations dropped a keypoint, fallback to original coords
                    x, y = keypoints[i]
                
                # Determine visibility - set to 0 if keypoint is outside image bounds
                v = vis[i]
                if x < 0 or x >= img_width or y < 0 or y >= img_height:
                    v = 0  # Keypoint is outside image bounds
                
                aug_kps_flat.extend([x, y, v])
            
            # Update bbox for geometric augmentations
            updated_bbox = ann["bbox"]
            applied_augs = [color_aug_name]
            if geometric_augs:
                applied_augs.append(chosen_geometric)
            if any(geom_aug in applied_augs for geom_aug in ["crop", "rotation", "scale", "translation"]):
                # Calculate new bbox from keypoints
                visible_keypoints = [(aug_kps_flat[i], aug_kps_flat[i+1]) for i in range(0, len(aug_kps_flat), 3) if aug_kps_flat[i+2] > 0]
                if visible_keypoints:
                    x_coords = [kp[0] for kp in visible_keypoints]
                    y_coords = [kp[1] for kp in visible_keypoints]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add 5% margin around the bbox (with safety check)
                    if x_max > x_min and y_max > y_min:
                        margin_x = (x_max - x_min) * 0.05
                        margin_y = (y_max - y_min) * 0.05
                        x_min -= margin_x
                        x_max += margin_x
                        y_min -= margin_y
                        y_max += margin_y
                        
                        # Clamp bbox to image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img_width, x_max)
                        y_max = min(img_height, y_max)
                        
                        updated_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    else:
                        # If all keypoints are at same position, use original bbox
                        updated_bbox = ann["bbox"]
                else:
                    # If no visible keypoints, use original bbox
                    updated_bbox = ann["bbox"]
            
            # Create filename with applied augmentations
            base_name, ext = os.path.splitext(img["file_name"])
            aug_suffix = f"{color_aug_name}_{test_type}"
            if geometric_augs:
                geom_value = "min" if use_min else "max"
                aug_suffix += f"_{chosen_geometric}_{geom_value}"
            new_fname = f"{base_name}_{aug_suffix}{ext}"
            
            # Write to output directory
            out_path = os.path.join(output_img_dir, new_fname)
            cv2.imwrite(out_path, current_image)
            
            coco_data["images"].append({
                "id": next_img_id,
                "file_name": new_fname,
                "height": current_image.shape[0],
                "width": current_image.shape[1]
            })
            
            coco_data["annotations"].append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": ann["category_id"],
                "bbox": updated_bbox,
                "area": updated_bbox[2] * updated_bbox[3],  # width * height
                "iscrowd": ann["iscrowd"],
                "keypoints": aug_kps_flat
            })
            
            images[next_img_id] = coco_data["images"][-1]
            annotations.append(coco_data["annotations"][-1])
            
            next_img_id += 1
            next_ann_id += 1
            added_count += 1
            
        except Exception as e:
            print(f"⚠️ Dependent augmentation {color_aug_name}_{test_type} failed: {e}")
            continue
    
    return added_count

def test_augmentation_ranges():
    """Apply min/max augmentations to test images."""
    
    # Paths
    json_path = "data_processing/test/annotations/active_val.json"
    img_dir = "data_processing/test/test/"
    aug_dir = "data_processing/test/augs/"
    
    # Ensure directories exist
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Load data
    try:
        coco_data, images, annotations, categories, info, licenses = load_data(json_path)
        
        # Get list of actual image files in the test directory
        actual_image_files = set()
        if os.path.exists(img_dir):
            for filename in os.listdir(img_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    actual_image_files.add(filename)
        
        print(f"Found {len(actual_image_files)} actual image files in {img_dir}")
        
        # Filter images to only include those that exist in the directory
        filtered_images = {}
        filtered_image_list = []
        for img_id, img_info in images.items():
            if img_info['file_name'] in actual_image_files:
                filtered_images[img_id] = img_info
                filtered_image_list.append(img_info)
        
        # Filter annotations to only include those for existing images
        filtered_annotations = []
        for ann in annotations:
            if ann['image_id'] in filtered_images:
                filtered_annotations.append(ann)
        
        # Update the data structures
        images = filtered_images
        annotations = filtered_annotations
        coco_data['images'] = filtered_image_list
        coco_data['annotations'] = filtered_annotations
        
        original_image_ids = set(images.keys())
        print(f"Filtered to {len(original_image_ids)} images that exist in directory")
        print(f"Filtered to {len(filtered_annotations)} annotations for existing images")
        
        # Save the filtered JSON immediately to remove non-existent images
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        print(f"Saved filtered JSON to {json_path}")
        
    except FileNotFoundError:
        print(f"Creating new annotation file: {json_path}")
        # Create minimal COCO structure if file doesn't exist
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
            "info": {},
            "licenses": []
        }
        images = {}
        annotations = []
        original_image_ids = set()
    
    # Check if augmentations are independent or dependent
    cfg = load_cfg("data_processing/augmentations.yaml")
    independent_augs = cfg.get("independent_augs", True)
    
    if independent_augs:
        # Original behavior - test each augmentation independently
        min_configs, max_configs = create_min_max_configs()
        
        print(f"Testing {len(min_configs)} augmentations with min/max values...")
        
        # Apply each augmentation with min and max values
        for aug_name in min_configs.keys():
            print(f"\nTesting {aug_name}...")
            
            # Test min value
            print(f"  Applying {aug_name}_min...")
            min_config = min_configs[aug_name]
            min_aug = build_augmentation(aug_name, min_config)
            if min_aug is not None:
                min_added = apply_augmentation(
                    f"{aug_name}_min", min_aug, 1.0, 
                    coco_data, images, annotations, img_dir, aug_dir, original_image_ids
                )
                print(f"  Added {min_added} images with {aug_name}_min")
            else:
                print(f"  Could not build {aug_name}_min")
            
            # Test max value
            print(f"  Applying {aug_name}_max...")
            max_config = max_configs[aug_name]
            max_aug = build_augmentation(aug_name, max_config)
            if max_aug is not None:
                max_added = apply_augmentation(
                    f"{aug_name}_max", max_aug, 1.0, 
                    coco_data, images, annotations, img_dir, aug_dir, original_image_ids
                )
                print(f"  Added {max_added} images with {aug_name}_max")
            else:
                print(f"  Could not build {aug_name}_max")
    else:
        # Dependent behavior - only test color augmentations with 1 geometric per color
        dependence_key = cfg.get("dependence_key", "")
        if dependence_key == "1 geometric per color":
            # Create min/max configs only for color augmentations
            augmentations = cfg["augmentations"]
            color_augs = []
            
            for aug_name, aug_config in augmentations.items():
                if not aug_config.get("is_geometric", False):
                    color_augs.append(aug_name)
            
            print(f"Testing {len(color_augs)} color augmentations with min/max values (with geometric pairs)...")
            
            # Apply each color augmentation with min and max values
            for color_aug_name in color_augs:
                print(f"\nTesting {color_aug_name}...")
                
                # Create min and max configs for this color augmentation
                aug_config = augmentations[color_aug_name]
                if color_aug_name == "brightness":
                    min_config = {"delta": [aug_config["delta"][0], aug_config["delta"][0]], "probability": 1.0}
                    max_config = {"delta": [aug_config["delta"][1], aug_config["delta"][1]], "probability": 1.0}
                elif color_aug_name == "color_overlay":
                    min_config = {"colors": aug_config["colors"], "alpha": [aug_config["alpha"][0], aug_config["alpha"][0]], "probability": 1.0}
                    max_config = {"colors": aug_config["colors"], "alpha": [aug_config["alpha"][1], aug_config["alpha"][1]], "probability": 1.0}
                elif color_aug_name == "contrast":
                    min_config = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
                    max_config = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
                elif color_aug_name == "gamma":
                    min_config = {"gamma_range": [aug_config["gamma_range"][0], aug_config["gamma_range"][0]], "probability": 1.0}
                    max_config = {"gamma_range": [aug_config["gamma_range"][1], aug_config["gamma_range"][1]], "probability": 1.0}
                elif color_aug_name == "gaussian_blur":
                    min_config = {"sigma": [aug_config["sigma"][0], aug_config["sigma"][0]], "probability": 1.0}
                    max_config = {"sigma": [aug_config["sigma"][1], aug_config["sigma"][1]], "probability": 1.0}
                elif color_aug_name == "hue_shift":
                    min_config = {"shift_limit": [aug_config["shift_limit"][0], aug_config["shift_limit"][0]], "probability": 1.0}
                    max_config = {"shift_limit": [aug_config["shift_limit"][1], aug_config["shift_limit"][1]], "probability": 1.0}
                elif color_aug_name == "motion_blur":
                    min_config = {"kernel_size": [aug_config["kernel_size"][0], aug_config["kernel_size"][0]], "probability": 1.0}
                    max_config = {"kernel_size": [aug_config["kernel_size"][1], aug_config["kernel_size"][1]], "probability": 1.0}
                elif color_aug_name == "occlusion":
                    min_config = {"area_pct": [aug_config["area_pct"][0], aug_config["area_pct"][0]], "probability": 1.0}
                    max_config = {"area_pct": [aug_config["area_pct"][1], aug_config["area_pct"][1]], "probability": 1.0}
                elif color_aug_name == "saturation":
                    min_config = {"ratio": [aug_config["ratio"][0], aug_config["ratio"][0]], "probability": 1.0}
                    max_config = {"ratio": [aug_config["ratio"][1], aug_config["ratio"][1]], "probability": 1.0}
                elif color_aug_name == "shadow":
                    min_config = {"intensity": [aug_config["intensity"][0], aug_config["intensity"][0]], "probability": 1.0}
                    max_config = {"intensity": [aug_config["intensity"][1], aug_config["intensity"][1]], "probability": 1.0}
                else:
                    continue
                
                # Test min and max for this color augmentation
                for test_type, test_config in [("min", min_config), ("max", max_config)]:
                    print(f"  Applying {color_aug_name}_{test_type}...")
                    
                    # Use the existing dependent augmentation function but for single color aug
                    added = apply_dependent_augmentation_ranges_single_color(
                        color_aug_name, test_config, test_type,
                        coco_data, images, annotations, img_dir, aug_dir, original_image_ids
                    )
                    print(f"  Added {added} images with {color_aug_name}_{test_type}")
        else:
            print(f"Unknown dependence_key: {dependence_key}")
            return
    
    # Save updated annotations
    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"\n✅ All augmentation tests completed!")
    print(f"Results saved to {aug_dir}")
    print(f"Annotations saved to {json_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")

if __name__ == "__main__":
    test_augmentation_ranges()
