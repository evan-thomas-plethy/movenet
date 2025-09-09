import json
import os
import random
import shutil
from collections import defaultdict

from data_processing.augment import run_all_augmentations

def make_training_dataset(
    json_path,
    input_frames_dir,
    train_dir,
    val_dir,
    train_json_out,
    val_json_out,
    val_ratio=0.15,
    general_val_dir="../data/general_val_500/",
    use_general_val=False,
    use_existing_train_set=False,
    use_existing_val=False,
    augmentations=None,
    augmentations_yaml=None
):
    """
    Splits dataset by video before augmentation, then augments only the train set.
    - If `use_existing_val=True`, reuses images and annotations in `val_dir` and `val_json_out`.
    - If `use_existing_train_set=True`, reuses images and annotations in `train_dir` and `train_json_out`.
    - Adds augmentation descriptions to train JSON's info['description'].
    - If `general_val_dir` is provided, adds its annotations and images to val set.
    """

    # Create output directories for JSON files
    os.makedirs(os.path.dirname(train_json_out), exist_ok=True)
    os.makedirs(os.path.dirname(val_json_out), exist_ok=True)

    # ----- Handle Validation -----
    val_frames = []
    val_annotations = []
    val_info = {}
    val_categories = []

    if use_existing_val and os.path.exists(val_json_out):
        print("🔄 Using existing validation set...")
        with open(val_json_out, 'r') as f:
            val_data = json.load(f)
        val_frames.extend(val_data["images"])
        val_annotations.extend(val_data["annotations"])
        val_info = val_data.get("info", {})
        val_categories = val_data.get("categories", [])
    else:
        print("✂️ Creating new validation set...")
        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        os.makedirs(val_dir, exist_ok=True)

        with open(json_path, 'r') as f:
            base_data = json.load(f)

        # Group images by video
        video_to_images = defaultdict(list)
        for img in base_data["images"]:
            video = os.path.basename(img["file_name"]).split("_frame_")[0]
            video_to_images[video].append(img)

        # Shuffle videos and split into train/val by frame count
        videos = list(video_to_images.keys())
        random.shuffle(videos)
        total_frames = sum(len(v) for v in video_to_images.values())
        val_count = 0
        for v in videos:
            if val_count < val_ratio * total_frames:
                val_frames.extend(video_to_images[v])
                val_count += len(video_to_images[v])

        val_ids = {img["id"] for img in val_frames}
        val_annotations.extend([a for a in base_data["annotations"] if a["image_id"] in val_ids])
        val_info = base_data.get("info", {})
        val_categories = base_data.get("categories", [])

        # Copy val images
        for img in val_frames:
            src = os.path.join(input_frames_dir, os.path.basename(img["file_name"]))
            dst = os.path.join(val_dir, os.path.basename(img["file_name"]))
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Add general_val_dir content
        if use_general_val:
            print(f"➕ Adding general validation images and annotations from {general_val_dir}")
            general_ann_path = os.path.join(general_val_dir, "annotations.json")
            with open(general_ann_path, 'r') as f:
                general_data = json.load(f)

            max_img_id = max((img["id"] for img in val_frames), default=0)
            max_ann_id = max((ann["id"] for ann in val_annotations), default=0)
            id_mapping = {}

            for img in general_data["images"]:
                new_id = max_img_id + 1
                id_mapping[img["id"]] = new_id
                img["id"] = new_id
                val_frames.append(img)
                src = os.path.join(general_val_dir, os.path.basename(img["file_name"]))
                dst = os.path.join(val_dir, os.path.basename(img["file_name"]))
                if os.path.exists(src):
                    shutil.copy(src, dst)
                max_img_id += 1

            for ann in general_data["annotations"]:
                ann["id"] = max_ann_id + 1
                ann["image_id"] = id_mapping[ann["image_id"]]
                val_annotations.append(ann)
                max_ann_id += 1

        # Save validation JSON
        json.dump(
            {"info": val_info, "images": val_frames, "annotations": val_annotations, "categories": val_categories},
            open(val_json_out, "w"), indent=4
        )

    # ----- Handle Training -----
    if use_existing_train_set and os.path.exists(train_json_out):
        print("🔄 Using existing train set...")
        with open(train_json_out, 'r') as f:
            train_data = json.load(f)
        all_train_images = train_data["images"]
        all_train_annotations = train_data["annotations"]
        train_info = train_data.get("info", {})
        train_categories = train_data.get("categories", [])
    else:
        print("✂️ Creating new train set...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir, exist_ok=True)

        with open(json_path, 'r') as f:
            base_data = json.load(f)

        val_filenames = {os.path.basename(img["file_name"]) for img in val_frames}
        train_frames = [img for img in base_data["images"] if os.path.basename(img["file_name"]) not in val_filenames]
        train_ids = {img["id"] for img in train_frames}
        train_annotations = [a for a in base_data["annotations"] if a["image_id"] in train_ids]

        for img in train_frames:
            src = os.path.join(input_frames_dir, os.path.basename(img["file_name"]))
            dst = os.path.join(train_dir, os.path.basename(img["file_name"]))
            if os.path.exists(src):
                shutil.copy(src, dst)

        # Augmentations
        all_train_images = list(train_frames)
        all_train_annotations = list(train_annotations)
        next_image_id = max(img["id"] for img in base_data["images"]) + 1
        next_ann_id = max(ann["id"] for ann in base_data["annotations"]) + 1
        base_info = base_data.get("info", {})
        base_categories = base_data.get("categories", [])
        base_description = base_info.get("description", "")
        aug_descriptions = []

        if augmentations:
            for aug in augmentations:
                aug_json = os.path.join(os.path.dirname(json_path), f"aug_{aug}.json")
                aug_dir = os.path.join(all_images_dir, aug)
                if not os.path.exists(aug_json):
                    print(f"⚠️ Skipping {aug} - missing {aug_json}")
                    continue
                with open(aug_json, 'r') as f:
                    aug_data = json.load(f)

                if "info" in aug_data and "description" in aug_data["info"]:
                    aug_descriptions.append(f"{aug}: {aug_data['info']['description']}")

                aug_name_map = {
                    img["file_name"].replace(f"{aug}_", "", 1): img
                    for img in aug_data["images"]
                }

                for orig_img in train_frames:
                    orig_name = os.path.basename(orig_img["file_name"])
                    if orig_name in aug_name_map:
                        aug_img = aug_name_map[orig_name]
                        new_img = {
                            **aug_img,
                            "id": next_image_id,
                            "file_name": f"{aug}_{orig_name}"
                        }
                        all_train_images.append(new_img)

                        src = os.path.join(aug_dir, orig_name)
                        dst = os.path.join(train_dir, f"{aug}_{orig_name}")
                        if os.path.exists(src):
                            shutil.copy(src, dst)

                        anns_for_img = [a for a in aug_data["annotations"] if a["image_id"] == aug_img["id"]]
                        for ann in anns_for_img:
                            new_ann = {
                                **ann,
                                "id": next_ann_id,
                                "image_id": next_image_id
                            }
                            all_train_annotations.append(new_ann)
                            next_ann_id += 1
                        next_image_id += 1
                    else:
                        print(f"⚠️ Missing augmented image for {orig_name} in {aug}")

        train_info = dict(base_info)
        train_categories = list(base_categories)
        if aug_descriptions:
            combined_desc = (base_description + "\nAugmentations:\n" + "\n".join(aug_descriptions)).strip()
        else:
            combined_desc = base_description
        train_info["description"] = combined_desc

        # Save new train JSON
        json.dump(
            {"info": train_info, "images": all_train_images, "annotations": all_train_annotations, "categories": train_categories},
            open(train_json_out, "w"), indent=4
        )

    print(f"✅ Split completed: {len(all_train_images)} train images, {len(val_frames)} val images")

    if not use_existing_train_set and not augmentations:
        run_all_augmentations(augmentations_yaml=f"data_processing/{augmentations_yaml}")