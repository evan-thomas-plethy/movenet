import json
import random

# Load the COCO JSON file
json_path = "active/annotations/KNEE_FLEXION_SUPINE_EXT_9.json"

# Choose the option: 'random' for a random subset or 'prefix' for file name prefix filtering
selection_method = 'random'  # Change to 'prefix' to use the file name prefix filtering
portion = 0.5
file_name_prefixes = ['']  # List of prefixes to match (only used if selection_method is 'prefix')

with open(json_path, "r") as f:
    data = json.load(f)

# Check the selection method
if selection_method == 'random':
    random.shuffle(data["images"])
    subset_size = int(portion * len(data["images"]))
    subset_images = data["images"][:subset_size]

elif selection_method == 'prefix':
    # Select images whose file name starts with either of the specified prefixes
    subset_images = [img for img in data["images"] if any(img["file_name"].startswith(prefix) for prefix in file_name_prefixes)]

else:
    raise ValueError("Invalid selection_method. Choose either 'random' or 'prefix'.")

# Get image IDs for filtering annotations
subset_ids = {img["id"] for img in subset_images}

# Filter annotations for the selected images
subset_annotations = [ann for ann in data["annotations"] if ann["image_id"] in subset_ids]

# Save the subset as a new JSON file
subset_data = {"images": subset_images, "annotations": subset_annotations}

# Output file name can be dynamic or hardcoded
output_file = f"active/annotations/KNEE_FLEXION_SUPINE_EXT_9_subset.json"

with open(output_file, "w") as f:
    json.dump(subset_data, f, indent=4)

print(f"✅ Subset created: {len(subset_images)} images and {len(subset_annotations)} annotations saved to {output_file}")