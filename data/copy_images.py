import os
import json
import shutil

# Paths
json_path = "active/annotations/active_train7x100.json"
images_root = "active/KNEE_FLEXION_SUPINE"  # Directory where images currently are
destination_dir = "active/train7x100"

# Create destination folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Move each image
for image_info in data["images"]:
    file_name = image_info["file_name"]
    source_path = os.path.join(images_root, file_name)
    dest_path = os.path.join(destination_dir, file_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path)
        print(f"Moved {file_name} → {destination_dir}")
    else:
        print(f"Warning: {file_name} not found in {images_root}")

print("Done moving images!")