import json
import os

# Paths
json_path = 'active/annotations/KNEE_FLEXION_SUPINE.json'
image_dir = 'active/KNEE_FLEXION_SUPINE'

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Get all valid image filenames from the JSON
valid_image_names = set(img['file_name'] for img in data['images'])

# List all files in the image directory
all_image_files = os.listdir(image_dir)

# Remove images not in the JSON
for image_file in all_image_files:
    if image_file not in valid_image_names:
        full_path = os.path.join(image_dir, image_file)
        if os.path.isfile(full_path):
            print(f"Deleting {full_path}")
            os.remove(full_path)

print("Cleanup complete!")