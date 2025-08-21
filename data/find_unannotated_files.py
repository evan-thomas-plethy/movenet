import os
import json

# Set your dataset directory here
DATASET_DIR = "dataset_2.0"
INPUT_FRAMES_DIR = os.path.join(DATASET_DIR, "input_frames")
ANNOTATION_JSON = os.path.join(DATASET_DIR, "person_keypoints.json")

# Load annotated filenames from JSON
with open(ANNOTATION_JSON, 'r') as f:
    data = json.load(f)
    annotated_files = set(os.path.basename(img["file_name"]) for img in data.get("images", []))

# List all .jpg files in input_frames directory
all_files = set(f for f in os.listdir(INPUT_FRAMES_DIR) if f.endswith(".jpg"))

# Find unannotated files
unannotated_files = sorted(all_files - annotated_files)

# Print result
if unannotated_files:
    print(f"⚠️ Found {len(unannotated_files)} unannotated image files:")
    for fname in unannotated_files:
        print(f"  - {fname}")
else:
    print("✅ All image files in input_frames are annotated.")