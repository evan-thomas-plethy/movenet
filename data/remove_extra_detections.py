import json

# Load the COCO JSON file
with open('active/annotations/KNEE_FLEXION_SUPINE_EXT_9.json', 'r') as f:
    data = json.load(f)

# Remove annotations with fewer than 17 keypoints
data['annotations'] = [ann for ann in data['annotations'] if ann['num_keypoints'] >= 17]

# Check if number of annotations == number of images
num_annotations = len(data['annotations'])
num_images = len(data['images'])

if num_annotations != num_images:
    print(f"Warning: Number of annotations ({num_annotations}) does NOT match number of images ({num_images})")
else:
    print(f"Success: Number of annotations matches number of images ({num_images})")

# Save the updated COCO JSON file
with open('active/annotations/KNEE_FLEXION_SUPINE_EXT_9.json', 'w') as f:
    json.dump(data, f, indent=4)