import json

# index_mapping = [0, 14, 15, 17, 16, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13]

index_mapping = list(range(17))
swap_pairs = [(1, 2), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
for i, j in swap_pairs:
    index_mapping[i], index_mapping[j] = index_mapping[j], index_mapping[i]

# Function to reorder the keypoints in a single annotation
def reorder_keypoints(keypoints):
    reordered = []
    for idx in index_mapping:
        # Extract 3 values for each keypoint (x, y, confidence)
        reordered.extend(keypoints[idx*3:idx*3+3])
    return reordered

# Load the COCO JSON file
with open('active/annotations/merged_data.json', 'r') as f:
    coco_data = json.load(f)

# Reorder the keypoints in each annotation
for annotation in coco_data['annotations']:
    annotation['keypoints'] = reorder_keypoints(annotation['keypoints'])

# Save the updated COCO JSON file
with open('active/annotations/merged_data.json', 'w') as f:
    json.dump(coco_data, f, indent=4)

print("Reordering completed successfully!")