import json

# Load the COCO JSON file
with open('active/annotations/KNEE_FLEXION_SUPINE_EXT_9.json', 'r') as f:
    data = json.load(f)

for ann in data['annotations']:
    kps = ann['keypoints']
    # 0: Nose
    # 1: Left eye
    # 2: Right eye
    # 3: Left ear
    # 4: Right ear
    # 5: Left shoulder
    # 6: Right shoulder
    # 7: Left elbow
    # 8: Right elbow
    # 9: Left wrist
    # 10: Right wrist
    # 11: Left hip
    # 12: Right hip
    # 13: Left knee
    # 14: Right knee
    # 15: Left ankle
    # 16: Right ankle

    kps[14] = 1
    # kps[17] = 1
    kps[26] = 1
    # kps[29] = 1

    # kps[24] = kps[21] 
    # kps[25] = kps[22] 
    # kps[30] = kps[27]
    # kps[31] = kps[28]
    

# Save the updated COCO JSON file
with open('active/annotations/KNEE_FLEXION_SUPINE_EXT_9.json', 'w') as f:
    json.dump(data, f, indent=4)

