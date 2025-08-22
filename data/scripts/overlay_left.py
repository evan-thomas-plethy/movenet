import json
import os
import cv2
import numpy as np

# COCO Keypoint indices for the left side of the body (based on the COCO ordering)
LEFT_KEYPOINTS = [
    1,  # Left Eye
    3,  # Left Ear
    5,  # Left Shoulder
    7,  # Left Elbow
    9,  # Left Wrist
    11, # Left Hip
    13, # Left Knee
    15  # Left Ankle
]

RIGHT_KEYPOINTS = [
    2,  # Left Eye
    4,  # Left Ear
    6,  # Left Shoulder
    8,  # Left Elbow
    10,  # Left Wrist
    12, # Left Hip
    14, # Left Knee
    16  # Left Ankle
]

# Function to overlay left-side keypoints on the image
def overlay_left_keypoints(image, annotations, image_id):
    # Iterate through annotations and find the relevant one for the given image_id
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            keypoints = annotation['keypoints']
            for i in RIGHT_KEYPOINTS:
                # Each keypoint is represented as (x, y, confidence) in the annotations
                x = keypoints[i * 3]  # x coordinate
                y = keypoints[i * 3 + 1]  # y coordinate
                confidence = keypoints[i * 3 + 2]  # confidence

                # If the confidence is greater than a threshold (e.g., 0.5), draw the keypoint
                if confidence > 0:
                    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green color

    return image

# Function to process a single image and overlay keypoints, then save the result
def process_and_overlay_single_image(annotation_file, image_dir, output_dir, target_image_filename):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    
    # Find the image with the specified filename
    for image_info in images:
        image_filename = image_info['file_name']
        if image_filename == target_image_filename:
            image_id = image_info['id']
            image_path = os.path.join(image_dir, image_filename)
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image {image_filename}. Skipping.")
                return
            
            # Overlay the left side keypoints
            img_with_keypoints = overlay_left_keypoints(img, annotations, image_id)
            
            # Save the image with overlaid keypoints to the current working directory
            output_image_path = os.path.join(output_dir, f"overlay_{image_filename}")
            cv2.imwrite(output_image_path, img_with_keypoints)
            print(f"Saved image with overlaid keypoints: {output_image_path}")
            return

    print(f"Image {target_image_filename} not found in annotations.")

# Set paths for annotations and images
annotation_file = 'active/annotations/KNEE_FLEXION_SUPINE_EXT_4.json'  # Input annotations JSON file
image_dir = 'active/KNEE_FLEXION_SUPINE'  # Directory containing the images
output_dir = os.getcwd()  # Save the processed images to the current working directory

# Specify the image filename you want to process
target_image_filename = 'KNEE_FLEXION_SUPINE_EXT_4_frame_0010.jpg'  # Replace with your target image filename

process_and_overlay_single_image(annotation_file, image_dir, output_dir, target_image_filename)