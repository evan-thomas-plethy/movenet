import json
import os
import cv2
import numpy as np

# Path configurations
IMAGE_DIR = 'dataset_1.0/input_frames'  # Directory containing images
ANNOTATION_FILE = 'dataset_1.0/person_keypoints.json'  # COCO keypoints JSON
OUTPUT_VIDEO = 'output_video.mp4'  # Output video file
FPS = 20  # Frames per second
OUTPUT_SIZE = (1280, 720)  # Width, Height for video

def overlay_keypoints(image, annotations, image_id):
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            keypoints = annotation['keypoints']
            num_keypoints = len(keypoints) // 3

            for i in range(num_keypoints):
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                confidence = keypoints[i * 3 + 2]

                if confidence > 0:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    return image

def create_video_from_annotations(image_dir, annotation_file, output_video, fps, output_size):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images_info = {img['file_name']: img['id'] for img in data['images']}
    annotations = data['annotations']

    image_filenames = sorted([f for f in os.listdir(image_dir) if f in images_info])
    
    print(f"Number of images in directory: {len(image_filenames)}")
    print(f"Number of images in annotation file: {len(images_info)}")

    if not image_filenames:
        print("No images found in the directory.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, output_size)

    for filename in image_filenames:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping unreadable image: {filename}")
            continue

        image_id = images_info.get(filename)
        if image_id is None:
            continue

        image_with_keypoints = overlay_keypoints(image, annotations, image_id)

        # Resize to fixed output size for consistent video dimensions
        resized_frame = cv2.resize(image_with_keypoints, output_size)

        video_writer.write(resized_frame)

    video_writer.release()
    print(f"✅ Video saved: {output_video}")

if __name__ == "__main__":
    create_video_from_annotations(IMAGE_DIR, ANNOTATION_FILE, OUTPUT_VIDEO, FPS, OUTPUT_SIZE)