import json
import os
import cv2
import numpy as np

# Path configurations
IMAGE_DIR = '../merged_dataset/input_frames'  # Directory containing images
ANNOTATION_FILE = '../merged_dataset/person_bboxes.json'  # COCO keypoints JSON
OUTPUT_VIDEO = '../bboxes_overlay.mp4'  # Output video file
FPS = 10  # Frames per second
OUTPUT_SIZE = (1280, 720)  # Width, Height for video

def overlay_keypoints_and_bbox(image, annotations, image_id):
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            # Overlay bounding box
            bbox = annotation.get('bbox', [])
            if bbox and len(bbox) >= 4:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = bbox[:4]
                x, y, w, h = int(x), int(y), int(w), int(h)
                # Draw bounding box rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Add bbox score if available
                bbox_score = annotation.get('bbox_score', 1.0)
                if bbox_score < 1.0:
                    cv2.putText(image, f'{bbox_score:.2f}', (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Overlay keypoints
            keypoints = annotation.get('keypoints', [])
            if keypoints:
                num_keypoints = len(keypoints) // 3

                for i in range(num_keypoints):
                    x = keypoints[i * 3]
                    y = keypoints[i * 3 + 1]
                    confidence = keypoints[i * 3 + 2]

                    if confidence > 0:
                        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    return image

def resize_with_padding(image, target_size, bg_color=(0, 0, 0)):
    """
    Resize image to fit within target_size while maintaining aspect ratio.
    Adds padding to fill the remaining space with bg_color.
    """
    target_width, target_height = target_size
    img_height, img_width = image.shape[:2]
    
    # Calculate scaling factor to fit image within target size
    scale = min(target_width / img_width, target_height / img_height)
    
    # Calculate new dimensions
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size and background color
    canvas = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    
    # Calculate position to center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas

def create_video_from_annotations(image_dir, annotation_file, output_video, fps, output_size):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images_info = {img['file_name']: img['id'] for img in data['images']}
    annotations = data['annotations']

    image_filenames = sorted([f for f in os.listdir(image_dir) if f in images_info])
    
    print(f"Number of images in directory: {len(image_filenames)}")
    print(f"Number of images in annotation file: {len(images_info)}")
    print(f"Output video size: {output_size[0]}x{output_size[1]}")

    if not image_filenames:
        print("No images found in the directory.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, output_size)

    for i, filename in enumerate(image_filenames):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping unreadable image: {filename}")
            continue

        image_id = images_info.get(filename)
        if image_id is None:
            continue

        # Overlay keypoints and bounding boxes on the original image
        image_with_overlay = overlay_keypoints_and_bbox(image, annotations, image_id)
        
        # Get original image dimensions for logging
        orig_height, orig_width = image.shape[:2]
        
        # Resize with padding to maintain aspect ratio
        final_frame = resize_with_padding(image_with_overlay, output_size)
        
        video_writer.write(final_frame)
        
        # Log progress for first few frames
        if i < 5:
            print(f"Frame {i+1}: {filename} ({orig_width}x{orig_height}) → {output_size[0]}x{output_size[1]}")

    video_writer.release()
    print(f"✅ Video saved: {output_video}")
    print(f"Processed {len(image_filenames)} frames with seamless size handling")

if __name__ == "__main__":
    create_video_from_annotations(IMAGE_DIR, ANNOTATION_FILE, OUTPUT_VIDEO, FPS, OUTPUT_SIZE)