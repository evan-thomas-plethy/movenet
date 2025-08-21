import os
import cv2

val_dir = "active/val/"
output_dir = "flipped_val/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all images in val_dir
for filename in os.listdir(val_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(val_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Skipping {filename} (not readable)")
            continue

        # Flip horizontally (1 = horizontal, 0 = vertical, -1 = both)
        flipped = cv2.flip(img, 1)

        # Save to flipped_val/ with same filename
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, flipped)

print(f"✅ Flipped images saved to {output_dir}")