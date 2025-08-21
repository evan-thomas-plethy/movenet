import os
import cv2
import hashlib

# Define directories
TRAIN_DIR = 'active/sitting-lying_train/'
VAL_DIR = 'active/sitting-lying_val/'

def compute_image_hash(image_path):
    """Computes a hash for an image file to compare image content."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return hashlib.md5(image.tobytes()).hexdigest()  # Hash image data
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def get_image_hashes(directory):
    """Returns a dictionary mapping image hashes to filenames in a directory."""
    image_hashes = {}
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image_hash = compute_image_hash(image_path)
        if image_hash:
            image_hashes.setdefault(image_hash, []).append(filename)
    return image_hashes

# Compute hashes for train and val images
train_hashes = get_image_hashes(TRAIN_DIR)
val_hashes = get_image_hashes(VAL_DIR)

# Find identical images by matching hashes
common_hashes = set(train_hashes.keys()) & set(val_hashes.keys())

# Print results
print(f"Total images in {TRAIN_DIR}: {sum(len(v) for v in train_hashes.values())}")
print(f"Total images in {VAL_DIR}: {sum(len(v) for v in val_hashes.values())}")
print(f"Identical images found in both directories: {len(common_hashes)}")

if common_hashes:
    print("\nIdentical images (same content, different filenames possible):")
    for img_hash in common_hashes:
        print(f"Hash: {img_hash}")
        print(f"  Train files: {train_hashes[img_hash]}")
        print(f"  Val files: {val_hashes[img_hash]}\n")