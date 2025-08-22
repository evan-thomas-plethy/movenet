import os

dir1 = "dataset_1.0/input_frames"
dir2 = "dataset_2.0/input_frames"

files_1 = set(os.listdir(dir1))
files_2 = set(os.listdir(dir2))

overlap = files_1 & files_2

if overlap:
    print(f"⚠️ {len(overlap)} overlapping image filenames found:")
    for fname in sorted(overlap):
        print(f"  - {fname}")
else:
    print("✅ No overlapping image filenames found.")