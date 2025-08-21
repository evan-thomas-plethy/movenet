import os
import re
import json

# List your augmentations here
augmentations = ['blur', 'brightness', 'crop', 'darkness', 'flip', 'hue', 'noise', 'rotate', 'scale']

def clean_filename(filename):
    # Remove the last _XXXX (digits) before .jpg, but not if preceded by 'frame'
    # e.g. NEW_HEEL_SLIDES_8_frame_0279_1182.jpg -> NEW_HEEL_SLIDES_8_frame_0279.jpg
    #      NEW_HEEL_SLIDES_5_frame_0000_1.jpg -> NEW_HEEL_SLIDES_5_frame_0000.jpg
    #      NEW_HEEL_SLIDES_5_frame_0000.jpg -> unchanged
    pattern = r'(?<!frame)(_\d+)(?=\.jpg$)'
    return re.sub(pattern, '', filename)

def process_directory(aug_name):
    dir_path = os.path.join(os.getcwd(), aug_name)
    if not os.path.isdir(dir_path):
        print(f"Directory {dir_path} does not exist, skipping.")
        return

    for fname in os.listdir(dir_path):
        if fname.lower().endswith('.jpg'):
            new_fname = clean_filename(fname)
            if new_fname != fname:
                src = os.path.join(dir_path, fname)
                dst = os.path.join(dir_path, new_fname)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                    print(f"Renamed: {fname} -> {new_fname}")
                else:
                    print(f"Target file {new_fname} already exists, skipping.")

def process_json(aug_name):
    json_path = os.path.join(os.getcwd(), f'aug_{aug_name}.json')
    if not os.path.isfile(json_path):
        print(f"JSON file {json_path} does not exist, skipping.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    changed = False
    if 'images' in data:
        for img in data['images']:
            old_name = img.get('file_name', '')
            new_name = clean_filename(old_name)
            if new_name != old_name:
                img['file_name'] = new_name
                changed = True

    if changed:
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print(f"Updated file names in {json_path}")

if __name__ == "__main__":
    for aug in augmentations:
        process_directory(aug)
        process_json(aug)