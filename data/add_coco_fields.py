import json

def add_fields(json_path, output_path=None):
    # Load the COCO annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    data['info'] = {
        "year": "2025",
        "version": "19",
        "description": "Exported from roboflow.com",
        "contributor": "",
        "url": "https://app.roboflow.com/datasets/vision-ai-b6jhb/19",
        "date_created": "2025-04-04T19:20:07.635798Z"
    },
    data['licenses'] = [
        {
            "id": 1,
            "url": "",
            "name": "Unknown"
        }
    ],
    data['categories'] = [
        {
            "id": 0,
            "name": "poses",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "person",
            "supercategory": "poses",
            "keypoints": [
                "nose",
                "neck",
                "left-shoulder",
                "left-elbow",
                "left-wrist",
                "right-shoulder",
                "right-elbow",
                "right-wrist",
                "left-hip",
                "left-knee",
                "left-ankle",
                "right-hip",
                "right-knee",
                "right-ankle",
                "left-eye",
                "right-eye",
                "right-ear",
                "left-ear"
            ],
            "skeleton": [
                [
                    1,
                    16
                ],
                [
                    3,
                    4
                ],
                [
                    4,
                    5
                ],
                [
                    16,
                    17
                ],
                [
                    6,
                    7
                ],
                [
                    7,
                    8
                ],
                [
                    10,
                    9
                ],
                [
                    9,
                    2
                ],
                [
                    15,
                    18
                ],
                [
                    12,
                    13
                ],
                [
                    13,
                    14
                ],
                [
                    1,
                    15
                ],
                [
                    3,
                    2
                ],
                [
                    2,
                    1
                ],
                [
                    2,
                    6
                ],
                [
                    2,
                    12
                ],
                [
                    11,
                    10
                ]
            ]
        }
    ]
    # Iterate through each annotation
    for ann in data.get('annotations', []):
        keypoints = ann.get('keypoints', [])
        if keypoints:
            # Count visible keypoints (visibility > 0)
            num_visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
            ann['num_keypoints'] = num_visible
        else:
            ann['num_keypoints'] = 0

    # Write the updated JSON
    out_path = json_path
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated annotations saved to {out_path}")

# Example usage
if __name__ == "__main__":
    input_json = 'active/annotations/active_val.json'  # Replace with your actual path
    add_fields(input_json)