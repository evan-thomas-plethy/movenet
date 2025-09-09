#!/usr/bin/env python3
"""
Script to recalculate bounding boxes from COCO keypoint annotations.
Reads person_keypoints.json and creates person_bboxes.json with updated bboxes
based on visible keypoints (visibility = 1 or 2).
"""

import json
import os
import sys
from typing import List, Tuple, Dict, Any


def calculate_bbox_from_keypoints(keypoints: List[float]) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from keypoints array.
    
    Args:
        keypoints: List of [x, y, visibility] for each keypoint (17 keypoints = 51 values)
        
    Returns:
        Tuple of (x, y, width, height) for the bounding box
    """
    # COCO person keypoints: 17 keypoints, each with [x, y, visibility]
    # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
    
    visible_points = []
    
    # Extract visible keypoints (visibility = 1 or 2)
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, vis = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if vis in [1, 2]:  # visible keypoints
                visible_points.append((x, y))
    
    if not visible_points:
        # If no visible keypoints, return a default small bbox
        return 0.0, 0.0, 1.0, 1.0
    
    # Calculate bounding box from visible points
    x_coords = [point[0] for point in visible_points]
    y_coords = [point[1] for point in visible_points]
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    # COCO bbox format: [x, y, width, height]
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # Add margin on all sides
    margin_x = bbox_width * 0.10
    margin_y = bbox_height * 0.05
    
    bbox_x = max(0, x_min - margin_x)  # Ensure x doesn't go negative
    bbox_y = max(0, y_min - margin_y)  # Ensure y doesn't go negative
    bbox_width = bbox_width + (2 * margin_x)
    bbox_height = bbox_height + (2 * margin_y)
    
    return bbox_x, bbox_y, bbox_width, bbox_height


def recalculate_bboxes(input_file: str, output_file: str) -> None:
    """
    Recalculate bounding boxes for all annotations based on visible keypoints.
    
    Args:
        input_file: Path to input COCO keypoints JSON file
        output_file: Path to output COCO JSON file with updated bboxes
    """
    print(f"Reading annotations from: {input_file}")
    
    # Load the COCO dataset
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Found {len(coco_data.get('annotations', []))} annotations")
    
    # Process each annotation
    updated_count = 0
    for annotation in coco_data.get('annotations', []):
        if 'keypoints' in annotation and len(annotation['keypoints']) > 0:
            # Calculate new bounding box from keypoints
            new_bbox = calculate_bbox_from_keypoints(annotation['keypoints'])
            
            # Update the annotation
            annotation['bbox'] = list(new_bbox)
            
            # Recalculate area based on new bbox
            annotation['area'] = new_bbox[2] * new_bbox[3]
            
            updated_count += 1
    
    print(f"Updated {updated_count} annotations with new bounding boxes")
    
    # Save the updated dataset
    print(f"Saving updated annotations to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("Done!")


def main():
    """Main function to run the bbox recalculation."""
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "merged_dataset", "person_keypoints.json")
    output_file = os.path.join(script_dir, "..", "merged_dataset", "person_bboxes.json")
    
    # Convert to absolute paths
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Recalculate bounding boxes
    recalculate_bboxes(input_file, output_file)


if __name__ == "__main__":
    main()
