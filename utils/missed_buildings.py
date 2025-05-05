import numpy as np
from shapely.geometry import Polygon
import os

def load_buildings(filename):
    """Load building coordinates from file and return as list of numpy arrays."""
    buildings = []
    with open(filename, 'r') as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            buildings.append(np.array(coords))
    return buildings

def load_objects(filename):
    """Load object data from file and return as list of [center_x, center_y, width, height]."""
    objects = []
    with open(filename, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            # If class_id is included in the data, skip it
            if len(data) == 5:
                data = data[1:]
            objects.append(data)
    return objects

def overlap_ratio(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes."""
    poly1 = Polygon(np.array(bbox1).reshape(4, 2))
    poly2 = Polygon(np.array(bbox2).reshape(4, 2))
    if poly1.area == 0 or poly2.area == 0:
        return 0
    return poly1.intersection(poly2).area / poly1.area

def object_to_bbox(obj):
    """Convert object [center_x, center_y, width, height] to corner coordinates."""
    cx, cy, w, h = obj
    half_w, half_h = w/2, h/2
    return [
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h]
    ]

def find_missed_buildings_with_objects(ground_truth_file, prediction_file, objects_file, 
                                     overlap_threshold=0.5, output_file="missed_buildings.txt"):
    """
    Find buildings that contain objects but were missed in prediction.
    
    Args:
        ground_truth_file: Path to ground truth buildings file
        prediction_file: Path to predicted buildings file
        objects_file: Path to objects file
        overlap_threshold: Minimum overlap ratio to consider a prediction match
        output_file: Path to output file for missed buildings
    """
    # Load data
    gt_buildings = load_buildings(ground_truth_file)
    pred_buildings = load_buildings(prediction_file)
    objects = load_objects(objects_file)
    
    # Find buildings with objects
    buildings_with_objects = []
    for gt_idx, gt_building in enumerate(gt_buildings):
        has_object = False
        for obj in objects:
            obj_bbox = object_to_bbox(obj)
            if overlap_ratio(gt_building, obj_bbox) > 0:  # Any overlap means object is in building
                has_object = True
                break
        
        if has_object:
            # Check if this building was missed in predictions
            was_detected = False
            for pred_building in pred_buildings:
                if overlap_ratio(gt_building, pred_building) > overlap_threshold:
                    was_detected = True
                    break
            
            if not was_detected:
                buildings_with_objects.append(gt_building)
    
    # Save missed buildings to file
    with open(output_file, 'w') as f:
        for building in buildings_with_objects:
            coords = ' '.join(map(str, building))
            f.write(f"{coords}\n")
    
    print(f"Found {len(buildings_with_objects)} missed buildings containing objects.")
    print(f"Results saved to {output_file}")
    
    return buildings_with_objects

# Kept for argument purpose
if __name__ == "__main__":
    ground_truth_file = os.path.abspath("resources/ClassifierOutput/Building Classification/Ground Truth/labels.txt")
    prediction_file = os.path.abspath("resources/ClassifierOutput/Building Classification/YOLO-v8m/labels.txt")
    objects_file = os.path.abspath("resources/ClassifierOutput/Object Detection/Ground Truth/object_labels.txt")
    output_file = os.path.abspath("output/missed_buildings/missed_buildings.txt")
    
    missed_buildings = find_missed_buildings_with_objects(
        ground_truth_file=ground_truth_file,
        prediction_file=prediction_file,
        objects_file=objects_file,
        overlap_threshold=0.5,  # Adjust this threshold as needed
        output_file=output_file
    )