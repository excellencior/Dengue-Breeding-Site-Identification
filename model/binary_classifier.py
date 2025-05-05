# For each of the building (found from zLabels/building_labels.txt) compare each of the objects (found from zLabels/object_labels.txt) and find the object with the highest IOU. If at least one object is found in the area of a building mask that building risky (class_id = 1), otherwise safe (class_id = 0). Save that building information (class_id building_bbox) to a new file (zLabels/building_labels_final.txt).

# NB: The building bbox format (rotated): [top_left_x top_left_y top_right_x top_right_y bottom_right_x bottom_right_y bottom_left_x bottom_left_y] and the object bbox format: [class_id center_x center_y width height]

# class_id map (object) => 0: construction_site, 1: flower_pot, 2: open_tank, 4: polythene, 5: reservoir, 6: tyres
"""
    Building Binary Classification into Risky(1) and Safe(0) based on the objects present on top of the building
"""
import os
import cv2
import numpy as np

def read_building_labels(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        labels.append(list(map(float, line.strip().split())))
    
    # Using min area rect format for buildings
    building_labels = []
    for label in labels:
        x1, y1, x2, y2, x3, y3, x4, y4 = label
        points = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
        building_labels.append(cv2.minAreaRect(points))
    return building_labels # FORMAT: ((center_x, center_y), (width, height), angle)

def read_object_labels(file_path):
    with open(file_path, 'r') as f:
        object_labels = f.readlines()
    return [list(map(float, obj.strip().split())) for obj in object_labels]

def is_within_building(building, obj):
    _, center_x, center_y, width, height = obj[0:5]
    obj_bbox = ((center_x, center_y), (width, height), 0)  # No rotation for objects

    obj_area = width * height
    
    inter_area = cv2.rotatedRectangleIntersection(obj_bbox, building)[1]
    if inter_area is None:
        return False
    
    overlap_area = cv2.contourArea(inter_area)
    overlap_ratio = overlap_area / obj_area
    if overlap_ratio > 0.9:
        return True
    return False


def contains_objects(building, objects, processed_objects):
    """
        Check if the building contains any of the objects
    """
    contain = False
    for obj in objects:
        obj_tuple = tuple(obj)  # Convert the object to a tuple for hashability
        if obj_tuple in processed_objects:
            continue

        if is_within_building(building, obj):
            contain = True
            processed_objects.add(obj_tuple)  # Add the tuple to the set
    return contain


# ------------------------------------=================---------------------------------
def classify(building_labels_path, object_labels_path, output_labels_path):
    """
        Classify the buildings into risky and safe based on the objects present on top of the building
    """
    # Absolute path conversion
    building_labels_path = os.path.abspath(building_labels_path)
    object_labels_path = os.path.abspath(object_labels_path)
    output_labels_path = os.path.abspath(output_labels_path)

    building_labels = read_building_labels(building_labels_path)
    object_labels = read_object_labels(object_labels_path)

    processed_objects = set()
    number_of_risky_buildings, number_of_safe_buildings = 0, 0

    with open(output_labels_path, 'w') as f_out:
        for building in building_labels:            
            risky = 0
            
            if contains_objects(building, object_labels, processed_objects):
                risky = 1 # Building is risky if it contains any (At least one) object
            
            if risky:
                number_of_risky_buildings += 1
            else:
                number_of_safe_buildings += 1
            
            # Save the building class and bounding box to the final file
            building = cv2.boxPoints(building).flatten().tolist()
            building = [round(pos) for pos in building]
            f_out.write(f"{risky} {' '.join(map(str, building))}\n")
            
    print(len(processed_objects), '/', len(object_labels), 'objects are on top of buildings')
    print(f"Number of risky buildings: {number_of_risky_buildings}")
    print(f"Number of safe buildings: {number_of_safe_buildings}")



