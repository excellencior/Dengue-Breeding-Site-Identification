"""
Overlaying building & object bboxes on the original orthophoto
"""
import os
import cv2
import numpy as np

# Define color and thickness for bounding boxes
RED = (0, 0, 255)  # Red color for risky buildings
GREEN = (0, 255, 0)  # Green color for safe buildings
VIOLET = (255, 0, 255)

# Read labels files and print the labels on the orthophoto
def read_building_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            values = line.strip().split()
            labels.append(list(map(float, values)))
    return labels

def read_object_lables(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            label = np.array(line.strip().split(), dtype=np.float32)
            class_id = int(label[0])
            center_x = label[1]
            center_y = label[2]
            width = label[3]
            height = label[4]
            x_min = center_x - width / 2
            y_min = center_y - height / 2
            x_max = center_x + width / 2
            y_max = center_y + height / 2
            points = np.array([class_id, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max], dtype=np.int32)
            labels.append(points)
    return labels

def draw(org_image_path: str, building_labels_path: str, object_labels_path: None, output_path: str):
    # Absolute Path Conversion
    org_image_path = os.path.abspath(org_image_path)
    building_labels_path = os.path.abspath(building_labels_path)
    output_path = os.path.abspath(output_path)

    org_image = cv2.imread(org_image_path)
    building_labels = read_building_labels(building_labels_path)
    object_labels = []
    if object_labels_path:
        object_labels_path = os.path.abspath(object_labels_path)
        object_labels = read_object_lables(object_labels_path)

    store_img = org_image.copy()
    for label in building_labels:
        # class_id = 
        class_id = int(label[0])
        label = label[1:]
        points = np.array(label).reshape(-1, 2).astype(np.int32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        if class_id == 1:
            cv2.fillPoly(org_image, [box], (0, 0, 255))
        else:
            cv2.fillPoly(org_image, [box], VIOLET)
        cv2.polylines(store_img, [points], isClosed=True, color=RED, thickness=6)
    
    for label in object_labels:
        class_id = int(label[0])
        points = np.array(label[1:]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(org_image, [points], (0, 0, 255))
        cv2.polylines(store_img, [points], isClosed=True, color=(0, 0, 255), thickness=4)
        
    alpha = 0.2
    output_img = cv2.addWeighted(store_img, 1, org_image, alpha, 0)
    cv2.imwrite(output_path, output_img)
    print(f"Bounding boxes drawn and saved to {output_path}")


def draw_canvas(org_image_path: str, building_labels_path: str, object_labels_path: str, output_path: str):
    # Absolute Path Conversion
    org_image_path = os.path.abspath(org_image_path)
    building_labels_path = os.path.abspath(building_labels_path)
    object_labels_path = os.path.abspath(object_labels_path)
    output_path = os.path.abspath(output_path)

    # Read the original image to get its dimensions
    org_image = cv2.imread(org_image_path)
    height, width, channels = org_image.shape

    # Create a blank canvas with the same dimensions as the original image
    canvas = np.zeros((height, width, 3), dtype=np.uint8)  # Black canvas
    # canvas.fill(255)  # Uncomment this to make the canvas white instead

    # Read labels
    building_labels = read_building_labels(building_labels_path)
    object_labels = read_object_lables(object_labels_path)

    # Draw building bounding boxes
    for label in building_labels:
        class_id = 2
        points = np.array(label).reshape(-1, 2).astype(np.int32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        if class_id == 1:
            cv2.fillPoly(canvas, [box], (0, 0, 255))  # Red for class_id 1
        else:
            cv2.fillPoly(canvas, [box], VIOLET)  # Violet for other classes
        cv2.polylines(canvas, [points], isClosed=True, color=RED, thickness=6)

    # Draw object bounding boxes
    for label in object_labels:
        class_id = int(label[0])
        points = np.array(label[1:]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(canvas, [points], (0, 0, 255))  # Red fill
        cv2.polylines(canvas, [points], isClosed=True, color=RED, thickness=4)

    # Save the canvas as the output image
    cv2.imwrite(output_path, canvas)
    print(f"Bounding boxes drawn on blank canvas and saved to {output_path}")

def draw_segmentation(image_path, label_path, output_path, bbox = False, opacity=0.3):
    VIOLET = (238, 130, 238) 
    RED = (0, 0, 255)  # Define Red color (BGR format)
    BLACK = (0, 0, 0)  # Define Black color (BGR format)
    WHITE = (255, 255, 255)  # Define White color (BGR format)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    overlay = image.copy()

    with open(label_path, 'r') as file:
        for line in file:
            label = list(map(float, line.strip().split()))
            if len(label) < 4:  # Check if there are enough points
                continue
            points = np.array(label[:-1]).reshape(-1, 2).astype(np.int32)  # Exclude the conf score
            
            if bbox:
                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.fillPoly(overlay, [box], VIOLET)
                cv2.polylines(image, [box], isClosed=True, color=WHITE, thickness=10)

            else:
                cv2.fillPoly(overlay, [points], VIOLET)
                cv2.polylines(image, [points], isClosed=True, color=WHITE, thickness=10)

    blended = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

    cv2.imwrite(output_path, blended)
    print(f"Segments drawn and saved to {output_path}")


