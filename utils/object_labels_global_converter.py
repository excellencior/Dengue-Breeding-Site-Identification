"""
Converting Annotated, Predicted Object Labels to Global Coordinates (In original orthophoto coordinate system)
"""

import os
# Function to read YOLOv9 label save_to and convert to pixel coordinates
def read_yolo_labels(label_file, image_width, image_height):
    with open(label_file, 'r') as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            center_x = float(parts[1]) * image_width
            center_y = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
            conf_score = float(parts[5])  # Extract the confidence score
            labels.append((class_id, center_x, center_y, width, height, conf_score))
        return labels

save_to = open('predicted_object_labels_v9gelanc.txt', 'w')
labels_dir = 'Objects/detect/exp2/labels'
# iterate over yolov9 labels gt_test_yolov9/labels
for label_file in os.listdir(labels_dir):
    image_width, image_height = 512, 512
    x, y = map(float, label_file.split('_')[2].split('-')[1:3])
    label_path = os.path.join(labels_dir, label_file)
    object_labels = read_yolo_labels(label_path, image_width, image_height)
    # iterate over object labels
    for label in object_labels:
        class_id, center_x, center_y, width, height, conf_score = label
        if conf_score < 0.14:
            continue
        # now split the label save_to
        center_x = center_x + x
        center_y = center_y + y
        save_to.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

save_to.close()
