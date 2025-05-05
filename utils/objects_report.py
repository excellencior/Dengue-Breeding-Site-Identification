import os
import csv

# List of class names
class_names = ["construction_site", "flower_pot", "open_tank", "polythene", "reservoir", "tyres"]

# Directory containing label files
gt_labels_dir = os.path.abspath('resources/Objects/gt_yolov9/labels')
pred_labels_dir = os.path.abspath('resources/Objects/detect/exp2/labels')

# pred_no_augmentation_labels_dir = 'detect/augmentation_v4/labels'
confidence_threshold = 0.14

def desc(labels_dir):
    # Dictionary to store counts of each class
    class_counts = {class_name: 0 for class_name in class_names}
    # Process each label file
    for label_file in os.listdir(labels_dir):
        label_file_path = os.path.join(labels_dir, label_file)
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                if "gt" not in labels_dir:
                    confidence = float(parts[5])
                    if confidence < confidence_threshold:
                        continue
                class_name = class_names[class_id]
                class_counts[class_name] += 1

    # Print the counts of each class
    if "gt" not in labels_dir:
        print(f"Confidence threshold: {confidence_threshold}\n---------------------------")
    print(f'Total labels: {sum(class_counts.values())}')
    for class_name, count in class_counts.items():
        print(f"{class_name} : {count}")

# # Process ground truth labels
print("Ground Truth Labels")
print("*******************")
desc(gt_labels_dir)

# Process predicted labels
print("\nPredicted Labels")
print("****************")
desc(pred_labels_dir)

# Ground Truth Labels
# *******************
# construction_site : 23
# flower_pot : 2707
# open_tank : 16
# polythene : 97
# reservoir : 5
# tyres : 33