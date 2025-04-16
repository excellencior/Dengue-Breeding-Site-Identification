"""
Pixel wise accuracy for segmentation 
"""
import os
import cv2
import numpy as np

def rename_files(directory):
    """
    Would rename images into imageno_slicetop.x-sliceleft.y
    """
    files = os.listdir(directory)
    
    # Extract slice positions and filenames
    file_info = []
    for filename in files:
        parts = filename.split('_')[2].split('-')
        x, y = int(parts[1]), int(parts[2])
        file_info.append((x, y, filename))
    
    # Sort based on x and then y
    file_info.sort(key=lambda info: (info[0], info[1]))
    
    # Rename files in sorted order
    for index, (x, y, filename) in enumerate(file_info, start=1):
        new_filename = f"{index}_{x}-{y}{os.path.splitext(filename)[1]}"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


def seg_accuracy_pixelwise(prediction_folder, ground_truth_folder, model_type="YOLO") -> tuple:
    """
    Image Format: Both image should represent the masks in which segmentation regions with dark background
    """
    # Initialize accumulators
    total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

    # Absolute path conversion
    prediction_folder = os.path.abspath(prediction_folder)
    ground_truth_folder = os.path.abspath(ground_truth_folder)

    for filename in os.listdir(prediction_folder):
        # Load model output and ground truth
        model_output = cv2.imread(os.path.join(prediction_folder, filename), cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(os.path.join(ground_truth_folder, filename), cv2.IMREAD_GRAYSCALE)

        if model_type == "yolo":
            _, pred_binary = cv2.threshold(model_output, 250, 255, cv2.THRESH_BINARY)
        else:
            # For model = SegGPT
            _, binary_output = cv2.threshold(model_output, 110, 255, cv2.THRESH_BINARY)
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            pred_binary = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel)            

        # Apply binary thresholding to the ground truth
        _, gt_binary = cv2.threshold(ground_truth, 250, 255, cv2.THRESH_BINARY)

        # Flatten the arrays
        pred_binary_flat = pred_binary.flatten()
        gt_binary_flat = gt_binary.flatten()
        
        # Calculate TP, TN, FP, FN for this image
        TP = np.sum((pred_binary_flat == 255) & (gt_binary_flat == 255))
        TN = np.sum((pred_binary_flat == 0) & (gt_binary_flat == 0))
        FP = np.sum((pred_binary_flat == 255) & (gt_binary_flat == 0))
        FN = np.sum((pred_binary_flat == 0) & (gt_binary_flat == 255))
      
        # Accumulate the results
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

    # Calculate precision and recall
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) if (total_TP + total_TN + total_FP + total_FN) > 0 else 0

    # with open('seg_accuracies.txt', 'a') as f:
    #     if model_type == "YOLO":
    #         f.write(f'------- Model : YOLO {output_folder[-3:]} -------\n')
    #     else:
    #         f.write(f'------- Model : SegGPT -------\n')
            
    #     f.write(f'Precision: {precision}\n')
    #     f.write(f'Recall: {recall}\n')
    #     f.write(f'Accuracy: {accuracy:.4f}\n\n')

    return precision, recall, accuracy

