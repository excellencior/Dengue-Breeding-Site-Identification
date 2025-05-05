"""
After combined building labels are generated, this script is used to process the labels
Sliding Window --- Inferenced on slices with 50% overlap
non max supression NMM --- reducing number of false positives
"""
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns


def read_labels(file_path):
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            values = line.strip().split()
            bbox = list(map(float, values))
            bboxes.append(bbox)
    return bboxes

# ---------------- x Added to check if this works better than iou x ---------------- #
def overlap_ratio(bbox1, bbox2):
    poly1 = Polygon(np.array(bbox1).reshape(4, 2))
    poly2 = Polygon(np.array(bbox2).reshape(4, 2))
    if poly1.area == 0 or poly2.area == 0:
        return 0
    return poly1.intersection(poly2).area / poly1.area

def get_aspect_ratio(bbox):
    # format of bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
    bbox = np.array(bbox).reshape(4, 2)
    width = np.linalg.norm(bbox[0] - bbox[1])
    height = np.linalg.norm(bbox[1] - bbox[2])
    if min(width, height) == 0:
        return max(width, height)
    return max(width, height) / min(width, height)

def rotated_iou(bbox1, bbox2):
    if len(bbox1) != 8 or len(bbox2) != 8:
        print("Problematic bbox1: ", bbox1)
        print("Problematic bbox2: ", bbox2, "\n")
    poly1 = Polygon(np.array(bbox1).reshape(4, 2))
    poly2 = Polygon(np.array(bbox2).reshape(4, 2))
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    # Avoid division by zero
    if union == 0:
        return 0
    return intersection / union

def calculate_area(bbox):
    return Polygon(np.array(bbox).reshape(4, 2)).area

def adjusted_confidence(bbox):
    # Confidence adjusted by area
    area = calculate_area(bbox['pos'])
    conf_score = bbox['conf']
    return conf_score * area  # Or use sqrt(area) * conf_score if desired

def merge_rotated_bboxes(bbox1, bbox2):
    # Convert bboxes to the format expected by cv2.minAreaRect
    points1 = np.array(bbox1).reshape(4, 2)
    points2 = np.array(bbox2).reshape(4, 2)
    # Combine points from both bboxes
    all_points = np.vstack([points1, points2]).astype(np.int32)    
    # Find the minimum area rectangle
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    # Convert back to the original format
    return box.flatten().tolist()

def check_alignment(bbox1, bbox2):
    # Check if bboxes are aligned (you may need to adjust this based on your specific alignment criteria)
    points1 = np.array(bbox1).reshape(4, 2)
    points2 = np.array(bbox2).reshape(4, 2)
    
    # Check if any of the sides are parallel
    for i in range(4):
        vec1 = points1[(i+1)%4] - points1[i]
        for j in range(4):
            vec2 = points2[(j+1)%4] - points2[j]
            if np.abs(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) > 0.95:  # Allow for small angle differences
                return True
    return False

def align_bboxes(bbox1, bbox2):
    # Implement bbox alignment logic here
    # For simplicity, we'll just merge them as before
    return merge_rotated_bboxes(bbox1, bbox2)

def post_process_containment(bboxes, containment_threshold=0.6):
    """
    containment_threshold: minimum overlap ratio for a bbox to be considered contained within another bbox (v8m = 0.8, v11m = 0.6)
    """
    final_bboxes = []
    sorted_bboxes = sorted(bboxes, key=lambda x: calculate_area(x), reverse=True)
    
    while sorted_bboxes:
        current_bbox = sorted_bboxes.pop(0)
        current_poly = Polygon(np.array(current_bbox).reshape(4, 2))
        
        contained = []
        i = 0
        while i < len(sorted_bboxes):
            other_bbox = sorted_bboxes[i]
            other_poly = Polygon(np.array(other_bbox).reshape(4, 2))
            
            # Avoid division by zero
            if other_poly.area == 0:
                i += 1
                continue
            overlap_ratio = other_poly.intersection(current_poly).area / other_poly.area
            
            if overlap_ratio > containment_threshold:
                contained.append(other_bbox)
                sorted_bboxes.pop(i)
            else:
                i += 1
        
        final_bboxes.append(current_bbox)
    
    # # delete bboxes with aspect ratio more than 4
    # final_bboxes = [bbox for bbox in final_bboxes if get_aspect_ratio(bbox) < 5]
    # # delete bbox with area less than 0.086x10^6
    # final_bboxes = [bbox for bbox in final_bboxes if calculate_area(bbox) > 0.086 * 10**6]

    return final_bboxes

def process_bounding_boxes(bboxes, or_threshold=0.4):
    """
        Process bounding boxes using a combination of NMM and alignment-based merging for model = SegGPT
    """
    merged_bboxes = []
    
    # Sort bboxes by area in descending order
    sorted_bboxes = sorted(bboxes, key=lambda x: calculate_area(x), reverse=True)
    
    while sorted_bboxes:
        current_bbox = sorted_bboxes.pop(0)
        
        i = 0
        while i < len(merged_bboxes):
            other_bbox = merged_bboxes[i]
            # iou = rotated_iou(current_bbox, other_bbox)
            or1 = overlap_ratio(current_bbox, other_bbox)
            or2 = overlap_ratio(other_bbox, current_bbox)
            final_or = max(or1, or2)
            
            if 0.2 <= final_or < or_threshold:
                if check_alignment(current_bbox, other_bbox):
                    merged_bbox = align_bboxes(current_bbox, other_bbox)
                    merged_bboxes[i] = merged_bbox
                    break
            elif final_or >= or_threshold:
                if calculate_area(current_bbox) > calculate_area(other_bbox):
                    merged_bboxes[i] = current_bbox
                break
            i += 1
        else:
            # If no overlap or alignment found, add the current bbox to merged_bboxes
            merged_bboxes.append(current_bbox)
    
    return merged_bboxes

def area_based_merging(bboxes, iou_threshold=0.5):
    """
        Process bounding boxes using a combination of NMM and alignment-based merging for model = YOLO
    """
    merged_bboxes = []
    
    # Sort bounding boxes by adjusted confidence score in descending order
    sorted_bboxes = sorted(bboxes, key=adjusted_confidence, reverse=True)

    for current_bbox in sorted_bboxes:
        i = 0
        while i < len(merged_bboxes):
            other_bbox = merged_bboxes[i]

            # Calculate IOU using only the positions
            iou1 = overlap_ratio(current_bbox['pos'], other_bbox['pos'])
            iou2 = overlap_ratio(other_bbox['pos'], current_bbox['pos'])
            iou = max(iou1, iou2)
            
            if 0.35 <= iou < iou_threshold:
                if check_alignment(current_bbox['pos'], other_bbox['pos']):
                    # Align and merge bounding boxes
                    merged_bbox_pos = align_bboxes(current_bbox['pos'], other_bbox['pos'])
                    # Update merged box in merged_bboxes with new position and higher confidence
                    merged_bboxes[i] = {"pos": merged_bbox_pos, "conf": max(current_bbox['conf'], other_bbox['conf'])}
                    break
            elif iou >= iou_threshold:
                # Keep the box with higher adjusted confidence
                if adjusted_confidence(current_bbox) > adjusted_confidence(other_bbox):
                    merged_bboxes[i] = current_bbox
                break
            i += 1
        else:
            # If no overlap/alignment found, add current box to merged_bboxes
            merged_bboxes.append(current_bbox)

    merged_bboxes = [bbox['pos'] for bbox in merged_bboxes]
    
    return merged_bboxes


def process_sliding_window(label_dir, save_dir, SEG_MODEL='yolo'):
    """
        Process bounding boxes generated from sliding window approach [50% overlap]
    """
    label_dir = os.path.abspath(label_dir)
    
    files = sorted(os.listdir(label_dir), key=lambda f: (int(f.split('_')[0]), int(f.split('_')[1].split('-')[0])))
    all_bboxes = []

    for file in files:
        file_path = os.path.join(label_dir, file)
        top_x, top_y = map(float, file[:-4].split('_')[1].split('-'))
        bboxes = read_labels(file_path)
        
        if SEG_MODEL == 'seggpt':
            for bbox in bboxes:
                abs_bbox = [pos + (top_x if i % 2 == 0 else top_y) for i, pos in enumerate(bbox)]
                all_bboxes.append(abs_bbox)
        else:
            for bbox in bboxes:
                conf_score = float(bbox[-1])
                bbox = bbox[:-1]
                abs_bbox = [pos + (top_x if i % 2 == 0 else top_y) for i, pos in enumerate(bbox)]
                all_bboxes.append({"pos": abs_bbox, "conf": conf_score})
        
    print("x ------- Processing all bounding boxes commenced ------- x")

    if SEG_MODEL == 'seggpt':
        final_bboxes = process_bounding_boxes(all_bboxes)
    else:
        final_bboxes = area_based_merging(all_bboxes)
    post_processed_bboxes = post_process_containment(final_bboxes)

    print(f"Total bounding boxes before NMM: {len(all_bboxes)}")
    print(f"Total bounding boxes after NMM: {len(final_bboxes)}")
    print(f"Total bounding boxes after containment post-processing: {len(post_processed_bboxes)}")

    # # # delete bboxes with aspect ratio more than 4
    # final_bboxes = [bbox for bbox in final_bboxes if get_aspect_ratio(bbox) < 4]
    # # # delete bbox with area less than 0.086x10^6
    # final_bboxes = [bbox for bbox in final_bboxes if calculate_area(bbox) > 0.086 * 10**6]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/labels.txt', 'w') as f:
        for bbox in post_processed_bboxes:
            bbox = [round(pos) for pos in bbox]
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{bbox_str}\n")
    # return post_processed_bboxes