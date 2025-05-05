import os
import cv2
import numpy as np
from shapely.geometry import Polygon

def read_labels(file_path):
    """Reads bounding boxes from a text file."""
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            values = line.strip().split()
            bbox = list(map(float, values))
            bboxes.append(bbox)
    return bboxes

def rotated_iou(bbox1, bbox2):
    """Computes the Intersection over Union (IoU) between two rotated bounding boxes."""
    poly1 = Polygon(np.array(bbox1).reshape(4, 2))
    poly2 = Polygon(np.array(bbox2).reshape(4, 2))
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0
    return intersection / union

def calculate_area(bbox):
    """Calculates the area of a bounding box."""
    return Polygon(np.array(bbox).reshape(4, 2)).area

def post_process_containment(bboxes, containment_threshold=0.8):
    """Removes bounding boxes that are mostly contained within larger ones."""
    final_bboxes = []
    sorted_bboxes = sorted(bboxes, key=calculate_area, reverse=True)
    
    while sorted_bboxes:
        current_bbox = sorted_bboxes.pop(0)
        current_poly = Polygon(np.array(current_bbox).reshape(4, 2))
        
        i = 0
        while i < len(sorted_bboxes):
            other_bbox = sorted_bboxes[i]
            other_poly = Polygon(np.array(other_bbox).reshape(4, 2))
            
            if other_poly.area == 0:
                i += 1
                continue
            
            overlap_ratio = other_poly.intersection(current_poly).area / other_poly.area
            
            if overlap_ratio > containment_threshold:
                sorted_bboxes.pop(i)
            else:
                i += 1
        
        final_bboxes.append(current_bbox)
    
    return final_bboxes

def greedy_nms(bboxes, iou_threshold=0.5, SEG_MODEL='yolo'):
    """
    Applies Greedy Non-Maximum Suppression (NMS) for YOLO-based bounding boxes.
    """
    if not bboxes:
        return []
    
    bboxes = sorted(bboxes, key=lambda x: x['conf'], reverse=True)

    final_bboxes = []
    
    while bboxes:
        best_bbox = bboxes.pop(0)
        final_bboxes.append(best_bbox['pos'])
        
        bboxes = [bbox for bbox in bboxes if rotated_iou(best_bbox['pos'], bbox['pos']) < iou_threshold]
    
    return final_bboxes

def process_sliding_window(label_dir, save_dir, iou_threshold=0.5):
    """
    Processes bounding boxes using Greedy NMS for YOLO or area-based merging for SegGPT.
    """
    label_dir = os.path.abspath(label_dir)
    files = sorted(os.listdir(label_dir))
    all_bboxes = []
    
    for file in files:
        file_path = os.path.join(label_dir, file)
        top_x, top_y = map(float, file[:-4].split('_')[1].split('-'))
        bboxes = read_labels(file_path)
        
        for bbox in bboxes:
                conf_score = float(bbox[-1])
                bbox = bbox[:-1]
                abs_bbox = [pos + (top_x if i % 2 == 0 else top_y) for i, pos in enumerate(bbox)]
                all_bboxes.append({"pos": abs_bbox, "conf": conf_score})
    
    print(f"Total bounding boxes before processing: {len(all_bboxes)}")
    
    final_bboxes = greedy_nms(all_bboxes, iou_threshold)
    final_bboxes = post_process_containment(final_bboxes)
    
    print(f"Total bounding boxes after processing: {len(final_bboxes)}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(f'{save_dir}/labels.txt', 'w') as f:
        for bbox in final_bboxes:
            bbox_str = ' '.join(map(str, map(round, bbox)))
            f.write(f"{bbox_str}\n")
