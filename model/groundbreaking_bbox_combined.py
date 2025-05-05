"""
After combined building labels are generated, this script is used to process the labels
Sliding Window --- Inferenced on slices with 50% overlap
non max supression nms --- reducing number of false positives
"""
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster


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

def post_process_containment(bboxes, containment_threshold=0.8):
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
    
    return final_bboxes

def process_bounding_boxes(bboxes, or_threshold=0.4):
    """
        Process bounding boxes using a combination of NMS and alignment-based merging for model = SegGPT
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
                    # Align and merge bboxes
                    merged_bbox = align_bboxes(current_bbox, other_bbox)
                    # merged_class = current_class * other_class
                    merged_bboxes[i] = merged_bbox
                    break
            elif final_or >= or_threshold:
                # Keep the larger bbox
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
        Process bounding boxes using a combination of NMS and alignment-based merging for model = YOLO
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

def enhanced_obb_merging(bboxes, iou_threshold=0.5, alignment_angle_threshold=15, proximity_factor=0.3, min_confidence=0.25):
    """
    Advanced algorithm for merging oriented bounding boxes with multi-criteria evaluation.
    
    This algorithm uses a comprehensive approach that considers:
    1. Multi-metric overlap evaluation (IOU, Containment, Coverage)
    2. Spatial and orientation relationship analysis
    3. Confidence-weighted merging with dynamic thresholds
    4. Hierarchical clustering for group-based merging
    
    Args:
        bboxes: List of dicts with 'pos' (8-point OBB) and 'conf' (confidence)
        iou_threshold: Base threshold for IOU consideration
        alignment_angle_threshold: Maximum angle difference in degrees for alignment
        proximity_factor: Factor to determine proximity threshold based on bbox dimensions
        min_confidence: Minimum confidence score to consider a detection valid
        
    Returns:
        List of merged bounding box positions
    """
    if not bboxes:
        return []
    
    # Filter out low confidence detections
    filtered_bboxes = [bbox for bbox in bboxes if bbox['conf'] >= min_confidence]
    
    if not filtered_bboxes:
        return []
    
    # Enhanced similarity matrix calculation
    def calculate_similarity_matrix(boxes):
        n = len(boxes)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                box_i, box_j = boxes[i], boxes[j]
                
                # Calculate multiple overlap metrics
                iou = calculate_iou(box_i['pos'], box_j['pos'])
                containment_i_in_j = calculate_containment(box_i['pos'], box_j['pos'])
                containment_j_in_i = calculate_containment(box_j['pos'], box_i['pos'])
                
                # Calculate spatial relationship metrics
                angle_diff = calculate_angle_difference(box_i['pos'], box_j['pos'])
                centroid_dist = calculate_normalized_centroid_distance(box_i['pos'], box_j['pos'])
                
                # Confidence-based weighting
                conf_weight = np.sqrt(box_i['conf'] * box_j['conf'])
                
                # Combined similarity score with adaptive weighting
                overlap_score = max(iou, containment_i_in_j, containment_j_in_i)
                alignment_score = 1.0 - min(1.0, angle_diff / alignment_angle_threshold)
                proximity_score = 1.0 - min(1.0, centroid_dist / proximity_factor)
                
                # Final weighted similarity score
                similarity = (0.6 * overlap_score + 0.2 * alignment_score + 0.2 * proximity_score) * conf_weight
                
                sim_matrix[i, j] = sim_matrix[j, i] = similarity
                
        return sim_matrix
    
    def calculate_iou(bbox1, bbox2):
        """Calculate IOU between two oriented bounding boxes"""
        poly1 = Polygon(np.array(bbox1).reshape(4, 2))
        poly2 = Polygon(np.array(bbox2).reshape(4, 2))
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
            
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def calculate_containment(bbox1, bbox2):
        """Calculate how much of bbox1 is contained within bbox2"""
        poly1 = Polygon(np.array(bbox1).reshape(4, 2))
        poly2 = Polygon(np.array(bbox2).reshape(4, 2))
        
        if not poly1.is_valid or not poly2.is_valid or poly1.area <= 0:
            return 0.0
            
        intersection_area = poly1.intersection(poly2).area
        return intersection_area / poly1.area
    
    def calculate_angle_difference(bbox1, bbox2):
        """Calculate the minimum angle difference between two OBBs"""
        # Extract primary axis orientations
        rect1 = cv2.minAreaRect(np.array(bbox1).reshape(4, 2).astype(np.float32))
        rect2 = cv2.minAreaRect(np.array(bbox2).reshape(4, 2).astype(np.float32))
        
        angle1 = rect1[2]
        angle2 = rect2[2]
        
        # Normalize angles to 0-90 degrees range since boxes have symmetry
        angle1 = angle1 % 90
        angle2 = angle2 % 90
        
        # Calculate minimum angle difference accounting for symmetry
        angle_diff = min(abs(angle1 - angle2), 90 - abs(angle1 - angle2))
        return angle_diff
    
    def calculate_normalized_centroid_distance(bbox1, bbox2):
        """Calculate distance between centroids normalized by box dimensions"""
        points1 = np.array(bbox1).reshape(4, 2)
        points2 = np.array(bbox2).reshape(4, 2)
        
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)
        
        # Calculate average box dimensions
        rect1 = cv2.minAreaRect(points1.astype(np.float32))
        rect2 = cv2.minAreaRect(points2.astype(np.float32))
        
        avg_width = (rect1[1][0] + rect2[1][0]) / 2
        avg_height = (rect1[1][1] + rect2[1][1]) / 2
        avg_dimension = max(1.0, np.sqrt(avg_width * avg_height))
        
        # Normalize distance by average dimension
        distance = np.linalg.norm(centroid1 - centroid2)
        return distance / avg_dimension
    
    def smart_merge_bboxes(group_indices):
        """Merge a group of bounding boxes using weighted combination"""
        if len(group_indices) == 1:
            return filtered_bboxes[group_indices[0]]['pos']
        
        group_boxes = [filtered_bboxes[i] for i in group_indices]
        
        # If high confidence box exists, prioritize it
        max_conf_idx = np.argmax([box['conf'] for box in group_boxes])
        max_conf_box = group_boxes[max_conf_idx]
        
        if max_conf_box['conf'] > 0.8:  # High confidence threshold
            return max_conf_box['pos']
        
        # Otherwise use advanced merging
        # Sort by confidence
        group_boxes.sort(key=lambda x: x['conf'], reverse=True)
        
        # Try to find consensus area by analyzing overlap patterns
        all_points = np.vstack([np.array(box['pos']).reshape(4, 2) for box in group_boxes])
        weights = np.array([box['conf'] for box in group_boxes])
        
        # Method 1: Weighted average of vertices with convex hull fitting
        weighted_points = np.zeros((4, 2))
        total_weight = np.sum(weights)
        
        # Find correspondence between points
        for i, box in enumerate(group_boxes):
            pts = np.array(box['pos']).reshape(4, 2)
            # Ensure consistent ordering of points
            rect = cv2.minAreaRect(pts.astype(np.float32))
            ordered_pts = cv2.boxPoints(rect).astype(np.float32)
            
            # Apply weight
            weighted_points += (ordered_pts * (weights[i] / total_weight))
        
        # Ensure the result is a proper rectangle
        rect = cv2.minAreaRect(weighted_points.astype(np.float32))
        merged_box = cv2.boxPoints(rect).flatten().tolist()
        
        return merged_box
    
    # Apply hierarchical clustering to group similar boxes
    if len(filtered_bboxes) <= 1:
        return [box['pos'] for box in filtered_bboxes]
    
    similarity_matrix = calculate_similarity_matrix(filtered_bboxes)
    
    # Convert similarity to distance for clustering
    distance_matrix = 1.0 - similarity_matrix
    
    # Use hierarchical clustering to group similar boxes
    # Convert distance matrix to condensed form required by linkage
    condensed_dist = []
    for i in range(len(filtered_bboxes)):
        for j in range(i + 1, len(filtered_bboxes)):
            condensed_dist.append(distance_matrix[i, j])
    
    # Perform hierarchical clustering
    Z = linkage(np.array(condensed_dist), method='average')
    
    # Determine optimal threshold adaptively based on data distribution
    threshold = 1.0 - (iou_threshold * 0.8)  # Dynamic threshold based on iou_threshold
    clusters = fcluster(Z, threshold, criterion='distance')
    
    # Group indices by cluster
    cluster_groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(i)
    
    # Merge boxes within each cluster
    final_bboxes = []
    for cluster_id, indices in cluster_groups.items():
        merged_bbox = smart_merge_bboxes(indices)
        final_bboxes.append(merged_bbox)
    
    return final_bboxes

def process_sliding_window(label_dir, save_dir, SEG_MODEL='yolo'):
    """
        Process bounding boxes generated from sliding window approach [50% overlap]
    """
    # Absolute path conversion
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

    # Choose the appropriate merging algorithm based on model type
    if SEG_MODEL == 'seggpt':
        final_bboxes = process_bounding_boxes(all_bboxes)
    else:
        # Use the enhanced OBB merging algorithm for YOLO
        final_bboxes = enhanced_obb_merging(all_bboxes)
    
    # Apply post-processing to handle containment relationships
    post_processed_bboxes = post_process_containment(final_bboxes)

    print(f"Total bounding boxes before NMS: {len(all_bboxes)}")
    print(f"Total bounding boxes after NMS: {len(final_bboxes)}")
    print(f"Total bounding boxes after containment post-processing: {len(post_processed_bboxes)}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/labels.txt', 'w') as f:
        for bbox in post_processed_bboxes:
            bbox = [round(pos) for pos in bbox]
            bbox_str = ' '.join(map(str, bbox))
            f.write(f"{bbox_str}\n")
    
    return post_processed_bboxes

# Example usage:
# process_sliding_window('path/to/labels', 'path/to/save', 'yolo')