"""
Per image slice based building labels generation from the model inference masks for both models (seggpt and yolo)
"""
import os
import cv2
import numpy as np

# Function to extract x and y coordinates from filenames
def get_coordinates_from_filename_buildings(filename):
    filename = os.path.basename(filename)[:-4]
    parts = filename.split('_')
    pos = parts[1].split('-')
    x = int(pos[0])
    y = int(pos[1])
    return x, y

def get_coordinates_from_filename_objects(filename):
    parts = filename.split('_')
    pos = parts[2].split('-')
    x = int(pos[1])
    y = int(pos[2])
    return x, y

# Function to check if two rotated bounding boxes overlap significantly
def check_overlap(rect1, rect2, threshold=0.7):
    inter_area = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    if inter_area is None:
        return False
    overlap_area = cv2.contourArea(inter_area)
    rect1_area = rect1[1][0] * rect1[1][1]
    rect2_area = rect2[1][0] * rect2[1][1]
    return (overlap_area / min(rect1_area, rect2_area)) > threshold

# Function to sort files based on x and y coordinates
def sort_files(files, coordinate_extractor):
    return sorted(files, key=lambda f: coordinate_extractor(f))



def seggpt_labels(mask_dir, save_dir):
    """
    Per image consists of inference from SegGPT model and by using binary thresholding and morphological operations masks are generated per image
    """
    for index, test_file in enumerate(sort_files(os.listdir(mask_dir), get_coordinates_from_filename_buildings)):
        # Load the x2048 image
        x2048_image_path = os.path.join(mask_dir, test_file)
        x2048_image = cv2.imread(x2048_image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(x2048_image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold to segment the buildings
        binary_threshold = 110
        _, binary_mask = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to close small gaps in the thresholded image
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a list to store rotated bounding boxes of detected buildings
        rotated_bounding_boxes = []
        total_building_count = 0

        # Filter and store bounding boxes, removing smaller overlapping bboxes
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                rect = cv2.minAreaRect(cnt)
                keep = True
                for existing_rect in rotated_bounding_boxes:
                    if check_overlap(existing_rect, rect):
                        if existing_rect[1][0] * existing_rect[1][1] > rect[1][0] * rect[1][1]:
                            keep = False
                            break
                        else:
                            rotated_bounding_boxes.remove(existing_rect)
                            total_building_count -= 1
                if keep:
                    rotated_bounding_boxes.append(rect)
                    total_building_count += 1

        # Create a label file for the current image
        label_file_path = os.path.join(save_dir, f'{index + 1}_{get_coordinates_from_filename_buildings(test_file)[0]}-{get_coordinates_from_filename_buildings(test_file)[1]}.txt')
        if index % 10 == 0:
            print(f'Processed {index} images')
            
        # Save the labels in the labels directory
        with open(label_file_path, 'w') as label_file:
            for rect in rotated_bounding_boxes:
                # Get the rotated bounding box coordinates
                box = cv2.boxPoints(rect)
                box = np.int_(box)
                label = f"{box[0][0]} {box[0][1]} {box[1][0]} {box[1][1]} {box[2][0]} {box[2][1]} {box[3][0]} {box[3][1]}"
                label_file.write(f"{label}\n")


def yolo_labels(mask_dir, save_dir):
    """
    Generate YOLO labels from the mask files.
    If save_dir doesn't exist, it will be created.
    Existing files will be overwritten.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_bboxes = []

    for index, filename in enumerate(sort_files(os.listdir(mask_dir), get_coordinates_from_filename_buildings)):
        with open(os.path.join(mask_dir, filename), 'r') as f:
            xpos, ypos = get_coordinates_from_filename_buildings(filename)
            label_file_path = os.path.join(save_dir, f'{index+1}_{xpos}-{ypos}.txt')

            # Always open in 'w' mode first to overwrite any existing file
            open(label_file_path, 'w').close()

            lines = f.readlines()
            for line in lines:
                points = line.strip().split()
                conf_score = float(points[-1])
                seg_labels = np.array(points[1:-1], dtype=np.float32) * 2048
                seg_labels = seg_labels.reshape(-1, 2)
                polygon = cv2.minAreaRect(seg_labels)
                rbox = cv2.boxPoints(polygon)
                rbox = np.intp(rbox)

                all_bboxes.append(rbox.flatten())

                with open(label_file_path, 'a') as label_file:
                    label_file.write(f"{rbox[0][0]} {rbox[0][1]} {rbox[1][0]} {rbox[1][1]} "
                                     f"{rbox[2][0]} {rbox[2][1]} {rbox[3][0]} {rbox[3][1]} {conf_score}\n")

    print(f'Processed All Labels: {len(all_bboxes)}')


# Given image dimensions and tile sizes
image_width = 16460
image_height = 14590
tile_size = 2048
small_tile_size = 512
print(f'Dataset Info - Image Width: {image_width}, Image Height: {image_height}, Tile Size: {tile_size}, Small Tile Size: {small_tile_size}')