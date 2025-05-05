"""
FORMAT: coco segmentation annotation
"""
import numpy as np
import json
import cv2

# Function to create binary mask from polygons
def create_binary_mask_for_image(annotations, height, width):
    # Initialize a blank black mask (single mask for all objects in the image)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Loop through all annotations for this image and combine them into one mask
    for ann in annotations:
        segmentation = ann['segmentation']
        # Check if the segmentation is in polygon format
        if isinstance(segmentation, list):
            # Loop through each polygon in the segmentation
            for polygon in segmentation:
                # Convert the polygon into a NumPy array and reshape it
                poly = np.array(polygon).reshape((-1, 2))
                # Draw and fill the polygon on the mask
                cv2.fillPoly(mask, [np.int32(poly)], 255)

    return mask

# Function to generate binary masks from COCO annotations
def generate_binary_masks(coco_annotation_path, output_folder):
    # Load the COCO annotations
    with open(coco_annotation_path) as f:
        coco = json.load(f)

    annotations = coco['annotations']
    images = {img['id']: img for img in coco['images']}

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Loop through each image and create a single binary mask for all objects in the image
    for image_id, image_info in images.items():
        image_filename = image_info['file_name']
        height = image_info['height']
        width = image_info['width']

        # Get annotations for this image
        image_annotations = annotations_by_image.get(image_id, [])

        # Create the combined binary mask for this image
        if image_annotations:
            mask = create_binary_mask_for_image(image_annotations, height, width)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)  # Black mask if no annotations

        # Keep the original filename and change the extension to .png
        mask_filename = os.path.splitext(image_filename)[0] + ".png"
        mask_filepath = os.path.join(output_folder, mask_filename)

        # Save the mask as an image (black background, white mask or just black if no masks)
        cv2.imwrite(mask_filepath, mask)
        print(f"Mask saved at: {mask_filepath}")

if __name__ == "__main__":
    coco_annotation_path = "Building/gt_all/_annotations.coco.json"
    output_folder = "Building/gt_masks"

    generate_binary_masks(coco_annotation_path, output_folder)
