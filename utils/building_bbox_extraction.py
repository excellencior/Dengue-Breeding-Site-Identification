# Current resolution: 8192x7261
# Original resolution: 16460x14590
"""
    Positioning Oritental Bounding Box around the building segmentations
"""
import json
import cv2
import numpy as np

def scale(x, y):
    x_scale = 16460 / 8192
    y_scale = 14590 / 7261
    return [float(x * x_scale), float(y * y_scale)]

# Read the coco json segmentations and scale each of the segmentations
coco_json = json.load(open('Building/BuildingSegmentation1.v6i.coco/train/_annotations.coco.json'))
org_image = cv2.imread('Orthophotos/Komlapur_Orthophoto.jpg')

store_img = np.zeros_like(org_image)
for annotation in coco_json['annotations']:
    segmentation = annotation['segmentation'][0] # The original one was containing list of lists (polygon in polygon)
    new_segmentation = []
    for i in range(0, len(segmentation), 2):
        new_segmentation.extend(scale(segmentation[i], segmentation[i + 1]))
    # annotation['segmentation'] = new_segmentation # Replace the old segmentation with the new one (only the points of the polygon)

    # Overlay the polygon
    points = np.array(new_segmentation).reshape(-1, 2).astype(np.int32)
    cv2.polylines(org_image, [points], True, (0, 0, 255), 2)

    # Overlay the minAreaRect around the polygon
    points = np.array(new_segmentation).reshape(-1, 2).astype(np.int32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(org_image, [box], 0, (0, 255, 0), 4)

    # Save the bboxes in a txt file
    points = np.array(new_segmentation).reshape(-1, 2).astype(np.int32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    with open('Labels_combined/building_ground_truth/labels.txt', 'a') as f:
        f.write(f'{box[0][0]} {box[0][1]} {box[1][0]} {box[1][1]} {box[2][0]} {box[2][1]} {box[3][0]} {box[3][1]}\n')

# Save the new image
cv2.imwrite('Orthophotos/Komlapur_Orthophoto_bboxed.jpg', org_image)

# json.dump(coco_json, open('Building/BuildingSegmentation1.v3i.coco/train/_annotations.coco.json', 'w'), indent=4)