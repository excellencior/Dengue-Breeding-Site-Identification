# write a code to iterate a folder full of images and increase their contrast
# by 20% and save them in a new folder

import cv2
import os

# Path to the folder containing images
path = 'test'

# Path to the folder where the new images will be saved
new_path = 'new_images'

# Create the new folder if it doesn't exist
if not os.path.exists(new_path):
    os.makedirs(new_path)

# Iterate over all files in the folder
for filename in os.listdir(path):
    # Read the image
    img = cv2.imread(os.path.join(path, filename))
    
    # Increase the contrast by 20%
    img = cv2.convertScaleAbs(img, alpha=1.4, beta=-10)
    
    # Save the new image in the new folder
    cv2.imwrite(os.path.join(new_path, filename), img)

print('Done!')