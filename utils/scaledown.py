import os
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

def resize_images_cv2(input_folder, output_folder, fx, fy, interpolation=cv2.INTER_LANCZOS4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Failed to read image {img_path}")
                continue
            resized_img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logging.info(f"Saved resized image to {output_path} using OpenCV")

def resize_images_pil(input_folder, output_folder, resample_filter=Image.LANCZOS, quality=95):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    resized_img = img.resize((img.width // 2, img.height // 2), resample=resample_filter)
                    output_path = os.path.join(output_folder, filename)
                    resized_img.save(output_path, quality=quality)
                    logging.info(f"Saved resized image to {output_path} using PIL")
            except Exception as e:
                logging.error(f"Error processing {filename} with PIL: {e}")

# Specify the folders
folder = 'Building/test_result_old copy'
resized_cv2 = 'new_test_result_old_cv2'

fx = 0.5  # horizontal scale factor
fy = 0.5  # vertical scale factor

# Resize images in both folders using OpenCV
resize_images_cv2(folder, resized_cv2, fx, fy)