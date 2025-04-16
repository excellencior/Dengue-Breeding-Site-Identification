import cv2

def convert_tif_jpg(tif_path, jpg_path):
    try:
        # Read the TIFF image
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        
        # Check if image was successfully loaded
        if img is None:
            raise ValueError("Error reading the TIFF file.")

        # Convert the image to JPEG and save
        cv2.imwrite(jpg_path, img)
        print("\nConversion complete.")
    except Exception as e:
        print("Error converting file")
        print(e)

# Example usage
tif_dir = "Komlapur_Orthophoto.tif"
jpg_dir = "JPG_Komlapur_Orthophoto.jpg"

convert_tif_jpg(tif_dir, jpg_dir)
