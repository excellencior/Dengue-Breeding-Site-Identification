# Note: It was used to rename the files from the kaggle to use these in the "test" folder in SegGPT
import os

# Define the directory containing the images
directory = "test"

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):  # Check if the file is a JPEG image
        # Replace the specified characters in the filename
        new_filename = filename.replace("(", "-").replace(")", "-").replace(",", "-").replace(".", "-", filename.count(".") - 1)
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

print("Renaming complete.")