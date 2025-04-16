# Note: Cleaning the roboflow naming (getting rid of the parts after .rf)

import os

def clean_filename(filename):
    parts = filename.split('_', 3)
    filename = parts[0] + '_' + parts[3]
    if '_jpg' in filename:
        base = filename.split('_jpg')[0]
        if '.' in filename.split('_jpg')[-1]:
            extension = filename.split('.')[-1]
            return f"{base}.{extension}"
        else:
            return base
    else:
        return filename

def clean_filenames_in_directory(directory):
    for filename in os.listdir(directory):
        new_filename = clean_filename(filename)
        original_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(original_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")

# Specify the directory containing the files
directory_path = "Building/output/test"

# Clean the filenames in the specified directory
clean_filenames_in_directory(directory_path)
