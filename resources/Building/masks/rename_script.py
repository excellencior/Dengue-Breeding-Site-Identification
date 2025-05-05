import os
from pathlib import Path

def extract_coordinates(filename):
    filename_parts = filename.split('_')
    if len(filename_parts) >= 5:
        try:
            x_coordinate = float(filename_parts[3])
            y_coordinate = float(filename_parts[4])
            return x_coordinate, y_coordinate
        except (ValueError, IndexError):
            return None, None
    return None, None

def rename_files(directory_path):
    file_path = Path(directory_path)
    text_files = [file for file in file_path.glob("*.txt") if file.is_file()]
    
    files_with_coordinates = []
    for text_file in text_files:
        x_coordinate, y_coordinate = extract_coordinates(text_file.name)
        if x_coordinate is not None and y_coordinate is not None:
            files_with_coordinates.append((text_file, x_coordinate, y_coordinate))
    
    sorted_files = sorted(files_with_coordinates, key=lambda item: (item[1], item[2]))
    
    for sort_order, (file_item, x_coordinate, y_coordinate) in enumerate(sorted_files, 1):
        new_filename = f"{sort_order}_{int(x_coordinate)}-{int(y_coordinate)}.txt"
        new_file_path = file_item.parent / new_filename
        
        file_item.rename(new_file_path)
        print(f"Renamed: {file_item.name} -> {new_filename}")

if __name__ == "__main__":
    directory = "labels_11m_25"
    rename_files(directory)
    print("File renaming complete!")