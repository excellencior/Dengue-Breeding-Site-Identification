input_file = "predicted_object_labels_v9gelanc.txt"  # Change this to your actual file name
output_file = "predicted_object_labels_v9gelanc_nofp.txt"

# Read the file and filter out lines with class ID 1
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if not line.startswith("1 "):
            outfile.write(line)