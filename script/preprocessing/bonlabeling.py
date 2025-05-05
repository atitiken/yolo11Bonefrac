import os
import pandas as pd

# Set the path to the folder containing the .txt files and images
txt_folder = "D:\Punya dede\RM\Dataset\BoneFractureYolo8\merge label"
image_folder = "D:\Punya dede\RM\Dataset\BoneFractureYolo8\merge image"
output_csv = "D:/Punya dede/RM/renamed_labels.csv"

# Define binary label mapping
def binary_label(label):
    return 0 if label == "6" else 1  # 0 for no fracture, 1 for fracture

# Lists to store the processed data
data = []
frac_count = 1  # Counter for fracture images
nofrac_count = 1  # Counter for no fracture images

# Process each .txt file
for filename in sorted(os.listdir(txt_folder)):  # Sort for consistency
    if filename.endswith(".txt"):
        old_txt_path = os.path.join(txt_folder, filename)
        old_image_path = os.path.join(image_folder, filename.replace(".txt", ".jpg"))  # Assuming images are .jpg
        
        # Read label from the .txt file
        with open(old_txt_path, "r") as file:
            line = file.readline().strip()  # Read single line
            
            if not line:  # If file is empty, label as no fracture (0)
                label_value = 0
            else:
                parts = line.split()  # Split by spaces
                label = parts[0] if len(parts) > 0 else "6"  # Default to no fracture if missing
                label_value = binary_label(label)

        # Determine new filename based on label
        if label_value == 1:
            new_filename = f"bon_frac{frac_count:04d}"
            frac_count += 1
        else:
            new_filename = f"bon_nofrac{nofrac_count:04d}"
            nofrac_count += 1
        
        new_txt_path = os.path.join(txt_folder, new_filename + ".txt")
        new_image_path = os.path.join(image_folder, new_filename + ".jpg")

        # Rename the txt file
        os.rename(old_txt_path, new_txt_path)

        # Rename the corresponding image if it exists
        if os.path.exists(old_image_path):
            os.rename(old_image_path, new_image_path)

        # Append formatted data
        data.append([new_filename + ".jpg", label_value])

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data, columns=["imagename", "label"])
df.to_csv(output_csv, index=False)

print(f"CSV file saved as {output_csv}")
