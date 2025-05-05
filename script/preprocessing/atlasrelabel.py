import os
import pandas as pd
from PIL import Image

# Define paths
fracture_folder = r"D:\Punya dede\RM\Dataset\FracAtlas\images\Non_fractured"
no_fracture_folder = r"D:\Punya dede\RM\Dataset\FracAtlas\images\Non_fractured"
output_csv = r"D:/Punya dede/RM/atlas_labels.csv"

# Counters for renaming
frac_count = 1
nofrac_count = 1

# List to store the processed data
data = []

# Define valid image extensions (all lower-case)
valid_extensions = (".jpg", ".jpeg", ".png")

# Helper function to process a folder
def process_folder(folder, label, prefix, start_count):
    count = start_count
    folder_data = []
    for filename in sorted(os.listdir(folder), key=lambda x: x.lower()):
        if not filename.lower().endswith(valid_extensions):
            print(f"Skipping non-image file: {filename}")
            continue

        # If the file already appears renamed (starts with the prefix), add it directly.
        if filename.startswith(prefix):
            print(f"File already renamed: {filename}")
            folder_data.append([filename, label])
            continue

        old_path = os.path.join(folder, filename)
        new_filename = f"{prefix}{count:04d}.jpg"
        new_path = os.path.join(folder, new_filename)
        
        # If target file already exists, skip conversion/renaming.
        if os.path.exists(new_path):
            print(f"Target file already exists: {new_filename}. Skipping conversion for {filename}.")
        else:
            ext = os.path.splitext(filename)[1].lower()
            print(f"Processing {filename} (ext: {ext}) -> {new_filename}")
            if ext != ".jpg":
                try:
                    with Image.open(old_path) as img:
                        img = img.convert("RGB")  # Ensure RGB mode
                        img.save(new_path, "JPEG")
                    os.remove(old_path)
                    print(f"Converted {filename} to {new_filename}")
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
                    new_filename = filename  # Fallback: use original filename
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed {filename} to {new_filename}")
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
        
        folder_data.append([new_filename, label])
        count += 1
    return folder_data, count

# Process fracture images (label 1)
frac_data, frac_count = process_folder(fracture_folder, 1, "atlas_frac", frac_count)
data.extend(frac_data)

# Process no fracture images (label 0)
nofrac_data, nofrac_count = process_folder(no_fracture_folder, 0, "atlas_nofrac", nofrac_count)
data.extend(nofrac_data)

# Save data to CSV
df = pd.DataFrame(data, columns=["imagename", "label"])
df.to_csv(output_csv, index=False)

print(f"CSV file saved as {output_csv}")
