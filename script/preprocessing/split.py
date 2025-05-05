import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# --- Configuration ---
source_folder = r"D:\Punya dede\RM\Dataset\Merged Dataset\image"  # Folder with all images
output_folder = r"D:\Punya dede\RM\Dataset\Merged Dataset\split"  # Where to save the split dataset

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Define class substrings (check 'nofrac' first to avoid conflict with 'frac')
def get_label(filename):
    fname = filename.lower()
    if "nofrac" in fname:
        return "no_fracture"
    elif "frac" in fname:
        return "fracture"
    else:
        return None  # Unknown class

# --- Create output directory structure ---
splits = ['train', 'val', 'test']
classes = ['fracture', 'no_fracture']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_folder, split, cls), exist_ok=True)

# --- Gather files by class ---
files_by_class = {cls: [] for cls in classes}
all_files = [f for f in os.listdir(source_folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for f in all_files:
    label = get_label(f)
    if label:
        files_by_class[label].append(f)
    else:
        print(f"Skipping file (unknown label): {f}")

# --- Split each class into train, val, test ---
for cls, files in files_by_class.items():
    random.shuffle(files)
    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits_files = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }
    
    for split, split_files in splits_files.items():
        for fname in split_files:
            src_path = os.path.join(source_folder, fname)
            dst_path = os.path.join(output_folder, split, cls, fname)
            shutil.copy2(src_path, dst_path)
        print(f"Class '{cls}' - {split}: {len(split_files)} files")

print("Dataset split complete.")
