import os
import pandas as pd

# Paths to the CSV files
old_csv = r"D:\Punya dede\RM\Dataset\BoneFractureYolo8\renamed_labels.csv"       # your original labels CSV
new_csv = r"D:/Punya dede/RM/Dataset/BoneFractureYolo8augmented_labels.csv"    # new CSV for augmented images

# Read the original CSV
df_old = pd.read_csv(old_csv)

# Prepare a list to collect the augmented image entries
augmented_entries = []

# For each row in the old CSV, create two new rows for the augmented variants
for _, row in df_old.iterrows():
    original_filename = row["imagename"]  # e.g., "last_frac0001.jpg"
    label = row["label"]
    
    # Remove extension to get the base name (e.g., "last_frac0001")
    base_name = os.path.splitext(original_filename)[0]
    
    # Create new filenames for both augmentation variants
    new_filename_p15 = f"aug_{base_name}_p15.jpg"  # rotated +15°
    new_filename_n15 = f"aug_{base_name}_n15.jpg"  # rotated -15°
    
    # Append both augmented entries with the same label as original
    augmented_entries.append({"imagename": new_filename_p15, "label": label})
    augmented_entries.append({"imagename": new_filename_n15, "label": label})

# Create a new DataFrame from the augmented entries
df_augmented = pd.DataFrame(augmented_entries)

# Save the new CSV file
df_augmented.to_csv(new_csv, index=False)

print(f"Augmented CSV saved as {new_csv}")
