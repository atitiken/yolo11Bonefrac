import os
import pandas as pd

# Update these paths as needed
csv_file = r"D:\Punya dede\RM\Dataset\Merged Dataset\merge_labels.csv"
nofrac_folder = r"D:\Punya dede\RM\Dataset\Merged Dataset\image"  # folder where atlas_nofrac images are stored

# Load the CSV file
df = pd.read_csv(csv_file)

# Filter for no-fracture images.
# We assume these images have label 0 and filenames that start with "atlas_nofrac"
mask = (df["label"] == 0) & (df["imagename"].str.startswith("atlas_nofrac"))
df_nofrac = df[mask]

print(f"Total no-fracture images: {len(df_nofrac)}")

# Ensure there are at least 2000 images to remove
num_to_remove = 2000
if len(df_nofrac) < num_to_remove:
    print(f"There are fewer than {num_to_remove} no-fracture images. No action taken.")
else:
    # Select 2000 images to remove. (Here we choose randomly; remove .sample(n=...) if you prefer the first 2000)
    remove_df = df_nofrac.sample(n=num_to_remove, random_state=42)
    remove_filenames = remove_df["imagename"].tolist()
    
    # Remove each image file from the folder
    for filename in remove_filenames:
        file_path = os.path.join(nofrac_folder, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed file: {filename}")
            else:
                print(f"File not found (skipped): {filename}")
        except Exception as e:
            print(f"Error removing {filename}: {e}")
    
    # Remove these rows from the CSV
    df_updated = df.drop(remove_df.index)
    df_updated.to_csv(csv_file, index=False)
    print(f"Removed {num_to_remove} no-fracture images and updated the CSV file.")
