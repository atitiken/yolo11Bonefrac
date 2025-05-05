import os
import pandas as pd
import logging

# Configure logging to output to a file
logging.basicConfig(
    filename='dataset_check.log',  # log file name
    level=logging.INFO,            # minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Starting dataset consistency check.")

# Update these paths accordingly
csv_file = r"D:\Punya dede\RM\Dataset\Merged Dataset\merge_labels.csv"  # Path to your CSV file
image_folder = r"D:\Punya dede\RM\Dataset\Merged Dataset\image"  # Folder containing images

# Load CSV data
try:
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded CSV file with {len(df)} entries.")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    raise

# Get a set of all image filenames listed in the CSV
labeled_files = set(df['imagename'].tolist())
logging.info(f"CSV contains {len(labeled_files)} unique image names.")

# Get a set of image files from the folder (filtering common image extensions)
all_files = set(
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
)
logging.info(f"Found {len(all_files)} image files in folder.")

# Identify files that are in the folder but missing in the CSV
missing_in_csv = all_files - labeled_files
if missing_in_csv:
    logging.warning("Files present in folder but missing in CSV:")
    for f in missing_in_csv:
        logging.warning(f"Missing in CSV: {f}")
else:
    logging.info("All files in the folder are present in the CSV.")

# Identify files that are in the CSV but not found in the folder
missing_in_folder = labeled_files - all_files
if missing_in_folder:
    logging.warning("Files listed in CSV but not found in folder:")
    for f in missing_in_folder:
        logging.warning(f"Missing in folder: {f}")
else:
    logging.info("All files listed in the CSV are present in the folder.")

# Check and log the label distribution
label_counts = df['label'].value_counts()
logging.info("Label Distribution:")
for label, count in label_counts.items():
    logging.info(f"Label {label}: {count}")

logging.info("Dataset consistency check complete.")
