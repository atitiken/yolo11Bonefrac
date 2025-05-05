import os
import pandas as pd
import logging

# Configure logging to output to a file
logging.basicConfig(
    filename='unique_labels.log',  # log file name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Starting duplicate check and removal.")

# Update the CSV file path accordingly
csv_file = r"D:\Punya dede\RM\Dataset\Merged Dataset\merge_labels.csv"

# Load CSV data
try:
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded CSV file with {len(df)} entries.")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    raise

# Identify duplicate entries based on 'imagename'
duplicates = df[df.duplicated(subset='imagename', keep=False)]
if not duplicates.empty:
    logging.warning(f"Found {len(duplicates)} duplicate entries:")
    # Log the duplicates
    for imagename in duplicates['imagename'].unique():
        dup_count = (df['imagename'] == imagename).sum()
        logging.warning(f"Image {imagename} appears {dup_count} times.")
    
    # Remove duplicates: keep first occurrence
    df_unique = df.drop_duplicates(subset='imagename', keep='first')
    logging.info(f"After deduplication, CSV has {len(df_unique)} entries.")
    
    # Save the unique CSV (this will overwrite the original file)
    df_unique.to_csv(csv_file, index=False)
    logging.info(f"CSV file updated with unique entries and saved to {csv_file}.")
else:
    logging.info("No duplicate entries found in CSV.")

logging.info("Duplicate check and removal complete.")
