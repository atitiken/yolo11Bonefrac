import pandas as pd

# Replace with your CSV file path
csv_file = r"Dataset/Merged Dataset/merge_labels.csv"  # or augmented_labels.csv

# Load the CSV
df = pd.read_csv(csv_file)

# Count the occurrences of each label
label_counts = df["label"].value_counts()
print("Label Distribution:")
print(label_counts)

# Optionally, print percentages:
print("\nPercentage Distribution:")
print(df["label"].value_counts(normalize=True) * 100)
