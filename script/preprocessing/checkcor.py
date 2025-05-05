import os
from PIL import Image

def is_image_valid(file_path):
    """
    Check if an image file can be opened and verified.
    Returns True if valid; otherwise, False.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify does not load the image fully but checks for corruption.
        return True
    except Exception as e:
        print(f"Corrupt image detected: {file_path}\nError: {e}")
        return False

def remove_corrupt_images(root_dir, valid_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
    """
    Recursively scan `root_dir` for images with extensions in valid_extensions.
    Remove images that fail the validity check.
    """
    removed_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(subdir, file)
                if not is_image_valid(file_path):
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                        print(f"Removed file: {file_path}")
                    except Exception as remove_err:
                        print(f"Error removing {file_path}: {remove_err}")
    return removed_files

if __name__ == '__main__':
    # Specify the root folder of your dataset (this can be your train/val/test root)
    dataset_root = r"D:\Punya dede\RM\Dataset\Merged Dataset\split"
    
    print("Scanning dataset for corrupt images...")
    removed = remove_corrupt_images(dataset_root)
    print(f"Total corrupt images removed: {len(removed)}")
