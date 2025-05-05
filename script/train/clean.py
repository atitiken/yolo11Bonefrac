import os
from PIL import Image, UnidentifiedImageError
import shutil
from tqdm import tqdm

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

def move_corrupted_image(src_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    shutil.move(src_path, dest_path)
    print(f"Moved corrupted image to: {dest_path}")

def check_and_move_corrupted_images(root_dir, corrupted_dir):
    total_images = 0
    corrupted_count = 0

    for subdir, _, files in os.walk(root_dir):
        image_files = [f for f in files if is_image_file(f)]
        for filename in tqdm(image_files, desc=f"Checking {subdir}", unit="image"):
            total_images += 1
            filepath = os.path.join(subdir, filename)
            try:
                with Image.open(filepath) as img:
                    img.convert("RGB")  # Fully load the image
            except (IOError, OSError, UnidentifiedImageError) as e:
                corrupted_count += 1
                print(f"Corrupted image: {filepath} | Error: {e}")
                move_corrupted_image(filepath, corrupted_dir)

    print(f"\n✅ Done. Total images checked: {total_images}")
    print(f"❌ Corrupted images found and moved: {corrupted_count}")
    print(f"✅ Clean images remaining: {total_images - corrupted_count}")

if __name__ == "__main__":
    # Set these paths before running
    INPUT_DIR = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split\split\train"
    CORRUPTED_DIR = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split\split\corrupted"

    check_and_move_corrupted_images(INPUT_DIR, CORRUPTED_DIR)
