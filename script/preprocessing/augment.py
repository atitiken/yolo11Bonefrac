import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import glob
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

# Define paths - adjust these to your actual paths
BASE_DIR = r"D:\Punya dede\RM\Dataset\Merged Dataset"  # Your base dataset directory
TRAIN_DIR = os.path.join(BASE_DIR, "split", "train")
VAL_DIR = os.path.join(BASE_DIR, "split", "val")
TEST_DIR = os.path.join(BASE_DIR, "split", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create dataset YAML file
def create_dataset_yaml():
    """Create a YAML file for YOLOv11s-cls training"""
    data = {
        "path": BASE_DIR,
        "train": os.path.join("split", "train"),
        "val": os.path.join("split", "val"),
        "test": os.path.join("split", "test"),
        "names": {
            0: "fracture",
            1: "no_fracture"
        }
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, "bone_fracture.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created dataset YAML at: {yaml_path}")
    return yaml_path

def check_dataset_structure():
    """Verify dataset structure and print summary"""
    splits = ["train", "val", "test"]
    classes = ["fracture", "no_fracture"]
    
    for split in splits:
        print(f"Split: {split}")
        split_dir = os.path.join(BASE_DIR, "split", split)
        if not os.path.exists(split_dir):
            print(f"  WARNING: Directory not found: {split_dir}")
            continue
            
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                image_files = glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                              glob.glob(os.path.join(cls_dir, "*.jpeg")) + \
                              glob.glob(os.path.join(cls_dir, "*.png"))
                count = len(image_files)
                print(f"  Class '{cls}': {count} images")
            else:
                print(f"  WARNING: Class directory not found: {cls_dir}")
    
    print("\nMake sure your dataset follows this structure:")
    print(f"{BASE_DIR}/split/train/fracture/")
    print(f"{BASE_DIR}/split/train/no_fracture/")
    print(f"{BASE_DIR}/split/val/fracture/")
    print(f"{BASE_DIR}/split/val/no_fracture/")
    print(f"{BASE_DIR}/split/test/fracture/")
    print(f"{BASE_DIR}/split/test/no_fracture/")

def train_yolo_model():
    """Train YOLOv11s-cls model with best practices for medical image classification"""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create YAML file
    yaml_path = create_dataset_yaml()
    
    # Try to load pre-trained YOLOv11s-cls model
    try:
        model = YOLO(r"D:\Punya dede\RM\yolo11s-cls.pt")
        print("Successfully loaded YOLOv11s-cls model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Downloading YOLOv11s-cls model...")
        # If model file doesn't exist, download it
        import subprocess
        subprocess.run("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11s-cls.pt", shell=True)
        model = YOLO("yolov11s-cls.pt")
    
    # Set training parameters
    batch_size = 32
    img_size = 640
    epochs = 100
    
    print("\n=== Starting Training ===")
    print(f"Training with YAML file: {yaml_path}")
    print(f"Batch size: {batch_size}, Image size: {img_size}, Epochs: {epochs}")
    
    # Train the model
    model.train(
        data=yaml_path,
        epochs=epochs,
        patience=20,  # Early stopping patience
        batch=batch_size,
        imgsz=img_size,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        weight_decay=0.0005,
        project=OUTPUT_DIR,
        name="bone_fracture_model",
        exist_ok=True,
        verbose=True,
        task="classify",  # Important: specify task=classify for classification
        plots=True,  # Generate plots during training
        save=True,  # Save training checkpoints
        device=device,
        cache=True,  # Cache images for faster training
        amp=True  # Use mixed precision for faster training if possible
    )
    
    print("\n=== Training Complete ===")
    
    # Validate on test set
    print("\n=== Evaluating on Test Set ===")
    test_results = model.val(data=yaml_path, split="test")
    print(f"Test accuracy: {test_results.top1:.4f}")
    
    # Generate confusion matrix and classification report
    print("\n=== Generating Evaluation Metrics ===")
    results = model.predict(TEST_DIR, save=True, save_txt=True, verbose=False)
    
    # Collect true and predicted labels
    true_labels = []
    pred_labels = []
    
    # Process test folder to get true labels
    for class_idx, class_name in enumerate(["fracture", "no_fracture"]):
        class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_dir):
            image_files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                          glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                          glob.glob(os.path.join(class_dir, "*.png"))
            true_labels.extend([class_idx] * len(image_files))
    
    # Get predicted labels
    for result in results:
        # YOLOv11s-cls outputs the class with highest probability
        pred_labels.append(int(result.probs.top1))
    
    # Make sure we have the same number of true and predicted labels
    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["fracture", "no_fracture"], 
                yticklabels=["fracture", "no_fracture"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # Generate classification report
    report = classification_report(true_labels, pred_labels, 
                                  target_names=["fracture", "no_fracture"],
                                  output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))
    print("\nClassification Report:")
    print(report_df)
    
    print(f"\nAll results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    print("=== Bone Fracture Classification with YOLOv11s-cls ===\n")
    
    # First check dataset structure
    print("Checking dataset structure...")
    check_dataset_structure()
    
    # Confirm to proceed
    response = input("\nDo you want to proceed with training? (y/n): ")
    if response.lower() == 'y':
        train_yolo_model()
    else:
        print("Training aborted.")