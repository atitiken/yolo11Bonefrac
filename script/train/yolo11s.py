import os
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

# ── 1. Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = r"D:\Punya dede\RM\Dataset\Merged Dataset"
SPLIT_DIR  = os.path.join(BASE_DIR, "split")
TRAIN_DIR  = os.path.join(SPLIT_DIR, "train")
VAL_DIR    = os.path.join(SPLIT_DIR, "val")
TEST_DIR   = os.path.join(SPLIT_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Load YOLOv11s‑cls backbone ───────────────────────────────────────────────
model = YOLO(r"D:\Punya dede\RM\yolo11s-cls.pt")  # Pretrained classification model

# ── 3. Training function ─────────────────────────────────────────────────────────
def train_yolo_fixed(run_name="final_run", save_results=True):
    # Fixed hyperparameters
    batch_size = 8
    img_size = 320
    epochs = 100
    patience = 10
    lr0 = 1e-3
    weight_decay = 5e-4

    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Train the model — YOLO handles full training loop
    results = model.train(
        data=SPLIT_DIR,
        task="classify",
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        optimizer="SGD",
        lr0=lr0,
        weight_decay=weight_decay,
        save=True,
        save_period=1,
        project=OUTPUT_DIR,
        name=run_name,
        exist_ok=True,
        verbose=True
    )

    if save_results:
        # ── Test evaluation
        test_metrics = model.val(split='test')

        # ── Predictions on test set
        preds = model.predict(TEST_DIR, save=True, save_txt=True, conf=0.25)

        # ── Build true & pred labels
        class_names = sorted(os.listdir(TRAIN_DIR))
        true_labels, pred_labels = [], []
        for idx, cname in enumerate(class_names):
            imgs = os.listdir(os.path.join(TEST_DIR, cname))
            true_labels += [idx] * len(imgs)
        pred_labels = [int(r.probs.top1) for r in preds]

        # ── Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
        plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))

        # ── Classification report to CSV
        report = classification_report(true_labels, pred_labels,
                                       target_names=class_names, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(run_dir, "classification_report.csv")
        )

    return results

# ── 4. Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    train_yolo_fixed()
    print("✅ Training completed!")
