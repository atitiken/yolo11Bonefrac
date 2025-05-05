#!/usr/bin/env python3
"""
evaluate_yolo_test.py

Standalone evaluation of your YOLO11n-cls model on the test dataset.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model():
    # ── Paths (HARD-CODED) ────────────────────────────────────────────────
    MODEL_PATH  = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split\output\kfold_run_fold4\weights\best.pt"
    SPLIT_DIR   = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split\split"
    TEST_DIR    = os.path.join(SPLIT_DIR, "test")
    OUTPUT_DIR  = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split\output\eval_test"
    CONF_THRESH = 0.5

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load Model ────────────────────────────────────────────────────────
    print(f"Loading model from {MODEL_PATH}…")
    model = YOLO(MODEL_PATH)

    # ── Identify Classes ───────────────────────────────────────────────────
    class_names = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])
    print(f"Detected classes: {class_names}")

    # ── Overall Metrics via model.val() ───────────────────────────────────
    print("Running model.val() for overall metrics…")
    results = model.val(data=SPLIT_DIR, split="test")
    accuracy = getattr(results, "top1", None)
    print(f"✔ Top-1 Accuracy: {accuracy:.4f}")

    # ── Gather All Test Images ────────────────────────────────────────────
    # Recursively find all image files under TEST_DIR
    pattern = os.path.join(TEST_DIR, "**", "*.*")
    all_test_images = [
        f for f in glob.glob(pattern, recursive=True)
        if os.path.splitext(f)[1].lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        }
    ]
    print(f"Found {len(all_test_images)} test images.")

    # ── Predictions ───────────────────────────────────────────────────────
    print("Generating predictions on test images…")
    preds = model.predict(
        source=all_test_images,
        save=False,
        save_txt=False,
        conf=CONF_THRESH
    )

    # ── Build True & Predicted Labels ────────────────────────────────────
    true_labels, pred_labels, pred_scores = [], [], []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(TEST_DIR, cls)
        imgs = [
            f for f in os.listdir(cls_dir)
            if os.path.isfile(os.path.join(cls_dir, f))
        ]
        true_labels += [idx] * len(imgs)

    for r in preds:
        pred_labels.append(int(r.probs.top1))
        pred_scores.append(float(r.probs.top1conf))

    # ── Confusion Matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # ── Classification Report ────────────────────────────────────────────
    report_dict = classification_report(
        true_labels, pred_labels,
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    report_df.to_csv(report_path, index=True)
    print(f"Saved classification report to {report_path}")

    # ── mAP@0.5 Calculation ──────────────────────────────────────────────
    precisions = []
    for cls_idx, cls_name in enumerate(class_names):
        tp = sum(
            (t == cls_idx and p == cls_idx and conf >= CONF_THRESH)
            for t, p, conf in zip(true_labels, pred_labels, pred_scores)
        )
        fp = sum(
            (t != cls_idx and p == cls_idx and conf >= CONF_THRESH)
            for t, p, conf in zip(true_labels, pred_labels, pred_scores)
        )
        fn = sum(
            (t == cls_idx and (p != cls_idx or conf < CONF_THRESH))
            for t, p, conf in zip(true_labels, pred_labels, pred_scores)
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)
    map50 = float(np.mean(precisions))
    print(f"✔ mAP@0.5: {map50:.4f}")

    # ── Summary JSON ─────────────────────────────────────────────────────
    summary = {
        "accuracy": accuracy,
        "mAP50": map50,
        "per_class_precision": dict(zip(class_names, precisions))
    }
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    evaluate_model()
