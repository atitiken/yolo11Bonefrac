import os
import torch
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, f1_score, accuracy_score,
                             precision_score, recall_score, roc_curve, roc_auc_score,
                             average_precision_score)
from tqdm import tqdm

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def evaluate_model(model_path, test_dir, output_dir, conf_threshold=0.25, chunk_size=8):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Loading model from {model_path}")
    model = YOLO(model_path)

    class_names = sorted(os.listdir(test_dir))
    print(f"🏷️ Classes: {class_names}")

    all_images = []
    path_to_class = {}
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(class_dir, img_file)
                all_images.append(img_path)
                path_to_class[img_path] = idx

    print(f"Found {len(all_images)} test images across {len(class_names)} classes")

    true_labels, pred_labels, pred_probs = [], [], []

    print("🔍 Running chunked predictions...")
    for i in tqdm(range(0, len(all_images), chunk_size), desc="Evaluating"):
        chunk = all_images[i:i+chunk_size]

        try:
            results = model.predict(
                source=chunk,
                conf=conf_threshold,
                half=True,
                stream=True,
                retina_masks=False,
                verbose=False
            )

            for r in results:
                img_path = r.path
                label = path_to_class.get(img_path, -1)
                true_labels.append(label)
                pred_labels.append(int(r.probs.top1))
                if len(r.probs.data) > 1:
                    pred_probs.append(float(r.probs.data[1]))
                else:
                    pred_probs.append(float(r.probs.data[0]))

            del results
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"⚠️ Error during batch {i}-{i+chunk_size}: {e}")
            continue

    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    y_prob = np.array(pred_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Handle binary ROC/PR curves
    if len(class_names) == 2:
        if "fracture" in class_names[0].lower() or "positive" in class_names[0].lower():
            y_prob_binary = y_prob
        else:
            y_prob_binary = 1 - y_prob
        try:
            roc_auc = roc_auc_score(y_true, y_prob_binary)
            avg_precision = average_precision_score(y_true, y_prob_binary)
        except:
            roc_auc = None
            avg_precision = None
    else:
        roc_auc = None
        avg_precision = None

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    if len(class_names) == 2 and roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob_binary)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
        plt.close()

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob_binary)
        plt.figure(figsize=(10, 8))
        plt.plot(recall_curve, precision_curve, label=f'PR (AP = {avg_precision:.3f})', color='green')
        plt.axhline(y=sum(y_true)/len(y_true), linestyle='--', color='navy',
                    label=f'Baseline (ratio = {sum(y_true)/len(y_true):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
        plt.close()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    pd.DataFrame({"Metric": list(metrics.keys())[:6], "Value": list(metrics.values())[:6]}).to_csv(
        os.path.join(output_dir, "metrics_summary.csv"), index=False)

    print("\n" + "="*50)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"✓ Accuracy:          {accuracy:.4f}")
    print(f"✓ Precision:         {precision:.4f}")
    print(f"✓ Recall:            {recall:.4f}")
    print(f"✓ F1 Score:          {f1:.4f}")
    if roc_auc is not None:
        print(f"✓ ROC AUC:           {roc_auc:.4f}")
    if avg_precision is not None:
        print(f"✓ Average Precision: {avg_precision:.4f}")
    print("="*50)
    print(f"📄 Saved all results to {output_dir}")

    return metrics

if __name__ == "__main__":
    BASE_DIR = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split"
    MODEL_PATH = os.path.join(BASE_DIR, "output", "kfold_run_fold4", "weights", "best.pt")
    TEST_DIR = os.path.join(BASE_DIR, "split", "test")
    OUTPUT_DIR = os.path.join(BASE_DIR, "eval tes kfold")

    print(f"Using GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    metrics = evaluate_model(MODEL_PATH, TEST_DIR, OUTPUT_DIR, conf_threshold=0.25, chunk_size=8)
    print("✅ Evaluation completed!")
