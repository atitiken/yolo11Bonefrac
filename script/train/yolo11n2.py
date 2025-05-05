import os
import shutil
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from collections import Counter
import glob
from pathlib import Path
import json

# ── NEW IMPORTS FOR BALANCED SAMPLER ─────────────────────────────────────────
from torch.utils.data import DataLoader, WeightedRandomSampler
from ultralytics.models.yolo.classify.train import ClassificationTrainer

# ── 0. CUSTOM TRAINER FOR CLASS BALANCING ─────────────────────────────────────
class BalancedClassificationTrainer(ClassificationTrainer):
    def get_dataloader(self, dataset, batch_size, rank=0, mode='train', **kwargs):
        """
        Overrides base get_dataloader signature:
          dataset    – an instance of the dataset
          batch_size – batch size
          rank       – for distributed training
          mode       – 'train' or 'val'
          **kwargs   – any other arguments (e.g., workers, shuffle, etc.)
        """
        # First, get the default dataloader
        dl = super().get_dataloader(dataset, batch_size, rank, mode, **kwargs)

        # If we're in training mode, inject our WeightedRandomSampler
        if mode == 'train':
            # Extract labels from the underlying dataset
            # `dataset.samples` should be a list of (path, class_idx)
            labels = [item[1] for item in dl.dataset.samples]

            # Compute inverse-frequency class weights
            counts = np.bincount(labels)
            class_weights = 1.0 / counts
            sample_weights = [class_weights[lbl] for lbl in labels]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            return DataLoader(
                dataset=dl.dataset,
                batch_size=dl.batch_size,
                sampler=sampler,
                num_workers=dl.num_workers,
                pin_memory=dl.pin_memory
            )

        # Otherwise (mode='val'), just return the default loader
        return dl

# ── 1. Paths and Configuration ─────────────────────────────────────────────────
BASE_DIR   = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split"
SPLIT_DIR  = os.path.join(BASE_DIR, "split")
TRAIN_DIR  = os.path.join(SPLIT_DIR, "train")
VAL_DIR    = os.path.join(SPLIT_DIR, "val")
TEST_DIR   = os.path.join(SPLIT_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Configuration ───────────────────────────────────────────────────────────
MODEL_PATH   = r"D:\Punya dede\RM\yolo11n-cls.pt"
NUM_FOLDS    = 5
RANDOM_SEED  = 42

# ── 3. Helper Functions ────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)

def analyze_dataset(train_dir):
    class_counts = {c: len(os.listdir(os.path.join(train_dir, c)))
                    for c in sorted(os.listdir(train_dir))}
    print("Class distribution:")
    for c, cnt in class_counts.items():
        print(f"  {c}: {cnt}")
    total = sum(class_counts.values())
    class_weights = {cls: total/(len(class_counts)*cnt)
                     for cls, cnt in class_counts.items()}
    return class_counts, class_weights, sorted(class_counts.keys())

def prepare_kfold_datasets(train_dir, k=5):
    class_names = sorted(os.listdir(train_dir))
    all_images  = []
    for idx, cls in enumerate(class_names):
        for img in os.listdir(os.path.join(train_dir, cls)):
            all_images.append((os.path.join(train_dir, cls, img), idx, cls))
    seed_everything(RANDOM_SEED)
    random.shuffle(all_images)
    img_paths = [x[0] for x in all_images]
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for train_idx, val_idx in kf.split(img_paths):
        folds.append(([all_images[i] for i in train_idx],
                      [all_images[i] for i in val_idx]))
    return folds, class_names

def create_fold_dataset(fold_data, fold_idx, base_dir):
    fold_dir       = os.path.join(base_dir, f"fold_{fold_idx}")
    fold_train_dir = os.path.join(fold_dir, "train")
    fold_val_dir   = os.path.join(fold_dir, "val")
    for cls in os.listdir(TRAIN_DIR):
        os.makedirs(os.path.join(fold_train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(fold_val_dir,   cls), exist_ok=True)
    for img_path, _, cls in fold_data[0]:
        shutil.copy(img_path, os.path.join(fold_train_dir, cls, os.path.basename(img_path)))
    for img_path, _, cls in fold_data[1]:
        shutil.copy(img_path, os.path.join(fold_val_dir,   cls, os.path.basename(img_path)))
    return fold_dir

def cleanup_fold_dataset(fold_dir):
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

def train_yolo_fold(
    fold_idx,
    fold_dir,
    class_weights,
    run_name="kfold_run",
    freeze_layers=True,
    train_workers=4,
    val_workers=0,
    accumulate=2,
    use_cos_lr=True,
    dropout=0.2
):
    # Build overrides including project/name
    overrides = {
        'data':        fold_dir,
        'model':       MODEL_PATH,
        'task':        'classify',
        'epochs':      100,
        'batch':       8,
        'imgsz':       320,
        'optimizer':   'AdamW',
        'lr0':         5e-4,
        'weight_decay':1e-4,
        'dropout':     dropout,
        'cos_lr':      use_cos_lr,
        'augment':     False,
        'patience':    15,
        'workers':     train_workers,

        # ← Direct logs here:
        'project': OUTPUT_DIR,
        'name':    f"{run_name}_fold{fold_idx}"
    }

    # Initialize balanced trainer
    trainer = BalancedClassificationTrainer(overrides=overrides)
    print(f"▶ Training fold {fold_idx} → outputs in {OUTPUT_DIR}/{overrides['name']}")
    trainer.train()  # now logs to OUTPUT_DIR/{name}

    # Load best weights and validate as before…
    best_weights = os.path.join(OUTPUT_DIR, overrides['name'], "weights", "best.pt")
    model = YOLO(best_weights)
    torch.cuda.empty_cache()
    val_metrics = model.val(data=fold_dir, workers=val_workers)
    return model, val_metrics

def evaluate_model(model, test_dir, class_names, run_dir, conf_thresh=0.5):
    """Evaluate the model on the test set with detailed metrics and save all outputs."""
    # 1) Overall metrics via model.val()
    print("Running model.val() for overall metrics…")
    results = model.val(data=SPLIT_DIR, split="test")
    accuracy = getattr(results, "top1", None)
    print(f"✔ Top-1 Accuracy: {accuracy:.4f}")

    # 2) Gather all test image paths
    all_test_images = []
    true_labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(test_dir, cls)
        imgs = [
            os.path.join(cls_dir, f) 
            for f in os.listdir(cls_dir) 
            if os.path.splitext(f)[1].lower() in 
               {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
        ]
        all_test_images.extend(imgs)
        true_labels += [idx] * len(imgs)
    print(f"Found {len(all_test_images)} test images across {len(class_names)} classes.")

    # 3) Predictions
    print("Generating predictions on test images…")
    preds = model.predict(
        source=all_test_images,
        conf=conf_thresh,
        save=False,
        save_txt=False,
        verbose=False
    )

    pred_labels = [int(r.probs.top1) for r in preds]
    pred_scores = [float(r.probs.top1conf) for r in preds]

    # 4) Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # 5) Classification report
    report_dict = classification_report(
        true_labels, pred_labels,
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(run_dir, "classification_report.csv")
    report_df.to_csv(report_path, index=True)
    print(f"Saved classification report to {report_path}")

    # 6) mAP@0.5 calculation
    precisions = []
    for cls_idx, cls_name in enumerate(class_names):
        tp = sum(
            (t == cls_idx and p == cls_idx and s >= conf_thresh)
            for t, p, s in zip(true_labels, pred_labels, pred_scores)
        )
        fp = sum(
            (t != cls_idx and p == cls_idx and s >= conf_thresh)
            for t, p, s in zip(true_labels, pred_labels, pred_scores)
        )
        fn = sum(
            (t == cls_idx and (p != cls_idx or s < conf_thresh))
            for t, p, s in zip(true_labels, pred_labels, pred_scores)
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)
    map50 = float(np.mean(precisions))
    print(f"✔ mAP@0.5: {map50:.4f}")

    # 7) Save JSON summary
    summary = {
        "accuracy": accuracy,
        "mAP50": map50,
        "per_class_precision": dict(zip(class_names, precisions))
    }
    summary_path = os.path.join(run_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    return results, report_df, summary

def run_kfold_cross_validation(
    train_workers: int = 4,
    val_workers: int = 0,
    accumulate: int = 2,
    use_cos_lr: bool = True,
    dropout: float = 0.2
):
    """Run k-fold cross-validation with detailed metrics tracking
    and custom train/validation settings."""
    print(f"Starting {NUM_FOLDS}-fold cross-validation...")
    
    # 1) Reproducibility
    seed_everything(RANDOM_SEED)
    
    # 2) Dataset analysis
    class_counts, class_weights, class_names = analyze_dataset(TRAIN_DIR)
    
    # 3) Prepare folds
    folds, class_names = prepare_kfold_datasets(TRAIN_DIR, k=NUM_FOLDS)
    
    # 4) Create output directory
    ensemble_dir = os.path.join(OUTPUT_DIR, "kfold_ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    fold_metrics = []
    best_models  = []
    
    # 5) Loop over folds
    for fold_idx, fold_data in enumerate(folds):
        print(f"\n{'='*80}\nTraining Fold {fold_idx+1}/{NUM_FOLDS}\n{'='*80}")
        
        # a) Set up fold directories
        fold_dir = create_fold_dataset(fold_data, fold_idx, SPLIT_DIR)
        
        # b) Train & validate
        model, val_metrics = train_yolo_fold(
            fold_idx=fold_idx,
            fold_dir=fold_dir,
            class_weights=class_weights,
            run_name="kfold_run",
            freeze_layers=True,
            train_workers=train_workers,
            val_workers=val_workers,
            accumulate=accumulate,
            use_cos_lr=use_cos_lr,
            dropout=dropout
        )
        
        # c) Record metrics
        cls_loss = getattr(val_metrics, 'loss', float('nan'))
        fold_metrics.append({
            'fold':     fold_idx,
            'accuracy': val_metrics.top1,
            'loss':     cls_loss,
            'top5':     val_metrics.top5
        })
        best_models.append(model)
        
        # d) Cleanup
        cleanup_fold_dataset(fold_dir)
    
    # 6) Save fold metrics DataFrame
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(ensemble_dir, "fold_metrics.csv"), index=False)
    
    # 7) Compute & print averages
    avg_acc  = fold_df['accuracy'].mean()
    avg_loss = fold_df['loss'][~fold_df['loss'].isna()].mean()
    print(f"\nAverage Accuracy across folds: {avg_acc:.4f}")
    print(f"Average Loss     across folds: {avg_loss:.4f}")
    
    # 8) Select best fold
    best_idx   = int(fold_df['accuracy'].idxmax())
    best_model = best_models[best_idx]
    print(f"Best model from fold {best_idx+1} (accuracy={fold_df.loc[best_idx, 'accuracy']:.4f})")
    
    # 9) Final evaluation on test set
    print("\nEvaluating best model on test set...")
    test_metrics, report_df, detailed_metrics = evaluate_model(
        best_model,
        TEST_DIR,
        class_names,
        ensemble_dir
    )
    
    # 10) Export best model
    best_path = os.path.join(ensemble_dir, "best_model.torchscript")
    best_model.export(format="torchscript", fname=best_path, optimize=True)
    print(f"Best model exported to {best_path}")
    
    # 11) Plot fold‐level metrics
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fold_df['fold'] + 1, fold_df['accuracy'], 'o-', label='Accuracy')
    plt.axhline(avg_acc, color='r', linestyle='--', label=f'Avg {avg_acc:.4f}')
    plt.title('Accuracy per Fold'); plt.xlabel('Fold'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(fold_df['fold'] + 1, fold_df['loss'], 'o-', label='Loss')
    plt.axhline(avg_loss, color='r', linestyle='--', label=f'Avg {avg_loss:.4f}')
    plt.title('Loss per Fold'); plt.xlabel('Fold'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ensemble_dir, "fold_metrics_plot.png"))
    plt.close()
    
    # 12) Class‐wise performance
    print("\nClass-wise performance:")
    display_df = report_df.loc[class_names, ['precision', 'recall', 'f1-score']]
    print(display_df)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(class_names))
    w = 0.25
    plt.bar(x - w, display_df['precision'], w, label='Precision')
    plt.bar(x,     display_df['recall'],    w, label='Recall')
    plt.bar(x + w, display_df['f1-score'],  w, label='F1-Score')
    plt.xlabel('Class'); plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.xticks(x, class_names, rotation=45)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(ensemble_dir, "per_class_metrics.png"))
    plt.close()
    
    # 13) Return structured results
    return {
        'fold_metrics_df':  fold_df,
        'best_fold_idx':    best_idx,
        'test_report_df':   report_df,
        'detailed_metrics': detailed_metrics,
        'best_model':       best_model
    }


# ── 6. Model Ensembling (Optional) ─────────────────────────────────────────────────
def ensemble_prediction(models, test_dir, class_names, ensemble_dir):
    """Perform ensemble prediction by averaging predictions from all models"""
    print("\nPerforming ensemble predictions...")
    
    # Get all test images
    test_images = []
    true_labels = []
    for idx, cname in enumerate(class_names):
        class_path = os.path.join(test_dir, cname)
        imgs = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        test_images.extend(imgs)
        true_labels.extend([idx] * len(imgs))
    
    # Initialize ensemble predictions
    ensemble_preds = np.zeros((len(test_images), len(class_names)))
    
    # Get predictions from each model
    for i, model in enumerate(models):
        print(f"Getting predictions from model {i+1}/{len(models)}...")
        
        preds = model.predict(test_images, verbose=False)
        
        # Accumulate predictions
        for j, pred in enumerate(preds):
            ensemble_preds[j] += pred.probs.data.numpy()
    
    # Average predictions
    ensemble_preds /= len(models)
    
    # Get final predictions
    pred_labels = np.argmax(ensemble_preds, axis=1)
    pred_scores = np.max(ensemble_preds, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Ensemble Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(ensemble_dir, "ensemble_confusion_matrix.png"))
    plt.close()
    
    # Classification report
    report = classification_report(true_labels, pred_labels,
                                 target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(ensemble_dir, "ensemble_classification_report.csv"))
    
    # Calculate mAP50
    class_metrics = {cls_name: {"TP": 0, "FP": 0, "FN": 0, "precision": 0, "recall": 0} 
                     for cls_name in class_names}
    
    for true_label, pred_label, confidence in zip(true_labels, pred_labels, pred_scores):
        true_cls = class_names[true_label]
        pred_cls = class_names[pred_label]
        
        if confidence >= 0.5:
            if true_cls == pred_cls:
                class_metrics[true_cls]["TP"] += 1
            else:
                class_metrics[true_cls]["FN"] += 1
                class_metrics[pred_cls]["FP"] += 1
        else:
            class_metrics[true_cls]["FN"] += 1
    
    map50_values = []
    for cls_name, metrics in class_metrics.items():
        tp = metrics["TP"]
        fp = metrics["FP"]
        fn = metrics["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        class_metrics[cls_name]["precision"] = precision
        class_metrics[cls_name]["recall"] = recall
        map50_values.append(precision)
    
    map50 = np.mean(map50_values) if map50_values else 0
    
    print(f"\nEnsemble Model Results:")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  mAP50: {map50:.4f}")
    
    return report_df, map50

# ── 7. Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    
    # Set seeds for reproducibility
    seed_everything(RANDOM_SEED)
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Model: {os.path.basename(MODEL_PATH)}")
    print(f"  Number of Folds: {NUM_FOLDS}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Using Layer Freezing: Yes (70% of backbone)")
    print(f"  Medical-Safe Augmentations: Rotation (±5°), Horizontal Flip")
    print(f"  Output Directory: {OUTPUT_DIR}")
    
    # Run k-fold cross-validation
    best_model, test_metrics, detailed_metrics = run_kfold_cross_validation()
    
    print("\nResults Summary:")
    print(f"  Test Accuracy: {test_metrics.top1:.4f}")
    print(f"  mAP50: {detailed_metrics['mAP50']:.4f}")
    print(f"  F1-Score: {detailed_metrics.get('macro avg', {}).get('f1-score', 'N/A')}")
    print(f"  Classification Loss: {detailed_metrics['classification_loss']:.4f}")
    
    print("\n✅ K-fold cross-validation training completed!")