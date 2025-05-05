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

# ── 1. Paths and Configuration ───────────────────────────────────────────────────
BASE_DIR   = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split"
SPLIT_DIR  = os.path.join(BASE_DIR, "split")
TRAIN_DIR  = os.path.join(SPLIT_DIR, "train")
VAL_DIR    = os.path.join(SPLIT_DIR, "val")
TEST_DIR   = os.path.join(SPLIT_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Configuration ───────────────────────────────────────────────────────────
# Smaller model for faster training & better generalization with small datasets
MODEL_PATH = r"D:\Punya dede\RM\yolo11n-cls.pt"  # Using yolo11n-cls instead of yolo11s-cls
NUM_FOLDS = 5  # Number of folds for cross-validation
RANDOM_SEED = 42

# ── 3. Helper Functions ───────────────────────────────────────────────────────
def seed_everything(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def analyze_dataset(train_dir):
    """Analyze class distribution in the dataset"""
    class_counts = {}
    class_names = sorted(os.listdir(train_dir))
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        class_counts[class_name] = len(os.listdir(class_path))
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Calculate weights for class balancing
    total = sum(class_counts.values())
    class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
    
    return class_counts, class_weights, class_names

def prepare_kfold_datasets(train_dir, k=5):
    """Prepare datasets for k-fold cross-validation"""
    class_names = sorted(os.listdir(train_dir))
    all_images = []
    
    # Collect all image paths with their class labels
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_dir, class_name)
        image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        for img_path in image_paths:
            all_images.append((img_path, idx, class_name))
    
    # Shuffle the dataset
    seed_everything(RANDOM_SEED)
    random.shuffle(all_images)
    
    # Split into k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    
    # Convert to path objects for easier handling
    img_paths = [item[0] for item in all_images]
    
    for train_idx, val_idx in kf.split(img_paths):
        fold_train = [all_images[i] for i in train_idx]
        fold_val = [all_images[i] for i in val_idx]
        folds.append((fold_train, fold_val))
    
    return folds, class_names

def create_fold_dataset(fold_data, fold_idx, base_dir):
    """Create a temporary dataset directory for the current fold"""
    fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
    fold_train_dir = os.path.join(fold_dir, "train")
    fold_val_dir = os.path.join(fold_dir, "val")
    
    # Create directories
    for class_name in os.listdir(TRAIN_DIR):
        os.makedirs(os.path.join(fold_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(fold_val_dir, class_name), exist_ok=True)
    
    # Copy training images
    for img_path, _, class_name in fold_data[0]:  # training data
        dst_path = os.path.join(fold_train_dir, class_name, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
    
    # Copy validation images
    for img_path, _, class_name in fold_data[1]:  # validation data
        dst_path = os.path.join(fold_val_dir, class_name, os.path.basename(img_path))
        shutil.copy(img_path, dst_path)
    
    return fold_dir

def cleanup_fold_dataset(fold_dir):
    """Clean up temporary fold directory"""
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

# ── 4. Training function ─────────────────────────────────────────────────────────
def train_yolo_fold(
    fold_idx,
    fold_dir,
    class_weights,
    run_name="kfold_run",
    freeze_layers=True,
    train_workers=4,
    val_workers=0,
    accumulate=2
):
    """Train YOLO model on a specific fold"""
    # Create a new model instance for each fold
    model = YOLO(MODEL_PATH)
    
    # Calculate class weights for focal loss
    weight_str = ", ".join([f"{w:.3f}" for w in class_weights.values()])
    print(f"Using class weights: [{weight_str}]")
    
    # Hyperparameters - Adjusted for small datasets
    batch_size = 8
    img_size = 320
    epochs = 100
    patience = 15  # Increased patience for smoother convergence
    lr0 = 5e-4  # Lower learning rate for better stability
    weight_decay = 1e-4  # Increased regularization
    
    fold_run_name = f"{run_name}_fold{fold_idx}"
    run_dir = os.path.join(OUTPUT_DIR, fold_run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Freeze layers if specified - Compatible with YOLOv11 cls architecture
    if freeze_layers:
        try:
            # Attempt to access the model architecture
            model_params = list(model.model.parameters())
            # Calculate how many layers to freeze (70% of layers)
            freeze_count = int(len(model_params) * 0.7)
            
            # Freeze the first 70% of layers
            for i, param in enumerate(model_params):
                if i < freeze_count:
                    param.requires_grad = False
            
            print(f"Successfully froze {freeze_count} layers ({freeze_count}/{len(model_params)} - 70%)")
        except AttributeError as e:
            # Fallback method if the above approach doesn't work
            print(f"Cannot access model layers directly. Using alternative freeze method.")
            
            # Try accessing model through named parameters
            frozen = 0
            total = 0
            for name, param in model.model.named_parameters():
                total += 1
                # Freeze parameters in early layers (usually start with features, conv, bn, etc.)
                if any(x in name for x in ['features.0', 'features.1', 'features.2', 'features.3', 'features.4']):
                    param.requires_grad = False
                    frozen += 1
            
            print(f"Froze {frozen}/{total} parameters in early layers")
        except Exception as e:
            print(f"Warning: Layer freezing failed with error: {e}")
            print("Continuing without layer freezing")
    
    # Train the model with conservative augmentations for medical images
    results = model.train(
        data=fold_dir,
        task="classify",
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        optimizer="AdamW",  # AdamW often performs better than SGD for small datasets
        lr0=lr0,
        weight_decay=weight_decay,
        save=True,
        save_period=20,
        project=OUTPUT_DIR,
        name=fold_run_name,
        exist_ok=True,
        verbose=True,
        augment=False,  # Enable augmentations
        dropout=0.2,   # Add dropout for regularization
        cos_lr=True,   # Cosine learning rate schedule
        degrees=5.0,   # Conservative rotation for medical images
        fliplr=0.5, 
        workers=train_workers,
        accumulate=accumulate,   # Horizontal flip (often acceptable for bone images)
    )
    print(f"\n▶ Fold {fold_idx}: CLEAR CACHE & VALIDATE (workers={val_workers})")
    torch.cuda.empty_cache()
    val_metrics = model.val(
        data=fold_dir,
        workers=val_workers
    )
    
    return model, val_metrics

def evaluate_model(model, test_dir, class_names, run_dir):
    """Evaluate the model on the test set with detailed metrics"""
    # Test evaluation
    test_metrics = model.val(data=test_dir)
    
    # Predictions on test set
    preds = model.predict(test_dir, save=True, save_txt=True, conf=0.25)
    
    # Build true & pred labels
    true_labels, pred_labels = [], []
    pred_scores = []
    for idx, cname in enumerate(class_names):
        imgs = os.listdir(os.path.join(test_dir, cname))
        true_labels += [idx] * len(imgs)
    
    for r in preds:
        pred_labels.append(int(r.probs.top1))
        pred_scores.append(float(r.probs.top1conf))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
    plt.close()
    
    # Classification report with precision, recall, F1
    report = classification_report(true_labels, pred_labels,
                                  target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(run_dir, "classification_report.csv"))
    
    # Calculate mAP50 (in classification this is similar to precision at threshold 0.5)
    # Initialize counters for each class
    class_metrics = {cls_name: {"TP": 0, "FP": 0, "FN": 0, "precision": 0, "recall": 0} 
                     for cls_name in class_names}
    
    # Calculate TP, FP, FN for each class at confidence threshold 0.5
    for true_label, pred_label, confidence in zip(true_labels, pred_labels, pred_scores):
        true_cls = class_names[true_label]
        pred_cls = class_names[pred_label]
        
        if confidence >= 0.5:  # Only count predictions with confidence >= 0.5
            if true_cls == pred_cls:
                class_metrics[true_cls]["TP"] += 1
            else:
                class_metrics[true_cls]["FN"] += 1
                class_metrics[pred_cls]["FP"] += 1
        else:
            class_metrics[true_cls]["FN"] += 1
    
    # Calculate precision and recall for each class
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
    
    # Calculate mAP50 (mean Average Precision at 0.5 threshold)
    map50 = np.mean(map50_values) if map50_values else 0
    
    # Extract classification loss from the test metrics
    cls_loss = getattr(test_metrics, 'loss', None)
    
    # Create a detailed metrics dictionary
    detailed_metrics = {
        "accuracy": test_metrics.top1,
        "mAP50": map50,
        "classification_loss": cls_loss,
        "per_class": class_metrics
    }
    
    # Save detailed metrics to a CSV file
    pd.DataFrame(detailed_metrics).to_csv(os.path.join(run_dir, "detailed_metrics.csv"))
    
    # Print the metrics
    print(f"\nDetailed Metrics:")
    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  mAP50: {detailed_metrics['mAP50']:.4f}")
    print(f"  Classification Loss: {detailed_metrics['classification_loss']:.4f}")
    print("\nPer-class metrics at threshold 0.5:")
    for cls_name, metrics in class_metrics.items():
        print(f"  {cls_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return test_metrics, report_df, detailed_metrics

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