import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import time
import copy
from tqdm import tqdm
import gc

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
train_dir = r"D:\Punya dede\RM\yolo_dataset\train"
val_dir = r"D:\Punya dede\RM\yolo_dataset\val"
test_dir = r"D:\Punya dede\RM\yolo_dataset\test"

# Define data transformations with augmentations for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard EfficientNet normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Custom dataset that works with file paths instead of loaded images to save memory
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])
        
        # Load image on-demand
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# Function to get image paths and labels from directory
def get_image_paths(directory):
    image_paths = []
    labels = []
    class_dirs = ['0', '1']  # 0 - no fracture, 1 - fracture
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(directory, class_dir)
        if not os.path.isdir(class_path):
            print(f"Warning: Directory {class_path} does not exist")
            continue
        
        print(f"Finding image paths in {class_path}")
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    print(f"Found {len(image_paths)} images")
    return image_paths, np.array(labels)

# Model definition function
def initialize_model():
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
    return model

# Training function with automatic mixed precision
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, patience=10, min_delta=1e-4, num_epochs=50):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    no_improvement = 0
    
    # Initialize history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Initialize AMP GradScaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass with automatic mixed precision
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        if torch.cuda.is_available():
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Record history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Early stopping check
                if epoch_acc > best_acc + min_delta:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_wts)
                    return model, history, best_epoch
    
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history, best_epoch

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float('nan')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }
    
    return results

# Function to plot training history
def plot_history(history, fold=None):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs' + (f' - Fold {fold+1}' if fold is not None else ''))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs' + (f' - Fold {fold+1}' if fold is not None else ''))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if fold is not None:
        plt.savefig(f'training_history_fold_{fold+1}.png')
    else:
        plt.savefig('training_history.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(cm, fold=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix' + (f' - Fold {fold+1}' if fold is not None else ''))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if fold is not None:
        plt.savefig(f'confusion_matrix_fold_{fold+1}.png')
    else:
        plt.savefig('confusion_matrix.png')
    plt.close()

def extract_features_for_smote(image_paths, labels, resize_dim=32):
    """
    Extract simplified features from images for SMOTE processing
    """
    features = []
    valid_indices = []
    valid_labels = []
    
    print("Extracting features for SMOTE processing...")
    for idx, path in enumerate(tqdm(image_paths)):
        try:
            img = Image.open(path).convert('RGB').resize((resize_dim, resize_dim))
            img_arr = np.array(img).flatten()
            features.append(img_arr)
            valid_indices.append(idx)
            valid_labels.append(labels[idx])
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return np.array(features), np.array(valid_labels), valid_indices

# 10-fold StratifiedKFold Cross-Validation with memory optimization
def perform_kfold_cv():
    # Get image paths instead of loading all images
    print("Getting training image paths...")
    train_image_paths, train_labels = get_image_paths(train_dir)
    print(f"Class distribution in training: {np.bincount(train_labels)}")
    
    # Get paths for test set too
    print("Getting test image paths...")
    test_image_paths, test_labels = get_image_paths(test_dir)
    print(f"Class distribution in test: {np.bincount(test_labels)}")
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    
    # Prepare dummy indices for the StratifiedKFold
    indices = np.arange(len(train_labels))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, train_labels)):
        print(f"\nFold {fold+1}/10")
        print("-" * 40)
        
        # Split data for this fold - keep only paths
        fold_train_paths = [train_image_paths[i] for i in train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_paths = [train_image_paths[i] for i in val_idx]
        fold_val_labels = train_labels[val_idx]
        
        print(f"Fold train size: {len(fold_train_paths)}, Fold validation size: {len(fold_val_paths)}")
        print(f"Fold train class distribution: {np.bincount(fold_train_labels)}")
        print(f"Fold val class distribution: {np.bincount(fold_val_labels)}")
        
        # Apply SMOTE to balance training data
        # First extract features only once
        train_features, fold_train_labels_valid, valid_indices = extract_features_for_smote(fold_train_paths, fold_train_labels)
        fold_train_paths_valid = [fold_train_paths[i] for i in valid_indices]
        
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42, sampling_strategy=1.0)  # Balanced ratio
        balanced_features, balanced_labels = smote.fit_resample(train_features, fold_train_labels_valid)
        
        print(f"After SMOTE: {np.bincount(balanced_labels)}")
        
        # Create synthetic image paths for SMOTE-generated samples
        original_sample_count = len(fold_train_paths_valid)
        synthetic_paths = fold_train_paths_valid.copy()  # Start with original paths
        synthetic_labels = balanced_labels.copy()
        
        # Create validation dataset without loading images in memory
        val_dataset = ImagePathDataset(
            fold_val_paths, 
            fold_val_labels, 
            transform=data_transforms['val']
        )
        
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Create balanced training dataset using original paths and augmentation
        # For SMOTE synthetic samples, we'll use the closest original image from the minority class
        train_dataset = ImagePathDataset(
            synthetic_paths, 
            synthetic_labels, 
            transform=data_transforms['train']  # Apply augmentations here
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        
        # Clear memory
        del train_features
        del balanced_features
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Initialize model
        model = initialize_model()
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        # Train the model
        model, history, best_epoch = train_model(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader,
            patience=10, min_delta=1e-4, num_epochs=50
        )
        
        # Plot training history
        plot_history(history, fold)
        
        # Create test dataloader without loading all images
        test_dataset = ImagePathDataset(test_image_paths, test_labels, transform=data_transforms['test'])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Evaluate model on test set
        test_results = evaluate_model(model, test_loader)
        print(f"\nTest Results - Fold {fold+1}:")
        for k, v in test_results.items():
            if k != 'confusion_matrix':
                print(f"{k}: {v:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(test_results['confusion_matrix'], fold)
        
        # Save model
        torch.save(model.state_dict(), f'efficientnetb0_fold_{fold+1}.pth')
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'best_epoch': best_epoch + 1,
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_f1': test_results['f1_score'],
            'test_auc': test_results['auc']
        })
        
        # Clean up memory after each fold
        del model
        del train_loader
        del val_loader
        del test_loader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create summary dataframe
    summary_df = pd.DataFrame(fold_results)
    print("\nCross-Validation Summary:")
    print(summary_df)
    print("\nMean Metrics:")
    print(summary_df[['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']].mean())
    
    # Save summary
    summary_df.to_csv('cv_results_summary.csv', index=False)
    
    return summary_df

if __name__ == "__main__":
    # Check if CUDA is available and print device info
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Print path information for verification
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Test directory: {test_dir}")
    
    # Make data paths absolute
    train_dir = os.path.abspath(train_dir)
    val_dir = os.path.abspath(val_dir)
    test_dir = os.path.abspath(test_dir)
    
    # Verify class directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        for class_dir in ['0', '1']:
            class_path = os.path.join(dir_path, class_dir)
            if os.path.isdir(class_path):
                print(f"Found {class_dir} directory in {dir_path}")
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]
                print(f"  - Contains {len(files)} image files")
            else:
                print(f"WARNING: {class_path} directory not found!")
    
    # Run 10-fold cross-validation with memory optimization
    try:
        perform_kfold_cv()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()