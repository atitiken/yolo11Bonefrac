import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

def run_eigencam(
    weights_path: str,
    test_dir: str,
    output_dir: str,
    imgsz: int = 224,
    target_layer_idx: int = -2
):
    """
    Apply EigenCAM from rigvedrs/YOLO-V11-CAM to YOLO11s-cls 
    over 'fracture' and 'nofracture' test folders.
    """
    # 1) Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    for cls in ('fracture', 'nofracture'):
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    # 2) Load YOLO11s-cls in classification mode
    yolo = YOLO(weights_path, task='classify')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_model = yolo.model.to(device).eval()

    # 3) Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 4) Select the target convolutional layer
    backbone = list(pt_model.model.children())
    target_layer = backbone[target_layer_idx]

    # 5) Initialize EigenCAM
    cam = EigenCAM(model=pt_model, target_layers=[target_layer])

    # 6) Process each class folder
    for cls in ('fracture', 'nofracture'):
        src_folder = os.path.join(test_dir, cls)
        dst_folder = os.path.join(output_dir, cls)

        for fname in os.listdir(src_folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue

            img_path = os.path.join(src_folder, fname)
            try:
                # Load and normalize original image for overlay
                orig = np.float32(Image.open(img_path).convert('RGB')) / 255.0

                # Prepare input tensor
                inp = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

               