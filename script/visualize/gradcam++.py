import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def visualize_with_gradcam2_and_cm(
    weights_path,
    input_dir,
    output_dir,
    img_size=224,
    target_layer_idx=-2,
    cm_path="confusion_matrix.png"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLO in classification mode
    yolo = YOLO(weights_path, task='classify')
    model = yolo.model.to(device).eval()

    # Preprocessing (must exactly match eval pipeline)
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Pick the target layer
    target_layer = list(model.model.children())[target_layer_idx]

    y_true, y_pred = [], []

    # Use GradCAM++
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        for cls in sorted(os.listdir(input_dir)):
            cls_in  = os.path.join(input_dir, cls)
            cls_out = os.path.join(output_dir, cls)
            os.makedirs(cls_out, exist_ok=True)

            for img_name in sorted(os.listdir(cls_in)):
                if not img_name.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff')):
                    continue

                img_path = os.path.join(cls_in, img_name)
                try:
                    # Load and prep
                    orig_small = np.array(
                        Image.open(img_path).convert('RGB').resize((img_size, img_size))
                    ) / 255.0
                    inp = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    inp.requires_grad_()   # <-- ENABLE grad on the input!

                    # Compute CAM
                    grayscale_cam = cam(input_tensor=inp)[0]  # shape (H,W)
                    grayscale_cam = scale_cam_image(grayscale_cam)

                    # Colored overlay
                    overlay = show_cam_on_image(orig_small, grayscale_cam, use_rgb=True)
                    overlay_pil = Image.fromarray(overlay)

                    # Plain grayscale heatmap → 3‐channel
                    gray_uint8 = (grayscale_cam * 255).astype(np.uint8)
                    heatmap_pil = Image.fromarray(
                        np.stack([gray_uint8]*3, axis=-1)
                    )

                    # Prediction
                    with torch.no_grad():
                        logits = model(inp)
                        probs  = torch.softmax(logits, dim=1)
                        idx    = probs.argmax(dim=1).item()
                    pred_label = yolo.names[idx]

                    # Record for CM
                    y_true.append(cls)
                    y_pred.append(pred_label)

                    # Compose final image
                    orig_pil = Image.fromarray((orig_small * 255).astype(np.uint8))
                    w, h = orig_pil.size
                    canvas = Image.new("RGB", (w*3, h+50), "white")
                    canvas.paste(orig_pil,      (0,   0))
                    canvas.paste(heatmap_pil,   (w,   0))
                    canvas.paste(overlay_pil,   (2*w, 0))

                    # Draw text
                    draw = ImageDraw.Draw(canvas)
                    font = ImageFont.load_default()
                    text = f"GT: {cls} → Pred: {pred_label}"
                    bbox = draw.textbbox((0,0), text, font=font)
                    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                    draw.text(((3*w-tw)//2, h+10), text, fill="black", font=font)

                    # Save
                    out_name = f"{os.path.splitext(img_name)[0]}_cam.jpg"
                    canvas.save(os.path.join(cls_out, out_name))

                except Exception as e:
                    print(f"❌ Error {img_path}: {e}")

    # Build and save confusion matrix
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"📊 Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    BASE_DIR   = r"D:\Punya dede\RM\Dataset\Merged Dataset"
    WEIGHTS    = os.path.join(BASE_DIR, "output", "run1", "weights", "best.pt")
    INPUT_DIR  = os.path.join(BASE_DIR, "split", "test")
    OUTPUT_DIR = os.path.join(BASE_DIR, "gradcam2_plus_cm")

    print("Using device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    visualize_with_gradcam2_and_cm(
        weights_path=WEIGHTS,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        img_size=224,
        target_layer_idx=-2,
        cm_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
