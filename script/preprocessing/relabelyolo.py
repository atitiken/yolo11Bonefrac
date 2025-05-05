#!/usr/bin/env python3
from pathlib import Path

# adjust to your root
ROOT = Path(r"D:\Punya dede\RM\Dataset\Merged Dataset\split\images")
SPLITS = ["train","val","test"]
CLASS_MAP = {"no_fracture":0, "fracture":1}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff"}

for split in SPLITS:
    for cls, idx in CLASS_MAP.items():
        img_dir = ROOT / split / cls
        lbl_dir = ROOT / "labels" / split / cls
        if not img_dir.exists():
            print(f"⚠️  Missing {img_dir}, skipping")
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS: continue
            # full‐image box: x_center=0.5,y_center=0.5,w=1.0,h=1.0
            (lbl_dir / f"{img.stem}.txt").write_text(f"{idx} 0.5 0.5 1.0 1.0\n")
print("✅ Labels written under split/labels/") 
