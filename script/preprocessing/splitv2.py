import os
import shutil
import random

def split_dataset(
    src_dir,         # path to folder containing class subfolders
    dst_dir,         # path to root of new train/val/test folders
    classes,         # list of class folder names, e.g. ["fracture","no_fracture"]
    ratios=(0.7,0.15,0.15),
    seed=42
):
    random.seed(seed)
    assert sum(ratios)==1.0, "ratios must sum to 1.0"

    # Create the train/val/test/<class> directories
    for split in ["train","val","test"]:
        for cls in classes:
            os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

    # For each class, split and copy
    for cls in classes:
        cls_src = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(cls_src)
                 if os.path.isfile(os.path.join(cls_src,f))]
        random.shuffle(files)

        n = len(files)
        n_train = int(ratios[0]*n)
        n_val   = int(ratios[1]*n)
        # remaining goes to test
        train_files = files[:n_train]
        val_files   = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]

        for fname in train_files:
            shutil.copy(os.path.join(cls_src,fname),
                        os.path.join(dst_dir,"train",cls,fname))
        for fname in val_files:
            shutil.copy(os.path.join(cls_src,fname),
                        os.path.join(dst_dir,"val",cls,fname))
        for fname in test_files:
            shutil.copy(os.path.join(cls_src,fname),
                        os.path.join(dst_dir,"test",cls,fname))

        print(f"{cls}: {len(train_files)} train, "
              f"{len(val_files)} val, {len(test_files)} test")

if __name__=="__main__":
    SRC_DIR = r"D:\Punya dede\RM\Dataset\newtest set\FracAtlas\images"
    DST_DIR = r"D:\Punya dede\RM\Dataset\newtest set\fracAtlas_split"
    CLASSES = ["Fractured","Non_fractured"]

    split_dataset(SRC_DIR, DST_DIR, CLASSES, ratios=(0.7,0.15,0.15), seed=42)
