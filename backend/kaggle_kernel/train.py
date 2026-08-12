#!/usr/bin/env python
"""RetinaSense training on Kaggle GPU. Loads public fundus datasets,
merges to 6 classes, CLAHE preprocesses, splits, and trains EfficientNetB3."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import shutil
import glob
import csv
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DATA = WORK / "data"
TRAIN = DATA / "train"
VAL = DATA / "val"
TEST = DATA / "test"

CLASS_NAMES = [
    "Diabetic Retinopathy", "Glaucoma", "AMD",
    "Hypertensive Retinopathy", "Macular Edema", "Healthy",
]

LABEL_MAP = {
    "Normal": "Healthy",
    "Diabetes": "Diabetic Retinopathy",
    "Glaucoma": "Glaucoma",
    "AMD": "AMD",
    "Hypertension": "Hypertensive Retinopathy",
}

INPUT_SIZE = 300
BATCH_SIZE = 32
PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 60
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
PHASE1_PATIENCE = 6
PHASE2_PATIENCE = 10
DROPOUT = 0.4
UNFREEZE_LAYERS = 80
SEED = 42

MODEL_OUT = WORK / "retinasense_model.pth"
ONNX_OUT = WORK / "retinasense_model.onnx"


def setup_data():
    print("=== Setting up data ===")
    print(f"Input contents: {[str(p) for p in INPUT.iterdir()]}")
    DATA.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (TRAIN / cls).mkdir(parents=True, exist_ok=True)
        (VAL / cls).mkdir(parents=True, exist_ok=True)
        (TEST / cls).mkdir(parents=True, exist_ok=True)

    # Recursively find all directories matching known labels anywhere under /kaggle/input
    label_dirs = {}  # label -> list of dirs
    csv_path = None
    images_dir = None
    for root in INPUT.rglob("*"):
        if root.is_dir():
            if root.name in LABEL_MAP:
                label_dirs.setdefault(LABEL_MAP[root.name], []).append(root)
            if root.name == "images" and images_dir is None:
                images_dir = root
        elif root.is_file() and root.name.lower().endswith(".csv") and "label" in root.name.lower():
            if csv_path is None:
                csv_path = root
    print(f"Found label dirs: {[(k, len(v)) for k, v in label_dirs.items()]}")
    print(f"CSV: {csv_path}  images_dir: {images_dir}")
    added = 0
    for mapped, dirs in label_dirs.items():
        for folder in dirs:
            dest = DATA / "raw" / mapped
            dest.mkdir(parents=True, exist_ok=True)
            for img in folder.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    shutil.copy2(img, dest / img.name)
                    added += 1
    print(f"Fundus dataset: {added} images mapped")

    # ---- Dataset 2: Combined Fundus (CSV-labeled) ----
    mapped2 = 0
    if csv_path and csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # find image column and label column
                img_name = None
                label_val = None
                for k, v in row.items():
                    if not v:
                        continue
                    kk = k.lower().strip()
                    if "image" in kk or "file" in kk or "id" in kk or "img" in kk:
                        img_name = v
                    if "label" in kk or "class" in kk or "disease" in kk or "diagnosis" in kk:
                        label_val = v
                if img_name is None or label_val is None:
                    continue
                label_key = label_val.strip().split()[0] if label_val.strip() else ""
                mapped = LABEL_MAP.get(label_key)
                if mapped is None:
                    continue
                src = images_dir / img_name
                if not src.exists():
                    for ext in (".png", ".jpg", ".jpeg"):
                        cand = images_dir / f"{img_name}{ext}"
                        if cand.exists():
                            src = cand
                            break
                if not src.exists():
                    continue
                dest = DATA / "raw" / mapped
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / src.name)
                mapped2 += 1
    print(f"Combined dataset: {mapped2} images mapped")

    # ---- Merge raw into 6 class folders with pHash dedup ----
    raw = DATA / "raw"
    print("=== Deduplicating with pHash ===")
    seen = {}
    totals = Counter()
    for cls in CLASS_NAMES:
        cls_raw = raw / cls
        if not cls_raw.exists():
            continue
        dest = TRAIN / cls  # placeholder; moved later
        for img_path in cls_raw.iterdir():
            try:
                img = Image.open(img_path).convert("RGB")
                small = img.resize((16, 16))
                g = np.asarray(small.convert("L"), dtype=np.float32)
                g -= g.mean()
                bits = (g > 0).flatten()
                ph = bits @ (1 << np.arange(bits.size)) % (2 ** 64)
                key = int(ph)
            except Exception:
                continue
            if key in seen:
                continue
            seen[key] = True
            totals[cls] += 1
            (DATA / "unique" / cls).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, DATA / "unique" / cls / img_path.name)
    print(f"Unique per class: {dict(totals)}")

    # ---- CLAHE preprocess ----
    print("=== CLAHE preprocessing ===")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for cls, count in totals.items():
        src_dir = DATA / "unique" / cls
        out_dir = DATA / "proc" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.iterdir():
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img)
            l = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = Image.fromarray(img).resize((512, 512), Image.LANCZOS)
            out.save(out_dir / (img_path.stem + ".png"))

    # ---- 70/15/15 split ----
    print("=== Splitting 70/15/15 ===")
    rng = np.random.RandomState(SEED)
    for cls, count in totals.items():
        proc_dir = DATA / "proc" / cls
        files = sorted(proc_dir.iterdir())
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        for f in files[:n_train]:
            shutil.copy2(f, TRAIN / cls / f.name)
        for f in files[n_train:n_train + n_val]:
            shutil.copy2(f, VAL / cls / f.name)
        for f in files[n_train + n_val:]:
            shutil.copy2(f, TEST / cls / f.name)
        print(f"  {cls}: train={len(files[:n_train])} val={len(files[n_train:n_train+n_val])} test={len(files[n_train+n_val:])}")


class RetinalDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, class_names, transform=None):
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        for cls_name in class_names:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    self.samples.append((p, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class RetinaSenseModel(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def get_transforms(train=True):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), norm])


def compute_class_weights(ds):
    counts = Counter(label for _, label in ds.samples)
    total = sum(counts.values())
    w = {}
    for i in range(len(CLASS_NAMES)):
        n = counts.get(i, 1)
        w[i] = total / (len(CLASS_NAMES) * max(n, 1))
    return torch.tensor([w[i] for i in range(len(CLASS_NAMES))], dtype=torch.float32)


def freeze_backbone(model, freeze=True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def unfreeze_last(model, num=80):
    layers = list(model.backbone.features)
    total = len(layers)
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = i >= total - num


def train_phase(model, name, train_loader, val_loader, device, class_weights, start_epoch=0):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    lr = PHASE1_LR if name == "1" else PHASE2_LR
    max_epochs = PHASE1_EPOCHS if name == "1" else PHASE2_EPOCHS
    patience = PHASE1_PATIENCE if name == "1" else PHASE2_PATIENCE
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-7)
    for _ in range(start_epoch):
        scheduler.step()

    best_acc = 0.0
    wait = 0
    best_path = WORK / (f"retinasense_phase1.pth" if name == "1" else "retinasense_best.pth")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"[Phase {name}] Epoch {epoch+1}/{max_epochs} "
              f"train_loss={train_loss/train_total:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss/val_total:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}", flush=True)

        with open(WORK / f"training_log_phase{name}.csv", "a") as f:
            if epoch == start_epoch:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
            f.write(f"{epoch+1},{train_loss/train_total:.4f},{train_acc:.4f},{val_loss/val_total:.4f},{val_acc:.4f},{lr_now:.2e}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), str(best_path))
            print(f"  [BEST] val_acc={val_acc:.4f} saved", flush=True)
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}", flush=True)
                break

    if best_path.exists():
        model.load_state_dict(torch.load(str(best_path), weights_only=True))
    return model, best_acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    setup_data()

    # Determine usable device: some Kaggle GPUs (P100 sm_60) are unsupported by
    # the preinstalled PyTorch build. Fall back to CPU in that case.
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 7:
                device = torch.device("cuda")
                print(f"CUDA capability {cap} supported, using GPU", flush=True)
            else:
                print(f"CUDA capability {cap} < 7.0 unsupported by this PyTorch, using CPU", flush=True)
        except Exception as e:
            print(f"GPU check failed ({e}), using CPU", flush=True)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_ds = RetinalDataset(TRAIN, CLASS_NAMES, get_transforms(train=True))
    val_ds = RetinalDataset(VAL, CLASS_NAMES, get_transforms(train=False))
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)
    class_weights = compute_class_weights(train_ds)
    print(f"Class weights: {class_weights.tolist()}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = RetinaSenseModel(num_classes=len(CLASS_NAMES), dropout=DROPOUT).to(device)

    phase1_path = WORK / "retinasense_phase1.pth"
    if phase1_path.exists():
        print("Phase 1 checkpoint found, skipping", flush=True)
        model.load_state_dict(torch.load(str(phase1_path), weights_only=True))
        p1 = 0.0
    else:
        freeze_backbone(model, True)
        model, p1 = train_phase(model, "1", train_loader, val_loader, device, class_weights)

    unfreeze_last(model, UNFREEZE_LAYERS)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params Phase 2: {n_trainable:,}", flush=True)
    model, p2 = train_phase(model, "2", train_loader, val_loader, device, class_weights)

    torch.save(model.state_dict(), str(MODEL_OUT))
    print(f"Saved: {MODEL_OUT}", flush=True)

    # ONNX export
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    torch.onnx.export(model, dummy, str(ONNX_OUT), input_names=["input"], output_names=["output"], opset_version=18)
    print(f"Saved ONNX: {ONNX_OUT}", flush=True)

    with open(WORK / "training_results.json", "w") as f:
        json.dump({"phase1_best": p1, "phase2_best": p2, "device": str(device)}, f)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
