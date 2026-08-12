"""RetinaSense AI - PyTorch GPU Training Script
Trains EfficientNetB3 on retinal disease dataset using CUDA.
Two-phase training: head training → fine-tune backbone.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import torchvision.models as models
from pathlib import Path
import json
import csv
import time
import numpy as np
from PIL import Image
from collections import Counter


CONFIG = {
    "dataset_path": Path("D:/Projects/Proj/datasets"),
    "model_dir": Path("D:/Projects/Retinasense Ai/backend/app/ai/models"),
    "input_size": 300,
    "num_classes": 6,
    "class_names": [
        "Diabetic Retinopathy",
        "Glaucoma",
        "AMD",
        "Hypertensive Retinopathy",
        "Macular Edema",
        "Healthy",
    ],
    "batch_size": 16,
    "phase1_epochs": 12,
    "phase2_epochs": 60,
    "phase1_lr": 1e-3,
    "phase2_lr": 1e-4,
    "phase1_patience": 6,
    "phase2_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
    "dropout": 0.4,
    "backbone_unfreeze_layers": 80,
    "seed": 42,
}


class RetinalDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.samples = []
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        for cls_name in class_names:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

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
        features = self.backbone(x)
        return self.head(features)


def compute_class_weights(dataset: RetinalDataset) -> torch.Tensor:
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = []
    for i in range(num_classes):
        n = counts.get(i, 1)
        weights.append(total / (num_classes * max(n, 1)))
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"  Class weights: {dict(zip(CONFIG['class_names'], [round(w, 2) for w in weights.tolist()]))}")
    return weights


def get_transforms(train=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.Resize((CONFIG["input_size"], CONFIG["input_size"])),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((CONFIG["input_size"], CONFIG["input_size"])),
            T.ToTensor(),
            normalize,
        ])


def freeze_backbone(model, freeze=True):
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def unfreeze_last_layers(model, num_layers=80):
    layers = list(model.backbone.features)
    total = len(layers)
    for i, layer in enumerate(layers):
        if i >= total - num_layers:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False


def train_phase(model, phase_name, train_loader, val_loader, device, config, class_weights, start_epoch=0):
    print(f"\n{'='*60}")
    print(f"Phase {phase_name}: {config['phase1_epochs'] if phase_name == '1' else config['phase2_epochs']} max epochs")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["phase1_lr"] if phase_name == "1" else config["phase2_lr"],
        weight_decay=1e-4,
    )

    max_epochs = config["phase1_epochs"] if phase_name == "1" else config["phase2_epochs"]
    patience = config["phase1_patience"] if phase_name == "1" else config["phase2_patience"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=config["min_lr"])
    for _ in range(start_epoch):
        scheduler.step()

    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = config["model_dir"] / (f"retinasense_phase1.pth" if phase_name == "1" else "retinasense_best.pth")
    log_path = config["model_dir"] / f"training_log_phase{phase_name}.csv"

    log_rows = []
    if start_epoch > 0 and log_path.exists():
        with open(log_path) as f:
            reader = csv.DictReader(f)
            log_rows = list(reader)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_preds, train_labels = [], []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [{epoch+1}/{max_epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={train_loss/train_total:.4f} acc={train_correct/train_total:.4f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step()

        per_class = {}
        for i, cls in enumerate(config["class_names"]):
            cls_mask = np.array(all_labels) == i
            if cls_mask.sum() > 0:
                per_class[cls] = round((np.array(all_preds)[cls_mask] == i).mean(), 4)

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        log_rows.append(row)
        print(f"  Epoch {epoch+1}/{max_epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  Per-class: {per_class}")

        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(log_rows)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), str(best_model_path))
            print(f"  [BEST] New best: val_acc={val_acc:.4f} saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    if best_model_path.exists():
        model.load_state_dict(torch.load(str(best_model_path), weights_only=True))
        print(f"Loaded best weights from {best_model_path}")
    return model, best_val_acc


def export_onnx(model, config, device):
    model.eval()
    dummy = torch.randn(1, 3, config["input_size"], config["input_size"]).to(device)
    onnx_path = config["model_dir"] / "retinasense_model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {onnx_path}")
    return onnx_path


def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    train_ds = RetinalDataset(CONFIG["dataset_path"] / "train", CONFIG["class_names"], get_transforms(train=True))
    val_ds = RetinalDataset(CONFIG["dataset_path"] / "val", CONFIG["class_names"], get_transforms(train=False))

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    class_weights = compute_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = RetinaSenseModel(num_classes=CONFIG["num_classes"], dropout=CONFIG["dropout"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total params, {trainable:,} trainable")

    # Phase 1: train head only
    phase1_path = CONFIG["model_dir"] / "retinasense_phase1.pth"
    if phase1_path.exists():
        print(f"\n[RetinaSense AI] Phase 1 checkpoint found at {phase1_path}. Skipping Phase 1.")
        model.load_state_dict(torch.load(str(phase1_path), weights_only=True))
        p1_acc = 0.3174  # Known from previous log
    else:
        print("\n[RetinaSense AI] Phase 1: Training head (backbone frozen)...")
        freeze_backbone(model, freeze=True)
        model, p1_acc = train_phase(model, "1", train_loader, val_loader, device, CONFIG, class_weights)

    # Phase 2: unfreeze last N layers, fine-tune
    print(f"\n[RetinaSense AI] Phase 2: Fine-tuning last {CONFIG['backbone_unfreeze_layers']} backbone layers...")
    unfreeze_last_layers(model, CONFIG["backbone_unfreeze_layers"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    phase2_log = CONFIG["model_dir"] / "training_log_phase2.csv"
    resume_epoch = 0
    if phase2_log.exists():
        with open(phase2_log) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                resume_epoch = int(rows[-1]["epoch"])
                print(f"  Resuming Phase 2 from epoch {resume_epoch + 1}")

    best_p2 = CONFIG["model_dir"] / "retinasense_best.pth"
    if resume_epoch > 0 and best_p2.exists():
        print(f"  Loading best Phase 2 weights from {best_p2} for resume")
        model.load_state_dict(torch.load(str(best_p2), map_location=device, weights_only=True))

    model, p2_acc = train_phase(model, "2", train_loader, val_loader, device, CONFIG, class_weights, start_epoch=resume_epoch)

    final_path = CONFIG["model_dir"] / "retinasense_model.pth"
    torch.save(model.state_dict(), str(final_path))
    print(f"\nFinal model saved: {final_path}")

    export_onnx(model, CONFIG, device)

    results = {
        "phase1_best_acc": p1_acc,
        "phase2_best_acc": p2_acc,
        "device": str(device),
        "model": "EfficientNetB3",
        "num_classes": CONFIG["num_classes"],
        "class_names": CONFIG["class_names"],
    }
    results_path = CONFIG["model_dir"] / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    print(f"\nDone! Phase 1 best: {p1_acc:.4f}, Phase 2 best: {p2_acc:.4f}")


if __name__ == "__main__":
    main()
