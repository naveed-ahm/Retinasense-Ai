import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import csv
import itertools

from train_v2 import RetinaSenseModel, RetinalDataset, get_transforms, CONFIG


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = CONFIG
    config["model_dir"].mkdir(parents=True, exist_ok=True)

    test_ds = RetinalDataset(config["dataset_path"] / "test", config["class_names"], get_transforms(train=False))
    print(f"Test: {len(test_ds)} images")

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True,
    )

    model = RetinaSenseModel(num_classes=config["num_classes"], dropout=config["dropout"]).to(device)
    best_path = config["model_dir"] / "retinasense_best.pth"
    if not best_path.exists():
        print(f"ERROR: No checkpoint found at {best_path}")
        return
    model.load_state_dict(torch.load(str(best_path), map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {best_path}")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config["class_names"], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        print(f"\nMacro AUC-ROC: {auc:.4f}")
    except Exception as e:
        print(f"AUC-ROC error: {e}")

    per_class = {}
    for i, cls in enumerate(config["class_names"]):
        cls_mask = all_labels == i
        if cls_mask.sum() > 0:
            cls_acc = (all_preds[cls_mask] == i).mean()
            per_class[cls] = round(float(cls_acc), 4)
    print(f"\nPer-class accuracy: {per_class}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks(range(len(config["class_names"])))
    axes[0].set_yticks(range(len(config["class_names"])))
    axes[0].set_xticklabels(config["class_names"], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(config["class_names"], fontsize=8)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")
    fig.colorbar(im, ax=axes[0])

    for i, cls in enumerate(config["class_names"]):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        axes[1].plot(fpr, tpr, label=f"{cls} (AUC)", lw=1.5)
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.5)
    axes[1].set_title("ROC Curves (One-vs-Rest)")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    eval_dir = config["model_dir"] / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    fig.savefig(eval_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    fig.savefig(eval_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots saved to {eval_dir}/")

    report = {
        "test_accuracy": round(float(accuracy), 4),
        "macro_auc": round(float(auc), 4) if "auc" in dir() else None,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
        "device": str(device),
        "model": "EfficientNetB3 (PyTorch)",
        "num_classes": config["num_classes"],
        "class_names": config["class_names"],
    }
    report_path = config["model_dir"] / "evaluation" / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
