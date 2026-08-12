"""
RetinaSense AI — Model Verification on unseen images.

Predicts on every image in the held-out test set (6 classes),
records predicted class + confidence, and writes a JSON report
to app/ai/models/evaluation/verification_report.json.

Usage:
    python verify_model.py
"""
import os, json, sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from app.ai.config import ModelConfig
from app.ai.inference import ModelLoader


def main() -> None:
    config = ModelConfig()
    loader = ModelLoader(config)
    if loader.model is None:
        print("ERROR: trained model not found; cannot verify.")
        sys.exit(1)

    test_dir = config.dataset_path / "test"
    if not test_dir.exists():
        print(f"ERROR: test dir not found: {test_dir}")
        sys.exit(1)

    results = []
    per_class = {name: {"correct": 0, "total": 0, "conf": []} for name in config.class_names}

    for class_name in config.class_names:
        cls_dir = test_dir / class_name
        if not cls_dir.exists():
            print(f"WARNING: test folder missing for class: {class_name}")
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred = loader.predict(image_rgb)
            predicted = pred["diagnosis"]
            confidence = pred["confidence"]
            correct = predicted == class_name
            per_class[class_name]["total"] += 1
            per_class[class_name]["conf"].append(confidence)
            if correct:
                per_class[class_name]["correct"] += 1
            results.append({
                "image": str(img_path),
                "true_label": class_name,
                "predicted_label": predicted,
                "confidence": confidence,
                "correct": correct,
            })

    summary = {}
    total_correct = 0
    total_samples = 0
    conf_all = []
    for name in config.class_names:
        pc = per_class[name]
        acc = pc["correct"] / pc["total"] if pc["total"] else 0.0
        avg_conf = float(np.mean(pc["conf"])) if pc["conf"] else 0.0
        summary[name] = {
            "samples": pc["total"],
            "correct": pc["correct"],
            "accuracy": round(acc, 4),
            "avg_confidence_correct": round(avg_conf, 2) if acc else None,
        }
        total_correct += pc["correct"]
        total_samples += pc["total"]
        conf_all += pc["conf"]

    overall = {
        "samples": total_samples,
        "correct": total_correct,
        "accuracy": round(total_correct / total_samples, 4) if total_samples else 0.0,
        "avg_confidence_all": round(float(np.mean(conf_all)), 2) if conf_all else 0.0,
    }

    out = {
        "model": str(config.deploy_path),
        "input_size": list(config.input_size),
        "overall": overall,
        "per_class": summary,
        "predictions": results,
    }
    out_path = config.model_dir / "evaluation" / "verification_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n[Model Verification — Unseen Test Images]")
    print(f"  Overall accuracy: {overall['accuracy']:.2%} ({overall['correct']}/{overall['samples']})")
    print(f"  Avg confidence (all): {overall['avg_confidence_all']:.1f}%")
    for name in config.class_names:
        s = summary[name]
        print(f"  {name:30s} acc={s['accuracy']:.2%} ({s['correct']}/{s['samples']})  avg_conf={s['avg_confidence_correct']}%")
    print(f"\n  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
