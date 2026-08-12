import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from pathlib import Path
from app.ai.config import ModelConfig


def evaluate_model(
    model: tf.keras.Model,
    config: ModelConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    if config is None:
        config = ModelConfig()
    if output_dir is None:
        output_dir = config.model_dir / "evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dir = config.dataset_path / "test"
    if not test_dir.exists():
        return {"error": f"Test directory not found: {test_dir}"}
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=config.class_names,
        batch_size=config.batch_size,
        image_size=config.input_size[:2],
        shuffle=False,
        interpolation="lanczos3",
    )
    test_ds = test_ds.map(
        lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    y_pred_list, y_true_list, y_prob_list = [], [], []
    for x, y in test_ds:
        prob = model.predict(x, verbose=0)
        y_pred_list.append(np.argmax(prob, axis=1))
        y_true_list.append(np.argmax(y.numpy(), axis=1))
        y_prob_list.append(prob)
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    y_pred_prob = np.concatenate(y_prob_list)
    class_names = list(config.class_names)
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    # ROC Curves (one-vs-rest)
    n_classes = len(class_names)
    fpr, tpr, roc_auc = {}, {}, {}
    y_true_bin = tf.keras.utils.to_categorical(y_true, n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2563EB", "#4F46E5", "#0EA5E9", "#F59E0B", "#10B981", "#EF4444"]
    for i, color in enumerate(colors[:n_classes]):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    roc_path = output_dir / "roc_curves.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    # Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, color in enumerate(colors[:n_classes]):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_prob[:, i])
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{class_names[i]} (AP = {ap:.3f})")
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05],
           xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)
    pr_path = output_dir / "precision_recall_curves.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    # Compile results
    metrics_summary = {
        "accuracy": float(report.get("accuracy", 0)),
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
        "per_class": {name: report[name] for name in class_names if name in report},
        "auc_per_class": {class_names[i]: float(roc_auc[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "test_samples": len(y_true),
        "output_files": {
            "confusion_matrix": str(cm_path),
            "roc_curves": str(roc_path),
            "precision_recall": str(pr_path),
        },
    }
    print("\n[Evaluation Results]")
    print(f"  Accuracy:  {metrics_summary['accuracy']:.4f}")
    print(f"  Macro Avg Precision: {metrics_summary['macro_avg'].get('precision', 0):.4f}")
    print(f"  Macro Avg Recall:    {metrics_summary['macro_avg'].get('recall', 0):.4f}")
    print(f"  Macro Avg F1:        {metrics_summary['macro_avg'].get('f1-score', 0):.4f}")
    print(f"  Macro Avg AUC:       {np.mean(list(roc_auc.values())):.4f}")
    for cls in class_names:
        if cls in report:
            print(f"  {cls:30s} P={report[cls]['precision']:.3f} R={report[cls]['recall']:.3f} F1={report[cls]['f1-score']:.3f} AUC={roc_auc[class_names.index(cls)]:.3f}")
    return metrics_summary
