import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
from pathlib import Path
from app.ai.config import ModelConfig
from app.ai.model import build_model

PRETRAINED_PREPROCESS = tf.keras.applications.efficientnet.preprocess_input


def _make_dataset(directory: Path, config: ModelConfig, augment: bool, seed: int) -> tf.data.Dataset:
    """tf.data pipeline with explicit class ordering matching config.class_names."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        labels="inferred",
        label_mode="categorical",
        class_names=config.class_names,
        batch_size=config.batch_size,
        image_size=config.input_size[:2],
        interpolation="lanczos3",
        shuffle=True,
        seed=seed,
    )
    if augment:
        aug = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomTranslation(0.1, 0.1),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.15),
                layers.RandomContrast(0.15),
            ],
            name="augmentation",
        )
        ds = ds.map(
            lambda x, y: (aug(tf.cast(x, tf.float32), training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(lambda x, y: (PRETRAINED_PREPROCESS(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Compute balanced class weights from training directory counts."""
    counts = []
    for cls in class_names:
        cls_dir = train_dir / cls
        n = len([f for f in cls_dir.iterdir() if f.is_file()]) if cls_dir.exists() else 0
        counts.append(n)
    total = sum(counts)
    weights = {}
    for i, n in enumerate(counts):
        weights[i] = total / (len(class_names) * max(n, 1))
    print(f"  Class weights: {dict(zip(class_names, [round(w, 2) for w in weights.values()]))}")
    return weights


def _compile(model: tf.keras.Model, config: ModelConfig, learning_rate: float) -> None:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc", multi_label=False),
        ],
    )


def _train_phase(
    model: tf.keras.Model,
    phase_name: str,
    config: ModelConfig,
    learning_rate: float,
    epochs: int,
    patience: int,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    checkpoint_path: Path,
    append_log: bool,
    class_weights: dict[int, float] | None = None,
) -> tf.keras.Model:
    _compile(model, config, learning_rate)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1,
        ),
        CSVLogger(str(config.model_dir / "training_log.csv"), append=append_log),
    ]
    print(f"\n=== Phase {phase_name}: LR={learning_rate:.0e}, up to {epochs} epochs ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    return model


def train_model(config: ModelConfig | None = None, resume: bool = False) -> tf.keras.Model:
    if config is None:
        config = ModelConfig()
    config.model_dir.mkdir(parents=True, exist_ok=True)
    train_dir = config.dataset_path / "train"
    val_dir = config.dataset_path / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}. Run dataset pipeline first.")

    print(f"[RetinaSense AI] Building {config.backbone} model for {config.num_classes} classes...")
    print(f"  Classes: {config.class_names}")
    train_ds = _make_dataset(train_dir, config, augment=config.augmentation, seed=config.random_seed)
    val_ds = _make_dataset(val_dir, config, augment=False, seed=config.random_seed)
    class_weights = compute_class_weights(train_dir, config.class_names)

    if resume and config.deploy_path.exists():
        print(f"[RetinaSense AI] Resuming: fine-tuning existing model from {config.deploy_path}")
        model = tf.keras.models.load_model(str(config.deploy_path))
        model = _train_phase(
            model, "2 (resume)", config, config.phase2_learning_rate,
            config.phase2_epochs, config.early_stopping_patience,
            train_ds, val_ds, config.deploy_path, append_log=True,
            class_weights=class_weights,
        )
    else:
        skip_phase1 = config.phase1_path.exists()
        if skip_phase1:
            print(f"[RetinaSense AI] Phase 1 checkpoint found: {config.phase1_path}")
            print("[RetinaSense AI] Skipping Phase 1, loading existing checkpoint...")
            model = tf.keras.models.load_model(str(config.phase1_path))
        else:
            # Phase 1: backbone fully frozen, train the head only.
            print("[RetinaSense AI] Phase 1: training head (backbone frozen)...")
            model = build_model(config, train_base=False)
            model.summary()
            model = _train_phase(
                model, "1", config, config.phase1_learning_rate,
                config.phase1_epochs, config.phase1_early_stopping_patience,
                train_ds, val_ds, config.phase1_path, append_log=False,
                class_weights=class_weights,
            )
            best_phase1 = config.phase1_path if config.phase1_path.exists() else None
            if best_phase1:
                print(f"[RetinaSense AI] Loading best Phase 1 weights from {best_phase1}")
                model = tf.keras.models.load_model(str(best_phase1))

        # Phase 2: unfreeze the last N backbone layers, fine-tune with low LR.
        print(f"[RetinaSense AI] Phase 2: fine-tuning last {config.trainable_base_layers} backbone layers...")
        fine_model = build_model(config, train_base=True)
        fine_model.set_weights(model.get_weights())
        model = _train_phase(
            fine_model, "2", config, config.phase2_learning_rate,
            config.phase2_epochs, config.early_stopping_patience,
            train_ds, val_ds, config.deploy_path, append_log=True,
            class_weights=class_weights,
        )

    if config.deploy_path.exists():
        print(f"[RetinaSense AI] Loading best Phase 2 checkpoint: {config.deploy_path}")
        model = tf.keras.models.load_model(str(config.deploy_path))
    else:
        model.save(str(config.deploy_path))
    print(f"[RetinaSense AI] Deploy model saved to {config.deploy_path}")
    return model
