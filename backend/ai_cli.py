"""
RetinaSense AI Model CLI.

Usage:
    python ai_cli.py build          # Build and compile model (no training)
    python ai_cli.py train          # Train model (requires dataset)
    python ai_cli.py train --resume # Resume training from checkpoint
    python ai_cli.py evaluate       # Evaluate on test set
    python ai_cli.py predict <img>  # Predict on a single image
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ai.config import ModelConfig
from app.ai.model import build_model
from app.ai.train import train_model
from app.ai.evaluate import evaluate_model
from app.ai.inference import ModelLoader


def cmd_build():
    config = ModelConfig()
    model = build_model(config)
    model.summary()
    print(f"\nModel built: {model.count_params():,} parameters")


def cmd_train():
    config = ModelConfig()
    resume = "--resume" in sys.argv
    model = train_model(config, resume=resume)
    print(f"Training complete. Model saved to {config.deploy_path}")


def cmd_evaluate():
    import tensorflow as tf
    config = ModelConfig()
    if config.deploy_path.exists():
        model = tf.keras.models.load_model(str(config.deploy_path))
    else:
        model = build_model(config)
        print("WARNING: No trained model found, using untrained model for structure check")
    results = evaluate_model(model, config)
    print(json.dumps({k: v for k, v in results.items()
                       if k != "confusion_matrix"}, indent=2))


def cmd_predict():
    config = ModelConfig()
    if len(sys.argv) < 3:
        print("Usage: python ai_cli.py predict <image_path>")
        return
    image_path = sys.argv[2]
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    import cv2
    loader = ModelLoader(config)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = loader.predict(image_rgb)
    print(json.dumps(result, indent=2))
    if result.get("overlay") is not None:
        out_path = "gradcam_output.png"
        cv2.imwrite(out_path, cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM overlay saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    command = sys.argv[1]
    cmds = {"build": cmd_build, "train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict}
    if command in cmds:
        cmds[command]()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
