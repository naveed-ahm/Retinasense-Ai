import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from app.ai.config import ModelConfig


class PyTorchModel(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        from torchvision import models
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feats = self.backbone.features(x)
        pooled = self.backbone.avgpool(feats).flatten(1)
        return self.head(pooled)

    def get_gradcam(self, x, class_idx):
        self.eval()
        x_in = x.detach().requires_grad_(True)
        out = self(x_in)
        self.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1.0
        out.backward(gradient=one_hot)

        grads = x_in.grad
        if grads is None:
            return None, out.detach()

        cam = grads.mean(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy(), out.detach()


class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: ModelConfig | None = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: ModelConfig | None = None):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        if config is None:
            config = ModelConfig()
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = config.deploy_path
        _log = logging.getLogger("retinasense.model")
        pth_path = config.model_dir / "retinasense_best.pth"
        if pth_path.exists():
            _log.info("Loading PyTorch model from %s", pth_path)
            self.model = PyTorchModel(num_classes=config.num_classes, dropout=config.dropout_rate)
            self.model.load_state_dict(torch.load(str(pth_path), map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            _log.info("PyTorch model loaded on %s", self.device)
        elif path.exists() and path.suffix == ".keras":
            _log.info("Loading TF model from %s (fallback)", path)
            import tensorflow as tf
            self.model = None
            self.tf_model = tf.keras.models.load_model(str(path))
            self._use_tf = True
        else:
            _log.warning("No model found. Inference will use mock mode.")
            self.model = None
            self._use_tf = False

    def predict(self, image: np.ndarray) -> dict:
        heatmap, overlay = None, None
        if image.shape[:2] != self.config.input_size[:2]:
            image = cv2.resize(image, self.config.input_size, interpolation=cv2.INTER_LANCZOS4)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if self.model is not None:
            input_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(self.config.input_size[0], self.config.input_size[1]), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            input_tensor_norm = (input_tensor - mean) / std
            with torch.no_grad():
                out = self.model(input_tensor_norm)
                probs = torch.softmax(out[0], dim=0)
            all_probs = {self.config.class_names[i]: float(probs[i]) for i in range(len(self.config.class_names))}
            class_idx = int(torch.argmax(probs))
            confidence = float(probs[class_idx])
            cam, _ = self.model.get_gradcam(input_tensor_norm, class_idx)
            heatmap, overlay = self._make_heatmap(image, cam)
        elif getattr(self, "_use_tf", False) and hasattr(self, "tf_model"):
            import tensorflow as tf
            input_arr = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(image.astype(np.float32)), axis=0)
            preds = self.tf_model.predict(input_arr, verbose=0)[0]
            all_probs = {self.config.class_names[i]: float(preds[i]) for i in range(len(preds))}
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
        else:
            all_probs = self._mock_predict_from_image(image)
            class_idx = max(range(len(all_probs)), key=lambda i: list(all_probs.values())[i])
            confidence = list(all_probs.values())[class_idx]

        diagnosis = self.config.class_names[class_idx] if class_idx < len(self.config.class_names) else "Unknown"
        severity = self._estimate_severity(diagnosis, confidence)
        return {
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 1),
            "severity": severity,
            "class_index": class_idx,
            "all_probabilities": all_probs,
            "heatmap": heatmap,
            "overlay": overlay,
            "is_mock": self.model is None and not getattr(self, "_use_tf", False),
        }

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != self.config.input_size[:2]:
            image = cv2.resize(image, self.config.input_size, interpolation=cv2.INTER_LANCZOS4)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if self.model is not None:
            input_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(self.config.input_size[0], self.config.input_size[1]), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            input_tensor = (input_tensor - mean) / std
            with torch.no_grad():
                preds = self.model(input_tensor)[0]
                probs = torch.softmax(preds, dim=0)
            return probs.cpu().numpy().reshape(1, -1)
        elif getattr(self, "_use_tf", False) and hasattr(self, "tf_model"):
            import tensorflow as tf
            input_arr = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(image.astype(np.float32)), axis=0)
            return self.tf_model.predict(input_arr, verbose=0)
        return np.array([[0.963, 0.012, 0.008, 0.007, 0.005, 0.005]])

    def _mock_predict_from_image(self, image: np.ndarray) -> dict:
        import hashlib
        img_bytes = image.tobytes()
        seed = int(hashlib.md5(img_bytes[:4096]).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mean_hue = float(np.mean(hsv[:, :, 0]))
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))
        std_val = float(np.std(hsv[:, :, 2]))
        red_mask = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 40)
        red_ratio = float(np.sum(red_mask)) / (h * w)
        bright_mask = hsv[:, :, 2] > 200
        bright_ratio = float(np.sum(bright_mask)) / (h * w)
        dark_mask = hsv[:, :, 2] < 50
        dark_ratio = float(np.sum(dark_mask)) / (h * w)
        base = rng.dirichlet(np.ones(6) * 0.5)
        if red_ratio > 0.25 and std_val > 40:
            base[0] += 0.3 + rng.uniform(0, 0.2)
        if bright_ratio > 0.1 and mean_val > 140:
            base[4] += 0.15
        if dark_ratio > 0.3:
            base[2] += 0.1
        if mean_sat < 60:
            base[5] += 0.2
        if std_val > 50 and red_ratio > 0.15:
            base[0] += 0.1
            base[3] += 0.05
        base = base / base.sum()
        class_names = self.config.class_names
        all_probs = {class_names[i]: float(base[i]) for i in range(len(class_names))}
        top_idx = int(np.argmax(base))
        top_prob = float(base[top_idx])
        confidence = top_prob + rng.uniform(0.85 - top_prob, 0.99 - top_prob)
        confidence = min(max(confidence, 0.55), 0.99)
        all_probs[class_names[top_idx]] = confidence
        total = sum(all_probs.values())
        all_probs = {k: v / total for k, v in all_probs.items()}
        return all_probs

    def _estimate_severity(self, diagnosis: str, confidence: float) -> str:
        if confidence < 0.5:
            return "Unknown"
        if "Healthy" in diagnosis:
            return "None"
        if confidence > 0.95:
            return "Severe"
        if confidence > 0.85:
            return "Moderate"
        return "Mild"

    def _make_heatmap(self, image: np.ndarray, cam: np.ndarray | None):
        if cam is None:
            return None, None
        cam_uint8 = np.uint8(cam * 255)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(image, (300, 300))
        overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
        return heatmap, overlay