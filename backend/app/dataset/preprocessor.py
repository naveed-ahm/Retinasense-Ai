from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from app.dataset.config import DatasetConfig


def _green_channel_extraction(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] >= 3:
        green = img[:, :, 1].copy()
        return cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
    return img


def _apply_clahe(img: np.ndarray, clip_limit: float, grid_size: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(
    image_path: Path | str,
    output_path: Path | str | None = None,
    config: DatasetConfig | None = None,
) -> np.ndarray:
    if config is None:
        config = DatasetConfig()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.resize(img, config.target_size, interpolation=cv2.INTER_LANCZOS4)
    if config.green_channel_only:
        img = _green_channel_extraction(img)
    img = _apply_clahe(img, config.clahe_clip_limit, config.clahe_grid_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_rgb).save(str(out), format=config.output_format.upper())
    return img_rgb


def batch_preprocess(
    source_dir: Path,
    dest_dir: Path,
    config: DatasetConfig | None = None,
) -> list[dict]:
    if config is None:
        config = DatasetConfig()
    results = []
    for ext in config.valid_extensions:
        for src_path in source_dir.rglob(f"*{ext}"):
            rel = src_path.relative_to(source_dir)
            dest = dest_dir / rel.with_suffix(f".{config.output_format}")
            try:
                preprocess_image(src_path, dest, config)
                results.append({"source": str(src_path), "destination": str(dest), "status": "ok"})
            except Exception as e:
                results.append({"source": str(src_path), "destination": str(dest), "status": "error", "error": str(e)})
    return results
