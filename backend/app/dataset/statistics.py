from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from app.dataset.config import DatasetConfig


def _analyze_single(path: Path) -> dict:
    try:
        with Image.open(path) as img:
            w, h = img.size
            arr = np.array(img.convert("RGB"))
            return {
                "path": str(path),
                "width": w,
                "height": h,
                "channels": arr.shape[2] if len(arr.shape) == 3 else 1,
                "mean_r": float(np.mean(arr[:, :, 0])),
                "mean_g": float(np.mean(arr[:, :, 1])),
                "mean_b": float(np.mean(arr[:, :, 2])),
                "std_r": float(np.std(arr[:, :, 0])),
                "std_g": float(np.std(arr[:, :, 1])),
                "std_b": float(np.std(arr[:, :, 2])),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "file_size": path.stat().st_size,
            }
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def generate_statistics(config: DatasetConfig) -> dict:
    all_images = []
    for split_name, split_dir in config.split_dirs.items():
        if split_dir.exists():
            for ext in config.valid_extensions:
                all_images.extend([(split_name, p) for p in split_dir.rglob(f"*{ext}")])
    if not all_images:
        for disease_name, disease_dir in config.disease_dirs.items():
            if disease_dir.exists():
                for ext in config.valid_extensions:
                    all_images.extend([(disease_name, p) for p in disease_dir.rglob(f"*{ext}")])
    if not all_images:
        return {"message": "No images found", "total": 0}
    per_category: dict[str, list[dict]] = defaultdict(list)
    all_results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_analyze_single, p): (cat, p) for cat, p in all_images}
        for future in as_completed(futures):
            cat = futures[future][0]
            result = future.result()
            per_category[cat].append(result)
            all_results.append(result)
    stats = {}
    for category, results in per_category.items():
        valid = [r for r in results if "error" not in r]
        if not valid:
            stats[category] = {"count": len(results), "errors": len([r for r in results if "error" in r])}
            continue
        widths = [r["width"] for r in valid]
        heights = [r["height"] for r in valid]
        means_r = [r["mean_r"] for r in valid]
        means_g = [r["mean_g"] for r in valid]
        means_b = [r["mean_b"] for r in valid]
        stats[category] = {
            "count": len(valid),
            "errors": len(results) - len(valid),
            "width": {"min": min(widths), "max": max(widths), "mean": float(np.mean(widths))},
            "height": {"min": min(heights), "max": max(heights), "mean": float(np.mean(heights))},
            "pixel_mean": {"r": float(np.mean(means_r)), "g": float(np.mean(means_g)), "b": float(np.mean(means_b))},
            "total_pixels": sum(r["width"] * r["height"] for r in valid),
            "total_size_mb": sum(r["file_size"] for r in valid) / (1024 * 1024),
        }
    return {
        "total_images": len(all_results),
        "valid_images": len([r for r in all_results if "error" not in r]),
        "categories": stats,
    }
