from pathlib import Path
from PIL import Image
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from app.dataset.config import DatasetConfig


def _check_image(path: Path) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "valid": False,
        "size_bytes": 0,
        "dimensions": None,
        "error": None,
        "hash": None,
    }
    if not result["exists"]:
        result["error"] = "File missing"
        return result
    result["size_bytes"] = path.stat().st_size
    if result["size_bytes"] == 0:
        result["error"] = "Empty file"
        return result
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            result["dimensions"] = img.size
            result["valid"] = True
    except Exception as e:
        result["error"] = f"Corrupted: {e}"
        return result
    try:
        with open(path, "rb") as f:
            result["hash"] = hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        result["error"] = f"Hash error: {e}"
    return result


def validate_images(config: DatasetConfig) -> dict:
    image_paths = []
    for disease_dir in config.disease_dirs.values():
        if disease_dir.exists():
            for ext in config.valid_extensions:
                image_paths.extend(disease_dir.rglob(f"*{ext}"))
    if not image_paths:
        return {
            "total": 0,
            "valid": 0,
            "missing": 0,
            "corrupted": 0,
            "duplicates": 0,
            "empty": 0,
            "images": [],
            "message": "No images found in dataset directories",
        }
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_check_image, p): p for p in image_paths}
        for future in as_completed(futures):
            results.append(future.result())
    valid = [r for r in results if r["valid"]]
    corrupted = [r for r in results if r["error"] and "Corrupted" in r["error"]]
    empty = [r for r in results if r["error"] and "Empty" in r["error"]]
    missing = [r for r in results if r["error"] and "Missing" in r["error"]]
    hash_map = {}
    duplicates = []
    for r in valid:
        if r["hash"]:
            if r["hash"] in hash_map:
                duplicates.append(r)
            else:
                hash_map[r["hash"]] = r
    return {
        "total": len(results),
        "valid": len(valid),
        "corrupted": len(corrupted),
        "missing": len(missing),
        "empty": len(empty),
        "duplicates": len(duplicates),
        "images": results,
        "hash_map": hash_map,
    }
