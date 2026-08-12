from pathlib import Path
import random
import shutil
from app.dataset.config import DatasetConfig


def split_dataset(config: DatasetConfig, dry_run: bool = True) -> dict:
    random.seed(config.random_seed)
    class_images: dict[str, list[Path]] = {}
    for class_name, disease_name in config.class_dirs.items():
        disease_dir = config.root / disease_name
        if not disease_dir.exists():
            continue
        preprocessed_dir = config.root / "preprocessed" / disease_name
        if preprocessed_dir.exists():
            disease_dir = preprocessed_dir
        images = []
        for ext in config.valid_extensions:
            images.extend(sorted(disease_dir.rglob(f"*{ext}")))
        if images:
            class_images[class_name] = images
    if not class_images:
        return {"message": "No images found in disease directories", "splits": {}}
    splits = {}
    for class_name, images in class_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        train_set = images[:n_train]
        val_set = images[n_train : n_train + n_val]
        test_set = images[n_train + n_val :]
        splits[class_name] = {
            "total": n,
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        }
        if not dry_run:
            for split_name, split_images in [("train", train_set), ("val", val_set), ("test", test_set)]:
                dest_base = config.split_dirs[split_name] / class_name
                dest_base.mkdir(parents=True, exist_ok=True)
                for img_path in split_images:
                    dest = dest_base / img_path.name
                    shutil.copy2(img_path, dest)
    return {
        "dry_run": dry_run,
        "message": "Split computed (dry run)" if dry_run else "Split executed",
        "splits": splits,
    }
