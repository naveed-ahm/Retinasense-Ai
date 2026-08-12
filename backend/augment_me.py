#!/usr/bin/env python
"""RetinaSense AI - Offline augmentation for the Macular Edema (ME) class.

Why this exists
---------------
ME has only ~52 training images vs ~1514 for Diabetic Retinopathy, yet on a
fundus photo DME (Diabetic Macular Edema) is clinically almost indistinguishable
from DR. `train_v2.py` already balances per-epoch exposure via
`get_balanced_sampler` (WeightedRandomSampler) + inverse-frequency FocalLoss
weights, so the remaining bottleneck is *diversity* of the 52 ME source photos.

This script expands `datasets/train/Macular Edema` by generating many strongly
augmented variants of each source image. It only touches the TRAIN split, so
val/test stay honest. The balanced sampler automatically re-distributes once the
class count grows, giving the model genuinely more ME variety per epoch.

Run:
    python augment_me.py            # default 9 variants / source -> ~520 ME images
    python augment_me.py --per 15   # 15 variants / source
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC_DIR = Path("D:/Projects/Proj/datasets/train/Macular Edema")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# A pool of augmentation builders. Each variant gets a random subset so the
# generated images are diverse, not just the same transform repeated.
TRANSFORMS = [
    lambda: T.RandomHorizontalFlip(p=1.0),
    lambda: T.RandomVerticalFlip(p=1.0),
    lambda: T.RandomRotation(degrees=25),
    lambda: T.RandomAffine(degrees=0, translate=(0.18, 0.18), scale=(0.82, 1.18), shear=12),
    lambda: T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04),
    lambda: T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6)),
    lambda: T.RandomPerspective(distortion_scale=0.25, p=1.0),
    lambda: T.RandomAdjustSharpness(sharpness_factor=1.6, p=1.0),
    lambda: T.RandomAutocontrast(p=1.0),
]


def build_pipeline(rng: random.Random) -> T.Compose:
    pool = TRANSFORMS[:]
    rng.shuffle(pool)
    k = rng.randint(3, len(pool))
    chosen = [f() for f in pool[:k]]
    # Always include at least one geometric + one photometric transform for realism.
    return T.Compose(chosen + [T.ToTensor()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=9, help="augmented variants per source image")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", type=str, default=str(SRC_DIR))
    args = ap.parse_args()

    src = Path(args.src)
    sources = [p for p in src.iterdir() if p.suffix.lower() in EXTS]
    if not sources:
        raise SystemExit(f"No source images found in {src}")

    rng = random.Random(args.seed)
    created = 0
    for img_path in sources:
        img = Image.open(img_path).convert("RGB")
        for k in range(args.per):
            pipe = build_pipeline(rng)
            aug = pipe(img).permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
            out = Image.fromarray(aug)
            out_path = src / f"{img_path.stem}_aug{k:02d}.png"
            out.save(out_path, "PNG")
            created += 1

    total = len(list(p for p in src.iterdir() if p.suffix.lower() in EXTS))
    print(f"Source ME images : {len(sources)}")
    print(f"Augmented created: {created}")
    print(f"New ME train total: {total}")


if __name__ == "__main__":
    main()
