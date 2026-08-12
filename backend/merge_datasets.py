"""Merge Kaggle datasets into existing dataset folders, filter to 6 classes, deduplicate with pHash."""
import os
import csv
import shutil
import hashlib
from pathlib import Path
from PIL import Image
import imagehash

BASE = Path(r"D:\Projects\Proj\datasets")
KAGGLE = BASE / "kaggle_downloads"
DEST = BASE  # AMD, EyePACS, Glaucoma, Healthy, HypertensiveRetinopathy, MacularEdema

# Label mapping: Kaggle folder/label -> our class folder
LABEL_MAP = {
    "AMD": "AMD",
    "A": "AMD",
    "Diabetes": "EyePACS",
    "D": "EyePACS",
    "Glaucoma": "Glaucoma",
    "G": "Glaucoma",
    "Normal": "Healthy",
    "N": "Healthy",
    "Hypertension": "HypertensiveRetinopathy",
    "H": "HypertensiveRetinopathy",
    # Discarded:
    "Cataract": None,
    "C": None,
    "Myopia": None,
    "M": None,
    "Other": None,
    "O": None,
}

def compute_phash(img_path):
    try:
        img = Image.open(img_path)
        return str(imagehash.phash(img))
    except:
        return None

def copy_image(src, dest_dir, existing_hashes):
    """Copy image if not duplicate."""
    phash = compute_phash(src)
    if phash is None:
        return False
    if phash in existing_hashes:
        return False
    existing_hashes.add(phash)
    dest = dest_dir / src.name
    if dest.exists():
        return False
    shutil.copy2(src, dest)
    return True

# Track existing hashes from current dataset
print("Computing hashes for existing dataset images...")
existing_hashes = set()
for class_dir in ["AMD", "EyePACS", "Glaucoma", "Healthy", "HypertensiveRetinopathy", "MacularEdema"]:
    d = DEST / class_dir
    if d.exists():
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
                h = compute_phash(f)
                if h:
                    existing_hashes.add(h)
print(f"Existing dataset: {len(existing_hashes)} unique images (by pHash)")

# === Dataset 1: Fundus Dataset (folder-based) ===
fundus_dir = KAGGLE / "Fundus Dataset" / "Fundus Dataset"
stats = {}
for folder_name in os.listdir(fundus_dir):
    folder_path = fundus_dir / folder_name
    if not folder_path.is_dir():
        continue
    dest_class = LABEL_MAP.get(folder_name)
    if dest_class is None:
        print(f"  SKIP: {folder_name} (not in our 6 classes)")
        continue
    dest_dir = DEST / dest_class
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_file in folder_path.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
            if copy_image(img_file, dest_dir, existing_hashes):
                count += 1
    stats[dest_class] = stats.get(dest_class, 0) + count
    print(f"  Fundus: {folder_name} -> {dest_class}: +{count} images")

# === Dataset 2: Multi Eye Disease Combined (CSV-based) ===
csv_path = KAGGLE / "Dataset" / "label_images.csv"
images_dir = KAGGLE / "Dataset" / "images"
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Get label from the 'label' column like ['N'], ['D'], etc.
        label_str = row.get('label', '').strip()
        if not label_str:
            continue
        # Parse label: ['N'] -> N
        label = label_str.strip("[]'\" ")
        dest_class = LABEL_MAP.get(label)
        if dest_class is None:
            continue
        dest_dir = DEST / dest_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        img_name = row.get('images', '').strip()
        if not img_name:
            continue
        img_path = images_dir / img_name
        if img_path.exists():
            copy_image(img_path, dest_dir, existing_hashes)

print("\n=== Final counts ===")
for cls in ["AMD", "EyePACS", "Glaucoma", "Healthy", "HypertensiveRetinopathy", "MacularEdema"]:
    d = DEST / cls
    count = len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')]) if d.exists() else 0
    print(f"  {cls}: {count}")
print(f"\nTotal unique images (by pHash): {len(existing_hashes)}")
