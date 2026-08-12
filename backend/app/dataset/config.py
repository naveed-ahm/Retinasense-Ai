from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DatasetConfig:
    root: Path = Path("D:/Projects/Proj/datasets")
    target_size: tuple[int, int] = (512, 512)
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    valid_extensions: set[str] = field(default_factory=lambda: {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"})
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    output_format: Literal["png", "jpg", "tiff"] = "png"
    green_channel_only: bool = False

    # Display class name -> source directory name (Healthy was previously omitted!)
    class_dirs: dict[str, str] = field(default_factory=lambda: {
        "Diabetic Retinopathy": "EyePACS",
        "Glaucoma": "Glaucoma",
        "AMD": "AMD",
        "Hypertensive Retinopathy": "HypertensiveRetinopathy",
        "Macular Edema": "MacularEdema",
        "Healthy": "Healthy",
    })

    @property
    def disease_dirs(self) -> dict[str, Path]:
        return {
            "EyePACS": self.root / "EyePACS",
            "Glaucoma": self.root / "Glaucoma",
            "AMD": self.root / "AMD",
            "HypertensiveRetinopathy": self.root / "HypertensiveRetinopathy",
            "MacularEdema": self.root / "MacularEdema",
            "Healthy": self.root / "Healthy",
        }

    @property
    def split_dirs(self) -> dict[str, Path]:
        return {
            "train": self.root / "train",
            "val": self.root / "val",
            "test": self.root / "test",
        }
