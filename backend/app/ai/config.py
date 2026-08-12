from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_dir: Path = Path("D:/Projects/Retinasense Ai/backend/app/ai/models")
    input_size: tuple[int, int] = (300, 300)
    input_shape: tuple[int, int, int] = (300, 300, 3)
    num_classes: int = 6
    class_names: list[str] = field(default_factory=lambda: [
        "Diabetic Retinopathy",
        "Glaucoma",
        "AMD",
        "Hypertensive Retinopathy",
        "Macular Edema",
        "Healthy",
    ])
    class_to_disease_dir: dict[str, str] = field(default_factory=lambda: {
        "Diabetic Retinopathy": "EyePACS",
        "Glaucoma": "Glaucoma",
        "AMD": "AMD",
        "Hypertensive Retinopathy": "HypertensiveRetinopathy",
        "Macular Edema": "MacularEdema",
        "Healthy": "Healthy",
    })
    backbone: str = "EfficientNetB3"
    weights: str = "imagenet"
    trainable_base_layers: int = 80
    dropout_rate: float = 0.3
    learning_rate: float = 1e-4
    phase1_learning_rate: float = 1e-3
    phase2_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    batch_size: int = 16
    epochs: int = 50
    phase1_epochs: int = 12
    phase2_epochs: int = 60
    early_stopping_patience: int = 10
    phase1_early_stopping_patience: int = 6
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    augmentation: bool = True
    model_filename: str = "retinasense_model.keras"
    deploy_filename: str = "retinasense_model.keras"
    phase1_filename: str = "retinasense_model_phase1.keras"
    dataset_path: Path = Path("D:/Projects/Proj/datasets")

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def deploy_path(self) -> Path:
        return self.model_dir / self.deploy_filename

    @property
    def phase1_path(self) -> Path:
        return self.model_dir / self.phase1_filename
