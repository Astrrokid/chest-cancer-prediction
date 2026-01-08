from dataclasses import dataclass
from pathlib import Path
from pydantic.dataclasses import dataclass as pyd_dataclass
from pydantic import AnyUrl


@pyd_dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: str
    unzip_dir: Path


@pyd_dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    update_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@pyd_dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    update_base_model_path: Path
    training_data: Path
    params_image_size: list
    params_is_augmentation: bool
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float


@pyd_dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int