"""
This file holds all the global params in the form of
dataclasses. Eventually, these can be converted to yaml config files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from src.utils import generate_uuid4

from configs import config


@dataclass(frozen=False, init=True)
class Data:
    """Class for data related params."""

    root_dir: Optional[Path] = config.DATA_DIR  # data/MNIST/...
    url: Optional[
        str
    ] = "https://storage.googleapis.com/peekingduck/data/castings_data.zip"  # https://...
    blob_file: Optional[str] = "castings.zip"
    # data_csv: str = (
    #     "/Users/reighns/gaohn/gaohn-pytorch-classification/castings_data/df.csv"
    # )
    data_csv: Union[str, Path] = field(init=False)
    data_dir: Union[str, Path] = field(init=False)

    def __post_init__(self) -> None:
        """Post init method for dataclass."""
        self.data_dir = Path(config.DATA_DIR) / "castings_data"
        self.data_csv = self.data_dir / "df.csv"


@dataclass(frozen=False, init=True)
class DataModuleParams:
    """Class to keep track of the data loader parameters."""

    debug: bool = True

    test_loader: Optional[Dict[str, Any]] = None

    debug_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )


@dataclass(frozen=False, init=True)
class AugmentationParams:
    """Class to keep track of the augmentation parameters."""

    image_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    mixup: bool = False
    mixup_params: Optional[Dict[str, Any]] = None


@dataclass(frozen=False, init=True)
class ModelParams:
    """Class to keep track of the model parameters."""

    model_name: str = "resnet18"
    # TODO: well know backbone models are usually from torchvision or timm.
    # adaptor: str = "torchvision/timm"
    pretrained: bool = True
    num_classes: int = 2  # 2
    dropout: float = 0.3  # 0.5


@dataclass(frozen=False, init=True)
class Stores:
    """A class to keep track of model artifacts."""

    project_name: str = "pytorch-training-pipeline"
    unique_id: str = field(default_factory=generate_uuid4)
    logs_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create the logs directory."""
        self.logs_dir = Path(config.LOGS_DIR) / self.project_name / self.unique_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.artifacts_dir = (
            Path(config.MODEL_ARTIFACTS) / self.project_name / self.unique_id
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CriterionParams:
    """A class to track loss function parameters."""

    train_criterion_name: str = "CrossEntropyLoss"
    valid_criterion_name: str = "CrossEntropyLoss"
    train_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.0,
        }
    )
    valid_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.0,
        }
    )


@dataclass
class OptimizerParams:
    """A class to track optimizer parameters."""

    # batch size increase 2, lr increases a factor of 2 as well.
    optimizer_name: str = "AdamW"
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "amsgrad": False,
            "weight_decay": 1e-6,
            "eps": 1e-08,
        }
    )


@dataclass
class SchedulerParams:
    """A class to track Scheduler Params."""

    scheduler_name: str = "CosineAnnealingWarmRestarts"  # Debug
    scheduler_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "T_0": 10,
            "T_mult": 1,
            "eta_min": 1e-6,
            "last_epoch": -1,
            "verbose": False,
        }
    )

    def __post_init__(self) -> None:
        """Initialize the scheduler params based on some choices given."""


@dataclass(frozen=False, init=True)
class GlobalTrainParams:
    """Global training parameters."""

    debug: bool = True
    debug_multiplier: int = 128
    epochs: int = 6  # 10 when not debug
    use_amp: bool = True
    mixup: bool = AugmentationParams().mixup
    patience: int = 3
    model_name: str = ModelParams().model_name
    num_classes: int = 2
    classification_type: str = "multiclass"


class CallbackParams:
    """Class to keep track of the callback parameters."""

    callbacks: List[str]  # = ["EarlyStopping", "ModelCheckpoint", "LRScheduler"]


@dataclass
class PipelineConfig:
    """The pipeline configuration class."""

    data: Data = Data()
    datamodule: DataModuleParams = DataModuleParams()
    augmentation: AugmentationParams = AugmentationParams()
    model: ModelParams = ModelParams()
    stores: Stores = Stores()
    global_train_params: GlobalTrainParams = GlobalTrainParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    scheduler_params: SchedulerParams = SchedulerParams()
    criterion_params: CriterionParams = CriterionParams()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # see utils.set_device


if __name__ == "__main__":
    datamodule = DataModuleParams()
    print(isinstance(datamodule, DataModuleParams))
