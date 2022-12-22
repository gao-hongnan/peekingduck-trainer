"""MNIST configurations."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torchvision.transforms as T

from configs import config
from configs.base_params import AbstractPipelineConfig
from src.utils.general_utils import generate_uuid4
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.history import History
from src.callbacks.metrics_meter import MetricMeter
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.callbacks.wandb_logger import WandbLogger
from src.callbacks.callback import Callback


@dataclass(frozen=False, init=True)
class Data:
    """Class for data related params."""

    root_dir: Path = Path(config.DATA_DIR)  # data/
    urls: Optional[Union[str, List[str]]] = None
    blob_file: Optional[str] = None
    pretrained_weights: Optional[str] = None  # "mnist_cnn.pt"
    data_csv: Optional[Union[str, Path]] = None
    class_name_to_id: Optional[Dict[str, int]] = None
    class_id_to_name: Optional[Dict[int, str]] = None
    download: bool = True


@dataclass(frozen=False, init=True)
class Resampling:
    """Class for cross validation."""

    # scikit-learn resampling strategy
    resample_strategy: str = "train_test_split"  # same name as in scikit-learn
    resample_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "train_size": 0.9,
            "test_size": 0.1,
            "random_state": 42,
            "shuffle": True,
        }
    )


@dataclass(frozen=False, init=True)
class DataModuleParams:
    """Class to keep track of the data loader parameters."""

    debug: bool = True
    num_debug_samples: int = 1280

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

    image_size: int = 28
    mean: List[float] = field(
        default_factory=lambda: [
            0.1307,
        ]
    )
    std: List[float] = field(
        default_factory=lambda: [
            0.3081,
        ]
    )

    mixup: bool = False
    mixup_params: Optional[Dict[str, Any]] = None

    train_transforms: Optional[T.Compose] = field(init=False, default=None)
    valid_transforms: Optional[T.Compose] = field(init=False, default=None)
    test_transforms: Optional[T.Compose] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Post init method for dataclass."""
        self.train_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(self.mean, self.std)]
        )
        self.valid_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(self.mean, self.std)]
        )
        self.test_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(self.mean, self.std)]
        )


@dataclass(frozen=False, init=True)
class ModelParams:
    """Class to keep track of the model parameters."""

    model_name: str = "custom"
    # adaptor: str = "torchvision/timm"
    # pretrained: bool = True
    num_classes: int = 10  # 2
    dropout: float = 0.3  # 0.5


@dataclass(frozen=False, init=True)
class Stores:
    """A class to keep track of model artifacts."""

    project_name: str = "MNIST"
    unique_id: str = field(default_factory=generate_uuid4)
    logs_dir: Path = field(init=False)
    model_artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create the logs directory."""
        self.logs_dir = Path(config.LOGS_DIR) / self.project_name / self.unique_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logs directory: {self.logs_dir}")

        self.model_artifacts_dir = (
            Path(config.MODEL_ARTIFACTS) / self.project_name / self.unique_id
        )
        self.model_artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=False, init=True)
class CriterionParams:
    """A class to track loss function parameters."""

    train_criterion: str = "CrossEntropyLoss"
    valid_criterion: str = "CrossEntropyLoss"
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


@dataclass(frozen=False, init=True)
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


@dataclass(frozen=False, init=True)
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
    """Train params, a lot of overlapping.
    FIXME: overlapping with other params.
    """

    epochs: int = 20  # 10 when not debug
    use_amp: bool = True
    patience: int = 1
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {
            "monitor": "val_Accuracy",
            "mode": "max",
        }
    )


@dataclass(frozen=False, init=True)
class CallbackParams:
    """Callback params."""

    callbacks: List[Callback] = field(
        default_factory=lambda: [
            History(),
            MetricMeter(),
            ModelCheckpoint(mode="max", monitor="val_Accuracy"),
            EarlyStopping(mode="max", monitor="val_Accuracy", patience=3),
            # WandbLogger(
            #     project="MNIST",
            #     entity="reighns",
            #     name="MNIST_EXP_1",
            #     config=pipeline_config.all_params,
            # ),
        ]
    )


@dataclass(frozen=False, init=True)
class PipelineConfig(AbstractPipelineConfig):
    """The pipeline configuration class."""

    device: str = field(init=False)
    seed: int = 1992
    all_params: Dict[str, Any] = field(default_factory=dict)

    data: Data = Data()
    resample: Resampling = Resampling()
    datamodule: DataModuleParams = DataModuleParams()
    augmentation: AugmentationParams = AugmentationParams()
    model: ModelParams = ModelParams()
    stores: Stores = Stores()
    global_train_params: GlobalTrainParams = GlobalTrainParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    scheduler_params: SchedulerParams = SchedulerParams()
    criterion_params: CriterionParams = CriterionParams()
    callback_params: CallbackParams = CallbackParams()
