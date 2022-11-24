"""MNIST configurations."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from configs import config
from src.utils.general_utils import generate_uuid4


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


@dataclass(frozen=False, init=True)
class ModelParams:
    """Class to keep track of the model parameters."""

    # TODO: well know backbone models are usually from torchvision or timm.
    # model_name: str = "resnet18"
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
    artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create the logs directory."""
        self.logs_dir = Path(config.LOGS_DIR) / self.project_name / self.unique_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.artifacts_dir = (
            Path(config.MODEL_ARTIFACTS) / self.project_name / self.unique_id
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=False, init=True)
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

    debug: bool = False
    debug_multiplier: int = 128
    epochs: int = 3  # 10 when not debug
    use_amp: bool = True
    mixup: bool = False
    patience: int = 3
    model_name: str = "custom"
    num_classes: int = 10
    classification_type: str = "multiclass"


@dataclass(frozen=False, init=True)
class CallbackParams:
    """Callback params."""

    callbacks: List[str]  # = ["EarlyStopping", "ModelCheckpoint", "LRScheduler"]


@dataclass(frozen=False, init=True)
class PipelineConfig:
    """Pipeline config."""

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
