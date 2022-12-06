"""RSNA Brast Cancer 2022 configurations."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys
import torch
import torchvision.transforms as T

from configs import config
from configs.base_params import AbstractPipelineConfig
from src.utils.general_utils import generate_uuid4


@dataclass(frozen=False, init=True)
class Data:
    """Class for data related params."""

    root_dir: Optional[Path] = config.DATA_DIR  # data/
    url: Optional[str] = ""
    blob_file: Optional[str] = ""
    train_csv: Union[str, Path] = field(init=False)
    test_csv: Union[str, Path] = field(init=False)
    train_dir: Union[str, Path] = field(init=False)
    test_dir: Union[str, Path] = field(init=False)
    data_dir: Union[str, Path] = field(init=False)
    image_col_name: str = "image_id_final"
    image_path_col_name: str = "image_path"
    group_by: str = "patient_id"
    stratify_by: Optional[str] = None
    target_col_name: str = "cancer"
    image_extension: str = ".png"
    class_name_to_id: Optional[Dict[str, int]] = field(
        default_factory=lambda: {"benign": 0, "malignant": 1}
    )
    class_id_to_name: Optional[Dict[int, str]] = field(init=False)

    def __post_init__(self) -> None:
        """Post init method for dataclass."""
        self.data_dir = Path(config.DATA_DIR) / "rsna_breast_cancer_detection"
        self.train_dir = self.data_dir / "train"
        self.train_csv = self.data_dir / "train.csv"
        self.test_dir = self.data_dir / "test"
        self.test_csv = self.data_dir / "test.csv"
        self.class_id_to_name = {v: k for k, v in self.class_name_to_id.items()}

    @property
    def download(self) -> bool:
        """Return True if data is not downloaded."""
        return not Path(self.data_dir).exists()


@dataclass(frozen=False, init=True)
class Resampling:
    """Class for cross validation."""

    # scikit-learn resampling strategy
    resample_strategy: str = "StratifiedGroupKFold"  # same name as in scikit-learn
    resample_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_splits": 5,
            "random_state": 42,
            "shuffle": True,
        }
    )


@dataclass(frozen=False, init=True)
class DataModuleParams:
    """Class to keep track of the data loader parameters."""

    debug: bool = False

    test_loader: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

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
    pre_center_crop: int = 256
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    mixup: bool = False
    mixup_params: Optional[Dict[str, Any]] = None

    train_transforms: Optional[T.Compose] = field(init=False, default=None)
    valid_transforms: Optional[T.Compose] = field(init=False, default=None)
    test_transforms: Optional[T.Compose] = field(init=False, default=None)
    debug_transforms: Optional[T.Compose] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Post init method for dataclass."""
        self.train_transforms = T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(self.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        self.valid_transforms = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )


@dataclass(frozen=False, init=True)
class ModelParams:
    """Class to keep track of the model parameters."""

    model_name: str = "resnet50"
    # adaptor: str = "torchvision/timm"
    pretrained: bool = True
    num_classes: int = 2
    # dropout: float = 0.3  # 0.5


@dataclass(frozen=False, init=True)
class Stores:
    """A class to keep track of model artifacts."""

    project_name: str = "RSNA-Breast-Cancer-Detection-2022"
    unique_id: str = field(default_factory=generate_uuid4)
    logs_dir: Path = field(init=False)
    model_artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create the logs directory."""
        self.logs_dir = Path(config.LOGS_DIR) / self.project_name / self.unique_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.model_artifacts_dir = (
            Path(config.MODEL_ARTIFACTS) / self.project_name / self.unique_id
        )
        self.model_artifacts_dir.mkdir(parents=True, exist_ok=True)


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
    num_classes: int = 2
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

    callbacks: List[str]  # = ["EarlyStopping", "ModelCheckpoint", "LRScheduler"]


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

    def __post_init__(self) -> None:
        # see utils.set_device
        self.os = sys.platform
        if self.os == "darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_params = self.to_dict()
