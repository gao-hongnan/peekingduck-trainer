"""RSNA Brast Cancer 2022 configurations."""
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
from src.callbacks.base import Callback
from configs.base_params import (
    AbstractPipelineConfig,
    CallbackConfig,
    CriterionConfig,
    DataConfig,
    DataModuleConfig,
    ModelConfig,
    OptimizerConfig,
    ResamplingConfig,
    SchedulerConfig,
    StoresConfig,
    TrainConfig,
    TransformConfig,
)


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
    stratify_by: Optional[str] = "cancer"
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
        self.train_csv = self.train_dir / "train.csv"
        self.test_dir = self.data_dir / "test"
        self.test_csv = self.test_dir / "test.csv"
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
            "n_splits": 4,
            "random_state": 42,
            "shuffle": True,
        }
    )


@dataclass(frozen=False, init=True)
class DataModuleParams:
    """Class to keep track of the data loader parameters."""

    debug: bool = False
    num_debug_samples: int = 128

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
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 16,
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

    image_size: int = 512
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

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
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
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
        self.test_transforms = T.Compose(
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

    adaptor: str = "timm"
    model_name: str = "seresnext50_32x4d"
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

    train_criterion: str = "CrossEntropyLoss"
    valid_criterion: str = "CrossEntropyLoss"
    # train_criterion_params: Dict[str, Any] = field(
    #     default_factory=lambda: {
    #         "weight": torch.tensor([1, 10]).cuda().float(),
    #         "size_average": None,
    #         "ignore_index": -100,
    #         "reduce": None,
    #         "reduction": "mean",
    #         "label_smoothing": 0.3,
    #     }
    # )
    # valid_criterion_params: Dict[str, Any] = field(
    #     default_factory=lambda: {
    #         "weight": torch.tensor([1, 10]).cuda().float(),
    #         "size_average": None,
    #         "ignore_index": -100,
    #         "reduce": None,
    #         "reduction": "mean",
    #         "label_smoothing": 0.3,
    #     }
    # )
    train_criterions: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": torch.tensor([1, 10]).cuda().float(),
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.1,
        }
    )
    valid_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": torch.tensor([1, 10]).cuda().float(),
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.1,
        }
    )


@dataclass(frozen=False, init=True)
class OptimizerParams:
    """A class to track optimizer parameters."""

    # batch size increase 2, lr increases a factor of 2 as well.
    optimizer_name: str = "AdamW"
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4 / 2,  # 32 -> 16 so lr/2
            "betas": (0.9, 0.999),
            "amsgrad": False,
            "weight_decay": 1e-6,
            "eps": 1e-08,
            # "warmup_prop": 0.1,
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
    epochs: int = 10  # 10 when not debug
    use_amp: bool = True
    patience: int = 10
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {
            "monitor": "val_AUROC",
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
            ModelCheckpoint(mode="max", monitor="val_AUROC"),
            EarlyStopping(mode="max", monitor="val_AUROC", patience=10),
        ]
    )


@dataclass(frozen=False, init=True)
class PipelineConfig(AbstractPipelineConfig):
    """The pipeline configuration class."""

    data: DataConfig = Data()
    resample: ResamplingConfig = Resampling()
    datamodule: DataModuleConfig = DataModuleParams()
    transforms: TransformConfig = AugmentationParams()
    model: ModelConfig = ModelParams()
    stores: StoresConfig = Stores()
    global_train_params: TrainConfig = GlobalTrainParams()
    optimizer_params: OptimizerConfig = OptimizerParams()
    scheduler_params: SchedulerConfig = SchedulerParams()
    criterion_params: CriterionParams = CriterionParams()
    callback_params: CriterionConfig = CallbackParams()

    device: str = field(init=False)
    seed: int = 1992
    all_params: Dict[str, Any] = field(default_factory=dict)
