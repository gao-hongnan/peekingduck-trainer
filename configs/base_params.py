"""Base class for params class. Similar to AbstractNode and Node."""
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pprint
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import torch
import torchvision.transforms as T

from configs import config
from src.utils.general_utils import generate_uuid4


@dataclass
class DataConfig(ABC):
    """Abstract Base Class."""

    root_dir: Path
    train_dir: Path
    valid_dir: Optional[Path] = field(default=None)
    test_dir: Optional[Path] = field(default=None)

    url: Optional[str] = field(default=None)


@dataclass
class ResamplingConfig(ABC):
    """Abstract Base Class."""

    resample_strategy: str
    resample_params: Dict[str, Any]


@dataclass
class DataModuleConfig(ABC):
    """Abstract Base Class."""

    debug: bool
    num_debug_samples: int

    train_loader: Dict[str, Any]
    valid_loader: Dict[str, Any]
    test_loader: Dict[str, Any]


@dataclass
class TransformConfig(ABC):
    """Abstract Base Class."""

    image_size: int

    mean: Optional[List[float]] = field(default=None)
    std: Optional[List[float]] = field(default=None)

    mixup: Optional[bool] = field(default=False)
    mixup_params: Optional[Dict[str, Any]] = field(default=None)

    cutmix: Optional[bool] = field(default=False)
    cutmix_params: Optional[Dict[str, Any]] = field(default=None)

    train_transforms: Union[T.Compose, A.Compose] = field(init=False, default=None)
    valid_transforms: Union[T.Compose, A.Compose] = field(init=False, default=None)
    test_transforms: Union[T.Compose, A.Compose] = field(init=False, default=None)


@dataclass
class ModelConfig(ABC):
    """Abstract Base Class."""

    adapter: str
    model_name: str
    num_classes: int
    pretrained: bool
    model_params: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class OptimizerConfig(ABC):
    """Abstract Base Class."""

    optimizer: str
    optimizer_params: Dict[str, Any]


@dataclass
class SchedulerConfig(ABC):
    """Abstract Base Class."""

    scheduler: str
    scheduler_params: Dict[str, Any]


@dataclass
class CriterionConfig(ABC):
    """Abstract Base Class."""

    train_criterion: str
    valid_criterion: str
    train_criterion_params: Dict[str, Any]
    valid_criterion_params: Dict[str, Any]


@dataclass
class CallbackConfig(ABC):
    """Abstract Base Class."""

    callbacks: List[str]
    # callbacks_params: Dict[str, Any]


@dataclass
class TrainConfig(ABC):
    """Abstract Base Class."""

    epochs: int
    classification_type: str
    monitored_metric: Dict[str, Any]
    use_amp: bool = field(default=False)
    patience: Literal["inf"] = field(default=float("inf"))


@dataclass
class StoresConfig(ABC):
    """Abstract Base Class."""

    project_name: str
    unique_id: str = field(init=False, default_factory=generate_uuid4)
    logs_dir: Path = field(init=False)
    model_artifacts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the store config."""
        self.logs_dir = Path(config.LOGS_DIR) / self.project_name / self.unique_id
        self.model_artifacts_dir = (
            Path(config.MODEL_ARTIFACTS) / self.project_name / self.unique_id
        )

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.model_artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=False, init=True)
class AbstractPipelineConfig(ABC):
    """The pipeline configuration class."""

    data: DataConfig
    resample: ResamplingConfig
    datamodule: DataModuleConfig
    transforms: TransformConfig
    model: ModelConfig
    optimizer_params: OptimizerConfig
    scheduler_params: SchedulerConfig
    criterion_params: CriterionConfig
    callback_params: CallbackConfig
    global_train_params: TrainConfig
    stores: StoresConfig

    device: str = field(init=False, repr=False, default="cpu")
    os: str = sys.platform
    all_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert dataclass obj as dict."""
        return asdict(self)

    def set_device(self) -> str:
        """Get the device."""
        if self.os == "darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_all_params(self) -> Dict[str, Any]:
        """Get all params."""
        return self.to_dict()

    def __post_init__(self) -> None:
        """Initialize the pipeline config.
        Currently only initializes the device."""
        self.set_device()  # assign device
        self.all_params = self.get_all_params()

    def __str__(self) -> str:
        """Get the string representation of the pipeline config."""
        return pprint.pformat(self.all_params, depth=4)

    @property
    @abstractmethod
    def seed(self) -> int:
        """The seed for the experiment."""


@dataclass(frozen=False, init=True)
class PipelineConfig(AbstractPipelineConfig):
    """The pipeline configuration class."""
