"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Union, List
from pathlib import Path
import torch
import torchvision.transforms as T
import albumentations as A


@dataclass
class DataConfig(ABC):
    """Abstract Base Class."""

    root_dir: Path
    train_dir: Path
    valid_dir: Path
    test_dir: Path

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
class TrainConfig(ABC):
    """Abstract Base Class."""

    epochs: int
    classification_type: str
    monitored_metric: Dict[str, Any]
    use_amp: bool = field(default=False)
    patience: Literal["inf"] = field(default=float("inf"))


@dataclass(frozen=False, init=True)
class AbstractPipelineConfig(ABC):
    """The pipeline configuration class."""

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

    @property
    @abstractmethod
    def seed(self) -> int:
        """The seed for the experiment."""


@dataclass(frozen=False, init=True)
class PipelineConfig(AbstractPipelineConfig):
    """The pipeline configuration class."""


# @dataclass(frozen=False, init=True)
# class AbstractPipelineConfig(ABC):
#     """The pipeline configuration class."""

#     device: str = field(init=False, repr=False, default="cpu")
#     os: str = sys.platform
#     all_params: Dict[str, Any] = field(default_factory=dict)
#     data: Data

#     def to_dict(self) -> Dict[str, Any]:
#         """Recursively convert dataclass obj as dict."""
#         return asdict(self)
