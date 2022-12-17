"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import torch

# @dataclass(frozen=False, init=True)
# class CallbackParams:
#     """Callback params."""
#     def some_meth(self):
#         pass

#     callbacks: List[Callback] = field(
#         default_factory=lambda: [
#             self.some_meth()
#             History(),
#             MetricMeter(),
#             ModelCheckpoint(mode="max", monitor="val_Accuracy"),
#             EarlyStopping(mode="max", monitor="val_Accuracy", patience=3),
#         ]
#     )


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


@dataclass(frozen=False, init=True)
class GlobalTrainParams:
    """Train params, a lot of overlapping.
    FIXME: overlapping with other params.
    """

    epochs: int = field(init=False)

    def get_epoch(self):
        return 10

    def __post_init__(self):
        self.epochs = self.get_epoch()


# def __init__(self, epochs=self.get_epoch()):
@dataclass(frozen=False, init=True)
class AbstractGlobalTrainParams:
    """Train params, a lot of overlapping.
    FIXME: overlapping with other params.
    """

    epochs: int # uninitialized
class GlobalTrainParams(Abstract...):
   def get_epoch():
       
    def __post_init__(self):
        self.epochs = self.get_epoch()
        
# shud not use get_epoch class method
# property can be epoch