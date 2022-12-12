"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import torch


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
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")
        return device

    def get_all_params(self) -> Dict[str, Any]:
        """Get all params."""
        return self.to_dict()

    def __post_init__(self) -> None:
        """Initialize the pipeline config.
        Currently only initializes the device."""
        self.device = self.set_device()
        self.all_params = self.get_all_params()

    @property
    @abstractmethod
    def seed(self) -> int:
        """The seed for the experiment."""


@dataclass(frozen=False, init=True)
class PipelineConfig(AbstractPipelineConfig):
    """The pipeline configuration class."""
