"""Base class for params class.
"""
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Dict, Any


@dataclass(frozen=False, init=True)
class AbstractPipelineConfig(ABC):
    """The pipeline configuration class."""

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert dataclass obj as dict."""
        return asdict(self)

    @abstractmethod
    def __post_init__(self) -> None:
        """Initialize the pipeline config.
        Currently only initializes the device."""
        raise NotImplementedError


@dataclass(frozen=False, init=True)
class PipelineConfig(AbstractPipelineConfig):
    """The pipeline configuration class."""
