"""Base class for params class.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=False, init=True)
class AbstractPipelineConfig(ABC):
    """The pipeline configuration class."""

    @abstractmethod
    def __post_init__(self) -> None:
        """Initialize the pipeline config.
        Currently only initializes the device."""
        raise NotImplementedError
