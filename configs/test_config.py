"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, MISSING
from typing import Any, Dict

import torch


@dataclass
class TrainConfig(ABC):
    """Abstract Base Class."""

    epochs: int = field(init=False)
    use_amp: bool = field(init=False, default=False)
    patience: int = field(init=False, default=3)
    classification_type: str = field(init=False)
    monitored_metric: Dict[str, Any] = field(init=False)


@dataclass
class MyTrain(TrainConfig):
    epochs: int = 10
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {"monitor": "a", "mode": "max"}
    )


if __name__ == "__main__":

    train = MyTrain()
    print(train)
