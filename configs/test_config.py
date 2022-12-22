"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Dict, Literal
from src.utils.general_utils import init_logger
import torch
from src.utils.validation_utils import enforce_types, Validator


@enforce_types
@dataclass
class TrainConfig(ABC):
    """Abstract Base Class."""

    epochs: int
    classification_type: str
    monitored_metric: Dict[str, Any]
    use_amp: bool = field(default=False)
    patience: Literal["inf"] = field(default=float("inf"))


@enforce_types
@dataclass
class MyTrain(TrainConfig):
    epochs: int
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {"monitor": "val_Accuracy", "mode": "max"}
    )


if __name__ == "__main__":

    train = MyTrain(
        epochs=1.1, monitored_metric={"monitor": "val_Accuracy", "mode": "max"}
    )
    print(train)
    patience = train.patience
