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
class TrainConfig(Validator, ABC):
    """Abstract Base Class."""

    epochs: int
    classification_type: str
    monitored_metric: Dict[str, Any]
    use_amp: bool = field(default=False)
    patience: float = field(default=float("inf"))

    def validate_epochs(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("epochs must be >= 0")
        return value


@enforce_types
@dataclass
class MyTrain(TrainConfig):
    epochs: int
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {"monitor": "val_Accuracy", "mode": "max"}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.a: int = 2


if __name__ == "__main__":

    train = MyTrain(
        epochs=2, monitored_metric={"monitor": "val_Accuracy", "mode": "max"}
    )
    print(train)
    print(train.a)
    patience = train.patience
