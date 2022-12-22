"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Dict, Literal
from src.utils.general_utils import init_logger
import torch
from src.utils.validation_utils import enforce_types, Validator


@dataclass
class TrainConfig(Validator, ABC):
    """Abstract Base Class."""

    epochs: int = field(init=False)
    use_amp: bool = False
    patience: float = field(init=False, default=float("inf"))
    classification_type: str = field(init=False)
    monitored_metric: Dict[str, Any] = field(init=False)

    def validate_epochs(self, value: int, **_):
        """Validate epochs."""
        if value < 0:
            raise ValueError("Epochs cannot be negative")
        return value


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
        epochs=1, monitored_metric={"monitor": "val_Accuracy", "mode": "max"}
    )
    print(train)
    patience = train.patience
