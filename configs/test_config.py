"""Base class for params class. Similar to AbstractNode and Node."""
import sys
from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Dict, Literal

import torch


class Validations:
    def __post_init__(self):
        """Run validation methods if declared.
        The validation method can be a simple check
        that raises ValueError or a transformation to
        the field value.
        The validation is performed by calling a function named:
            `validate_<field_name>(self, value, field) -> field.type`
        """
        for name, field in self.__dataclass_fields__.items():
            if method := getattr(self, f"validate_{name}", None):
                setattr(self, name, method(getattr(self, name), field=field))


@dataclass
class TrainConfig(Validations, ABC):
    """Abstract Base Class."""

    epochs: int = field(init=False)
    use_amp: bool = field(init=False, default=False)
    patience: float = field(init=False, default=float("inf"))
    classification_type: str = field(init=False)
    monitored_metric: Dict[str, Any] = field(init=False)


@dataclass
class MyTrain(TrainConfig):
    epochs: int = -1
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {"monitor": "a", "mode": "max"}
    )

    def validate_epochs(self, value, **_):
        if value < 0:
            raise ValueError("Epochs cannot be negative")
        return value


if __name__ == "__main__":

    train = MyTrain()
    print(train)
    patience = train.patience
    print(type(patience))
