"""This module is the high-level business logic of the project. This module will only
depend on the abstract interface module. This module will not depend on any low-level
concrete implementations."""
from typing import Any
from src.dii_base import Transforms  # from abstract interface import Transforms


class CustomDataset:
    """Dummy class for custom dataset."""

    def __init__(self, transforms: Transforms, stage: str = "train") -> None:
        self.stage = stage
        self.transforms = transforms

    def apply_transforms(self, dummy_data: Any = None) -> Any:
        """Apply transforms to dataset based on stage."""
        if self.stage == "train":
            transformed = self.transforms.get_train_transforms()(dummy_data)
        else:
            transformed = self.transforms.get_test_transforms()(dummy_data)
        return transformed

    def getitem(self, dummy_data: Any = None) -> Any:
        """Replace __getitem__ method as normal method for simplicity."""
        return self.apply_transforms(dummy_data=dummy_data)
