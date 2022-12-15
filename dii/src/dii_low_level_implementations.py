from src.dii_base import Transforms  # from abstract interface import Transforms
from typing import Callable


class ImageClassificationTransforms(Transforms):
    """Dummy class for image classification transforms."""

    def get_train_transforms(self) -> Callable:
        """Get train transforms."""
        print("Getting image classification train transforms.")
        return lambda x: None

    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
        print("Getting image classification test transforms.")
        return lambda x: None


class ImageSegmentationTransforms(Transforms):
    """Dummy class for image segmentation transforms."""

    def get_train_transforms(self) -> Callable:
        """Get train transforms."""
        print("Getting image segmentation train transforms.")
        return lambda x: None

    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
        print("Getting image segmentation test transforms.")
        return lambda x: None
