"""Transforms for data augmentation."""

from abc import ABC, abstractmethod
from typing import Any

import torchvision.transforms as T

from configs.base_params import PipelineConfig


class Transforms(ABC):
    """Create a Transforms class that can take in albumentations
    and torchvision transforms.

    TODO:
        1. Do we really need to enforce all params to be passed in?
        2. We need to add mixup/cutmix support, these SOTA techniques are common now.
    """

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__()
        self.pipeline_config = pipeline_config

    @property
    @abstractmethod
    def train_transforms(self):
        """Get the training transforms."""

    @property
    @abstractmethod
    def valid_transforms(self):
        """Get the validation transforms."""

    @property
    def test_transforms(self):
        """Get the test transforms."""

    @property
    def gradcam_transforms(self):
        """Get the gradcam transforms."""

    @property
    def debug_transforms(self):
        """Get the debug transforms."""

    @property
    def test_time_augmentations(self):
        """Get the test time augmentations."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the transforms."""


class ImageClassificationTransforms(Transforms):
    """General Image Classification Transforms.

    FIXME: This part definitely needs to be changed as it is all hard coded.
    """

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)

        self.image_size = pipeline_config.augmentation.image_size
        self.pre_center_crop = pipeline_config.augmentation.pre_center_crop
        self.mean = pipeline_config.augmentation.mean
        self.std = pipeline_config.augmentation.std
        self.pre_center_crop = pipeline_config.augmentation.pre_center_crop

    @property
    def train_transforms(self) -> T.Compose:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(self.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    @property
    def valid_transforms(self) -> T.Compose:
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.pre_center_crop),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    @property
    def test_transforms(self):
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

    @property
    def debug_transforms(self) -> T.Compose:
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
