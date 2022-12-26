"""Transforms for data augmentation."""
from abc import ABC, abstractmethod
from typing import Any

import torchvision.transforms as T

from configs.base_params import PipelineConfig
from src.transforms.base import Transforms


class ImageClassificationTransforms(Transforms):
    """General Image Classification Transforms."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

    @property
    def train_transforms(self) -> T.Compose:
        return self.pipeline_config.transforms.train_transforms

    @property
    def valid_transforms(self) -> T.Compose:
        return self.pipeline_config.transforms.valid_transforms

    @property
    def test_transforms(self):
        return self.pipeline_config.transforms.test_transforms

    @property
    def debug_transforms(self) -> T.Compose:
        return self.pipeline_config.transforms.debug_transforms
