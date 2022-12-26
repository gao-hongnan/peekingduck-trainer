"""DataModule class with reference to PyTorch Lightning's Data Hooks.
1. RTD: https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
2. Github: https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/hooks.py
"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from configs.base_params import PipelineConfig


class CustomizedDataModule(ABC):
    """Base class for custom data module."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        self.pipeline_config = pipeline_config

    @abstractmethod
    def setup(self, stage: str) -> None:
        """See docstring in PyTorch Lightning."""
        raise NotImplementedError

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""
        raise NotImplementedError

    @abstractmethod
    def valid_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        raise NotImplementedError

    def cross_validation_split(
        self, df: pd.DataFrame, fold: Optional[int] = None
    ) -> None:
        """Split the dataset into train, valid and test."""

    def prepare_data(self, fold: Optional[int] = None) -> None:
        """See docstring in PyTorch Lightning."""

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""

    def debug_train_dataloader(self) -> DataLoader:
        """Debug dataloader"""

    def debug_valid_dataloader(self) -> DataLoader:
        """Debug dataloader"""
