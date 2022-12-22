"""Model Interface that follows the Strategy Pattern."""
from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torchinfo
from torch import nn
from torchinfo.model_statistics import ModelStatistics

from configs.base_params import PipelineConfig


class Model(ABC, nn.Module):
    """Model Base Class."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__()
        self.backbone: Optional[nn.Module]
        self.head: Optional[nn.Module]
        self.model: nn.Module
        self.pipeline_config = pipeline_config

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create the model.
        Note that users can implement anything they want, as long
        as the shape matches.
        """
        raise NotImplementedError("Please implement your own model.")

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model."""

    def modify_head(self) -> nn.Module:
        """Modify the head of the model."""

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model.

        Note need to be private since it is only used internally.
        """

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract the embeddings from the model.

        NOTE:
            Users can use feature embeddings to do metric learning or clustering,
            as well as other tasks.

        Sample Implementation:
            ```python
            def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.backbone(inputs)
            ```
        """

    def model_summary(
        self, input_size: Optional[Tuple[int, int, int, int]] = None, **kwargs: Any
    ) -> ModelStatistics:
        """Wrapper for torchinfo package to get the model summary."""
        if input_size is None:
            input_size = (
                1,
                3,
                self.pipeline_config.augmentation.image_size,
                self.pipeline_config.augmentation.image_size,
            )
        return torchinfo.summary(self.model, input_size=input_size, **kwargs)

    def get_last_layer(self) -> Tuple[list, int, nn.Module]:
        """Get the last layer information of TIMM Model."""
        last_layer_name = None
        for name, _param in self.model.named_modules():
            last_layer_name = name

        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        linear_layer = functools.reduce(getattr, last_layer_attributes, self.model)
        # reduce applies to a list recursively and reduce
        in_features = functools.reduce(
            getattr, last_layer_attributes, self.model
        ).in_features
        return last_layer_attributes, in_features, linear_layer
