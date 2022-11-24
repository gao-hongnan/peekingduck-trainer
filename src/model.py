"""Model Interface.
This module defines the interface for all models.
It follows the Strategy Pattern: https://github.com/msaroufim/ml-design-patterns.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(1, os.getcwd())

import functools
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torchinfo
import torchvision
from torch import nn

from configs.global_params import PipelineConfig
from src.utils import seed_all


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

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model.

        Note:
            1. This typically is loaded from timm or torchvision.
            2. This is not mandatory since users can just create it in create_model.
        """

    def modify_head(self) -> nn.Module:
        """Modify the head of the model."""

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model.

        Note need to be private since it is only used internally.
        """

    def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract the embeddings from the model.

        Note:
            This is used for metric learning or clustering.

        Sample Implementation:
            ```python
            def extract_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.backbone(inputs)
            ```
        """

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

    def model_summary(
        self, input_size: Optional[Tuple[int, int, int, int]] = None, **kwargs: Any
    ) -> torchinfo.model_statistics.ModelStatistics:
        """Wrapper for torchinfo package to get the model summary.
        FIXME: Potentially silent error because it equips model to device
        without explicitly doing to(device).
        """
        if input_size is None:
            input_size = (
                1,
                3,
                self.pipeline_config.augmentation.image_size,
                self.pipeline_config.augmentation.image_size,
            )
        return torchinfo.summary(self.model, input_size=input_size, **kwargs)

    def get_last_layer(self):
        # FIXME: Implement this properly, works for TIMM.
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


class ImageClassificationModel(Model):
    """An generic image classification model.

    Note:
        1. This is generic in the sense that it can be used for any image classification by just
            modifying the head.
    """

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)

        self.backbone = self.load_backbone()
        self.head = self.modify_head()
        self.model = self.create_model()
        # self.model.apply(self._init_weights) # TODO activate if init weights

    def create_model(self) -> nn.Module:
        """Create the model."""
        # TODO: Ask Yier and David whether it is sensible to use self here even though
        # create_model takes in backbone and head.
        # TODO: Might check TIMM on he does it elegantly?
        self.backbone.fc = self.head
        return self.backbone

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model.

        Note:
            1. This typically is loaded from timm or torchvision.
            2. This is not mandatory since users can just create it in create_model.
        """

        backbone = getattr(torchvision.models, self.pipeline_config.model.model_name)(
            pretrained=self.pipeline_config.model.pretrained
        )
        return backbone

    def modify_head(self) -> nn.Module:
        """Modify the head of the model.

        NOTE: This part is very tricky, to modify the head,
        the penultimate layer of the backbone is taken, but different
        models will have different names for the penultimate layer.
        Maybe see my old code for reference where I check for it?
        """
        in_features = self.backbone.fc.in_features  # fc is hardcoded
        out_features = self.pipeline_config.model.num_classes

        head = nn.Linear(in_features=in_features, out_features=out_features)
        return head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        y = self.model(inputs)
        print(f"X: {inputs.shape}, Y: {y.shape}")
        print("Forward Pass Successful")


class MNISTModel(Model):
    """MNIST Model."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.create_model()

    def create_model(self) -> MNISTModel:
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=self.pipeline_config.model.dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.pipeline_config.model.num_classes)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.relu(F.max_pool2d(self.conv1(inputs), 2))
        inputs = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(inputs)), 2))
        inputs = inputs.view(-1, 320)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.dropout(inputs, training=self.training)
        inputs = self.fc2(inputs)
        return inputs


if __name__ == "__main__":
    seed_all(42)
    pipeline_config = PipelineConfig()
    model = ImageClassificationModel(pipeline_config).to(pipeline_config.device)
    print(model.model_summary(device=pipeline_config.device))

    inputs = torch.randn(1, 3, 224, 224).to(pipeline_config.device)

    model.forward_pass(inputs)
