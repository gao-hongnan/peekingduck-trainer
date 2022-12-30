"""Model Interface that follows the Strategy Pattern."""
from __future__ import annotations

import os
import sys

sys.path.insert(1, os.getcwd())

import timm
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from configs.base_params import PipelineConfig
from src.models.base import Model
from src.utils.general_utils import seed_all


class ImageClassificationModel(Model):
    """A generic image classification model. This is generic in the sense that
    it can be used for any image classification by just modifying the head.
    """

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)

        self.adapter = self.pipeline_config.model.adapter
        self.backbone = self.load_backbone()
        self.head = self.modify_head()
        self.model = self.create_model()

        # self.model.apply(self._init_weights) # activate if init weights
        print(f"Successfully created model: {self.pipeline_config.model.model_name}")

    def create_model(self) -> nn.Module:
        """Create the model."""
        self.backbone.fc = self.head
        return self.backbone

    def load_backbone(self) -> nn.Module:
        """Load the backbone of the model.

        NOTE:
            1. Backbones are usually loaded from timm or torchvision.
            2. This is not mandatory since users can just create it in create_model.
        """
        if self.adapter == "torchvision":
            backbone = getattr(
                torchvision.models, self.pipeline_config.model.model_name
            )(pretrained=self.pipeline_config.model.pretrained)
        elif self.adapter == "timm":
            backbone = timm.create_model(
                self.pipeline_config.model.model_name,
                pretrained=self.pipeline_config.model.pretrained,
                # in_chans=3,
            )
        return backbone

    def modify_head(self) -> nn.Module:
        """Modify the head of the model.

        NOTE/TODO:
            This part is very tricky, to modify the head,
            the penultimate layer of the backbone is taken, but different
            models will have different names for the penultimate layer.
            Maybe see my old code for reference where I check for it?
        """  # fully connected
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
