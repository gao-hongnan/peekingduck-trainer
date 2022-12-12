from __future__ import generators, print_function

import importlib
import pprint

from pytorch_grad_cam import GradCAM
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError
import argparse

from configs.base_params import PipelineConfig
from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.history import History
from src.callbacks.metrics_meter import MetricMeter
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.callbacks.wandb_logger import WandbLogger
from src.dataset import (
    ImageClassificationDataModule,
    MNISTDataModule,
    RSNABreastDataModule,
)
from src.model import ImageClassificationModel, MNISTModel
from src.trainer import Trainer
from src.utils.general_utils import seed_all, free_gpu_memory


def train_generic(pipeline_config: PipelineConfig) -> None:
    """Train on Generic Image Dataset with Train-Valid-Test Split."""
    num_classes = pipeline_config.global_train_params.num_classes

    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()

    model = ImageClassificationModel(pipeline_config).to(pipeline_config.device)
    metrics_collection = MetricCollection(
        [
            Accuracy(num_classes=num_classes),
            Precision(num_classes=num_classes, average="macro"),
            Recall(num_classes=num_classes, average="macro"),
            AUROC(num_classes=num_classes, average="macro"),
            MulticlassCalibrationError(
                num_classes=num_classes
            ),  # similar to brier loss
        ]
    )
    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        metrics=metrics_collection,
        callbacks=[
            History(),
            MetricMeter(),
            ModelCheckpoint(mode="max", monitor="val_Accuracy"),
            EarlyStopping(mode="max", monitor="val_Accuracy", patience=3),
        ],
    )

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()
    history = trainer.fit(train_loader, valid_loader, fold=None)
    print("Valid Loss", history["valid_loss"])
    print("Valid Acc", history["val_Accuracy"])
    print("Valid AUROC", history["val_AUROC"])
