from __future__ import generators, print_function

import importlib

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
from src.dataset import ImageClassificationDataModule, MNISTDataModule
from src.model import ImageClassificationModel, MNISTModel
from src.trainer import Trainer
from src.utils.general_utils import seed_all, free_gpu_memory


def train_generic(pipeline_config: PipelineConfig) -> None:
    """Train Steel Defect."""
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


def train_mnist(pipeline_config: PipelineConfig) -> None:
    """Train MNIST."""
    num_classes = pipeline_config.global_train_params.num_classes  # 10
    dm = MNISTDataModule(pipeline_config)
    dm.prepare_data()

    model = MNISTModel(pipeline_config).to(pipeline_config.device)
    metrics_collection = MetricCollection(
        [
            Accuracy(num_classes=num_classes),
            Precision(num_classes=num_classes),
            Recall(num_classes=num_classes),
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
            EarlyStopping(mode="max", monitor="val_Accuracy", patience=2),
            # WandbLogger(
            #     project="MNIST",
            #     entity="reighns",
            #     name="MNIST_EXP_1",
            #     config=pipeline_config.all_params,
            # ),
        ],
    )

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()
    history = trainer.fit(train_loader, valid_loader, fold=None)
    # history = trainer.history
    print(history.keys())
    print(history["valid_loss"])
    print(history["val_Accuracy"])
    print(history["val_AUROC"])
    # print(trainer.history["valid_probs"][0].shape)
    # print(trainer.history["valid_probs"][1].shape)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        default="cifar_params",
        help="The name of the config file.",
    )
    return parser.parse_args()


# TODO: maybe when compiling pipeline config, we can save state of the config as callables,
# like torchflare's self.state dict.
def run(opt: argparse.Namespace) -> None:
    base_config_path = "configs."
    config_name = opt.config_name
    print(f"Running config: {config_name}")

    config_path = base_config_path + config_name
    project = importlib.import_module(config_path)
    print(project)

    pipeline_config = project.PipelineConfig()
    print(f"Pipeline config: {pipeline_config.all_params}")
    if config_name == "mnist_params":
        train_mnist(pipeline_config)
    else:
        train_generic(pipeline_config)


if __name__ == "__main__":
    seed_all(1992)
    opt = parse_opt()
    run(opt)

    # mnist_config = mnist_params.PipelineConfig()
    # print(f"Pipeline Config: {mnist_config}")
    # train_mnist(mnist_config)
