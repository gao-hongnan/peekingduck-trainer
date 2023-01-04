from __future__ import generators, print_function

import importlib

from pytorch_grad_cam import GradCAM
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError
import pprint
from tabulate import tabulate
import pandas as pd
import torch
from argparse import ArgumentParser, Namespace
from configs.base_params import PipelineConfig
from src.datamodule.dataset import (
    ImageClassificationDataModule,
    MNISTDataModule,
    RSNABreastDataModule,
)
from src.models.model import ImageClassificationModel, MNISTModel
from src.trainer import Trainer
from src.utils.general_utils import seed_all, free_gpu_memory
from src.inference import inference_all_folds, inference_one_fold


def train_generic(pipeline_config: PipelineConfig) -> None:
    """Train on Generic Image Dataset with Train-Valid-Test Split."""
    artifacts_dir = pipeline_config.stores.model_artifacts_dir
    num_classes = pipeline_config.model.num_classes

    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()

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

    callbacks = pipeline_config.callback_params.callbacks

    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        metrics=metrics_collection,
        callbacks=callbacks,
    )

    history = trainer.fit(train_loader, valid_loader, fold=None)
    print("Valid Loss", history["valid_loss"])
    print("Valid Acc", history["val_Accuracy"])
    print("Valid AUROC", history["val_AUROC"])
    print(history.keys())
    # history_df = pd.DataFrame(
    #     {
    #         v: [history[v]]
    #         for k in history.keys()
    #         if k in ["valid_loss", "val_Accuracy", "val_AUROC"]
    #         for v in k
    #     }
    # )
    # print(tabulate(history_df, headers="keys", tablefmt="psql"))


def inference_generic(pipeline_config: PipelineConfig) -> None:
    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    model = ImageClassificationModel(pipeline_config).to(pipeline_config.device)

    weight = "/Users/reighns/gaohn/peekingduck-trainer/stores/model_artifacts/CIFAR-10/74502c5e-d25e-48c2-8b86-a690d33372f8/resnet18_best_val_Accuracy_fold_None_epoch9.pt"

    state_dict = torch.load(weight)["model_state_dict"]
    predictions = inference_one_fold(
        model=model,
        state_dict=state_dict,
        test_loader=test_loader,
        pipeline_config=pipeline_config,
    )
    print(predictions)


# FIXME: find out if checkpoint is indeed saving the best epoch and not the last epoch
# FIXME: history object's valid_probs return list of 10 instead but in the checkpoint it is 1
# FIXME: rename oof_... to valid_... in the checkpoint
# FIXME: OOF should be for all folds, not just one fold
def create_oof_df(pipeline_config: PipelineConfig) -> None:
    """Create OOF dataframe for Generic Image Dataset with a Resampling Strategy."""
    num_classes = pipeline_config.model.num_classes

    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()
    dm.setup(stage="fit")

    df_oof = dm.oof_df

    model = ImageClassificationModel(pipeline_config).to(pipeline_config.device)

    weight = "/Users/reighns/gaohn/peekingduck-trainer/stores/model_artifacts/CIFAR-10/74502c5e-d25e-48c2-8b86-a690d33372f8/resnet18_best_val_Accuracy_fold_None_epoch9.pt"

    oof_probs = torch.load(weight)["oof_probs"]
    oof_trues = torch.load(weight)["oof_trues"]
    oof_preds = torch.load(weight)["oof_preds"]

    df_oof[[f"class_{str(c)}_oof" for c in range(num_classes)]] = (
        oof_probs.detach().cpu().numpy()
    )
    df_oof["oof_trues"] = oof_trues.detach().cpu().numpy()
    df_oof["oof_preds"] = oof_preds.detach().cpu().numpy()
    print(df_oof.head())

    accuracy = Accuracy(num_classes=num_classes)(oof_preds, oof_trues)
    print("OOF Accuracy", accuracy)  # 0.3281 confirms that it is the best epoch


def train_one_fold(pipeline_config: PipelineConfig, fold: int) -> None:
    """Train one fold on a Generic Image Dataset with a Resampling Strategy.
    This is the precursor to training on all folds."""
    num_classes = pipeline_config.model.num_classes
    num_folds = pipeline_config.resample.resample_params["n_splits"]

    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data(fold)

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

    callbacks = pipeline_config.callback_params.callbacks

    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        metrics=metrics_collection,
        callbacks=callbacks,
    )

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()
    history = trainer.fit(train_loader, valid_loader, fold=fold)
    print("Valid Loss", history["valid_loss"])
    print("Valid Acc", history["val_Accuracy"])
    print("Valid AUROC", history["val_AUROC"])


def train_mnist(pipeline_config: PipelineConfig) -> None:
    """Train MNIST."""
    num_classes = pipeline_config.model.num_classes
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

    callbacks = pipeline_config.callback_params.callbacks

    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        metrics=metrics_collection,
        callbacks=callbacks,
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


def train_one_fold_rsna(pipeline_config: PipelineConfig, fold: int) -> None:
    """Train one fold on a Generic Image Dataset with a Resampling Strategy.
    This is the precursor to training on all folds."""
    num_classes = pipeline_config.model.num_classes
    num_folds = pipeline_config.resample.resample_params["n_splits"]

    dm = RSNABreastDataModule(pipeline_config)
    dm.prepare_data(fold)

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

    callbacks = pipeline_config.callback_params.callbacks

    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        metrics=metrics_collection,
        callbacks=callbacks,
    )

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()
    history = trainer.fit(train_loader, valid_loader, fold=fold)

    print("Valid Loss", history["valid_loss"])
    print("Valid Acc", history["val_Accuracy"])
    print("Valid AUROC", history["val_AUROC"])


def parse_opt() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        default="cifar10_params",
        help="The name of the config file.",
    )
    return parser.parse_args()


# TODO: maybe when compiling pipeline config, we can save state of the config as callables,
# like torchflare's self.state dict.
def run(opt: Namespace) -> None:
    """Run the pipeline."""
    base_config_path = "configs."
    config_name = opt.config_name
    print(f"Running config: {config_name}")

    config_path = base_config_path + config_name
    project = importlib.import_module(config_path)

    pipeline_config = project.PipelineConfig()
    print("Pipeline Config:")
    print(str(pipeline_config))
    # print(str(pipeline_config))
    # pp = pprint.PrettyPrinter(depth=4)
    # pp.pprint(pipeline_config.all_params)

    if config_name == "mnist_params":
        train_mnist(pipeline_config)
    elif "rsna_breast" in config_name:
        train_one_fold_rsna(pipeline_config, fold=4)
    elif "_cv_" in config_name:
        train_one_fold(pipeline_config, fold=1)
    else:
        train_generic(pipeline_config)
        inference_generic(pipeline_config)
        create_oof_df(pipeline_config)


if __name__ == "__main__":
    seed_all(1992)
    opt = parse_opt()
    run(opt)
