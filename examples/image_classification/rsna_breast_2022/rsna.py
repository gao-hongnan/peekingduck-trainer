from typing import Optional

import pandas as pd
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError

from configs.base_params import PipelineConfig
from src.datamodule.dataset import ImageClassificationDataModule
from src.models.model import ImageClassificationModel
from src.trainer import Trainer
from src.transforms.augmentations import ImageClassificationTransforms
from src.utils.general_utils import return_filepath, return_list_of_files


class RSNABreastDataModule(ImageClassificationDataModule):
    """Data module for RSBA Breast image classification dataset."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.transforms = ImageClassificationTransforms(pipeline_config)

    def prepare_data(self, fold: Optional[int] = None) -> None:
        train_dir = self.pipeline_config.data.train_dir
        test_dir = self.pipeline_config.data.test_dir

        train_images = return_list_of_files(
            train_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        test_images = return_list_of_files(
            test_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        print(f"Total number of images: {len(train_images)}")
        print(f"Total number of test images: {len(test_images)}")

        df = pd.read_csv(self.pipeline_config.data.train_csv)
        df[self.pipeline_config.data.image_col_name] = (
            df["patient_id"].astype(str) + "_" + df["image_id"].astype(str)
        )
        df["image_path"] = df[self.pipeline_config.data.image_col_name].apply(
            lambda x: return_filepath(
                image_id=x,
                folder=train_dir,
                extension=self.pipeline_config.data.image_extension,
            )
        )
        df.to_csv(f"{self.pipeline_config.data.data_dir}/df.csv", index=False)
        print(df.head())

        self.train_df, self.valid_df = self.cross_validation_split(df, fold)
        if self.pipeline_config.datamodule.debug:
            num_debug_samples = self.pipeline_config.datamodule.num_debug_samples
            print(f"Debug mode is on, using {num_debug_samples} images for training.")
            self.train_df = self.train_df.sample(num_debug_samples)
            self.valid_df = self.valid_df.sample(num_debug_samples)

        # hardcode
        test_df = pd.read_csv(self.pipeline_config.data.test_csv)
        test_df[self.pipeline_config.data.image_col_name] = (
            test_df["patient_id"].astype(str) + "/" + test_df["image_id"].astype(str)
        )
        test_df["image_path"] = test_df[self.pipeline_config.data.image_col_name].apply(
            lambda x: return_filepath(
                image_id=x,
                folder=test_dir,
                extension=".dcm",
            )
        )


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
