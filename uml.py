"""Dataset class with reference to PyTorch Lightning's Data Hooks.
1. RTD: https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
2. Github: https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/hooks.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(1, os.getcwd())

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
import torchvision.transforms as T
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST

from configs.base_params import PipelineConfig
from src.augmentations import ImageClassificationTransforms
from src.utils.general_utils import (
    create_dataframe_with_image_info,
    download_to,
    extract_file,
    return_filepath,
    return_list_of_files,
    seed_all,
    show,
)

TransformTypes = Optional[Union[A.Compose, T.Compose]]


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


# pylint: disable=invalid-name
class ImageClassificationDataset(Dataset):
    """A sample template for Image Classification Dataset."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        df: Optional[pd.DataFrame] = None,
        path: Optional[Union[str, Path]] = None,
        transforms: TransformTypes = None,
        stage: str = "train",
        **kwargs,
    ) -> None:
        """Constructor for the dataset class.
        Note:
            image_path, image_id and class_id are hardcoded as the column names
            of df are assumed to be such.
        Args:
            df (pd.DataFrame): Dataframe for either train, valid or test.
                This holds the image infos such as image path, image id, target etc.
            path (str): Path to the directory where the images are stored.
            transforms (TransformTypes): torchvision/albumentations transforms
                to apply to the images.
            stage (str): Defaults to "train". One of ['train', 'valid', 'test', 'gradcam']
        """
        super().__init__(**kwargs)
        self.image_path = df[pipeline_config.data.image_path_col_name].values
        self.image_ids = df[pipeline_config.data.image_col_name].values
        self.targets = (
            df[pipeline_config.data.target_col_name].values if stage != "test" else None
        )
        self.df = df
        self.transforms = transforms
        self.stage = stage
        self.path = path
        self.pipeline_config = pipeline_config

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def check_correct_dtype(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Check if the datatypes of X and y are correct. This is important as
        the loss function expects a certain datatype. See my repo/src/dataset.py."""
        raise NotImplementedError

    def check_correct_shape(self) -> None:
        """Check the shapes of X and y, for eg channels first."""
        raise NotImplementedError

    def apply_image_transforms(
        self, image: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # float32
        return torch.tensor(image, dtype=dtype)

    # pylint: disable=no-self-use # not yet!
    def apply_target_transforms(
        self, target: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Apply transforms to the target.
        Note:
            This is useful for tasks such as segmentation object detection where
            targets are in the form of bounding boxes, segmentation masks etc.
        """
        return torch.tensor(target, dtype=dtype)

    def __getitem__(
        self, index: int
    ) -> Union[torch.FloatTensor, Union[torch.FloatTensor, torch.LongTensor]]:
        """Implements the getitem method.
        Note:
            The following target dtype is expected:
            - BCEWithLogitsLoss expects a target.float()
            - CrossEntropyLoss expects a target.long()
        Args:
            index (int): index of the dataset.
        Returns:
            image (torch.FloatTensor): The image tensor.
            target (Union[torch.FloatTensor, torch.LongTensor]]): The target tensor.
        """
        image_path = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.apply_image_transforms(image)

        # Get target for all modes except for test dataset.
        # If test, replace target with dummy ones as placeholder.
        target = self.targets[index] if self.stage != "test" else torch.ones(1)
        target = self.apply_target_transforms(target)

        if self.stage in ["train", "valid", "debug"]:
            return image, target
        elif self.stage == "test":
            return image
        elif self.stage == "gradcam":
            # get image id as well to show on matplotlib image!
            return image, target, self.image_ids[index]
        else:
            raise ValueError(f"Invalid stage {self.stage}.")

    @classmethod
    def from_df(
        cls,
        pipeline_config: PipelineConfig,
        df: pd.DataFrame,
        transforms: TransformTypes,
        stage: str,
    ) -> ImageClassificationDataset:
        """Creates an instance of the dataset class from a dataframe.
        This is the default way for now.
        Example:
            >>> from utils import create_dataframe_with_image_info
            >>> df = create_dataframe_with_image_info()
        """
        return cls(pipeline_config, df=df, transforms=transforms, stage=stage)

    @classmethod
    def from_folder(
        cls,
        pipeline_config: PipelineConfig,
        path: Union[str, Path],
        transforms: TransformTypes,
        stage: str,
        **kwargs,
    ) -> ImageClassificationDataset:
        """Create a dataset from a folder.
        Note:
            The folder structure for train is assumed to be as follows:
            - train
                - class1
                    - image1
                    - image2
                    - image3
                - class2
                    - image1
                    - image2
                    - image3
                - class3
                    - image1
                    - image2
                    - image3
            Same for valid and test.
        """
        return cls(
            pipeline_config, path=path, transforms=transforms, stage=stage, **kwargs
        )


class CustomizedDataModule(ABC):
    """Base class for custom data module."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        self.pipeline_config = pipeline_config

    def cross_validation_split(
        self, df: pd.DataFrame, fold: Optional[int] = None
    ) -> None:
        """Split the dataset into train, valid and test."""

    def prepare_data(self, fold: Optional[int] = None) -> None:
        """See docstring in PyTorch Lightning."""

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

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""

    def debug_train_dataloader(self) -> DataLoader:
        """Debug dataloader"""

    def debug_valid_dataloader(self) -> DataLoader:
        """Debug dataloader"""


class MNISTDataModule(CustomizedDataModule):
    """DataModule for MNIST dataset."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.transforms = ImageClassificationTransforms(pipeline_config)

    def prepare_data(self) -> None:
        # download data here
        self.train_transforms = self.transforms.train_transforms
        self.valid_transforms = self.transforms.valid_transforms

        self.path = self.pipeline_config.data.root_dir
        self.download = self.pipeline_config.data.download

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders."""

        if stage == "fit":
            self.train_dataset = MNIST(
                download=self.download,
                root=self.path,
                transform=self.train_transforms,
                train=True,
            )
            self.valid_dataset = MNIST(
                download=self.download,
                root=self.path,
                transform=self.valid_transforms,
                train=False,
            )
        if self.pipeline_config.datamodule.debug:
            self.train_dataset = Subset(
                self.train_dataset,
                indices=range(self.pipeline_config.datamodule.num_debug_samples),
            )
            self.valid_dataset = Subset(
                self.valid_dataset,
                indices=range(self.pipeline_config.datamodule.num_debug_samples),
            )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            **self.pipeline_config.datamodule.train_loader,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset, **self.pipeline_config.datamodule.valid_loader
        )


# pylint: disable=too-many-instance-attributes
class ImageClassificationDataModule(CustomizedDataModule):
    """Data module for generic image classification dataset."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.transforms = ImageClassificationTransforms(pipeline_config)

    def cross_validation_split(
        self, df: pd.DataFrame, fold: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into train and validation dataframes."""
        resample_strategy = self.pipeline_config.resample.resample_strategy
        resample_params = self.pipeline_config.resample.resample_params
        group_by = self.pipeline_config.data.group_by
        stratify_by = self.pipeline_config.data.stratify_by
        stratify = df[stratify_by].values if stratify_by else None
        groups = df[group_by].values if group_by else None

        if resample_strategy == "train_test_split":
            train_df, valid_df = getattr(model_selection, resample_strategy)(
                df,
                # FIXME: stratify is hard coded for now
                stratify=df[self.pipeline_config.data.target_col_name],
                **resample_params,
            )

        else:  # KFold, StratifiedKFold, GroupKFold etc.
            # TODO: notice every CV strat has groups in split even if it is not used
            # this is good for consistency
            try:
                cv = getattr(model_selection, resample_strategy)(**resample_params)

            except AttributeError as attr_err:
                raise ValueError(
                    f"{resample_strategy} is not a valid resample strategy."
                ) from attr_err
            except TypeError as type_err:
                raise ValueError(
                    f"Invalid resample params for {resample_strategy}."
                ) from type_err

            # FIXME: fatal step for now because if df is large,
            # then this step will be run EVERY single time, redundant!
            for _fold, (_train_idx, valid_idx) in enumerate(
                cv.split(df, stratify, groups)
            ):
                df.loc[valid_idx, "fold"] = _fold + 1
            df["fold"] = df["fold"].astype(int)
            df.to_csv("df.csv", index=False)
            print(
                df.groupby(["fold", self.pipeline_config.data.target_col_name]).size()
            )
            train_df = df[df.fold != fold].reset_index(drop=True)
            valid_df = df[df.fold == fold].reset_index(drop=True)
            print("fold", fold, "train", train_df.shape, "valid", valid_df.shape)

        return train_df, valid_df

    def prepare_data(self, fold: Optional[int] = None) -> None:
        """Download data here and prepare.
        TODO:
            1. Here needs to be more generic for users,
            for example, if user choose train_test_split, then we need to
            split accordingly. If user choose KFold, then we need to split
            to 5 folds. I will include examples I have done.
            2. Config should hold a "CV" strategy for users to choose from.
        """
        url = self.pipeline_config.data.url
        blob_file = self.pipeline_config.data.blob_file
        root_dir = self.pipeline_config.data.root_dir
        train_dir = self.pipeline_config.data.train_dir
        test_dir = self.pipeline_config.data.test_dir

        if self.pipeline_config.data.download:
            download_to(url, blob_file, root_dir)
            extract_file(root_dir, blob_file)

        train_images = return_list_of_files(
            train_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        test_images = return_list_of_files(
            test_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        print(f"Total number of images: {len(train_images)}")
        print(f"Total number of test images: {len(test_images)}")

        if Path(self.pipeline_config.data.train_csv).exists():
            df = pd.read_csv(self.pipeline_config.data.train_csv)
        else:
            df = create_dataframe_with_image_info(
                train_images,
                self.pipeline_config.data.class_name_to_id,
                save_path=self.pipeline_config.data.train_csv,
            )
        print(df.head())

        self.train_df, self.valid_df = self.cross_validation_split(df, fold)
        if self.pipeline_config.datamodule.debug:
            num_debug_samples = self.pipeline_config.datamodule.num_debug_samples
            print(f"Debug mode is on, using {num_debug_samples} images for training.")
            self.train_df = self.train_df.sample(num_debug_samples)
            self.valid_df = self.valid_df.sample(num_debug_samples)

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders."""
        if stage == "fit":
            train_transforms = self.transforms.train_transforms
            valid_transforms = self.transforms.valid_transforms

            self.train_dataset = ImageClassificationDataset(
                self.pipeline_config,
                df=self.train_df,
                stage="train",
                transforms=train_transforms,
            )
            self.valid_dataset = ImageClassificationDataset(
                self.pipeline_config,
                df=self.valid_df,
                stage="valid",
                transforms=valid_transforms,
            )

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            **self.pipeline_config.datamodule.train_loader,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset, **self.pipeline_config.datamodule.valid_loader
        )


class RSNABreastDataModule(CustomizedDataModule):
    """Data module for RSBA Breast image classification dataset."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.transforms = ImageClassificationTransforms(pipeline_config)

    def cross_validation_split(
        self, df: pd.DataFrame, fold: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataframe into train and validation dataframes."""
        resample_strategy = self.pipeline_config.resample.resample_strategy
        resample_params = self.pipeline_config.resample.resample_params
        group_by = self.pipeline_config.data.group_by
        stratify_by = self.pipeline_config.data.stratify_by
        stratify = df[stratify_by].values if stratify_by else None
        groups = df[group_by].values if group_by else None

        if resample_strategy == "train_test_split":
            train_df, valid_df = getattr(model_selection, resample_strategy)(
                df,
                # FIXME: stratify is hard coded for now
                stratify=df[self.pipeline_config.data.target_col_name],
                **resample_params,
            )
        else:
            try:
                cv = getattr(model_selection, resample_strategy)(**resample_params)
            except AttributeError as attr_err:
                raise ValueError(
                    f"{resample_strategy} is not a valid resample strategy."
                ) from attr_err
            except TypeError as type_err:
                raise ValueError(
                    f"Invalid resample params for {resample_strategy}."
                ) from type_err

            for _fold, (_train_idx, valid_idx) in enumerate(
                cv.split(df, stratify, groups)
            ):
                df.loc[valid_idx, "fold"] = _fold + 1
            df["fold"] = df["fold"].astype(int)

            print(
                df.groupby(["fold", self.pipeline_config.data.target_col_name]).size()
            )
            train_df = df[df.fold != fold].reset_index(drop=True)
            valid_df = df[df.fold == fold].reset_index(drop=True)
            print("fold", fold, "train", train_df.shape, "valid", valid_df.shape)

        return train_df, valid_df

    def prepare_data(self, fold: Optional[int] = None) -> None:
        url = self.pipeline_config.data.url
        blob_file = self.pipeline_config.data.blob_file
        root_dir = self.pipeline_config.data.root_dir
        train_dir = self.pipeline_config.data.train_dir
        test_dir = self.pipeline_config.data.test_dir

        if self.pipeline_config.data.download:
            download_to(url, blob_file, root_dir)
            extract_file(root_dir, blob_file)

        train_images = return_list_of_files(
            train_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        test_images = return_list_of_files(
            test_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        )
        print(f"Total number of images: {len(train_images)}")
        print(f"Total number of test images: {len(test_images)}")

        df = pd.read_csv(self.pipeline_config.data.train_csv)
        df["image_id_final"] = (
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
        test_df["image_id_final"] = (
            test_df["patient_id"].astype(str) + "/" + test_df["image_id"].astype(str)
        )
        test_df["image_path"] = test_df[self.pipeline_config.data.image_col_name].apply(
            lambda x: return_filepath(
                image_id=x,
                folder=test_dir,
                extension=".dcm",
            )
        )
        self.test_df = test_df.drop_duplicates(subset="prediction_id")
        # hardcode

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders."""
        if stage == "fit":
            train_transforms = self.transforms.train_transforms
            valid_transforms = self.transforms.valid_transforms

            self.train_dataset = ImageClassificationDataset(
                self.pipeline_config,
                df=self.train_df,
                stage="train",
                transforms=train_transforms,
            )
            self.valid_dataset = ImageClassificationDataset(
                self.pipeline_config,
                df=self.valid_df,
                stage="valid",
                transforms=valid_transforms,
            )
        elif stage == "test":
            print(self.test_df)
            test_transforms = self.transforms.test_transforms
            self.test_dataset = ImageClassificationDataset(
                self.pipeline_config,
                df=self.test_df,
                stage="test",
                transforms=test_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            **self.pipeline_config.datamodule.train_loader,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset, **self.pipeline_config.datamodule.valid_loader
        )

    def test_dataloader(self) -> DataLoader:
        pass


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


class Trainer:  # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-public-methods
    """Object used to facilitate training."""

    stop: bool  # from EarlyStopping

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        model: Model,
        callbacks: List[Callback] = None,
        metrics: Union[MetricCollection, List[str]] = None,
    ) -> None:
        """Initialize the trainer.
        TODO:
        1. state = {"model": ..., "optimizer": ...} Torchflare's state is equivalent
        to our pipeline_config, but his holds callable as values.

        monitored_metric = {
            "monitor": "val_Accuracy",
            "metric_score": None,
            "mode": "max",
        }

        history = {"train_loss": [...], "valid_loss": [...], # save all epoch
                    "train_acc": [...], "valid_acc": [...], # save all epoch
                    "train_auroc": [...], "valid_auroc": [...], # save all epoch
                    "train_logits": [...], "valid_logits": [...], # save only best epoch?
                    "train_preds": [...], "valid_preds": [...], # save only best epoch?
                    "train_probs": [...], "valid_probs": [...], # save only best epoch?
                    }
        """
        # Set params
        self.pipeline_config = pipeline_config
        self.train_params = self.pipeline_config.global_train_params
        self.model = model
        self.model_artifacts_dir = pipeline_config.stores.model_artifacts_dir
        self.device = self.pipeline_config.device

        self.callbacks = callbacks
        self.metrics = metrics
        # TODO: if isinstance(metrics, list): convert to MetricCollection
        # FIXME: init_logger should be a callback type, if not writing here suggests
        # it really depends on the type of logger in init_logger.
        self.logger = init_logger(
            log_file=Path.joinpath(
                self.pipeline_config.stores.logs_dir, "training.log"
            ),
            module_name="training",
        )
        self.initialize()  # init non init attributes, etc

    def get_classification_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
        # mode: str = "valid",
    ):
        """[summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        Returns:
            [type]: [description]
        """
        probablistic_f1 = pfbeta_torch(y_trues, y_preds, beta=1)
        print("probablistic_f1", probablistic_f1)

        self.train_metrics = self.metrics.clone(prefix="train_")
        self.valid_metrics = self.metrics.clone(prefix="val_")
        train_metrics_results = self.train_metrics(y_probs, y_trues.flatten())
        valid_metrics_results = self.valid_metrics(y_probs, y_trues.flatten())
        print(f"valid metrics: {valid_metrics_results}")
        return train_metrics_results, valid_metrics_results

    def run(self):
        self.on_trainer_start()
        # self.fit()
        self.on_trainer_end()

    def initialize(self) -> None:
        """Called when the trainer begins."""
        self.optimizer = self.get_optimizer(
            model=self.model,
            optimizer_params=self.pipeline_config.optimizer_params,
        )
        self.scheduler = self.get_scheduler(
            optimizer=self.optimizer,
            scheduler_params=self.pipeline_config.scheduler_params,
        )

        if self.train_params.use_amp:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.monitored_metric = self.train_params.monitored_metric

        # Metric to optimize, either min or max.
        self.best_valid_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )
        self.patience_counter = self.train_params.patience  # Early Stopping Counter
        self.current_epoch = 1
        self.train_epoch_dict = {}
        self.valid_epoch_dict = {}
        self.train_batch_dict = {}
        self.valid_batch_dict = {}
        self.train_history_dict = {}
        self.valid_history_dict = {}
        self.invoke_callbacks("on_trainer_start")

    def on_fit_start(self, fold: int) -> None:
        """Called AFTER fit begins."""
        self.logger.info(f"Fold {fold} started")
        self.best_valid_loss = np.inf
        self.current_fold = fold

    def on_fit_end(self) -> None:
        """Called AFTER fit ends."""
        # print(self.train_batch_dict)
        # print(self.valid_batch_dict)
        # print(self.train_epoch_dict)
        # print(self.valid_epoch_dict)
        # print(self.train_history_dict)
        # print(self.valid_history_dict)
        free_gpu_memory(
            self.optimizer,
            self.scheduler,
            self.valid_history_dict["valid_trues"],
            self.valid_history_dict["valid_logits"],
            self.valid_history_dict["valid_preds"],
            self.valid_history_dict["valid_probs"],
        )

    def invoke_callbacks(self, event_name: str) -> None:
        """Invoke the callbacks."""
        for callback in self.callbacks:
            try:
                getattr(callback, event_name)(self)
            except NotImplementedError:
                pass

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        fold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit the model and returns the history object."""
        self.on_fit_start(fold=fold)

        for _epoch in range(1, self.train_params.epochs + 1):
            self.train_one_epoch(train_loader, _epoch)
            self.valid_one_epoch(valid_loader, _epoch)

            # self.monitored_metric["metric_score"] = torch.clone(
            #     torch.tensor(self.valid_history_dict[self.monitored_metric["monitor"]])
            # ).detach()  # FIXME: one should not hardcode "valid_macro_auroc" here
            self.monitored_metric["metric_score"] = torch.clone(
                self.valid_history_dict[self.monitored_metric["monitor"]]
            ).detach()  # FIXME: one should not hardcode "valid_macro_auroc" here
            valid_loss = self.valid_history_dict["valid_loss"]

            if self.stop:  # from early stopping
                break  # Early Stopping

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            self.current_epoch += 1

        self.on_fit_end()
        # FIXME: here is finish fitting, whether to call it on train end or on fit end?
        # Currently only history uses on_trainer_end.
        for callback in self.callbacks:
            callback.on_trainer_end(self)
        return self.history

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        """Train one epoch of the model."""
        curr_lr = self.get_lr(self.optimizer)
        train_start_time = time.time()

        # set to train mode
        self.model.train()

        train_bar = tqdm(train_loader)

        # Iterate over train batches
        for _step, batch in enumerate(train_bar, start=1):
            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            _batch_size = inputs.shape[0]  # unused for now

            with torch.cuda.amp.autocast(
                enabled=self.train_params.use_amp,
                dtype=torch.float16,
                cache_enabled=True,
            ):
                logits = self.model(inputs)  # Forward pass logits
                curr_batch_train_loss = self.computer_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                    stage="train",
                )
            self.optimizer.zero_grad()  # reset gradients

            if self.scaler is not None:
                self.scaler.scale(curr_batch_train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                curr_batch_train_loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric, every batch is diff
            self.train_batch_dict["train_loss"] = curr_batch_train_loss.item()
            # train_bar.set_description(f"Train. {metric_monitor}")

            _y_train_prob = get_sigmoid_softmax(self.pipeline_config)(logits)

            _y_train_pred = torch.argmax(_y_train_prob, dim=1)

            self.invoke_callbacks("on_train_batch_end")

        self.invoke_callbacks("on_train_loader_end")
        # total time elapsed for this epoch
        train_time_elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - train_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Train. Epoch {epoch}:"
            f"\nAvg Train Summary Loss: {self.train_epoch_dict['train_loss']:.3f}"
            f"\nLearning Rate: {curr_lr:.5f}"
            f"\nTime Elapsed: {train_time_elapsed}\n"
        )
        self.train_history_dict = {**self.train_epoch_dict}
        self.invoke_callbacks("on_train_epoch_end")

    def valid_one_epoch(self, valid_loader: DataLoader, epoch: int) -> None:
        """Validate the model on the validation set for one epoch.
        Args:
            valid_loader (torch.utils.data.DataLoader): The validation set dataloader.
        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """
        val_start_time = time.time()  # start time for validation

        self.model.eval()  # set to eval mode

        valid_bar = tqdm(valid_loader)

        valid_logits, valid_trues, valid_preds, valid_probs = [], [], [], []

        with torch.no_grad():
            for _step, batch in enumerate(valid_bar, start=1):
                # unpack
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                _batch_size = inputs.shape[0]

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = get_sigmoid_softmax(self.pipeline_config)(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = self.computer_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                    stage="valid",
                )
                # Update loss metric, every batch is diff
                self.valid_batch_dict["valid_loss"] = curr_batch_val_loss.item()

                # valid_bar.set_description(f"Validation. {metric_monitor}")

                self.invoke_callbacks("on_valid_batch_end")
                # For OOF score and other computation.
                # TODO: Consider giving numerical example. Consider rolling back to targets.cpu().numpy() if torch fails.
                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())
                valid_preds.extend(y_valid_pred.cpu())
                valid_probs.extend(y_valid_prob.cpu())

        valid_trues, valid_logits, valid_preds, valid_probs = (
            torch.vstack(valid_trues),
            torch.vstack(valid_logits),
            torch.vstack(valid_preds),
            torch.vstack(valid_probs),
        )
        _, valid_metrics_dict = self.get_classification_metrics(
            valid_trues, valid_preds, valid_probs
        )

        self.invoke_callbacks("on_valid_loader_end")

        # total time elapsed for this epoch
        valid_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - val_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Validation. Epoch {epoch}:"
            f"\nAvg Val Summary Loss: {self.valid_epoch_dict['valid_loss']:.3f}"
            f"\nAvg Val Accuracy: {valid_metrics_dict['val_Accuracy']:.3f}"
            f"\nAvg Val Macro AUROC: {valid_metrics_dict['val_AUROC']:.3f}"
            f"\nTime Elapsed: {valid_elapsed_time}\n"
        )
        # here self.valid_epoch_dict only has valid_loss, we update the rest
        self.valid_epoch_dict.update(
            {
                "valid_trues": valid_trues,
                "valid_logits": valid_logits,
                "valid_preds": valid_preds,
                "valid_probs": valid_probs,
            }
        )  # FIXME: potential difficulty in debugging since valid_epoch_dict is called in metrics meter
        self.valid_epoch_dict.update(valid_metrics_dict)
        # temporary stores current valid epochs info
        # FIXME: so now valid epoch dict and valid history dict are the same lol.
        self.valid_history_dict = {**self.valid_epoch_dict, **valid_metrics_dict}

        # TODO: after valid epoch ends, for example, we need to call
        # our History callback to save the metrics into a list.
        self.invoke_callbacks("on_valid_epoch_end")

    @staticmethod
    def get_optimizer(
        model,
        optimizer_params: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        """Get the optimizer for the model.
        Note:
            Do not invoke self.model directly in this call as it may affect model initalization.
            https://stackoverflow.com/questions/70107044/can-i-define-a-method-as-an-attribute
        """
        return getattr(torch.optim, optimizer_params.optimizer)(
            model.parameters(), **optimizer_params.optimizer_params
        )

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_params: Dict[str, Any],
    ) -> torch.optim.lr_scheduler:
        """Get the scheduler for the optimizer."""
        return getattr(torch.optim.lr_scheduler, scheduler_params.scheduler)(
            optimizer=optimizer, **scheduler_params.scheduler_params
        )

    @staticmethod
    def computer_criterion(
        y_trues: torch.Tensor,
        y_logits: torch.Tensor,
        criterion_params: Dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.
        The below example applies for CrossEntropyLoss.
        Args:
            y_trues ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is
                $0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C-1$.
                If containing class probabilities, same shape as the input.
            stage (str): train or valid, sometimes people use different loss functions for
                train and valid.
        """

        if stage == "train":
            loss_fn = getattr(torch.nn, criterion_params.train_criterion)(
                **criterion_params.train_criterion_params
            )
        elif stage == "valid":
            loss_fn = getattr(torch.nn, criterion_params.valid_criterion)(
                **criterion_params.valid_criterion_params
            )
        loss = loss_fn(y_logits, y_trues)
        return loss

    @staticmethod
    def get_lr(optimizer: torch.optim) -> float:
        """Get the learning rate of optimizer for the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]


if __name__ == "__main__":
    seed_all(42)

    from configs.global_params import PipelineConfig as SteelPipelineConfig

    pipeline_config = SteelPipelineConfig()
    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()
    dm.setup(stage="fit")

    image, target = dm.train_dataset[0]
    print(image.shape, target)

    debug_train_loader = dm.train_dataloader()
    image_batch, target_batch = next(iter(debug_train_loader))
    image_grid = torchvision.utils.make_grid(image_batch)
    show(image_grid)
