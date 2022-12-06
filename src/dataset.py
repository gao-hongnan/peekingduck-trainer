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
from typing import Optional, Union, Tuple

import albumentations as A
import cv2
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from sklearn import model_selection

from configs.base_params import PipelineConfig
from src.augmentations import ImageClassificationTransforms
from src.utils.general_utils import (
    create_dataframe_with_image_info,
    download_to,
    extract_file,
    return_list_of_files,
    seed_all,
    show,
)

TransformTypes = Optional[Union[A.Compose, T.Compose]]

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
    """Base class for custom data module.

    References:
        - https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
        - https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/hooks.py
    """

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        self.pipeline_config = pipeline_config

    def cross_validation_split(
        self, df: pd.DataFrame, fold: Optional[int] = None
    ) -> None:
        """Split the dataset into train, valid and test."""

    def prepare_data(self, fold: Optional[int] = None) -> None:
        """See docstring in PyTorch Lightning."""
        # download data here

    @abstractmethod
    def setup(self, stage: str) -> None:
        """See docstring in PyTorch Lightning.

        Example:
            if stage == "fit":
                # assign train and valid datasets for use in dataloaders
                pass

            if stage == "test":
                # assign test dataset for use in dataloaders
                pass
        """
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
