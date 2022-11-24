"""Dataset class with reference to PyTorch Lightning's Data Hooks.
1. RTD: https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
2. Github: https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/hooks.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(1, os.getcwd())

from abc import ABC, abstractmethod
from typing import Optional, Union


import cv2
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
import albumentations as A
from pathlib import Path
from configs.global_params import PipelineConfig
from src.augmentations import ImageClassificationTransforms
from src.utils import (
    seed_all,
    show,
    download_to,
    extract_file,
    create_dataframe_with_image_info,
    return_list_of_files,
)
from torchvision.datasets import MNIST


TransformTypes = Optional[Union[A.Compose, T.Compose]]

# pylint: disable=invalid-name
class ImageClassificationDataset(Dataset):
    """A sample template for Image Classification Dataset."""

    def __init__(
        self,
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
        # TODO: it is hardcoded and can be put in config
        self.image_path = df["image_path"].values
        self.image_ids = df["image_id"].values
        self.targets = df["class_id"].values if stage != "test" else None
        self.df = df
        self.transforms = transforms
        self.stage = stage
        self.path = path

        # TODO: Add checks for correct input modes etc.

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, transforms: TransformTypes, stage: str
    ) -> ImageClassificationDataset:
        """Creates an instance of the dataset class from a dataframe.
        This is the default way for now.

        Example:
            >>> from utils import create_dataframe_with_image_info
            >>> df = create_dataframe_with_image_info()
        """

        return cls(df=df, transforms=transforms, stage=stage)

    @classmethod
    def from_folder(
        cls,
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
        return cls(path=path, transforms=transforms, stage=stage, **kwargs)

    def check_correct_dtype(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Check if the datatypes of X and y are correct. This is important as
        the loss function expects a certain datatype. See my repo/src/dataset.py."""
        raise NotImplementedError

    def check_correct_shape(self) -> None:
        """Check the shapes of X and y, for eg channels first."""
        raise NotImplementedError

    def apply_image_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the image."""
        if self.transforms and isinstance(self.transforms, A.Compose):
            image = self.transforms(image=image)["image"]
        elif self.transforms and isinstance(self.transforms, T.Compose):
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # float
        return image

    def apply_target_transforms(self, target: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the target."""
        target = torch.tensor(target, dtype=torch.long)  # FIXME: hardcoded torch.long
        # other transforms such as bbox transforms can be applied here
        return target

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


class CustomizedDataModule(ABC):
    """Base class for custom data module.

    Note:
        1. Consider extending this to segmentation and object detection.

    References:
        - https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
        - https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/hooks.py
    """

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        self.pipeline_config = pipeline_config

    def prepare_data(self) -> None:
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

            if stage == "debug":
                # assign debug dataset for use in dataloaders
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

    def prepare_data(self) -> None:
        # download data here
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        self.path = self.pipeline_config.data.root_dir
        self.download = self.pipeline_config.data.download

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders."""

        if stage == "fit":
            self.train_dataset = MNIST(
                download=self.download,
                root=self.path,
                transform=self.transform,
                train=True,
            )
            self.valid_dataset = MNIST(
                download=self.download,
                root=self.path,
                transform=self.transform,
                train=False,
            )
        if self.pipeline_config.datamodule.debug:
            self.train_dataset = Subset(self.train_dataset, indices=range(1280))
            self.valid_dataset = Subset(self.valid_dataset, indices=range(1280))

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

    def prepare_data(self) -> None:
        """Download data here and prepare."""
        # TODO: I modified the original castings data so no need download for now.
        # url = self.pipeline_config.data.url
        # blob_file = self.pipeline_config.data.blob_file
        # root_dir = self.pipeline_config.data.root_dir

        # download_to(url, blob_file, root_dir)
        # extract_file(root_dir, blob_file)

        # TODO: Here needs to be more generic for users,
        # for example, if user choose train_test_split, then we need to
        # split accordingly. If user choose KFold, then we need to split
        # to 5 folds.

        # data_dir = self.pipeline_config.data.data_dir

        # class_name_to_id = {"ok": 0, "defect": 1}

        # all_images = return_list_of_files(
        #     data_dir, extensions=[".jpg", ".png", ".jpeg"], return_string=False
        # )

        # df = create_dataframe_with_image_info(
        #     all_images, class_name_to_id, save_path=data_dir / "df.csv"
        # )

        self.df = pd.read_csv(self.pipeline_config.data.data_csv)
        self.train_df, self.valid_df = train_test_split(
            self.df, test_size=0.1, random_state=42
        )
        if self.pipeline_config.datamodule.debug:
            self.debug_train_df = self.train_df.sample(100)
            self.debug_valid_df = self.valid_df.sample(100)

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders."""

        if stage == "fit":
            train_transforms = self.transforms.train_transforms
            valid_transforms = self.transforms.valid_transforms

            self.train_dataset = ImageClassificationDataset(
                self.train_df,
                stage="train",
                transforms=train_transforms,
            )
            self.valid_dataset = ImageClassificationDataset(
                self.valid_df,
                stage="valid",
                transforms=valid_transforms,
            )

        if stage == "debug":
            debug_transforms = self.transforms.debug_transforms
            self.debug_train_dataset = ImageClassificationDataset(
                self.debug_train_df,
                stage="debug",
                transforms=debug_transforms,
            )

            self.debug_valid_dataset = ImageClassificationDataset(
                self.debug_valid_df, stage="debug", transforms=debug_transforms
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

    def debug_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.debug_train_dataset, **self.pipeline_config.datamodule.debug_loader
        )

    def debug_valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.debug_valid_dataset, **self.pipeline_config.datamodule.debug_loader
        )


if __name__ == "__main__":
    seed_all(42)

    pipeline_config = PipelineConfig()
    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()
    dm.setup(stage="debug")

    image, target = dm.debug_train_dataset[0]
    print(image.shape, target)

    debug_train_loader = dm.debug_train_dataloader()
    image_batch, target_batch = next(iter(debug_train_loader))
    image_grid = torchvision.utils.make_grid(image_batch)
    show(image_grid)
