from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torchvision.transforms as T
from src.utils.general_utils import generate_uuid4
import torchvision

# type: ignore
import albumentations as A
import omegaconf
from omegaconf import DictConfig

import collections
import importlib
from itertools import product
from typing import Any, Dict, Generator

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def load_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations
    Args:
        cfg:
    Returns:
        compose object
    """
    augs = []
    for a in cfg:
        if a["class_name"] == "albumentations.OneOf":
            small_augs = []
            for small_aug in a["params"]:
                # yaml can't contain tuples, so we need to convert manually
                params = {
                    k: (
                        v
                        if not isinstance(v, omegaconf.listconfig.ListConfig)
                        else tuple(v)
                    )
                    for k, v in small_aug["params"].items()
                }
                aug = load_obj(small_aug["class_name"])(**params)
                small_augs.append(aug)
            aug = load_obj(a["class_name"])(small_augs)
            augs.append(aug)

        else:
            params = {
                k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v))
                for k, v in a["params"].items()
            }
            aug = load_obj(a["class_name"])(**params)
            augs.append(aug)

    return T.Compose(augs)


@dataclass
class Data:
    """Class for data related params."""

    root_dir: Path = Path(".")
    data_dir: Path = "${data.root_dir}/data"
    dataset_name: str = "cifar10"
    train_dir: str = "${data.data_dir}/${data.dataset_name}/train"
    test_dir: str = "${data.data_dir}/${data.dataset_name}/test"
    train_csv: str = "${data.data_dir}/${data.dataset_name}/train.csv"
    test_csv: str = "${data.data_dir}/${data.dataset_name}/test.csv"

    url: Optional[
        str
    ] = "https://github.com/gao-hongnan/peekingduck-trainer/releases/download/v0.0.1-alpha/cifar10.zip"
    blob_file: Optional[str] = "cifar10.zip"
    download: bool = False

    image_col_name: str = "image_id"
    image_path_col_name: str = "image_path"
    label_col_name: str = "class_id"
    group_by: Optional[str] = None
    stratify_by: Optional[str] = None
    image_extension: str = ".png"
    class_name_to_id: Optional[Dict[str, int]] = field(
        default_factory=lambda: {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }
    )


@dataclass
class Resampling:
    """Class for cross validation."""

    # scikit-learn resampling strategy
    resample_strategy: str = "train_test_split"  # same name as in scikit-learn
    resample_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "train_size": 0.9,
            "test_size": 0.1,
            "random_state": 42,
            "shuffle": True,
        }
    )


@dataclass
class DataModuleParams:
    """Class to keep track of the data loader parameters."""

    debug: bool = True  # TODO: how to pass debug in argparse to here?
    num_debug_samples: int = 128

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    test_loader: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )


@dataclass
class AugmentationParams:
    """Class to keep track of the augmentation parameters."""

    image_size: int = 32
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    mixup: bool = False
    mixup_params: Optional[Dict[str, Any]] = None
    a: torchvision.transforms.Compose = 1
    # TODO: give warning since it defaults to None if not keyed in?
    train_transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"class_name": "torchvision.transforms.ToPILImage", "params": {}},
            {
                "class_name": "torchvision.transforms.RandomResizedCrop",
                "params": {"size": "${transforms.image_size}"},
            },
            {
                "class_name": "torchvision.transforms.RandomHorizontalFlip",
                "params": {"p": 0.5},
            },
            {"class_name": "torchvision.transforms.ToTensor", "params": {}},
            {
                "class_name": "torchvision.transforms.Normalize",
                "params": {"mean": "${transforms.mean}", "std": "${transforms.std}"},
            },
        ]
    )

    valid_transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"class_name": "torchvision.transforms.ToPILImage", "params": {}},
            {
                "class_name": "torchvision.transforms.Resize",
                "params": {"size": "${transforms.image_size}"},
            },
            {"class_name": "torchvision.transforms.ToTensor", "params": {}},
            {
                "class_name": "torchvision.transforms.Normalize",
                "params": {"mean": "${transforms.mean}", "std": "${transforms.std}"},
            },
        ]
    )
    test_transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"class_name": "torchvision.transforms.ToPILImage", "params": {}},
            {
                "class_name": "torchvision.transforms.Resize",
                "params": {"size": "${transforms.image_size}"},
            },
            {"class_name": "torchvision.transforms.ToTensor", "params": {}},
            {
                "class_name": "torchvision.transforms.Normalize",
                "params": {"mean": "${transforms.mean}", "std": "${transforms.std}"},
            },
        ]
    )


@dataclass
class ModelParams:
    """Class to keep track of the model parameters."""

    adapter: str = "torchvision"  # "torchvision" "timm"
    model_name: str = "resnet18"  # resnet18 "tf_efficientnetv2_s"
    pretrained: bool = True
    num_classes: int = 10  # 2


@dataclass
class CriterionParams:
    """A class to track loss function parameters."""

    train_criterion: str = "CrossEntropyLoss"
    valid_criterion: str = "CrossEntropyLoss"
    train_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.0,
        }
    )
    valid_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "size_average": None,
            "ignore_index": -100,
            "reduce": None,
            "reduction": "mean",
            "label_smoothing": 0.0,
        }
    )


@dataclass
class OptimizerParams:
    """A class to track optimizer parameters."""

    # batch size increase 2, lr increases a factor of 2 as well.
    optimizer: str = "AdamW"
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 3e-4,  # bs: 32 -> lr = 3e-4
            "betas": (0.9, 0.999),
            "amsgrad": False,
            "weight_decay": 1e-6,
            "eps": 1e-08,
        }
    )


@dataclass
class SchedulerParams:
    """A class to track Scheduler Params."""

    scheduler: str = "CosineAnnealingWarmRestarts"  # Debug
    scheduler_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "T_0": 10,
            "T_mult": 1,
            "eta_min": 1e-6,
            "last_epoch": -1,
            "verbose": False,
        }
    )


# @dataclass
# class CallbackParams:
#     """Callback params."""

#     callbacks: List[Callback] = field(
#         default_factory=lambda: [
#             History(),
#             MetricMeter(),
#             ModelCheckpoint(mode="max", monitor="val_Accuracy"),
#             EarlyStopping(mode="max", monitor="val_Accuracy", patience=3),
#             # WandbLogger(project="CIFAR-10", entity="hongnan-aisg"),
#         ]
#     )


@dataclass
class GlobalTrainParams:
    """Train params, a lot of overlapping.
    FIXME: overlapping with other params.
    """

    epochs: int = 10  # 10 when not debug
    use_amp: bool = True
    patience: int = 3
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {
            "monitor": "val_Accuracy",
            "mode": "max",
        }
    )


@dataclass
class Stores:
    """A class to keep track of model artifacts."""

    project_name: str = "CIFAR-10"
    unique_id: str = field(default_factory=generate_uuid4)
    # logs_dir: Path = field(init=False)
    # model_artifacts_dir: Path = field(init=False)

    def __post_init__(self):
        """Post init."""
        # self.logs_dir = Path("logs") / self.project_name / self.unique_id
        # self.model_artifacts_dir = Path("model_artifacts") / self.project_name / self.unique_id
        self.unique_id = torch.rand(1).item()


@dataclass
class Config:
    # We will populate db using composition.
    data: Data = Data()
    resample: Resampling = Resampling()
    datamodule: DataModuleParams = DataModuleParams()
    transforms: AugmentationParams = AugmentationParams()
    model: ModelParams = ModelParams()
    criterion_params: CriterionParams = CriterionParams()
    optimizer_params: OptimizerParams = OptimizerParams()
    scheduler_params: SchedulerParams = SchedulerParams()
    stores: Stores = Stores()


# Create config group `db` with options 'mysql' and 'postgreqsl'
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    data = cfg.data
    assert isinstance(data.root_dir, Path)
    train_dir = data.train_dir
    cat_dir = Path(train_dir) / "cat"
    images_in_train_dir = len(list(cat_dir.glob("*.png")))
    print(f"Images in train dir: {images_in_train_dir}")
    criterion = cfg.criterion_params
    print(criterion.train_criterion)

    stores = cfg.stores
    print(stores.unique_id)

    transforms = cfg.transforms
    print(transforms.train_transforms)
    train_transforms = load_augs(transforms.train_transforms)
    print(train_transforms)


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
