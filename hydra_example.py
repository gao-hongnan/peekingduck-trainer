from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@dataclass(frozen=False, init=True)
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


@dataclass(frozen=False, init=True)
class Data:
    """Class for data related params."""

    root_dir: Optional[Path] = Path("./data")
    dataset_name: str = "cifar10"
    train_dir: Union[str, Path] = "${data.root_dir}/${data.dataset_name}/train"


@dataclass
class Config:
    # We will populate db using composition.
    data: Data = Data()
    criterion: CriterionParams = CriterionParams()


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
    print(cat_dir)
    images_in_train_dir = len(list(cat_dir.glob("*.png")))
    print(f"Images in train dir: {images_in_train_dir}")
    criterion = cfg.criterion
    print(criterion.train_criterion)


if __name__ == "__main__":
    my_app()
