import hydra
import uuid
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path


@hydra.main(
    version_base=None, config_path="configs/hydra_configs", config_name="config"
)
def run(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    stores = config.stores
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"][
        "output_dir"
    ]
    logs_dir = hydra_output_dir + "/logs"

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    model_artifacts_dir = hydra_output_dir + "/artifacts"
    Path(model_artifacts_dir).mkdir(parents=True, exist_ok=True)

    stores.logs_dir = logs_dir
    stores.model_artifacts_dir = model_artifacts_dir

    train_transforms = hydra.utils.instantiate(config.transforms.train_transforms)
    valid_transforms = hydra.utils.instantiate(config.transforms.valid_transforms)
    config.transforms.train_transforms = train_transforms
    return config


class Controller:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def instantiate(self, obj):
        ...

    def get_datamodule(self, datamodule):
        ...

    def get_model(self, model):
        ...

    def get_metrics(self, metrics):
        ...

    def get_callbacks(self, callbacks):
        ...

    def get_trainer(self, trainer):
        ...

    def run(self):
        self.instantiate(self.config)
        self.get_datamodule(self.config.datamodule)
        self.get_model(self.config.model)
        self.get_metrics(self.config.metrics)
        self.get_callbacks(self.config.callbacks)
        self.get_trainer(self.config.trainer)
        self.run_trainer()


if __name__ == "__main__":
    run()
