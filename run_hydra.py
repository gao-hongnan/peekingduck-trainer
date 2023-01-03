import hydra
import uuid
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field

@dataclass(frozen=False, init=True)
class ModelParams:
    """Class to keep track of the model parameters."""

    adapter: str = "torchvision"  # "torchvision" "timm"
    model_name: str = "resnet18"  # resnet18 "tf_efficientnetv2_s"
    pretrained: bool = True
    num_classes: int = 10  # 2
    dropout: float = 0.3  # 0.5

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=ModelParams)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.data.root_dir)
    uuid4_num = hydra.utils.instantiate(cfg.stores)
    print("aaa", uuid4_num)
    adapter = cfg.model.adapter
    model_name = cfg.model.model_name
    pretrained = cfg.model.pretrained
    if adapter == "torchvision":

        backbone = getattr(torchvision.models, model_name)(pretrained=pretrained)
        # print(backbone)


if __name__ == "__main__":
    run()
