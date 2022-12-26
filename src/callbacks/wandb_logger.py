"""
Implements logger for weights and biases.



TODO:
    1. Is there a need for module available check here? Maybe cause wandb is not a dependency?
    2. This should have a base class, see PyTorch Lightning's Logger(ABC)
    3. Grossly simplified version here, see PyTorch Lightning:
        https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/loggers/wandb.py
"""

from abc import ABC
from typing import Dict, List, Optional, Any

from src.callbacks.base import Callback
from src.trainer import Trainer

try:
    import wandb
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb = None


class WandbLogger(Callback, ABC):
    """Callback to log your metrics and loss values to  wandb platform.

    For more information about wandb take a look at [Weights and Biases](https://wandb.ai/)

    Args:
            project: The name of the project where you're sending the new run
            entity:  An entity is a username or team name where you're sending runs.
            name: A short display name for this run
            config: The hyperparameters for your model and experiment as a dictionary
            tags:  List of strings.
            directory: where to save wandb local run directory.
                If set to None it will use experiments save_dir argument.
            notes: A longer description of the run, like a -m commit message in git

    Note:
            set os.environ['WANDB_SILENT'] = True to silence wandb log statements.
            If this is set all logs will be written to WANDB_DIR/debug.log

    Examples:
        .. code-block::

            from torchflare.callbacks import WandbLogger

            params = {"bs": 16, "lr": 0.3}

            logger = WandbLogger(
                project="Experiment",
                entity="username",
                name="Experiment_10",
                config=params,
                tags=["Experiment", "fold_0"])
    """

    def __init__(
        self,
        project: str,
        entity: str,
        name: str = None,
        config: Dict = None,
        dir: str = None,  # pylint: disable=redefined-builtin
        **kwargs: Dict[str, Any],
    ) -> None:
        """Constructor of WandbLogger."""
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."
            )

        super().__init__()
        self.entity = entity
        self.project = project
        self.name = name
        self.config = config
        self.dir = dir
        self.experiment = None
        self.kwargs = kwargs

    def on_trainer_start(self, trainer: Trainer) -> None:
        self.experiment = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            config=self.config,
            dir=self.dir,
            **self.kwargs,
        )

    # def on_valid_epoch_end(self, trainer: Trainer) -> None:
    #     """Method to log metrics and values at the end of very epoch."""
    #     logs = {
    #         k: v for k, v in experiment.exp_logs.items() if k != experiment.epoch_key
    #     }
    #     self.experiment.log(logs)

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Method to end experiment after training is done."""
        self.experiment.finish()
