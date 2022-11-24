"""Implements Model History."""


from src.callbacks.callback import Callback
from typing import Dict, Any, DefaultDict, List
from src.trainer import Trainer
from collections import defaultdict


class History(Callback):
    """Class to log metrics to console and save them to a CSV file."""

    def __init__(self):
        """Constructor class for History Class."""
        super().__init__()
        self.history: DefaultDict[str, List[Any]]

    def on_trainer_start(
        self, trainer: Trainer  # pylint: disable=unused-argument
    ) -> None:
        """When the trainer starts, we should initialize the history.
        This is init method of Trainer.
        """
        self.history = defaultdict(list)

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        """Method to update history object at the end of every epoch."""
        self._update(history=trainer.train_history_dict)

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Method to update history object at the end of every epoch."""
        self._update(history=trainer.valid_history_dict)

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Method assigns accumulated history to history attribute
        back to Trainer class."""
        trainer.history = self.history

    def _update(self, history: Dict[str, Any]) -> None:
        """Updates the history object with the latest metrics."""
        for key in history:
            if key not in self.history:
                self.history[key] = [history.get(key)]
            else:
                self.history[key].append(history.get(key))
