from typing import Any


class Callback:
    """Callback base class.

    Reference:
        - https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/callbacks/callback.py
    """

    def __init__(self) -> None:
        """Constructor for Callback base class."""

    @property
    def state_key(self) -> str:
        """Identifier for the state of the callback.
        Used to store and retrieve a callback's state from the checkpoint dictionary by
        ``checkpoint["callbacks"][state_key]``. Implementations of a callback need to provide a unique state key if 1)
        the callback has state and 2) it is desired to maintain the state of multiple instances of that callback.
        """
        return self.__class__.__qualname__

    def _generate_state_key(self, **kwargs: Any) -> str:
        """Formats a set of key-value pairs into a state key string with the callback class name prefixed. Useful
        for defining a :attr:`state_key`.
        Args:
            **kwargs: A set of key-value pairs. Must be serializable to :class:`str`.
        """
        return f"{self.__class__.__qualname__}{repr(kwargs)}"

    def setup(self, trainer, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""

    def teardown(self, trainer, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    def on_trainer_start(self, trainer) -> None:
        """Called when the trainer begins."""

    def on_trainer_end(self, trainer) -> None:
        """Called when the trainer ends."""

    def on_fit_start(self, trainer) -> None:
        """Called AFTER fit begins."""

    def on_fit_end(self, trainer) -> None:
        """Called AFTER fit ends."""

    def on_train_batch_start(self, trainer) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(self, trainer) -> None:
        """Called when the train batch ends.
        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """

    def on_train_loader_start(self, trainer) -> None:
        """Called when the train loader begins."""

    def on_valid_loader_start(self, trainer) -> None:
        """Called when the validation loader begins."""

    def on_train_loader_end(self, trainer) -> None:
        """Called when the train loader ends."""

    def on_valid_loader_end(self, trainer) -> None:
        """Called when the validation loader ends."""

    def on_train_epoch_start(self, trainer) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer) -> None:
        """Called when the train epoch ends.
        To access all batch outputs at the end of the epoch, either:
        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_valid_epoch_start(self, trainer) -> None:
        """Called when the val epoch begins."""

    def on_valid_epoch_end(self, trainer) -> None:
        """Called when the val epoch ends."""

    def on_valid_batch_start(self, trainer) -> None:
        """Called when the validation batch begins."""

    def on_valid_batch_end(self, trainer) -> None:
        """Called when the validation batch ends."""
