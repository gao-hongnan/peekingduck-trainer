"""Callbacks base class.
This follows the Observer pattern, where the callback is the observer and the
trainer is the subject. The callback is notified of events in the trainer and
can take action accordingly.

See https://github.com/msaroufim/ml-design-patterns.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING  # This solves circular import type hinting.

if TYPE_CHECKING:
    from src.trainer import Trainer


# pylint: disable=line-too-long, too-many-public-methods
class Callback:
    """Callback base class.

    TODO: Torchflare sorts the callback, maybe we should look into it.
    This might be because some default callbacks instantiate some global attributes
    for eg, self.model_artifacts_dir

    Note:
        Different concrete callbacks can invoke different `on_..._start` and `on_..._end` methods.
        Therefore it is not an ABC class.

    Reference:
        - https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/callbacks/callback.py
        - https://github.com/Atharva-Phatak/torchflare/tree/main/torchflare
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

    def setup(self, trainer: Trainer, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""

    def teardown(self, trainer: Trainer, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    def on_trainer_start(self, trainer: Trainer) -> None:
        """Called when the trainer: Trainer begins."""

    def on_trainer_end(self, trainer: Trainer) -> None:
        """Called when the trainer: Trainer ends."""

    def on_fit_start(self, trainer: Trainer) -> None:
        """Called AFTER fit begins."""

    def on_fit_end(self, trainer: Trainer) -> None:
        """Called AFTER fit ends."""

    def on_train_batch_start(self, trainer: Trainer) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(self, trainer: Trainer) -> None:
        """Called when the train batch ends.
        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t
            ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """

    def on_train_loader_start(self, trainer: Trainer) -> None:
        """Called when the train loader begins."""

    def on_valid_loader_start(self, trainer: Trainer) -> None:
        """Called when the validation loader begins."""

    def on_train_loader_end(self, trainer: Trainer) -> None:
        """Called when the train loader ends."""

    def on_valid_loader_end(self, trainer: Trainer) -> None:
        """Called when the validation loader ends."""

    def on_train_epoch_start(self, trainer: Trainer) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: Trainer) -> None:
        """Called when the train epoch ends.
        To access all batch outputs at the end of the epoch, either:
        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module
            OR
        2. Cache data across train batch hooks inside the callback implementation to
        post-process in this hook.
        """

    def on_valid_epoch_start(self, trainer: Trainer) -> None:
        """Called when the val epoch begins."""

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Called when the val epoch ends."""

    def on_valid_batch_start(self, trainer: Trainer) -> None:
        """Called when the validation batch begins."""

    def on_valid_batch_end(self, trainer: Trainer) -> None:
        """Called when the validation batch ends."""

    def on_inference_start(self, trainer: Trainer) -> None:
        """Called when the inference begins."""

    def on_inference_end(self, trainer: Trainer) -> None:
        """Called when the inference ends."""
