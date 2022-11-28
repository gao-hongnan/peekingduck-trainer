"""Model Checkpoint Callback.

Reference:
    1. pytorch_lightning/callbacks/model_checkpoint.py
    2. torchflare/callbacks/model_checkpoint.py
Logic:
    1. This is called after each epoch.
    2. We check if the current epoch score is better than the best score.
    3. If it is, we save the model.
    4. Additional logics such as save every batch, save top k models, etc
        can be enhancements.

Save the weights and states for the best evaluation metric and also the OOF scores.
valid_trues -> oof_trues: np.array of shape [num_samples, 1] and represent the true
    labels for each sample in current fold.
    i.e. oof_trues.flattened()[i] = true label of sample i in current fold.
valid_logits -> oof_logits: np.array of shape [num_samples, num_classes] and
    represent the logits for each sample in current fold.
    i.e. oof_logits[i] = [logit_of_sample_i_in_current_fold_for_class_0,
                            logit_of_sample_i_in_current_fold_for_class_1, ...]
valid_preds -> oof_preds: np.array of shape [num_samples, 1] and represent the
    predicted labels for each sample in current fold.
    i.e. oof_preds.flattened()[i] = predicted label of sample i in current fold.
valid_probs -> oof_probs: np.array of shape [num_samples, num_classes] and represent the
    probabilities for each sample in current fold. i.e. first row is
    the probabilities of the first class.
    i.e. oof_probs[i] = [probability_of_sample_i_in_current_fold_for_class_0,
                            probability_of_sample_i_in_current_fold_for_class_1, ...]
"""
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch

from src.callbacks.callback import Callback
from src.trainer import Trainer


# TODO: taken from torchflare, to check again.
def _is_min(score, best, min_delta):
    return score <= (best - min_delta)


def _is_max(score, best, min_delta):
    return score >= (best + min_delta)


def init_improvement(mode: str, min_delta: float):
    """Get the scoring function and the best value according to mode.

    Args:
        mode: one of min or max.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.

    Returns:
        The scoring function and best value according to mode.
    """
    if mode == "min":
        improvement = partial(_is_min, min_delta=min_delta)
        best_score = math.inf
    else:
        improvement = partial(_is_max, min_delta=min_delta)
        best_score = -math.inf
    return improvement, best_score


class ModelCheckpoint(Callback):
    """Callback for Checkpointing your model.
    Referenced from torchflare.

    Args:
            mode: One of {"min", "max"}.
                In min mode, training will stop when the quantity monitored has stopped decreasing
                in "max" mode it will stop when the quantity monitored has stopped increasing.
            metric_name: Name of the metric to monitor, should be one of the keys in metrics list.

    Raises:
        ValueError if monitor does not start with prefix ``val_`` or ``train_``.

    Example:
        .. code-block::

            from pkd.callbacks.model_checkpoint import ModelCheckpoint
            model_checkpoint = ModelCheckpoint(mode="max", metric_name="val_Accuracy")
    """

    def __init__(
        self,
        mode: str,
        metric_name: str,
    ) -> None:
        """Constructor for ModelCheckpoint class."""
        super().__init__()

        if metric_name.startswith("train_") or metric_name.startswith("val_"):
            self.metric_name = metric_name
        else:
            raise ValueError("Monitor must have a prefix either train_ or val_.")

        self.mode = mode
        self.eps = 1e-7

        self.improvement, self.best_val = init_improvement(
            mode=self.mode, min_delta=self.eps
        )
        # FIXME: do we need self.save_dir here? For now taking from model_artifacts_dir
        # ideally as an argument and set attribute for pipeline_configs.stores?

    @staticmethod
    def save_checkpoint(state_dict: Dict[str, Any], model_artifacts_path: Path) -> None:
        """Method to save the state dictionaries of model, optimizer,etc."""
        torch.save(state_dict, model_artifacts_path)

    def on_trainer_start(self, trainer: Trainer) -> None:
        """Initialize the best score as either -inf or inf depending on mode."""
        # self.best_valid_loss = math.inf  # this is always true and will be logged?
        # TODO: support saving for multiple metrics, i.e. best loss and best accuracy
        self.best_valid_score = (
            -math.inf if trainer.monitored_metric["mode"] == "max" else math.inf
        )
        self.state_dict = {
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "epoch": None,
            "best_score": None,
            "oof_trues": None,
            "oof_preds": None,
            "oof_scores": None,
            "oof_logits": None,
        }
        self.model_artifacts_dir = trainer.pipeline_config.stores.model_artifacts_dir

    def on_valid_epoch_end(self, trainer: Trainer) -> None:
        """Method to save best model depending on the monitored quantity."""
        valid_score = trainer.valid_epoch_dict.get(self.metric_name)

        if self.improvement(score=valid_score, best=self.best_valid_score):
            model_artifacts_path = Path.joinpath(
                self.model_artifacts_dir,
                f"{trainer.pipeline_config.model.model_name}_best_{self.metric_name}_fold_{trainer.current_fold}_epoch{trainer.current_epoch}.pt",
            ).as_posix()

            self.best_valid_score = valid_score
            # FIXME: not elegant to assign to dict like this
            self.state_dict["model_state_dict"] = trainer.model.state_dict()
            self.state_dict["optimizer_state_dict"] = trainer.optimizer.state_dict()
            self.state_dict["scheduler_state_dict"] = trainer.scheduler.state_dict()
            # self.state_dict["epoch"] = trainer.current_epoch
            # self.state_dict["best_score"] = self.best_valid_score
            self.state_dict["oof_trues"] = trainer.valid_history_dict["valid_trues"]
            self.state_dict["oof_preds"] = trainer.valid_history_dict["valid_preds"]
            self.state_dict["oof_probs"] = trainer.valid_history_dict["valid_probs"]
            self.state_dict["oof_logits"] = trainer.valid_history_dict["valid_logits"]
            self.state_dict["model_artifacts_path"] = model_artifacts_path
            self.save_checkpoint(self.state_dict, model_artifacts_path)
