"""Trainer class for training and validating models."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm.auto import tqdm

from configs.config import init_logger
from configs.base_params import PipelineConfig
from src.callbacks.callback import Callback
from src.model import Model
from src.utils import general_utils

# TODO: clean up val vs valid naming confusions.
def get_sigmoid_softmax(
    pipeline_config: PipelineConfig,
) -> Union[torch.nn.Sigmoid, torch.nn.Softmax]:
    """Get the sigmoid or softmax function depending on loss function."""
    if pipeline_config.criterion_params.train_criterion_name == "BCEWithLogitsLoss":
        return getattr(torch.nn, "Sigmoid")()

    if pipeline_config.criterion_params.train_criterion_name == "CrossEntropyLoss":
        return getattr(torch.nn, "Softmax")(dim=1)


class Trainer:  # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-public-methods
    """Object used to facilitate training."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        model: Model,
        early_stopping=None,
        callbacks: List[Callback] = None,
        metrics: Union[MetricCollection, List[str]] = None,
    ) -> None:
        """Initialize the trainer.
        TODO:
        1. state = {"model": ..., "optimizer": ...} Torchflare's state is equivalent
        to our pipeline_config, but his holds callable as values.

        monitored_metric = {
            "metric_name": "val_Accuracy",
            "metric_score": None,
            "mode": "max",
        }

        history = {"train_loss": [...], "valid_loss": [...], # save all epoch
                    "train_acc": [...], "valid_acc": [...], # save all epoch
                    "train_auroc": [...], "valid_auroc": [...], # save all epoch
                    "train_logits": [...], "valid_logits": [...], # save only best epoch?
                    "train_preds": [...], "valid_preds": [...], # save only best epoch?
                    "train_probs": [...], "valid_probs": [...], # save only best epoch?
                    }
        """
        # Set params
        self.pipeline_config = pipeline_config
        self.train_params = self.pipeline_config.global_train_params
        self.model = model
        self.model_artifacts_dir = pipeline_config.stores.model_artifacts_dir
        self.device = self.pipeline_config.device

        self.early_stopping = early_stopping
        self.callbacks = callbacks
        self.metrics = metrics
        # TODO: if isinstance(metrics, list): convert to MetricCollection

        self.logger = init_logger(
            log_file=Path.joinpath(
                self.pipeline_config.stores.logs_dir, "training.log"
            ),
            module_name="training",
        )
        self.initialize()  # init non init attributes, etc

    def get_classification_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
        # mode: str = "valid",
    ):
        """[summary]
        # https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);
            mode (str, optional): [description]. Defaults to "valid".
        Returns:
            [type]: [description]
        """

        self.train_metrics = self.metrics.clone(prefix="train_")
        self.valid_metrics = self.metrics.clone(prefix="val_")
        train_metrics_results = self.train_metrics(y_probs, y_trues.flatten())
        valid_metrics_results = self.valid_metrics(y_probs, y_trues.flatten())
        print(f"valid metrics: {valid_metrics_results}")
        return train_metrics_results, valid_metrics_results

    def run(self):
        self.on_trainer_start()
        # self.fit()
        self.on_trainer_end()

    def initialize(self) -> None:
        """Called when the trainer begins."""
        self.optimizer = self.get_optimizer(
            model=self.model,
            optimizer_params=self.pipeline_config.optimizer_params,
        )
        self.scheduler = self.get_scheduler(
            optimizer=self.optimizer,
            scheduler_params=self.pipeline_config.scheduler_params,
        )

        if self.train_params.use_amp:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.monitored_metric = self.train_params.monitored_metric

        # Metric to optimize, either min or max.
        self.best_valid_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )
        self.patience_counter = self.train_params.patience  # Early Stopping Counter
        self.current_epoch = 1
        self.current_fold = None
        self.train_epoch_dict = {}
        self.valid_epoch_dict = {}
        self.train_batch_dict = {}
        self.valid_batch_dict = {}
        self.train_history_dict = {}
        self.valid_history_dict = {}
        self.invoke_callbacks("on_trainer_start")

    def on_fit_start(self, fold: int) -> None:
        """Called AFTER fit begins."""
        self.logger.info(
            f"\nTraining on Fold {fold} and using {self.train_params.model_name}\n"
        )
        self.best_valid_loss = np.inf

    def on_fit_end(self) -> None:
        """Called AFTER fit ends."""
        # print(self.train_batch_dict)
        print(self.valid_batch_dict)
        # print(self.train_epoch_dict)
        print(self.valid_epoch_dict)
        # print(self.train_history_dict)
        # print(self.valid_history_dict)
        general_utils.free_gpu_memory(
            self.optimizer,
            self.scheduler,
            self.valid_history_dict["valid_trues"],
            self.valid_history_dict["valid_logits"],
            self.valid_history_dict["valid_preds"],
            self.valid_history_dict["valid_probs"],
        )

    def invoke_callbacks(self, event_name: str) -> None:
        """Invoke the callbacks."""
        for callback in self.callbacks:
            try:
                getattr(callback, event_name)(self)
            except NotImplementedError:
                pass

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        fold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fit the model and returns the history object."""
        self.on_fit_start(fold=fold)

        for _epoch in range(1, self.train_params.epochs + 1):
            self.train_one_epoch(train_loader, _epoch)
            self.valid_one_epoch(valid_loader, _epoch)

            self.monitored_metric["metric_score"] = torch.clone(
                self.valid_history_dict[self.monitored_metric["metric_name"]]
            ).detach()  # FIXME: one should not hardcode "valid_macro_auroc" here
            valid_loss = self.valid_history_dict["valid_loss"]

            if self.early_stopping is not None:
                # TODO: Implement this properly, Add save_model_artifacts here as well. Or rather, create a proper callback to reduce the complexity of code below.
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=valid_loss
                )
                self.best_valid_loss = best_score

                if early_stop:
                    self.logger.info("Stopping Early!")
                    break
            else:

                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss

                if self.monitored_metric["mode"] == "max":
                    if self.monitored_metric["metric_score"] > self.best_valid_score:
                        self.logger.info(
                            f"\nValidation {self.monitored_metric['metric_name']} improved from {self.best_valid_score} to {self.monitored_metric['metric_score']}"
                        )
                        self.best_valid_score = self.monitored_metric["metric_score"]
                        # Reset patience counter as we found a new best score
                        patience_counter_ = self.patience_counter

                        # TODO: see my wandb run save from siim project

                        self.logger.info(
                            f"\nSaving model with best valid {self.monitored_metric['metric_name']} score: {self.best_valid_score}\n"
                        )
                    else:
                        patience_counter_ -= 1
                        self.logger.info(f"Patience Counter {patience_counter_}")
                        if patience_counter_ == 0:
                            self.logger.info(
                                f"\n\nEarly Stopping, patience reached!\n\nbest valid {self.monitored_metric['metric_name']} score: {self.best_valid_score}"
                            )
                            break
                else:
                    if self.monitored_metric["metric_score"] < self.best_valid_score:
                        self.best_valid_score = self.monitored_metric["metric_score"]

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            self.current_epoch += 1

        self.on_fit_end()
        # FIXME: here is finish fitting, whether to call it on train end or on fit end?
        # Currently only history uses on_trainer_end.
        for callback in self.callbacks:
            callback.on_trainer_end(self)
        return self.history

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> None:
        """Train one epoch of the model."""
        curr_lr = self.get_lr(self.optimizer)
        train_start_time = time.time()

        # set to train mode
        self.model.train()

        train_bar = tqdm(train_loader)

        # Iterate over train batches
        for _step, batch in enumerate(train_bar, start=1):
            # unpack - note that if BCEWithLogitsLoss, dataset should do view(-1,1) and not here.
            inputs, targets = batch
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            _batch_size = inputs.shape[0] # unused for now

            with torch.cuda.amp.autocast(
                enabled=self.train_params.use_amp,
                dtype=torch.float16,
                cache_enabled=True,
            ):
                logits = self.model(inputs)  # Forward pass logits
                curr_batch_train_loss = self.computer_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                    stage="train",
                )
            self.optimizer.zero_grad()  # reset gradients

            if self.scaler is not None:
                self.scaler.scale(curr_batch_train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                curr_batch_train_loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric, every batch is diff
            self.train_batch_dict["train_loss"] = curr_batch_train_loss.item()
            # train_bar.set_description(f"Train. {metric_monitor}")

            _y_train_prob = get_sigmoid_softmax(self.pipeline_config)(logits)

            _y_train_pred = torch.argmax(_y_train_prob, dim=1)

            self.invoke_callbacks("on_train_batch_end")

        self.invoke_callbacks("on_train_loader_end")
        # total time elapsed for this epoch
        train_time_elapsed = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - train_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Train. Epoch {epoch}:"
            f"\nAvg Train Summary Loss: {self.train_epoch_dict['train_loss']:.3f}"
            f"\nLearning Rate: {curr_lr:.5f}"
            f"\nTime Elapsed: {train_time_elapsed}\n"
        )
        self.train_history_dict = {**self.train_epoch_dict}
        self.invoke_callbacks("on_train_epoch_end")

    def valid_one_epoch(self, valid_loader: DataLoader, epoch: int) -> None:
        """Validate the model on the validation set for one epoch.
        Args:
            valid_loader (torch.utils.data.DataLoader): The validation set dataloader.
        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """
        val_start_time = time.time()  # start time for validation

        self.model.eval()  # set to eval mode

        valid_bar = tqdm(valid_loader)

        valid_logits, valid_trues, valid_preds, valid_probs = [], [], [], []

        with torch.no_grad():
            for _step, batch in enumerate(valid_bar, start=1):
                # unpack
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                _batch_size = inputs.shape[0]

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = get_sigmoid_softmax(self.pipeline_config)(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = self.computer_criterion(
                    targets,
                    logits,
                    criterion_params=self.pipeline_config.criterion_params,
                    stage="valid",
                )
                # Update loss metric, every batch is diff
                self.valid_batch_dict["valid_loss"] = curr_batch_val_loss.item()

                # valid_bar.set_description(f"Validation. {metric_monitor}")

                self.invoke_callbacks("on_valid_batch_end")
                # For OOF score and other computation.
                # TODO: Consider giving numerical example. Consider rolling back to targets.cpu().numpy() if torch fails.
                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())
                valid_preds.extend(y_valid_pred.cpu())
                valid_probs.extend(y_valid_prob.cpu())

        valid_trues, valid_logits, valid_preds, valid_probs = (
            torch.vstack(valid_trues),
            torch.vstack(valid_logits),
            torch.vstack(valid_preds),
            torch.vstack(valid_probs),
        )
        _, valid_metrics_dict = self.get_classification_metrics(
            valid_trues, valid_preds, valid_probs
        )

        self.invoke_callbacks("on_valid_loader_end")

        # total time elapsed for this epoch
        valid_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - val_start_time)
        )
        self.logger.info(
            f"\n[RESULT]: Validation. Epoch {epoch}:"
            f"\nAvg Val Summary Loss: {self.valid_epoch_dict['valid_loss']:.3f}"
            f"\nAvg Val Accuracy: {valid_metrics_dict['val_Accuracy']:.3f}"
            f"\nAvg Val Macro AUROC: {valid_metrics_dict['val_AUROC']:.3f}"
            f"\nTime Elapsed: {valid_elapsed_time}\n"
        )
        # here self.valid_epoch_dict only has valid_loss, we update the rest
        self.valid_epoch_dict.update(
            {
                "valid_trues": valid_trues,
                "valid_logits": valid_logits,
                "valid_preds": valid_preds,
                "valid_probs": valid_probs,
            }
        )
        self.valid_epoch_dict.update(valid_metrics_dict)
        # temporary stores current valid epochs info
        # FIXME: so now valid epoch dict and valid history dict are the same lol.
        self.valid_history_dict = {**self.valid_epoch_dict, **valid_metrics_dict}

        # TODO: after valid epoch ends, for example, we need to call
        # our History callback to save the metrics into a list.
        self.invoke_callbacks("on_valid_epoch_end")

    @staticmethod
    def get_optimizer(
        model,
        optimizer_params: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        """Get the optimizer for the model.
        Note:
            Do not invoke self.model directly in this call as it may affect model initalization.
            https://stackoverflow.com/questions/70107044/can-i-define-a-method-as-an-attribute
        """
        return getattr(torch.optim, optimizer_params.optimizer_name)(
            model.parameters(), **optimizer_params.optimizer_params
        )

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_params: Dict[str, Any],
    ) -> torch.optim.lr_scheduler:
        """Get the scheduler for the optimizer."""
        return getattr(torch.optim.lr_scheduler, scheduler_params.scheduler_name)(
            optimizer=optimizer, **scheduler_params.scheduler_params
        )

    @staticmethod
    def computer_criterion(
        y_trues: torch.Tensor,
        y_logits: torch.Tensor,
        criterion_params: Dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.
        The below example applies for CrossEntropyLoss.
        Args:
            y_trues ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is
                $0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C-1$.
                If containing class probabilities, same shape as the input.
            stage (str): train or valid, sometimes people use different loss functions for
                train and valid.
        """

        if stage == "train":
            loss_fn = getattr(torch.nn, criterion_params.train_criterion_name)(
                **criterion_params.train_criterion_params
            )
        elif stage == "valid":
            loss_fn = getattr(torch.nn, criterion_params.valid_criterion_name)(
                **criterion_params.valid_criterion_params
            )
        loss = loss_fn(y_logits, y_trues)
        return loss

    @staticmethod
    def get_lr(optimizer: torch.optim) -> float:
        """Get the learning rate of optimizer for the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]
