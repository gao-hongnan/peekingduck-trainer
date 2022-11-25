"""
def log_metrics(self, epoch: int, history: Dict[str, Union[float, np.ndarray]]):
    """Log a scalar value to both MLflow and TensorBoard
    Args:
        history (Dict[str, Union[float, np.ndarray]]): A dictionary of metrics to log.
    """
    for metric_name, metric_values in history.items():
        self.wandb_run.log({metric_name: metric_values[epoch - 1]}, step=epoch)

def on_fit_start(self, fold: int) -> None:
    """Called AFTER fit begins."""
    # To automatically log gradients
    if self.wandb_run is not None:
        self.wandb_run.watch(self.model, log_freq=100)

for _epoch in range(1, self.params.epochs + 1):
    self.train_one_epoch(train_loader, _epoch)
    self.valid_one_epoch(valid_loader, _epoch)

    if self.wandb_run is not None:
        self.log_metrics(_epoch, self.history)
"""
