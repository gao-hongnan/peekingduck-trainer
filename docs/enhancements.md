## To Do

```
@dataclass(frozen=False, init=True)
class GlobalTrainParams:
    """Train params, a lot of overlapping.
    FIXME: overlapping with other params.
    """

    epochs: int = 10  # 10 when not debug
    use_amp: bool = True
    patience: int = 3
    classification_type: str = "multiclass"
    monitored_metric: Dict[str, Any] = field(
        default_factory=lambda: {
            "monitor": "val_Accuracy",
            "mode": "max",
        }
    )


@dataclass(frozen=False, init=True)
class CallbackParams:
    """Callback params."""

    callbacks: List[Callback] = field(
        default_factory=lambda: [
            History(),
            MetricMeter(),
            ModelCheckpoint(mode="max", monitor="val_Accuracy"),
            EarlyStopping(mode="max", monitor="val_Accuracy", patience=3),
        ]
    )
```

The problem is that we have a lot of overlapping params. We need to clean this up.
For example patience is used in both `GlobalTrainParams` and `CallbackParams`.
Furthermore, `monitored_metric` is used in both `GlobalTrainParams` and `CallbackParams`
in a different way. We need them to be consistent.

- LR finder
- LR Scheduler
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    - messy implementation, we need to clean it up in cross validation. 
    - SUPER NOT ELEGANT.
    - see https://github.com/Lightning-AI/lightning/blob/master/examples/pl_loops/kfold.py

## TODOs Discussion with Team

- Replace all `print` with `logging`.
- How to type hint say `Model` class and its children? Should I use `Generic` somewhere?
- Currently, `inference.py` is decoupled from `Trainer` class. I saw PyTorch Lightning
    uses `trainer.test()` to do inference. Should we do the same?
- Recall BCEWithLogitsLoss expects a target.float() and not target.long() so need change
accordingly in `Dataset` class.
- `CustomizedDataModule` This can extended to segmentation and object detection.
- Consider removing `debug_dataloader(s)`.
- Literal overloading? Model Class.
- Segregate raw/processed data if not unclean.
- Pass in project name as iportlib. config. dynamic loading + argparse, click. dec loader

Line 328, if else flag to indicate whether use data dir or csv.
- `from_df` etc should be moved to `CustomizedDataModule` class.
- need to enforce file structure.
  
## Enhancements

### MixUp and CutMix

Currently can implement in `Trainer` class:

```python
# Iterate over train batches
for _step, batch in enumerate(train_bar, start=1):
    if self.train_params.mixup:
        # TODO: Implement MIXUP logic.
        #  Refer here: https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix
        #  and my https://colab.research.google.com/drive/1sYkKG8O17QFplGMGXTLwIrGKjrgxpRt5#scrollTo=5y4PfmGZubYp
        #       MIXUP logic can be found in petfinder.
        pass
```

But unsure if this is the best way to implement it.

### Label Smoothing
