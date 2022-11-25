# Enhancements

## MixUp and CutMix

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

## Label Smoothing
