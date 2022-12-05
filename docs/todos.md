# TODOs

- model_checkpoint
- early_stopping
- LR finder
- LR Scheduler
- We need to add mixup/cutmix support, these SOTA techniques are common now.

# TODOs Discussion with Team

- Replace all `print` with `logging`.
- How to type hint say `Model` class and its children? Should I use `Generic` somewhere?
- Currently, `inference.py` is decoupled from `Trainer` class. I saw PyTorch Lightning
    uses `trainer.test()` to do inference. Should we do the same?
- Recall BCEWithLogitsLoss expects a target.float() and not target.long() so need change
accordingly in `Dataset` class.
- `CustomizedDataModule` This can extended to segmentation and object detection.
- Consider removing `debug_dataloader(s)`.
- Literal overloading? Model Class.
- Pass in project name as iportlib. config. dynamic loading + argparse, click. dec loader

Line 328, if else flag to indicate whether use data dir or csv.
- `from_df` etc should be moved to `CustomizedDataModule` class.
- need to enforce file structure.