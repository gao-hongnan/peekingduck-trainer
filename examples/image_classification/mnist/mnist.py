class MNISTDataModule(CustomizedDataModule):
    """DataModule for MNIST dataset."""

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.transforms = ImageClassificationTransforms(pipeline_config)

    def prepare_data(self) -> None:
        # download data here
        self.train_transforms = self.transforms.train_transforms
        self.valid_transforms = self.transforms.valid_transforms

        self.path = self.pipeline_config.data.root_dir
        self.download = self.pipeline_config.data.download

    def setup(self, stage: str) -> None:
        """Assign train/val datasets for use in dataloaders."""

        if stage == "fit":
            self.train_dataset = MNIST(
                download=self.download,
                root=self.path,
                transform=self.train_transforms,
                train=True,
            )
            self.valid_dataset = MNIST(
                download=self.download,
                root=self.path,
                transform=self.valid_transforms,
                train=False,
            )
        if self.pipeline_config.datamodule.debug:
            self.train_dataset = Subset(
                self.train_dataset,
                indices=range(self.pipeline_config.datamodule.num_debug_samples),
            )
            self.valid_dataset = Subset(
                self.valid_dataset,
                indices=range(self.pipeline_config.datamodule.num_debug_samples),
            )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            **self.pipeline_config.datamodule.train_loader,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset, **self.pipeline_config.datamodule.valid_loader
        )


def train_mnist(pipeline_config: PipelineConfig) -> None:
    """Train MNIST."""
    num_classes = pipeline_config.model.num_classes
    dm = MNISTDataModule(pipeline_config)
    dm.prepare_data()

    model = MNISTModel(pipeline_config).to(pipeline_config.device)
    metrics_collection = MetricCollection(
        [
            Accuracy(num_classes=num_classes),
            Precision(num_classes=num_classes),
            Recall(num_classes=num_classes),
            AUROC(num_classes=num_classes, average="macro"),
            MulticlassCalibrationError(
                num_classes=num_classes
            ),  # similar to brier loss
        ]
    )

    callbacks = pipeline_config.callback_params.callbacks

    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        metrics=metrics_collection,
        callbacks=callbacks,
    )

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.valid_dataloader()
    history = trainer.fit(train_loader, valid_loader, fold=None)
    # history = trainer.history
    print(history.keys())
    print(history["valid_loss"])
    print(history["val_Accuracy"])
    print(history["val_AUROC"])
