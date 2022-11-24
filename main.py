import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

BATCH_SIZE = 32


def create_model():
    model = torchvision.models.resnet18(pretrained=True, num_classes=1000)
    # model.conv1 = nn.Conv2d(
    #     3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    # )
    # model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        # print(x)
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def save_checkpoints(self):
        # TODO: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # steps_per_epoch = 45000 // BATCH_SIZE
        # scheduler_dict = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         0.1,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     "interval": "step",
        # }
        return {"optimizer": optimizer}
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":
    LR = 0.001
    model = LitResnet(lr=LR)

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )
    from src.dataset import CustomDataModule

    steel_dm = CustomDataModule()

    trainer.fit(model, steel_dm)
    trainer.test(model, datamodule=steel_dm)
