from __future__ import generators, print_function

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import wandb
from configs import config, global_params, mnist_params
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.callbacks.history import History
from src.callbacks.metrics_meter import MetricMeter

from src import (
    dataset,
    # inference,
    # lr_finder,
    metrics,
)
from src.trainer import Trainer
from src.dataset import ImageClassificationDataModule, MNISTDataModule
from src.model import ImageClassificationModel, MNISTModel
from src.utils import general_utils


# BASE_DIR = Path(__file__).parent.parent.absolute().__str__()
# sys.path.append(BASE_DIR)


device = config.DEVICE
logs_dir = global_params.PipelineConfig.stores.logs_dir
main_logger = config.init_logger(
    log_file=Path.joinpath(logs_dir, "main.log"),
    module_name="main",
)  # FIXME: follow trainer style logger

# shutil.copy(FILES.global_params_path, LOGS_PARAMS.LOGS_DIR_RUN_ID)


def wandb_init(fold: int, pipeline_config: global_params.PipelineConfig):
    """Initialize wandb run.
    Args:
        fold (int): [description]
        pipeline_config (global_params.PipelineConfig): The pipeline configuration.
    Returns:
        [type]: [description]
    """
    config = {
        "Train_Params": pipeline_config.global_train_params.to_dict(),
        "Model_Params": pipeline_config.model_params.to_dict(),
        "Loader_Params": pipeline_config.loader_params.to_dict(),
        "File_Params": pipeline_config.files.to_dict(),
        "Wandb_Params": pipeline_config.wandb_params.to_dict(),
        "Folds_Params": pipeline_config.folds.to_dict(),
        "Augment_Params": pipeline_config.transforms.to_dict(),
        "Criterion_Params": pipeline_config.criterion_params.to_dict(),
        "Scheduler_Params": pipeline_config.scheduler_params.to_dict(),
        "Optimizer_Params": pipeline_config.optimizer_params.to_dict(),
    }

    wandb_run = wandb.init(
        config=config,
        name=f"{pipeline_config.global_train_params.model_name}_fold_{fold}",
        **pipeline_config.wandb_params.to_dict(),
    )
    return wandb_run


def log_gradcam(
    curr_fold_best_checkpoint,
    df_oof,
    pipeline_config: global_params.PipelineConfig,
    plot_gradcam: bool = True,
):
    """Log gradcam images into wandb for error analysis.
    # TODO: Consider getting the logits for error analysis, for example, if a predicted image which is correct has high logits this means the model is very sure, conversely, if a predicted image has low logits and also wrong, we also check why.
    """

    wandb_table = wandb.Table(
        columns=[
            "image_id",
            "y_true",
            "y_pred",
            "y_prob",
            "original_image",
            "gradcam_image",
        ]
    )
    model = models.CustomNeuralNet(pretrained=False)

    # I do not need to do the following as the trainer returns a checkpoint model.
    # So we do not need to say: model = CustomNeuralNet(pretrained=False) -> state = torch.load(...)
    curr_fold_best_state = curr_fold_best_checkpoint["model_state_dict"]
    model.load_state_dict(curr_fold_best_state)
    model.to(device)
    model.eval()

    if "vit" in pipeline_config.model_params.model_name:
        # blocks[-1].norm1  # for vit models use this, note this is using TIMM backbone.
        target_layers = [model.backbone.blocks[-1].norm1]

    elif "efficientnet" in pipeline_config.model_params.model_name:
        target_layers = [model.backbone.conv_head]
        reshape_transform = None

    elif (
        "resnet" in pipeline_config.model_params.model_name
        or "resnext" in pipeline_config.model_params.model_name
    ):
        target_layers = [model.backbone.layer4[-1]]
        reshape_transform = None
    elif "swin" in pipeline_config.model_params.model_name:
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/swinT_example.py
        # TODO: Note this does not work for swin 384 as the size is not (7, 7)
        def reshape_transform(tensor, height=7, width=7):
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.permute(0, 3, 1, 2)
            return result

        target_layers = [model.backbone.layers[-1].blocks[-1].norm1]

    # load gradcam_dataset
    gradcam_dataset = dataset.CustomDataset(
        df=df_oof,
        pipeline_config=pipeline_config,
        transforms=transformation.get_gradcam_transforms(pipeline_config),
        mode="gradcam",
    )
    count = 0
    for data in gradcam_dataset:
        X, y, original_image, image_id = (
            data["X"],
            data["y"],
            data["original_image"],
            data["image_id"],
        )
        # original's shape = (224, 224, 3) with unnormalized tensors.
        # X's shape = (3, 224, 224)
        # X_unsqueeze's shape = (1, 3, 224, 224)
        X_unsqueezed = X.unsqueeze(0)
        gradcam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=device,
            reshape_transform=reshape_transform,
        )

        # # If targets is None, the highest scoring category will be used for every image in the batch.
        gradcam_output = gradcam(
            input_tensor=X_unsqueezed,
            target_category=None,
            aug_smooth=False,
            eigen_smooth=False,
        )
        original_image = original_image.cpu().detach().numpy() / 255.0
        y_true = y.cpu().detach().numpy()
        y_pred = df_oof.loc[
            df_oof[pipeline_config.folds.image_col_name] == image_id,
            "oof_preds",
        ].values[0]

        # Hardcoded
        y_prob = df_oof.loc[
            df_oof[pipeline_config.folds.image_col_name] == image_id,
            "class_1_oof",
        ].values[0]
        assert original_image.shape[-1] == 3, "Channel Last when passing into gradcam."

        gradcam_image = show_cam_on_image(
            original_image, gradcam_output[0], use_rgb=False
        )
        if plot_gradcam:
            _fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
            axes[0].imshow(original_image)
            axes[0].set_title(f"y_true={y_true:.4f}")
            axes[1].imshow(gradcam_image)
            axes[1].set_title(f"y_pred={y_pred}")
            plt.show()
            torch.cuda.empty_cache()

        # No idea why we must cast to float instead of just numpy.
        wandb_table.add_data(
            image_id,
            float(y_true),
            float(y_pred),
            float(y_prob),
            wandb.Image(original_image),
            wandb.Image(gradcam_image),
        )
        # TODO: take 10 correct predictions and 10 incorrect predictions.
        # TODO: needs modification if problem is say regression, or multilabel.
        count += 1
        if count == 20:
            break
    return wandb_table


def train_one_fold(
    df_folds: pd.DataFrame,
    fold: int,
    pipeline_config: global_params.PipelineConfig,
    is_plot: bool = False,
    is_forward_pass: bool = True,
    is_gradcam: bool = True,
    is_find_lr: bool = False,
):
    """Train the model on the given fold."""
    artifacts_dir = pipeline_config.stores.artifacts_dir
    ################################## W&B #####################################
    # wandb.login()
    wandb_run = wandb_init(fold=fold, pipeline_config=pipeline_config)

    train_loader, valid_loader, df_oof = prepare.prepare_loaders(
        df_folds, fold, pipeline_config=pipeline_config
    )

    if is_plot:
        image_grid = plot.show_image(
            loader=train_loader,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            one_channel=False,
        )

        images = wandb.Image(
            np.transpose(image_grid, (1, 2, 0)),
            caption="Top: Output, Bottom: Input",
        )
        wandb.log({"examples": images})

    # Model, cost function and optimizer instancing
    model = models.CustomNeuralNet().to(device)

    if is_forward_pass:
        # Forward Sanity Check
        # TODO: https://discuss.pytorch.org/t/why-cannot-i-call-dataloader-or-model-object-twice/137761
        # Find out why this will change model behaviour, use with caution, or maybe just put it outside this function for safety.
        _forward_X, _forward_y = models.forward_pass(loader=train_loader, model=model)
    if is_find_lr:
        lr_finder.find_lr(model, device, train_loader, valid_loader, use_valid=False)

    reighns_trainer: trainer.Trainer = trainer.Trainer(
        pipeline_config=pipeline_config,
        model=model,
        model_artifacts_path=artifacts_dir,
        device=device,
        wandb_run=wandb_run,
    )

    curr_fold_best_checkpoint = reighns_trainer.fit(train_loader, valid_loader, fold)

    # TODO: Note that for sigmoid on one class, the OOF score is the positive class.
    df_oof[
        [
            f"class_{str(c)}_oof"
            for c in range(pipeline_config.global_train_params.num_classes)
        ]
    ] = (curr_fold_best_checkpoint["oof_probs"].detach().numpy())

    df_oof["oof_trues"] = curr_fold_best_checkpoint["oof_trues"]
    df_oof["oof_preds"] = curr_fold_best_checkpoint["oof_preds"]

    df_oof.to_csv(Path(artifacts_dir, f"oof_fold_{fold}.csv"), index=False)
    if is_gradcam:
        # TODO: df_oof['error_analysis'] = todo - error analysis by ranking prediction confidence and plot gradcam for top 10 and bottom 10.
        gradcam_table = log_gradcam(
            curr_fold_best_checkpoint=curr_fold_best_checkpoint,
            df_oof=df_oof,
            pipeline_config=pipeline_config,
            plot_gradcam=False,
        )

        wandb_run.log({"gradcam_table": gradcam_table})
        general_utils.free_gpu_memory(gradcam_table)

    general_utils.free_gpu_memory(model)
    wandb_run.finish()  # Finish the run to start next fold.

    return df_oof


def train_loop(pipeline_config: global_params.PipelineConfig, *args, **kwargs):
    """Perform the training loop on all folds. Here The CV score is the average of the validation fold metric.
    While the OOF score is the aggregation of all validation folds."""

    df_oof = pd.DataFrame()

    for fold in range(1, pipeline_config.folds.num_folds + 1):
        _df_oof = train_one_fold(
            *args, fold=fold, pipeline_config=pipeline_config, **kwargs
        )
        df_oof = pd.concat([df_oof, _df_oof])

    cv_mean_d, cv_std_d = metrics.calculate_cv_metrics(df_oof)
    main_logger.info(f"\nMEAN CV: {cv_mean_d}\nSTD CV: {cv_std_d}")

    df_oof.to_csv(Path(pipeline_config.stores.artifacts_dir, "oof.csv"), index=False)

    return df_oof


def train_steel_defect(debug: bool = True):
    pipeline_config = global_params.PipelineConfig()
    model = ImageClassificationModel(pipeline_config).to(pipeline_config.device)
    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        model_artifacts_path=pipeline_config.stores.artifacts_dir,
        device=device,
        # wandb_run=wandb_run,
    )

    dm = ImageClassificationDataModule(pipeline_config)
    dm.prepare_data()

    if debug:
        dm.setup(stage="debug")
        debug_train_loader = dm.debug_train_dataloader()
        debug_valid_loader = dm.debug_valid_dataloader()
        _ = trainer.fit(debug_train_loader, debug_valid_loader, fold=None)
    else:
        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        valid_loader = dm.valid_dataloader()
        _ = trainer.fit(train_loader, valid_loader, fold=None)


def train_mnist(debug: bool = False):
    pipeline_config = mnist_params.PipelineConfig()

    dm = MNISTDataModule(pipeline_config)
    dm.prepare_data()

    model = MNISTModel(pipeline_config).to(pipeline_config.device)

    trainer = Trainer(
        pipeline_config=pipeline_config,
        model=model,
        model_artifacts_path=pipeline_config.stores.artifacts_dir,
        device=device,
        callbacks=[History(), MetricMeter()],
        # wandb_run=wandb_run,
    )

    if debug:
        pass
        # dm.setup(stage="debug")
        # debug_train_loader = dm.debug_train_dataloader()
        # debug_valid_loader = dm.debug_valid_dataloader()
        # _ = trainer.fit(debug_train_loader, debug_valid_loader, fold=None)
    else:
        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()
        valid_loader = dm.valid_dataloader()
        _ = trainer.fit(train_loader, valid_loader, fold=None)
        print(trainer.history)
        print(trainer.history["valid_probs"][0].shape)
        print(trainer.history["valid_probs"][1].shape)


if __name__ == "__main__":
    general_utils.seed_all(1992)
    # train_steel_defect(debug=True)
    train_mnist(debug=False)
    # on_trainer_start
    # on_fit_start
    # on_epoch_start
    # on_loader_start
    # on_batch_start
    # on_batch_end
    # on_loader_end # cleanup?
    # on_epoch_end
    # on_fit_end
    # decouple my trainer class to above
    # add 1 callback to see if it works.
