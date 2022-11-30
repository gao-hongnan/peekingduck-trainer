import collections
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from src.model import Model
from src.trainer import get_sigmoid_softmax
from configs.base_params import PipelineConfig


@torch.inference_mode(mode=True)
def inference_all_folds(
    model: Model,
    state_dicts: List[collections.OrderedDict],
    test_loader: DataLoader,
    pipeline_config: PipelineConfig,
) -> np.ndarray:
    """Inference the model on all K folds.
    Args:
        model (Model): The model to be used for inference. Note that pretrained should be set to False.
        state_dicts (List[collections.OrderedDict]): The state dicts of the models. Generally, K Fold means K state dicts.
        test_loader (DataLoader): The dataloader for the test set.

    Returns:
        mean_preds (np.ndarray): The mean of the predictions of all folds.
    """
    device = pipeline_config.device
    model.to(device)
    model.eval()

    all_folds_probs = []
    for _fold_num, state in enumerate(state_dicts):
        model.load_state_dict(state)

        current_fold_probs = []

        for batch in tqdm(test_loader, position=0, leave=True):
            images = batch.to(device, non_blocking=True)
            test_logits = model(images)
            test_probs = get_sigmoid_softmax(pipeline_config)(test_logits).cpu().numpy()
            current_fold_probs.append(test_probs)
        current_fold_probs = np.concatenate(current_fold_probs, axis=0)
        all_folds_probs.append(current_fold_probs)
    mean_preds = np.mean(all_folds_probs, axis=0)
    return mean_preds


def inference(
    df_test: pd.DataFrame,
    model_dir: Union[str, Path],
    model: Model,
    transform_dict: Dict[str, albumentations.Compose],
    pipeline_config: PipelineConfig,
    df_sub: Optional[pd.DataFrame] = None,
    path_to_save: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """Inference the model and perform TTA, if any.
    Dataset and Dataloader are constructed within this function because of TTA.
    model and transform_dict are passed as arguments to enable inferencing multiple different models.
    Args:
        df_test (pd.DataFrame): The test dataframe.
        model_dir (str, Path): model directory for the model.
        model (Union[models.CustomNeuralNet, Any]): The model to be used for inference. Note that pretrained should be set to False.
        transform_dict (Dict[str, albumentations.Compose]): The dictionary of transforms to be used for inference. Should call from get_inference_transforms().
        df_sub (pd.DataFrame, optional): The submission dataframe. Defaults to None.
    Returns:
        all_preds (Dict[str, np.ndarray]): {"normal": normal_preds, "tta": tta_preds}
    """

    if df_sub is None:
        config.logger.info(
            "No submission dataframe detected, setting df_sub to be df_test."
        )
        df_sub = df_test.copy()

    # a dict to keep track of all predictions [no_tta, tta1, tta2, tta3]
    all_preds = {}
    model = model.to(device)

    # Take note I always save my torch models as .pt files. Note we must return paths as str as torch.load does not support pathlib.
    weights = utils.return_list_of_files(
        directory=model_dir, return_string=True, extension=".pt"
    )

    state_dicts = [torch.load(path)["model_state_dict"] for path in weights]

    # Loop over each TTA transforms, if TTA is none, then loop once over normal inference_augs.
    for aug_name, aug_param in transform_dict.items():
        test_dataset = dataset.CustomDataset(
            df=df_test,
            pipeline_config=pipeline_config,
            transforms=aug_param,
            mode="test",
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **pipeline_config.loader_params.test_loader
        )
        predictions = inference_all_folds(
            model=model,
            state_dicts=state_dicts,
            test_loader=test_loader,
            pipeline_config=pipeline_config,
        )
        print(predictions)
        all_preds[aug_name] = predictions

        ################# To change when necessary depending on the metrics needed for submission #################
        # TODO: Consider returning a list of predictions ranging from np.argmax to preds, probs etc, and this way we can use whichever from the output? See my petfinder for more.
        df_sub[pipeline_config.folds.class_col_name] = predictions[:, 1]

        df_sub[
            [
                pipeline_config.folds.image_col_name,
                pipeline_config.folds.class_col_name,
            ]
        ].to_csv(
            Path(path_to_save, f"submission_{aug_name}.csv"),
            index=False,
        )

        print(df_sub.head())

        plt.figure(figsize=(12, 6))
        plt.hist(df_sub[pipeline_config.folds.class_col_name], bins=100)

    # for each value in the dictionary all_preds, we need to take the mean of all the values and assign it to a df and save it.
    df_sub[pipeline_config.folds.class_col_name] = np.mean(
        list(all_preds.values()), axis=0
    )[:, 1]
    df_sub[
        [
            pipeline_config.folds.image_col_name,
            pipeline_config.folds.class_col_name,
        ]
    ].to_csv(Path(path_to_save, "submission_mean.csv"), index=False)

    return all_preds
