"""Consider putting this in prepare data or something like that."""
from __future__ import annotations
import pandas as pd

from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


class SplitConfig:
    """A class to keep track of cross-validation schema.

    Note:
        Convert this class to yaml or dataclass or hydra in future.

    Attributes:
        seed (int): random seed for reproducibility.
        num_folds (int): number of folds.
        cv_schema (str): cross-validation schema.
        class_col_name (str): name of the target column.
        image_col_name (str): name of the image column.
        folds_csv (str): path to the folds csv.
    """

    cv_schema: str = "train_test_split"
    # cv_schema_params must match the schema's arguments
    cv_schema_params: dict = {
        "test_size": 0.2,
        "random_state": 1992,
        "shuffle": True,
        "stratify": True,
    }
    # sample for StratifiedKFold
    # cv_schema_params = {"n_splits": 5, "shuffle": True, "random_state": 1992}
    # sample for StratifiedGroupKFold
    # cv_schema_params = {"n_splits": 5, "shuffle": True, "random_state": 1992}
    # but when splitting the data, we need to pass the group column
    class_col_name: str = "class_id"
    image_col_name: str = "image_id"
    group_kfold_split: str = "patient_id"


# TypeVar


def make_folds(
    df: pd.DataFrame,
    pipeline_config: SplitConfig,
) -> pd.DataFrame:
    """Split the given dataframe into training folds.
    Note that sklearn now has StratifiedGroupKFold!

    Args:
        df (pd.DataFrame): The train dataframe.
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        cv_params (pipeline_config.folds): The cross validation parameters.

    Returns:
        df_folds (pd.DataFrame): The folds dataframe with an additional column "fold".
    """

    # cv_params = pipeline_config.folds
    df_folds = df.copy()

    if pipeline_config.cv_schema == "StratifiedKFold":

        skf = StratifiedKFold(**pipeline_config.cv_schema_params)

        for fold, (_train_idx, val_idx) in enumerate(
            skf.split(
                X=df_folds[pipeline_config.image_col_name],
                y=df_folds[pipeline_config.class_col_name],
            )
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)

    elif pipeline_config.cv_schema == "StratifiedGroupKFold":

        sgkf = StratifiedGroupKFold(**pipeline_config.cv_schema_params)

        groups = df_folds[pipeline_config.group_kfold_split].values

        for fold, (_train_idx, val_idx) in enumerate(
            sgkf.split(
                X=df_folds[pipeline_config.image_col_name],
                y=df_folds[pipeline_config.class_col_name],
                groups=groups,
            )
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)

    elif pipeline_config.cv_schema == "train_test_split":
        if pipeline_config.cv_schema_params["stratify"]:
            stratify = df_folds[pipeline_config.class_col_name]
            pipeline_config.cv_schema_params["stratify"] = stratify

        train_df, val_df = train_test_split(
            df_folds, **pipeline_config.cv_schema_params
        )
        df_folds.loc[train_df.index, "fold"] = "train"
        df_folds.loc[val_df.index, "fold"] = "valid"

    else:
        raise ValueError(
            f"cv_schema {pipeline_config.cv_schema} is not implemented. "
            "Please see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection. "
            "to implement your own cross-validation schema."
        )

    # df_folds["fold"] = df_folds["fold"].astype(int)
    print(df_folds.groupby(["fold", pipeline_config.class_col_name]).size())

    # df_folds.to_csv(pipeline_config.folds_csv, index=False)

    return df_folds
