from typing import Any, Callable
from src.dii_low_level_implementations import (
    ImageClassificationTransforms,
    ImageSegmentationTransforms,
)
from src.dii_high_level_business_logic import CustomDataset


def run_dataset() -> None:
    """Run ML pipeline."""
    dataset = CustomDataset(transforms=ImageClassificationTransforms(), stage="train")
    dataset.getitem(dummy_data=None)

    # you can change transforms from ImageClassification to ImageSegmentation
    dataset = CustomDataset(transforms=ImageSegmentationTransforms(), stage="train")
    dataset.getitem(dummy_data=None)


if __name__ == "__main__":
    run_dataset()
