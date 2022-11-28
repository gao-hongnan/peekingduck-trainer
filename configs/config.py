"""Configurations for the project."""
import warnings
from pathlib import Path

import torch

# Suppress User Warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Repository's Name
AUTHOR = "Hongnan G."
REPO = "peekingduck-trainer.git"

# Torch Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating Directories
BASE_DIR = Path(__file__).parent.parent.absolute()

CONFIG_DIR = Path(BASE_DIR, "configs")
DATA_DIR = Path(BASE_DIR, "data")
DOCS_DIR = Path(BASE_DIR, "docs")
SRC_DIR = Path(BASE_DIR, "src")
STORES_DIR = Path(BASE_DIR, "stores")
EXAMPLES_DIR = Path(BASE_DIR, "examples")
TESTS_DIR = Path(BASE_DIR, "tests")
SCRIPTS_DIR = Path(BASE_DIR, "scripts")

## Local stores
LOGS_DIR = Path(STORES_DIR, "logs")
BLOB_STORE = Path(STORES_DIR, "blob")
FEATURE_STORE = Path(STORES_DIR, "feature")
MODEL_ARTIFACTS = Path(STORES_DIR, "model_artifacts")
TENSORBOARD = Path(STORES_DIR, "tensorboard")
WANDB_DIR = Path(STORES_DIR, "wandb")

## Create dirs
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
TESTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)
FEATURE_STORE.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
WANDB_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD.mkdir(parents=True, exist_ok=True)
