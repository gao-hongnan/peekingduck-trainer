<div align="center">
<h1>PyTorch Training Pipeline</a></h1>
by Hongnan Gao
Oct, 2022
<br>
</div>

## Introduction

This repository contains a PyTorch training pipeline for computer vision tasks.

## Workflow

### Installation

```bash
~/gaohn $ git clone https://github.com/gao-hongnan/peekingduck-trainer.git
~/gaohn $ cd peekingduck-trainer
~/gaohn/peekingduck-trainer        $ python -m venv <venv_name> && <venv_name>\Scripts\activate 
~/gaohn/peekingduck-trainer (venv) $ python -m pip install --upgrade pip setuptools wheel
~/gaohn/peekingduck-trainer (venv) $ pip3 install torch torchvision torchaudio \    
                                    --extra-index-url https://download.pytorch.org/whl/cu113 
~/gaohn/peekingduck-trainer (venv) $ pip install -r requirements.txt
```

If you are using Conda, then you can do

```bash
~/gaohn/peekingduck-trainer (venv) $ conda install pytorch torchvision torchaudio \
                                     pytorch-cuda=11.6 -c pytorch -c nvidia
```

If your computer does not have GPU, then you can install the CPU version of PyTorch.

```bash
~/gaohn/peekingduck-trainer (venv) $ pip3 install torch torchvision torchaudio # 1.12.1
```

### Create Folder Structure

```bash
~/gaohn/peekingduck-trainer (venv) $ python configs/config.py
```

### Run Training

```bash
~/gaohn/peekingduck-trainer (venv) $ python main.py
```