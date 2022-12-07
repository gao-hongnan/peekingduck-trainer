# Design Patters

## Model

### **Base Model Class**

#### Design Pattern

This module defines the interface for all models.
It follows the Strategy Pattern: https://github.com/msaroufim/ml-design-patterns.

## Dataset

### **CustomizedDataModule**

#### Design Pattern

- This interface follows the [**Strategy Pattern**](https://github.com/msaroufim/ml-design-patterns).
- Liskov Substitution Principle
    - For example, in the normal method `prepare_data`, we have the argument 
        `fold: Optional[int] = None`. However, not everytime we need to use this,
        but we still need to have this argument in child class as it follows
        the Liskov Substitution Principle.



- An abstract class **interface** that takes in 
`pipeline_config` and setup the dataloaders for training, validation and testing.
- References: 
    - https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    - https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/hooks.py


## Folder Structure

if folder do by indices?
if df do by indices?