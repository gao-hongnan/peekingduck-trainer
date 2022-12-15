# Design Patters

## Questions

1. `AbstractPipelineConfig` is an **abstract** class that defines the **interface** for all pipeline configs. Note that this is an **abstraction** layer where both high-level and low-level modules depend on. 
2. Almost all high level modules in the whole entire `src` folder depends on `AbstractPipelineConfig`.
3. 
4. I think the better way 

## Dependency Injection

- Composition > Inheritance as the latter has high coupling.
- This is a design pattern (creational pattern) that allows us to implement loose coupling in software design?
- Class not responsible for creating its dependencies.
- Creation

If object A depends on object B, object A must not create import object B directly. Instead of this, object A must provide a way for injecting object B. The responsibility of object creation and dependency injection are delegated to external code.

Why use Dependency Injection in your code?

- Flexibility of configurable components — As the components are externally configured, there can be various definitions for a component(Control on application structure).
- Testing Made Easy — Instantiating mock objects and integrating with class definitions is easier.
- High cohesion — Code with reduced module complexity, increased module reusability.
- Minimalistic dependencies — As the dependencies are clearly defined, easier to eliminate/reduce unnecessary dependencies.

### Pipeline Config

- Inheritance: `PipelineConfig` inherits from `AbstractPipelineConfig` which 

## Model

### **Base Model Class**

#### Design Pattern

This module defines the interface for all models.
It follows the [**Strategy Pattern**](https://github.com/msaroufim/ml-design-patterns).

Users must implement the following methods:

- `create_model`: This method creates the model and returns it.
- `forward`: This method defines the forward pass of the model.

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

```python
class ImageClassificationDataModule(CustomizedDataModule):
    """Data module for generic image classification dataset."""

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config
        self.transforms = ImageClassificationTransforms(pipeline_config)
```

violates the Dependency Inversion Principle since it depends on `ImageClassificationTransforms` which is a concrete class.

## Folder Structure

if folder do by indices?
if df do by indices?


## Augmentations

### Transforms

#### Design Pattern

## References

- https://refactoring.guru/design-patterns/strategy
- https://refactoring.guru/design-patterns/strategy/python/example
- https://github.com/msaroufim/ml-design-patterns
