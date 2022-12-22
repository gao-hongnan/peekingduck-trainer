# Design Patterns

## New Ideas

### Pipeline Pattern

Compose/Pipeline example:

```python
import functools
from typing import Callable

ComposableFunction = Callable[[float], float]

# Helper function for composing functions
def compose(*functions: ComposableFunction) -> ComposableFunction:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def addThree(x: float) -> float:
    return x + 3


def multiplyByTwo(x: float) -> float:
    return x * 2


def addN(n: float) -> ComposableFunction:
    return lambda x: x + n


def main():
    x = 12
    # oldres = multiplyByTwo(multiplyByTwo(addThree(addThree(x))))
    myfunc = compose(addN(3), addN(3), multiplyByTwo, multiplyByTwo)
    result = myfunc(x)
    print(f"Result: {result}")
```

For our use case, a bit like scikit-learn or huggingface, pseudocode:

```python
pipeline = Pipeline.Compose([
    ImageClassificationDataModule(pipeline_config),
    ImageClassificationModel(pipeline_config),
    Trainer(pipeline_config),
])
pipeline.run()
```

This may require a bit of refactoring, such as ensure that input and output
of each component is consistent (i.e. `pipeline_config` is passed in and out).
This is currently not the case, for example, `ImageClassificationDataModule`
does not return anything...see example above. Worth exploring.




- Instead of user overriding base Transforms class, they can directly specify the transform augmentations in the config file. Yier pointed out that the current implementation makes the child class ImageClassificationTransforms moot. 
    - To reconcile this, we think of why we need the base class in the first place. 
    - The base class defines an abstract interface for low and high level modules to interact on. This serves as a bridge and also decouples the high and low level module. 
    - The base class can eventually have a common __call__ method so that torchvision and albu for eg can have a consistent transform interface. 
    - The base class can also become a strategy design pattern for training where user may change strategy mid way of training. It is common to use heavy transforms early and light transforms later in a NN. Users can have for eg Transform1-3 epoch and transform 4-6 epoch with mixup. 
    - With some reasoning and basis for the existence of this base class, how then do we reconcile the fact that it can technically be done in the config class via conditionals as well. 
- 


## Config Management

### Frameworks

Hydra and Pydantic are two frameworks that can be used to manage configs.
They are both heavily used in the ML community, with the former being
developed by Facebook.

- [Hydra](https://hydra.cc/docs/intro/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)

In the later stages of the project, we should consider using one of these
frameworks to manage our configs. Currently I am using a simple config system
with `dataclass` for each individual config class. Things soon get difficult
when you want to validate configs, or interpolate variables across configs.

### Design

> Need draw UML diagram for this.

- `AbstractPipelineConfig` is an **abstract** class that defines the **interface** for all pipeline configs. Note that this is an **abstraction** layer where both high-level and low-level modules depend on. 
- `PipelineConfig` is the main config class,
  - is **inherited** from `AbstractPipelineConfig`, which defines the **interface** for all pipeline configs.
  - is a **composition**[^1] of all other configs, such as `ModelConfig`, `DataConfig`, `OptimizerConfig`, etc,
  - is **injected** into all high level modules, such as `Trainer`, `Model`, `Dataset`, etc.
  - is a **service locator** for all other configs, such as `ModelConfig`, `DataConfig`, `OptimizerConfig`, etc.
  
- Overall, we are using the **Dependency Injection** / **Service Locator** pattern to **inject** `pipeline_config` into all high level modules, and for each high level module,
we "locate" the configs from the `pipeline_config` object.

- Each individual config class must have validation schema, to avoid invalid config being passed into the pipeline. 
  - What I propose: `Validator` to validate input **types** and **values**. Maybe something like Yier's `Mixin` class.
  - What I have: Bugged `Validator` and decorator `enforce_types`.

- Each individual config class must have interpolation schema, to avoid repeating the same config across multiple configs. 
  - What I propose: to follow hydra/pydantic's interpolation schema, which is to use `${}` to interpolate variables. 

- Other remarks:
  - All of my params file has a class called `PipelineConfig`, which inherits
from `AbstractPipelineConfig`. This is because I initially treated `PipelineConfig` as
the "low level concrete implementation", and other classes such as `Trainer` and `Model`
as "high level business logic", both of which depends on the abstract layer `AbstractPipelineConfig`. We then inject different variations of implementations of
`AbstractPipelineConfig` into the high level business logic. 


[^1]: See [here](https://suneeta-mall.github.io/2022/03/15/hydra-pydantic-config-management-for-training-application.html#an-extension-using-structured-config)
for example.

[^2]: If we only look at the main application in `main.py`, say `train_generic`, then `ImageClassificationModel` is now a low level module because it is a concrete implementation of the interface `Model`. 

### Questions and Problems

- Questions:
  - Interface/Abstraction:
    - `TrainConfig`.
    - `AbstractPipelineConfig` but it composes of all other smaller configs.
    - The interface provides a common interface for all high level modules and low level modules to interact on. This encourages **decoupling** of high and low level modules.
    - The interface also provides a contract for all concrete implementations of the interface. This encourages **consistency** of high and low level modules.
  
  - Low level modules:
    - `cifar10_params.GlobalTrainParams` is a concrete implementation of `TrainConfig` so it is a low level module(?)
    - `cifar10_params.PipelineConfig` is a concrete implementation of `AbstractPipelineConfig` so it is a low level module(?)

  - High level modules: 
    - `ImageClassificationModel` is a high level module in this context[^2] since it incorporates important business logic(?) 

- Problems:
    - Problem 1: Should my individual params class be child class of a parent such as
as `Params`? Or should each individual params class has its own parent class? 
    - `OptimizerParams(Params)` 
    - `OptimizerParams(AbstractParams)`
    - `OptimizerParams(AbstractOptimizerParams)`: The benefit of this we can validate
    the params in the parent class for each individual params class. The con is that
    we need to create a new parent class for each individual params class.

- Problem 2: Overlapping of params, `monitored_metric` must coincide with `EarlyStopping`
and `ModelCheckpoint` params. This can be solved by **interpolation**.
  
- Problem 3: Can consider moving `config.py`'s constant to a parent class such as 
`AbstractParams` as part of the config management. Maybe as `property`?

## Interface and Abstract Class

Quoted from [stackoverflow](https://softwareengineering.stackexchange.com/questions/371722/criticism-and-disadvantages-of-dependency-injection): 

Interfaces are a contract. They exist to limit how tightly coupled two objects can be. Not every dependency needs an interface, but they help with writing modular code.

When you add in the concept of unit testing, you may have two conceptual implementations for any given interface: the real object you want to use in your application, and the mocked or stubbed object you use for testing code that depends on the object. That alone can be justification enough for the interface. 

## Dependency Injection

Quoted from [stackoverflow](https://softwareengineering.stackexchange.com/questions/371722/criticism-and-disadvantages-of-dependency-injection): 

Dependency injection at its simplest and most fundamental level is simply:

> A parent object provides all the dependencies required to the child object.

The term parent and child, in the context of dependency injection:

> The parent is the object that instantiates and configures the child object it uses.
The child is the component that is designed to be passively instantiated. i.e. it is designed to use whatever dependencies are provided by the parent, and does not instantiate it's own dependencies. For our example, the parent is the `PipelineConfig` and the child is the `ImageClassificationModel`.

### Pros

- Making isolation in unit testing possible/easy.
- Explicitly defining dependencies of a class
- Enabling switching implementations quickly (`CifarConfig` vs `MNISTConfig`).
- A creational pattern that allows us to implement loose coupling in software design where
high level modules are not responsible for creating low level modules (dependencies). For example, `Trainer` is not responsible for creating `ImageClassificationModel` but instead it is injected into `Trainer`.
- Flexibility of configurable components — As the components are externally configured, there can be various definitions for a component(Control on application structure).
- Testing Made Easy — Instantiating mock objects and integrating with class definitions is easier.
- High cohesion — Code with reduced module complexity, increased module reusability.
- Minimalistic dependencies — As the dependencies are clearly defined, easier to eliminate/reduce unnecessary dependencies.

## Service Locator

Service Locator is a software design pattern that also allows us to develop loosely coupled code. It implements the DIP principle and easier to use with an existing codebase as it makes the overall design looser without forcing changes to the public interface.

The Service Locator pattern introduces a locator object that objects are used to resolve dependencies means it allows you to "resolve" a dependency within a class.

Our config management design closely resembles the Service Locator pattern when compared to the Dependency Injection. Both of which use the DIP principle, so they are quite similar.

The reason it is more similar to the Service Locator pattern is because for each high level module that takes in `pipeline_config`, we need to further
resolve the dependencies within the `pipeline_config` object (i.e. to get the `model_config`, `data_config`, `optimizer_config`, etc.).

When you use a service locator, every class will have a dependency on your service locator. This is not the case with dependency injection. The dependency injector will typically be called only once at startup, to inject dependencies into the main class.


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
