# Software Principles

Professor uses version of circle to walk us through

## Abstraction Barrier

- Consider the function/class called `trainer`, which is a function of `model` and `data`. 
- The abstraction barrier separates the role of the programmer into two parts: 
  - an ***implementer***, who implements the `trainer` function, and
  - a ***client***, who uses the `trainer` function in any capacity, i.e. an user can go to my
  website and upload a data, select a model, and click a button to train the model.
- The concept of abstraction is to separate the implementer from the client. 
- Below the barrier is the implementer's domain, and above the barrier is the client's domain.

## The Four Pillars of OOP

### Encapsulation

- The concept of keeping all the data and functions operating on the data related to a composite data type together within an abstraction barrier is called encapsulation.
- A class is an example of an encapsulation.

```python
class Circle:
    """A Circle object encapsulates a circle on a 2D plane.

    NOTE: Circle class with cartesian coordinates.
    """

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def get_area(self):
        """Return the area of the circle."""
        return math.pi * self.r * self.r

    def contains(self, x, y):
        """Return true if the given point (x, y) is within the circle."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2) <= self.r

c = Circle(0, 0, 1)
print(c.get_area())
print(c.contains(0, 0))
print(c.contains(1, 1))
```

### Composition
 
### Inheritance

The notes used circle as examples, come up with my own example on composition vs inheritance
and when to use which.

Ask on when he say use point to compose, does this mean the `Point` class is just cartesian? If it
entails polar, then isn't `get_area` and `contains` methods in `Circle` class not applicable using
the original methods?

Important message:

The message here is this: Use composition to model a has-a relationship; inheritance for a is-a relationship. Make sure inheritance preserves the meaning of subtyping.

#### Overriding

##### String Representation

This is basically python's `__str__` or `__repr__` method.

This is a classic example of overriding. Inheritance is not only good for extending the behavior of an existing class but through method overriding, we can alter the behavior of an existing class as well.

### Polymorphism

Method overriding enables polymorphism, the fourth and the last pillar of OOP, and arguably the most powerful one. It allows us to change how existing code behaves, without changing a single line of the existing code (or even having access to the code).

```python
def say(obj: callable) -> None:
    print("Hi, I am " + str(obj))
```

Notice that the `say` function takes in a callable object, and calls the `__str__` method on it. This is a perfect example of polymorphism. The `say` function does not care what the object is, as long as it is callable. It can be a function, a class, or even a module. The `say` function does not care. It just calls the `__str__` method on the object, and prints the result.


## SOLID Principles

### Liskov Substitution Principle

Let $\phi(x)$ be a property provable about objects $x$ of type $T$. 
Then $\phi(y)$ should be true for objects $y$ of type $S$ where $S$ is a subtype of $T$ 
(i.e. $S <: T$).

In other words, that if something is true for a parent class, it should also be true for a child class.

See [here](https://stackoverflow.com/questions/55477952/need-clarity-in-understanding-liskov-substitution-principle) and [here](https://betterprogramming.pub/the-liskov-substitution-principle-lsp-explained-in-python-6ab92b29d0b8).

## Abstraction Class

CS2030S notes [here](https://nus-cs2030s.github.io/2021-s2/14-abstract.html)
gives a good example on why abstraction class is useful.

## Concrete Class (Implementation Class)

This is the class that implements the abstract class.

## Interface

Pure vs Impure interface. A bit like Strategy pattern vs Template pattern.

## Difference between Interface and Abstract Class?

[This post](https://stackoverflow.com/questions/1913098/what-is-the-difference-between-an-interface-and-abstract-class)
highlights the difference between interface and abstract class.

```python
from abc import ABC, abstractmethod

# Interface is a contract, you have to follow it, and consists of only abstract methods.
class MotorVehicle(ABC):
    """Interface for all motor vehicles.
    I say all motor vehicles should look like this.
    """

    @abstractmethod
    def run(self) -> None:
        """Run the vehicle."""
        ...

    @abstractmethod
    def get_fuel(self) -> int:
        """Return the amount of fuel left in the tank."""
        ...


class Car(MotorVehicle):
    """My team mate complies and writes vehicle looking that way."""

    def __init__(self, fuel: int) -> None:
        self.fuel: int = fuel

    def run(self) -> None:
        print("Wrroooooooom, let's roll baby!")

    def get_fuel(self) -> int:
        return self.fuel

# Abstract class is a contract, you have to follow it, but can have default implementation.
class MotorVehicle(ABC):
    def __init__(self, fuel: int) -> None:

        self.fuel: int = fuel

    def get_fuel(self) -> int:
        """Return the amount of fuel left in the tank."""
        # NOTE: They all have fuel, so lets implement this for everybody.
        return self.fuel

    def run(self) -> None:
        """Run the vehicle."""
        # NOTE: That can be very different, force them to provide their own implementation.
        raise NotImplementedError


class Car(MotorVehicle):
    """My teammate complies and writes vehicle looking that way."""

    super().__init__()

    def run(self) -> None:
        print("Wrroooooooom, let's roll baby!")
```


## SOLID

