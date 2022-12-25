# Software Principles

## Abstraction Barrier

- Consider the function/class called `trainer`, which is a function of `model` and `data`. 
- The abstraction barrier separates the role of the programmer into two parts: 
  - an ***implementer***, who implements the `trainer` function, and
  - a ***client***, who uses the `trainer` function in any capacity, i.e. an user can go to my
  website and upload a data, select a model, and click a button to train the model.
- The concept of abstraction is to separate the implementer from the client. 
- Below the barrier is the implementer's domain, and above the barrier is the client's domain.

## Encapsulation

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

## Composition
 
