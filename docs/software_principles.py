import math
from dataclasses import dataclass


# class Circle:
#     """A Circle object encapsulates a circle on a 2D plane.

#     NOTE: Circle class with cartesian coordinates.
#     """

#     def __init__(self, x, y, r):
#         self.x = x
#         self.y = y
#         self.r = r

#     def get_area(self):
#         """Return the area of the circle."""
#         return math.pi * self.r * self.r

#     def contains(self, x, y):
#         """Return true if the given point (x, y) is within the circle."""
#         return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2) <= self.r

# c = Circle(0, 0, 1)
# print(c.get_area())
# print(c.contains(0, 0))
# print(c.contains(1, 1))

@dataclass
class Point:
    """A Point object encapsulates a point on a 2D plane."""

    x: float
    y: float


class Circle:
    """A Circle object encapsulates a circle on a 2D plane.

    NOTE: Circle class with Point object.
    """

    def __init__(self, c: Point, r: float) -> None:
        self.center: Point = c
        self.r: float = r

    def get_area(self):
        """Return the area of the circle."""
        return math.pi * self.r * self.r

    def contains(self, x, y):
        """Return true if the given point (x, y) is within the circle."""
        return math.sqrt((self.center.x - x) ** 2 + (self.center.y - y) ** 2) <= self.r


if __name__ == "__main__":
    c = Circle(0, 0, 1)
    print(c.get_area())
    print(c.contains(0, 0))
    print(c.contains(1, 1))
