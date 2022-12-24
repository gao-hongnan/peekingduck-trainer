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


## Composition


// version 0.4
import java.lang.Math;

/**
 * A Circle object encapsulates a circle on a 2D plane.  
 */
class Circle {
  private double x;  // x-coordinate of the center
  private double y;  // y-coordinate of the center
  private double r;  // the length of the radius

  /**
   * Create a circle centered on (x, y) with given radius
  */
  public Circle(double x, double y, double r) {
    this.x = x;
    this.y = y;
    this.r = r;
  }

  /**
   * Return the area of the circle.
   */
  public double getArea() {
    return Math.PI * this.r * this.r;
  }

  /**
   * Return true if the given point (x, y) is within the circle.
   */
  public boolean contains(double x, double y) {
    return false; 
    // TODO: Left as an exercise
  }
}

 in python
 