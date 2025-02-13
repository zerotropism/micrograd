import math
import numpy as np
import matplotlib.pyplot as plt
from graph import draw_dot


class Value:

    # takes single value that it wraps and keeps track of
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        # function to compute chain ruled gradients
        self._backward = lambda: None
        # set for optimization in showing the children of the value for operations
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # prints
    def __repr__(self):
        return f"Value(data={self.data})"

    # adds
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        # compute the gradients in the context of an addition operation
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward

        return out

    # multiplies
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        # compute the gradients in the context of a multiplication operation
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward

        return out

    # computes tanh of a value
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad = (1.0 - t**2) * out.grad

        out._backward = _backward

        return out


# inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# bias
b = Value(6.8813735870195432, label="b")
# values
x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"
# output
o = n.tanh()
o.label = "o"
# initialize output grad value as the first grad
o.grad = 1.0
o._backward()
draw_dot(o).render("graph", format="svg", cleanup=True, view=True)
