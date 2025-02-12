import math
import numpy as np
import matplotlib.pyplot as plt
from graph import draw_dot


class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
print(a + b)
print(a.__add__(b))
print(a * b + c)
print(a.__mul__(b).__add__(c))
e = a * b
e.label = "e"
print(e._prev)
print(e._op)
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = d * f
L.label = "L"
print(L)
# draw_dot(L)
draw_dot(L).render("graph", format="svg", cleanup=True, view=True)
