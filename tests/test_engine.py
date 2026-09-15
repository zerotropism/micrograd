"""The engine is checked against numerical gradients, not just against itself."""

import pytest

from micrograd.engine import Value

H = 1e-6
TOLERANCE = 1e-4


def expression(a, b, c):
    """Exercises add, mul, pow, div, tanh, exp and a node reached by two paths.

    The reused node is what makes the `+=` in every `_backward` necessary: a value that
    feeds two branches accumulates a gradient from each of them.
    """
    d = a * b + c
    e = (d * d) / (a + 2.0)
    return (e.tanh() + (a * c).exp() * 0.1 + d**3) * a


def evaluate(values: list[float]) -> float:
    return expression(*(Value(v) for v in values)).data


def numerical_gradient(values: list[float], index: int) -> float:
    """Central difference: second-order accurate, unlike the forward difference."""
    up, down = list(values), list(values)
    up[index] += H
    down[index] -= H
    return (evaluate(up) - evaluate(down)) / (2 * H)


def test_forward_pass() -> None:
    a, b = Value(2.0), Value(-3.0)
    assert (a * b).data == -6.0
    assert (a + b).data == -1.0
    assert (a - b).data == 5.0
    assert (a / b).data == pytest.approx(-2 / 3)
    assert (a**3).data == 8.0


@pytest.mark.parametrize("index,name", [(0, "a"), (1, "b"), (2, "c")])
def test_analytic_gradient_matches_the_numerical_one(index: int, name: str) -> None:
    """The property that makes an autodiff engine correct."""
    values = [2.0, -3.0, 0.5]
    inputs = [Value(v) for v in values]
    expression(*inputs).backward()

    numeric = numerical_gradient(values, index)
    assert inputs[index].grad == pytest.approx(numeric, rel=TOLERANCE, abs=TOLERANCE)


def test_a_reused_value_accumulates_gradients_from_every_path() -> None:
    """With `=` instead of `+=` in `_backward`, this returns 1.0 instead of 2.0."""
    a = Value(3.0)
    out = a + a
    out.backward()
    assert a.grad == 2.0


def test_the_output_gradient_starts_at_one() -> None:
    a = Value(2.0)
    a.backward()
    assert a.grad == 1.0


def test_topological_order_visits_children_before_parents() -> None:
    """A wrong order silently produces zero gradients deeper in the graph."""
    a = Value(2.0)
    b = a * 3.0
    c = b * 4.0
    c.backward()
    assert a.grad == 12.0
