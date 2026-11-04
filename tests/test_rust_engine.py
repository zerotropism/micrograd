"""The Rust engine gives the Python engine's numbers. Skipped where micrograd-rs is not built."""

import random

import pytest

from micrograd.engine import Value
from micrograd.nn import MLP

micrograd_rs = pytest.importorskip("micrograd_rs")

XS = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
YS = [1.0, -1.0, -1.0, 1.0]


def test_gradients_match_the_python_engine() -> None:
    """The expression of test_engine.py, built in both engines."""
    values = [2.0, -3.0, 0.5]
    a, b, c = (Value(v) for v in values)
    d = a * b + c
    out = (((d * d) / (a + 2.0)).tanh() + (a * c).exp() * 0.1 + d**3) * a
    out.backward()

    g = micrograd_rs.Graph()
    ra, rb, rc = (g.leaf(v) for v in values)
    rd = g.add(g.mul(ra, rb), rc)
    e = g.mul(g.mul(rd, rd), g.pow(g.add(ra, g.leaf(2.0)), -1.0))
    total = g.add(g.add(g.tanh(e), g.mul(g.exp(g.mul(ra, rc)), g.leaf(0.1))), g.pow(rd, 3.0))
    rout = g.mul(total, ra)
    g.backward(rout)

    assert g.data(rout) == pytest.approx(out.data, rel=1e-12)
    for python, rust in ((a, ra), (b, rb), (c, rc)):
        assert g.grad(rust) == pytest.approx(python.grad, rel=1e-12)


def test_training_matches_the_python_mlp_step_by_step() -> None:
    """Same starting parameters, same data: same losses and same final parameters."""
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    start = [p.data for p in model.parameters()]

    losses = []
    for _ in range(20):
        total = sum((model(x) - y) ** 2 for x, y in zip(XS, YS, strict=True))
        losses.append(total.data)
        for p in model.parameters():
            p.grad = 0.0
        total.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad

    rust_losses, rust_params = micrograd_rs.train_mlp([3, 4, 4, 1], start, XS, YS, 20, 0.05)
    assert rust_losses == pytest.approx(losses, rel=1e-9)
    assert rust_params == pytest.approx([p.data for p in model.parameters()], rel=1e-9)


def test_a_value_outside_the_graph_is_an_index_error() -> None:
    g = micrograd_rs.Graph()
    a = g.leaf(1.0)
    with pytest.raises(IndexError):
        g.add(a, 5)


def test_wrong_shapes_are_a_value_error() -> None:
    with pytest.raises(ValueError, match="parameters given"):
        micrograd_rs.train_mlp([3, 4, 1], [0.0] * 3, XS, YS, 1, 0.05)
