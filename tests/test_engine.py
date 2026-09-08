"""Smoke tests for the Value autograd engine."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from headers import Value  # noqa: E402


def test_forward() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    assert (a * b).data == -6.0
    assert (a + b).data == -1.0


def test_backward() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == -3.0
    assert b.grad == 2.0
