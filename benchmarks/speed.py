"""Train the notebook's perceptron three ways and time each: the Python engine, the Rust engine
driven from Python one operation at a time, and the Rust engine end to end.

The three start from the same parameters and must end on the same loss; the script stops if
they do not. Each is timed several times and the best run is kept. The result is printed and
written to benchmarks/speed-<stamp>.md.

    uv run python benchmarks/speed.py [--steps 1000] [--repeat 5]
"""

import argparse
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import micrograd_rs

from micrograd.nn import MLP

SIZES = [3, 4, 4, 1]
XS = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
YS = [1.0, -1.0, -1.0, 1.0]
LEARNING_RATE = 0.05


def start() -> list[float]:
    random.seed(0)
    return [p.data for p in MLP(3, SIZES[1:]).parameters()]


def python_engine(params: list[float], steps: int) -> float:
    random.seed(0)
    model = MLP(3, SIZES[1:])
    for p, value in zip(model.parameters(), params, strict=True):
        p.data = value
    for _ in range(steps):
        total = sum((model(x) - y) ** 2 for x, y in zip(XS, YS, strict=True))
        for p in model.parameters():
            p.grad = 0.0
        total.backward()
        for p in model.parameters():
            p.data -= LEARNING_RATE * p.grad
    return total.data


def rust_by_operation(params: list[float], steps: int) -> float:
    """The Rust graph, built from Python: every add, mul and tanh is one call across."""
    g = micrograd_rs.Graph()
    ids = [g.leaf(p) for p in params]
    for _ in range(steps):
        g.truncate(len(ids))
        total = g.leaf(0.0)
        for x, y in zip(XS, YS, strict=True):
            inputs, nxt = [g.leaf(v) for v in x], 0
            for nin, nout in zip(SIZES, SIZES[1:], strict=False):
                outputs = []
                for _ in range(nout):
                    act = ids[nxt + nin]
                    for w, xi in zip(ids[nxt : nxt + nin], inputs, strict=True):
                        act = g.add(act, g.mul(w, xi))
                    outputs.append(g.tanh(act))
                    nxt += nin + 1
                inputs = outputs
            total = g.add(total, g.pow(g.add(inputs[0], g.leaf(-y)), 2.0))
        loss = g.data(total)
        g.backward(total)
        for p in ids:
            g.set_data(p, g.data(p) - LEARNING_RATE * g.grad(p))
    return loss


def rust_end_to_end(params: list[float], steps: int) -> float:
    losses, _ = micrograd_rs.train_mlp(SIZES, params, XS, YS, steps, LEARNING_RATE)
    return losses[-1]


IMPLEMENTATIONS = {
    "Python engine": python_engine,
    "Rust engine, one call per operation": rust_by_operation,
    "Rust engine, one call per training": rust_end_to_end,
}


def best_of(run, params: list[float], steps: int, repeat: int) -> tuple[float, float]:
    timings, loss = [], 0.0
    for _ in range(repeat):
        started = time.perf_counter()
        loss = run(params, steps)
        timings.append(time.perf_counter() - started)
    return min(timings), loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    params = start()
    results = {
        name: best_of(run, params, args.steps, args.repeat) for name, run in IMPLEMENTATIONS.items()
    }
    losses = [loss for _, loss in results.values()]
    if max(losses) - min(losses) > 1e-9 * max(1.0, abs(losses[0])):
        raise SystemExit(f"The implementations disagree on the final loss: {losses}")

    baseline = results["Python engine"][0]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "# Speed of the three engines",
        "",
        f"- measured: {stamp}",
        f"- machine: {platform.machine()}, {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}",
        f"- perceptron {'-'.join(map(str, SIZES))}, {len(XS)} examples, {args.steps} steps, "
        f"best of {args.repeat} runs",
        f"- final loss, the same for all three: {losses[0]:.12f}",
        "",
        "| implementation | seconds | ms per step | speed-up |",
        "|---|---|---|---|",
    ]
    for name, (seconds, _) in results.items():
        lines.append(
            f"| {name} | {seconds:.3f} | {1000 * seconds / args.steps:.3f} | "
            f"{baseline / seconds:.0f}x |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    path = Path(__file__).parent / f"speed-{stamp}.md"
    path.write_text(text)
    print(path)


if __name__ == "__main__":
    main()
