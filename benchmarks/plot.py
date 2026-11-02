"""
uv run -m benchmarks.plot .benchmarks/<machine>/<run>.json [out.png]
"""

import json
import sys
from itertools import groupby
from typing import NamedTuple

import matplotlib.pyplot as plt

from benchmarks.run import BASE, Shape


class Result(NamedTuple):
    func: str
    backend: str
    data: str
    shape: Shape
    median: float


def result(benchmark: dict) -> Result:
    info = benchmark["extra_info"]
    shape = Shape._make(info[field] for field in Shape._fields)
    return Result(info["func"], info["backend"], info["data"], shape, benchmark["stats"]["median"])


def load(path: str) -> list[Result]:
    with open(path) as f:
        return [result(benchmark) for benchmark in json.load(f)["benchmarks"]]


def swept(shape: Shape, field: str) -> bool:
    """Whether `shape` differs from BASE in `field` alone, with a value that has an x position."""
    return getattr(shape, field) is not None and shape._replace(**{field: getattr(BASE, field)}) == BASE


def curves(results: list[Result], field: str):
    """(label, xs, ys) for each backend/data pair, along the sweep of `field`."""
    points = sorted((r.backend, r.data, getattr(r.shape, field), r.median) for r in results if swept(r.shape, field))
    for (backend, data), group in groupby(points, key=lambda p: p[:2]):
        _, _, xs, ys = zip(*group)
        yield f"{backend}/{data}", xs, ys


def main(path: str, out: str = "benchmarks.png") -> None:
    results = load(path)
    funcs = sorted({r.func for r in results})
    # fields swept over more than one value
    fields = [field for field in Shape._fields if len({getattr(r.shape, field) for r in results if swept(r.shape, field)}) > 1]

    fig, axes = plt.subplots(len(funcs), len(fields), figsize=(4 * len(fields), 3 * len(funcs)), squeeze=False)
    for row, func in zip(axes, funcs):
        for ax, field in zip(row, fields):
            for label, xs, ys in curves([r for r in results if r.func == func], field):
                ax.plot(xs, ys, marker="o", label=label)
            ax.set(xscale="log", yscale="log", title=f"{func} vs {field}", xlabel=field, ylabel="median [s]")
            ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
