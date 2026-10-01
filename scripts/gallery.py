from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from edmkit.generate import ctrnn, double_pendulum, duffing, lorenz, lorenz96, mackey_glass, rk45, rossler, to_xy

OUTPUT_DIR = Path(__file__).parent / "gallery"

# widths of one column and of the full text of a two-column conference paper (ICML), in inches
COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.75

matplotlib.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "figure.titlesize": 9,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.labelpad": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes3d.grid": False,
        "axes3d.xaxis.panecolor": "white",
        "axes3d.yaxis.panecolor": "white",
        "axes3d.zaxis.panecolor": "white",
        "image.interpolation": "nearest",
    }
)


def traces(t: np.ndarray, X: np.ndarray, labels: str | list[str]):
    """The columns of ``X`` against time, one panel each."""
    fig = Figure(figsize=(TEXT_WIDTH, 0.6 + 0.9 * len(labels)))
    axes = fig.subplots(len(labels), 1, sharex=True, squeeze=False)[:, 0]
    for ax, label, x in zip(axes, labels, X.T):
        ax.plot(t, x, linewidth=0.6)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("t")
    return fig


def panels(views: dict[str, dict[str, np.ndarray]]):
    """One panel per titled view, each a trajectory given as its coordinates by axis label; three coordinates are drawn in 3D."""
    width = min(TEXT_WIDTH, COLUMN_WIDTH * len(views))
    fig = Figure(figsize=(width, 0.5 + 0.9 * width / len(views)))
    for i, (title, coordinates) in enumerate(views.items()):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d" if len(coordinates) == 3 else None)
        ax.plot(*coordinates.values(), linewidth=0.3, alpha=0.8)
        ax.set(title=title, **dict(zip(("xlabel", "ylabel", "zlabel"), coordinates)))
        ax.locator_params(nbins=4)
        if isinstance(ax, Axes3D):
            ax.set_box_aspect(None, zoom=0.85)
    return fig


def heatmap(t: np.ndarray, X: np.ndarray, ylabel: str, **style):
    """Every column of ``X`` against time as an image."""
    fig = Figure(figsize=(TEXT_WIDTH, 2.6))
    ax = fig.subplots()
    image = ax.imshow(X.T, aspect="auto", origin="lower", extent=(t[0], t[-1], -0.5, X.shape[1] - 0.5), **style)
    ax.set(xlabel="t", ylabel=ylabel)
    fig.colorbar(image, label="x")
    return fig


def save(fig: Figure, name: str, title: str):
    fig.suptitle(title, x=0.02, ha="left")
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.05)


def settled(t: np.ndarray, X: np.ndarray, transient: float):
    """Drop the first ``transient`` time units, so that figures show the attractor rather than the approach to it."""
    return t[t >= transient], X[t >= transient]


def rescaled(f: Callable[[float, np.ndarray], np.ndarray], speed: float, scale: float = 1.0):
    """Vector field whose solutions are those of ``f`` run ``speed`` times faster and stretched ``scale`` times."""
    return lambda t, x: speed * scale * f(speed * t, x / scale)


def chain(coupling: float):
    """Vector field of the chain A -> B -> C <- D of a Lorenz, a Rössler, a Lorenz and a Duffing system, for the state ``(A, B, C, D)`` of shape ``(11,)``.

    Each driven variable is pulled towards the one driving it: x, y and z of A drive those of B, and those of B drive those of C; of D, x alone drives x of C.
    Rössler and Duffing oscillate about 8 and 7 times slower than Lorenz, so they are run that much faster to keep up with it.
    Duffing, which stays within 1.5 of zero, is also stretched: unstretched it leaves no trace on C.
    """
    a = c = lorenz(10.0, 28.0, 8.0 / 3.0)
    b = rescaled(rossler(0.2, 0.2, 5.7), speed=8.0)
    d = rescaled(duffing(-1.0, 1.0, 0.3, 0.5, 1.2), speed=7.0, scale=8.0)

    def f(t: float, x: np.ndarray):
        A, B, C, D = x[0:3], x[3:6], x[6:9], x[9:11]
        return np.concatenate(
            [
                a(t, A),
                b(t, B) + coupling * (A - B),
                c(t, C) + coupling * (B - C) + coupling * np.array([D[0] - C[0], 0.0, 0.0]),
                d(t, D),
            ]
        )

    return f


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    title = "Lorenz\n" + r"$\sigma = 10$, $\rho = 28$, $\beta = 8/3$"
    t, X = settled(*rk45(lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.01, 120), transient=20)
    save(traces(t[:4000], X[:4000], "xyz"), "lorenz_traces", title)
    save(panels({"": dict(zip("xyz", X.T))}), "lorenz_attractor", title)

    title = "Rössler\n$a = 0.2$, $b = 0.2$, $c = 5.7$"
    t, X = settled(*rk45(rossler(0.2, 0.2, 5.7), np.ones(3), 0.02, 600), transient=100)
    save(traces(t[:10000], X[:10000], "xyz"), "rossler_traces", title)
    save(panels({"": dict(zip("xyz", X.T))}), "rossler_attractor", title)

    title = "Duffing\n" + r"$\alpha = -1$, $\beta = 1$, $\delta = 0.3$, $\gamma = 0.5$, $\omega = 1.2$"
    t, X = settled(*rk45(duffing(-1.0, 1.0, 0.3, 0.5, 1.2), np.array([1.0, 0.0]), 0.05, 600), transient=100)
    save(traces(t[:6000], X[:6000], "xv"), "duffing_traces", title)
    save(panels({"": dict(zip("xv", X.T))}), "duffing_attractor", title)

    title = "Double pendulum\npath of the lower bob, $m_1 = m_2 = 1$, $L_1 = 1$, $L_2 = 1/3$"
    _, X = rk45(double_pendulum(1.0, 1.0, 1.0, 1.0 / 3.0, 9.81), np.array([2.0, 2.5, 0.0, 0.0]), 0.01, 30)
    _, _, x2, y2 = to_xy(1.0, 1.0 / 3.0, X[:, 0], X[:, 1])
    fig = panels({"": {"x": x2, "y": y2}})
    fig.axes[0].set_aspect("equal")
    save(fig, "double_pendulum_attractor", title)

    title = "Mackey–Glass\n" + r"$\tau = 17$, $n = 10$, $\beta = 0.2$, $\gamma = 0.1$"
    t, x = settled(*mackey_glass(17.0, 10, 0.2, 0.1, 1.2, 0.1, 2000), transient=500)
    save(traces(t[:6000], x[:6000, None], "x"), "mackey_glass_traces", title)
    save(panels({"": {"x(t)": x[170:], r"x(t − $\tau$)": x[:-170]}}), "mackey_glass_attractor", title)

    title = "Lorenz 96\n40 variables on a ring, $F = 8$"
    t, X = settled(*rk45(lorenz96(8.0), 8.0 + np.random.default_rng(0).normal(scale=0.01, size=40), 0.05, 50), transient=20)
    save(heatmap(t, X, "variable"), "lorenz96_heatmap", title)

    # at a coupling of 8 every node follows its driver closely enough to show in the coupling figure, while C still switches wings
    title = "Chain of Lorenz, Rössler and Duffing systems\nA: Lorenz → B: Rössler → C: Lorenz ← D: Duffing, coupling 8, B run 8× and D 7× faster, D stretched 8×"
    t, X = settled(*rk45(chain(8.0), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 0.0]), 0.01, 120), transient=20)
    A, B, C, D = X[:, 0:3], X[:, 3:6], X[:, 6:9], X[:, 9:11]
    save(traces(t[:4000], X[:4000, [0, 3, 6, 9]], ["A.x", "B.x", "C.x", "D.x"]), "chain_traces", title)
    save(
        panels({"A": dict(zip("xyz", A.T)), "B": dict(zip("xyz", B.T)), "C": dict(zip("xyz", C.T)), "D": dict(zip("xv", D.T))}),
        "chain_attractor",
        title,
    )
    coupling = {"A → B": {"A.x": A[:, 0], "B.x": B[:, 0]}, "B → C": {"B.x": B[:, 0], "C.x": C[:, 0]}, "D → C": {"D.x": D[:, 0], "C.x": C[:, 0]}}
    save(panels(coupling), "chain_coupling", title)

    # with weights this sparse and a gain of 10 the activity stays irregular
    title = "CTRNN\n100 units, each pair connected with probability 0.05, $g = 10$"
    rng = np.random.default_rng(0)
    J = rng.normal(size=(100, 100)) * (rng.random((100, 100)) < 0.05) / np.sqrt(5)
    t, X = settled(*rk45(ctrnn(10.0, J), rng.normal(size=100), 0.1, 400), transient=100)
    fig = Figure(figsize=(COLUMN_WIDTH, 3.2))
    ax = fig.subplots()
    fig.colorbar(ax.imshow(J, cmap="RdBu_r", norm=CenteredNorm()), label="J[i, j]")
    ax.set(xlabel="driving unit j", ylabel="driven unit i")
    save(fig, "ctrnn_connectivity", title)
    save(heatmap(t[:1000], X[:1000], "unit", cmap="RdBu_r", norm=CenteredNorm()), "ctrnn_heatmap", title)
    save(panels({f"units {i}, {i + 1}": {f"unit {i}": X[:, i], f"unit {i + 1}": X[:, i + 1]} for i in (0, 2, 4, 6)}), "ctrnn_slices", title)


if __name__ == "__main__":
    main()
