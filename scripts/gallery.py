from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.colors import CenteredNorm, LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d.axes3d import Axes3D

from edmkit.generate import (
    chua,
    ctrnn,
    double_pendulum,
    duffing,
    euler_spiral,
    fitzhugh_nagumo,
    hindmarsh_rose,
    kuramoto,
    lorenz,
    lorenz96,
    lotka_volterra,
    mackey_glass,
    morris_lecar,
    pyloric,
    rk45,
    rossler,
    rulkov_map,
    three_scroll,
    to_xy,
)
from edmkit.wavelet import cwt, morlet

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


def panels(views: dict[str, dict[str, np.ndarray]], **style):
    """One panel per titled view, each a trajectory given as its coordinates by axis label; three coordinates are drawn in 3D."""
    width = min(TEXT_WIDTH, COLUMN_WIDTH * len(views))
    fig = Figure(figsize=(width, 0.5 + 0.9 * width / len(views)))
    line: dict[str, Any] = {"linewidth": 0.3, "alpha": 0.8, **style}
    for i, (title, coordinates) in enumerate(views.items()):
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d" if len(coordinates) == 3 else None)
        ax.plot(*coordinates.values(), **line)
        ax.set(title=title, **dict(zip(("xlabel", "ylabel", "zlabel"), coordinates)))
        ax.locator_params(nbins=4)
        if isinstance(ax, Axes3D):
            ax.set_box_aspect(None, zoom=0.85)
            # look along the direction in which the trajectory, as stretched onto the box, varies least, so that its two widest directions face the viewer
            limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
            shown = np.stack(list(coordinates.values())) / np.ptp(limits, axis=1, keepdims=True) * np.asarray(ax.get_box_aspect())[:, None]
            v = np.linalg.eigh(np.cov(shown))[1][:, 0]
            x, y, z = v if v[2] >= 0 else -v
            ax.view_init(elev=np.degrees(np.arcsin(z)), azim=np.degrees(np.arctan2(y, x)))
    return fig


def heatmap(t: np.ndarray, X: np.ndarray, ylabel: str, **style):
    """Every column of ``X`` against time as an image."""
    fig = Figure(figsize=(TEXT_WIDTH, 2.6))
    ax = fig.subplots()
    image = ax.imshow(X.T, aspect="auto", origin="lower", extent=(t[0], t[-1], -0.5, X.shape[1] - 0.5), **style)
    ax.set(xlabel="t", ylabel=ylabel)
    fig.colorbar(image, label="x")
    return fig


def spectrum(t: np.ndarray, rows: dict[str, np.ndarray], window: int, marks: dict[str, float] | None = None, unit: str = "1 / time"):
    """Per row, the Morlet scalogram of the first ``window`` samples beside the frequency profile of the whole series.

    The scalogram shows the amplitude ``|W|`` of each frequency over time, washed out within the cone of influence; the profile is its root mean
    square over the reliable times, so a sinusoid of amplitude A reads A in both. A row given as ``(T, k)`` holds k statistically alike series,
    such as the variables of a ring: the scalogram shows the first and the profile pools all of them.
    Frequencies run from those reliable over most of the series and over at least the second half of the window, up to where every profile has
    fallen 100-fold below its peak or to its numerical floor.
    """
    dt = t[1] - t[0]
    wavelet = morlet()
    lowest = max(4 * wavelet.width / (t[-1] - t[0]), 2 * wavelet.width / (window * dt))
    frequencies = np.geomspace(lowest, 0.5 / dt, 160)
    transforms = []
    for x in rows.values():
        W, reliable = cwt(x, dt=dt, frequencies=frequencies, wavelet=wavelet)
        power = np.abs(W.reshape(len(x), -1, len(frequencies))) ** 2
        profile = np.sqrt(np.nanmean(np.where(reliable[:, None], power, np.nan), axis=(0, 1)))
        transforms.append((np.sqrt(power[:window, 0]), profile))
    # up to the first frequency above each peak where the profile falls 100-fold below it, or else reaches its numerical floor
    top = 0
    for _, profile in transforms:
        peak = np.argmax(profile)
        below = np.flatnonzero(profile[peak:] < 1e-2 * profile[peak])
        top = max(top, peak + (below[0] if below.size else np.argmin(profile[peak:])))
    shown = slice(0, top + 1)
    f = frequencies[shown]

    fig = Figure(figsize=(TEXT_WIDTH, 0.5 + 1.1 * len(rows)))
    grid = fig.add_gridspec(len(rows), 2, width_ratios=[5, 1])
    t_window = t[:window]
    # downsample time to about 1500 columns, keeping the root mean square amplitude of each block
    stride = max(1, window // 1500)
    columns = window // stride
    t_shown = t_window[: columns * stride : stride] + 0.5 * stride * dt
    for i, (label, (amplitude, profile)) in enumerate(zip(rows, transforms)):
        scalogram = fig.add_subplot(grid[i, 0], sharex=fig.axes[0] if i else None)
        side = fig.add_subplot(grid[i, 1], sharey=scalogram)
        A = np.sqrt((amplitude[: columns * stride, shown] ** 2).reshape(columns, stride, -1).mean(axis=1))
        vmax = profile[shown].max() * 2
        norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax, clip=True)
        mesh = scalogram.pcolormesh(t_shown, f, A.T, norm=norm, cmap="magma", shading="nearest", rasterized=True)
        # cone of influence of the whole series, of which the window is the start
        scalogram.fill_betweenx(f, t[0], np.minimum(t[0] + wavelet.width / f, t_window[-1]), color="white", alpha=0.5, linewidth=0)
        if t[-1] - wavelet.width / f[0] < t_window[-1]:
            scalogram.fill_betweenx(f, np.maximum(t[-1] - wavelet.width / f, t[0]), t_window[-1], color="white", alpha=0.5, linewidth=0)
        scalogram.set(yscale="log", ylabel=f"{label}\nf ({unit})", xlim=(t_window[0], t_window[-1]), ylim=(f[0], f[-1]))
        side.plot(profile[shown], f, linewidth=0.8, color="black")
        side.set(xscale="log", xlim=(norm.vmin, norm.vmax))
        side.tick_params(labelleft=False)
        scalogram.yaxis.set_minor_formatter(NullFormatter())
        for name, frequency in (marks or {}).items():
            for ax in (scalogram, side):
                ax.axhline(frequency, color="tab:cyan", linewidth=0.6, linestyle="--")
            side.annotate(
                name, (1, frequency), xycoords=("axes fraction", "data"), xytext=(2, 0), textcoords="offset points", va="center", fontsize=6
            )
        fig.colorbar(mesh, ax=side, label="|W|", pad=0.02)
        if i < len(rows) - 1:
            scalogram.tick_params(labelbottom=False)
            side.tick_params(labelbottom=False)
    scalogram.set_xlabel("t")
    side.set_xlabel("RMS |W|")
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


def coupled(units: list[Callable[[float, np.ndarray], np.ndarray]], coupling: float):
    """Vector field of ``units`` coupled all-to-all through their first variable, each pulled towards the mean (electrical coupling), for their states stacked."""

    def f(t: float, x: np.ndarray):
        X = x.reshape(len(units), -1)
        dX = np.stack([unit(t, state) for unit, state in zip(units, X)])
        dX[:, 0] += coupling * (X[:, 0].mean() - X[:, 0])
        return dX.ravel()

    return f


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
    save(spectrum(t, dict(zip("xyz", X.T)), 4000), "lorenz_spectrum", title)

    title = "Rössler\n$a = 0.2$, $b = 0.2$, $c = 5.7$"
    t, X = settled(*rk45(rossler(0.2, 0.2, 5.7), np.ones(3), 0.02, 600), transient=100)
    save(traces(t[:10000], X[:10000], "xyz"), "rossler_traces", title)
    save(panels({"": dict(zip("xyz", X.T))}), "rossler_attractor", title)
    save(spectrum(t, {"x": X[:, 0], "z": X[:, 2]}, 10000), "rossler_spectrum", title)

    # Li's c = 11/6 settles onto a limit cycle, so c is raised to 1.9; this chaotic attractor rarely makes large excursions, none within this window
    title = "Li three-scroll\n$a = 40$, $k = 55$, $c = 1.9$, $d = 0.16$, $e = 0.65$, $f = 20$"
    t, X = settled(*rk45(three_scroll(40.0, 55.0, 1.9, 0.16, 0.65, 20.0), np.array([2.0, 2.0, 2.0]), 0.002, 45), transient=10)
    save(traces(t[:5000], X[:5000], "xyz"), "three_scroll_traces", title)
    save(panels({"": dict(zip("xyz", X.T))}), "three_scroll_attractor", title)
    save(spectrum(t, dict(zip("xyz", X.T)), 5000), "three_scroll_spectrum", title)

    title = "Chua\n" + r"$\alpha = 15.6$, $\beta = 28$, $m_0 = -8/7$, $m_1 = -5/7$"
    t, X = settled(*rk45(chua(15.6, 28.0, -8.0 / 7.0, -5.0 / 7.0), np.array([0.7, 0.0, 0.0]), 0.02, 300), transient=50)
    save(traces(t[:5000], X[:5000], "xyz"), "chua_traces", title)
    save(panels({"": dict(zip("xyz", X.T))}), "chua_attractor", title)
    save(spectrum(t, dict(zip("xyz", X.T)), 5000), "chua_spectrum", title)

    title = "Duffing\n" + r"$\alpha = -1$, $\beta = 1$, $\delta = 0.3$, $\gamma = 0.5$, $\omega = 1.2$"
    t, X = settled(*rk45(duffing(-1.0, 1.0, 0.3, 0.5, 1.2), np.array([1.0, 0.0]), 0.05, 600), transient=100)
    save(traces(t[:6000], X[:6000], "xv"), "duffing_traces", title)
    save(panels({"": dict(zip("xv", X.T))}), "duffing_attractor", title)
    save(spectrum(t, {"x": X[:, 0]}, 6000, marks={r"$\omega / 2\pi$": 1.2 / (2 * np.pi)}), "duffing_spectrum", title)

    title = "Double pendulum\npath of the lower bob, $m_1 = m_2 = 1$, $L_1 = 1$, $L_2 = 1/3$"
    _, X = rk45(double_pendulum(1.0, 1.0, 1.0, 1.0 / 3.0, 9.81), np.array([2.0, 2.5, 0.0, 0.0]), 0.01, 30)
    _, _, x2, y2 = to_xy(1.0, 1.0 / 3.0, X[:, 0], X[:, 1])
    fig = panels({"": {"x": x2, "y": y2}})
    fig.axes[0].set_aspect("equal")
    save(fig, "double_pendulum_attractor", title)

    title = "Euler spiral\ncurvature $\\kappa(s) = a s$, $a = 1$, arc length $s \\in [-8, 8]$"
    _, X = rk45(euler_spiral(1.0), np.zeros(3), 0.01, 8)
    fig = panels({"": {"x": np.concatenate([-X[:0:-1, 0], X[:, 0]]), "y": np.concatenate([-X[:0:-1, 1], X[:, 1]])}})
    fig.axes[0].set_aspect("equal")
    save(fig, "euler_spiral_attractor", title)

    title = "Hindmarsh–Rose\n" + r"$a = 1$, $b = 3$, $c = 1$, $d = 5$, $r = 0.006$, $s = 4$, $x_R = -1.6$, $I = 3.25$"
    t, X = settled(*rk45(hindmarsh_rose(1.0, 3.0, 1.0, 5.0, 0.006, 4.0, -1.6, 3.25), np.array([-1.6, 0.0, 0.0]), 0.05, 2500), transient=500)
    save(traces(t[:20000], X[:20000], "xyz"), "hindmarsh_rose_traces", title)
    save(panels({"": dict(zip("xyz", X.T))}), "hindmarsh_rose_attractor", title)
    save(spectrum(t, {"x": X[:, 0], "z": X[:, 2]}, 20000), "hindmarsh_rose_spectrum", title)

    # a single unit is two-dimensional and so at most periodic; three units with different currents and time scales, coupled through v, are chaotic
    # v jumps between two branches, so the attractor is drawn in the smooth recovery variables w
    title = "FitzHugh–Nagumo\n3 units coupled through $v$ with strength 0.1, " + r"$\epsilon = 0.12, 0.18, 0.27$, $I = 0.2, 0.5, 0.8$"
    units = [fitzhugh_nagumo(0.7, 0.8, epsilon, I) for epsilon, I in zip((0.12, 0.18, 0.27), (0.2, 0.5, 0.8))]
    t, X = settled(*rk45(coupled(units, 0.1), np.array([-1.0, 1.0, 0.0, 0.5, 1.0, 0.0]), 0.1, 2500), transient=1000)
    save(traces(t[:4000], X[:4000, ::2], ["v1", "v2", "v3"]), "fitzhugh_nagumo_traces", title)
    save(panels({"": dict(zip(["w1", "w2", "w3"], X[:, 1::2].T))}), "fitzhugh_nagumo_attractor", title)
    save(spectrum(t, {"v1": X[:, 0], "v2": X[:, 2], "v3": X[:, 4]}, 4000), "fitzhugh_nagumo_spectrum", title)

    # likewise for three Morris–Lecar units, drawn in their potassium activations n
    title = "Morris–Lecar\n3 units coupled through $V$ with strength 0.06, " + r"$\phi = 0.04, 0.06, 0.09$, $I = 70, 100, 130$"
    units = [
        morris_lecar(20.0, 2.0, 4.4, 8.0, -60.0, 120.0, -84.0, -1.2, 18.0, 2.0, 30.0, phi, I)
        for phi, I in zip((0.04, 0.06, 0.09), (70.0, 100.0, 130.0))
    ]
    t, X = settled(*rk45(coupled(units, 0.06), np.array([-60.0, 0.0, -20.0, 0.2, 10.0, 0.4]), 1.0, 16000), transient=10000)
    save(traces(t[:3000], X[:3000, ::2], ["V1", "V2", "V3"]), "morris_lecar_traces", title)
    save(panels({"": dict(zip(["n1", "n2", "n3"], X[:, 1::2].T))}), "morris_lecar_attractor", title)
    save(spectrum(t, {"V1": X[:, 0], "V2": X[:, 2], "V3": X[:, 4]}, 3000, unit="1 / ms"), "morris_lecar_spectrum", title)

    # a map has no trajectory between steps, so its states are drawn as points; the slow y drifts while the fast x bursts (upper band) or rests (lower branch)
    title = "Rulkov map\n" + r"$\alpha = 4.1$, $\mu = 0.001$, $\sigma = -1.5$"
    n, X = rulkov_map(4.1, 0.001, -1.5, 0.0, np.array([[-1.0, -3.0]]), 60000)
    n, X = settled(n, X[:, 0], transient=10000)
    save(traces(n[:3000], X[:3000], "xy"), "rulkov_map_traces", title)
    save(panels({"": {"y": X[:, 1], "x": X[:, 0]}}, linestyle="none", marker=".", markersize=0.6, alpha=0.4), "rulkov_map_attractor", title)
    save(spectrum(n, {"x": X[:, 0], "y": X[:, 1]}, 3000, unit="1 / step"), "rulkov_map_spectrum", title)

    # the AB/PD pacemaker inhibits LP and PY, which rebound in turn, giving the rhythm PD → LP → PY; with these neurons and synapses LP's bursts vary
    # chaotically from cycle to cycle (largest Lyapunov exponent about 3 / s); the attractor is drawn in each neuron's calcium
    title = "Pyloric circuit\nAB/PD 2, LP 1 and PY 4 of Prinz et al. (2004), synapses of 3 to 100 nS"
    g_mem = np.array([[100, 2.5, 6, 50, 5, 100, 0.01, 0], [100, 0, 8, 40, 5, 75, 0.05, 0.02], [400, 2.5, 2, 50, 0, 75, 0.05, 0]])
    g_glu = np.array([[0, 3, 0], [3, 0, 100], [10, 100, 0]])
    g_ach = np.array([[0, 0, 0], [0, 0, 0], [100, 0, 0]])
    x0 = np.zeros(45)
    x0[:3], x0[3:6] = -50.0, 0.05
    t, X = settled(*rk45(pyloric(g_mem, g_glu, g_ach), x0, 0.2, 11000), transient=3000)
    save(traces(t[:20000], X[:20000, :3], ["V (AB/PD)", "V (LP)", "V (PY)"]), "pyloric_traces", title)
    save(panels({"": dict(zip(["Ca (AB/PD)", "Ca (LP)", "Ca (PY)"], X[:, 3:6].T))}), "pyloric_attractor", title)
    save(spectrum(t, {"V (AB/PD)": X[:, 0], "V (LP)": X[:, 1], "V (PY)": X[:, 2]}, 20000, unit="1 / ms"), "pyloric_spectrum", title)

    title = "Mackey–Glass\n" + r"$\tau = 17$, $n = 10$, $\beta = 0.2$, $\gamma = 0.1$"
    t, x = settled(*mackey_glass(17.0, 10, 0.2, 0.1, 1.2, 0.1, 2000), transient=500)
    save(traces(t[:6000], x[:6000, None], "x"), "mackey_glass_traces", title)
    save(panels({"": {"x(t)": x[170:], r"x(t − $\tau$)": x[:-170]}}), "mackey_glass_attractor", title)
    save(spectrum(t, {"x": x}, 6000), "mackey_glass_spectrum", title)

    title = "Lorenz 96\n40 variables on a ring, $F = 8$"
    t, X = settled(*rk45(lorenz96(8.0), 8.0 + np.random.default_rng(0).normal(scale=0.01, size=40), 0.05, 50), transient=20)
    save(heatmap(t, X, "variable"), "lorenz96_heatmap", title)
    # the variables are alike under rotation of the ring, so the profile pools all 40 while the scalogram shows the first
    save(spectrum(t, {"x1 (profile: all)": X}, len(t)), "lorenz96_spectrum", title)

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
    save(spectrum(t, {"A.x": A[:, 0], "B.x": B[:, 0], "C.x": C[:, 0], "D.x": D[:, 0]}, 4000), "chain_spectrum", title)

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
    save(spectrum(t, {f"unit {i}": X[:, i] for i in range(4)}, 1000), "ctrnn_spectrum", title)

    title = "Lotka–Volterra\n4 competing species (Vano et al. 2006), $r = (1, 0.72, 1.53, 1.27)$"
    r = np.array([1.0, 0.72, 1.53, 1.27])
    alpha = np.array([[1, 1.09, 1.52, 0], [0, 1, 0.44, 1.36], [2.33, 0, 1, 0.47], [1.21, 0.51, 0.35, 1]])
    t, X = settled(*rk45(lotka_volterra(r[:, None] * alpha, r, 0.0), np.array([0.3013, 0.4586, 0.1307, 0.3557]), 0.5, 3000), transient=500)
    save(traces(t[:3000], X[:3000], ["x1", "x2", "x3", "x4"]), "lotka_volterra_traces", title)
    save(panels({"": dict(zip(["x1", "x2", "x3"], X.T))}), "lotka_volterra_attractor", title)
    save(spectrum(t, {"x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3]}, 3000), "lotka_volterra_spectrum", title)

    # a Gaussian spread of frequencies of width 1 puts the transition at k = 2 / (π g(0)) ≈ 1.6, so k = 3 is well past it
    title = "Kuramoto\n50 oscillators with all-to-all coupling, sorted by natural frequency, " + r"$\omega_i \sim \mathcal{N}(0, 1)$, $k = 3$"
    rng = np.random.default_rng(0)
    N = 50
    omega = np.sort(rng.normal(size=N))
    t, X = settled(*rk45(kuramoto(omega, np.full((N, N), 3.0 / N)), rng.uniform(0, 2 * np.pi, N), 0.1, 150), transient=50)
    fig = heatmap(t, np.sin(X), "oscillator", cmap="RdBu_r", norm=CenteredNorm())
    fig.axes[1].set_ylabel(r"$\sin \theta$")
    save(fig, "kuramoto_heatmap", title)

    # a phase has no amplitude to analyse, so its frequency is read off directly; against the natural frequencies, the mean frequencies of
    # locked oscillators form a plateau, and those of drifting ones follow the diagonal. At k = 3 every oscillator is locked, so the
    # instantaneous frequencies stay constant and a spectrogram would show nothing
    effective = (X[-1] - X[0]) / (2 * np.pi * (t[-1] - t[0]))
    fig = Figure(figsize=(COLUMN_WIDTH, 2.8))
    ax = fig.subplots()
    lim = np.abs(omega).max() / (2 * np.pi) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.5, linestyle="--")
    ax.plot(omega / (2 * np.pi), effective, linestyle="none", marker=".", markersize=3, color="black")
    ax.set(xlabel=r"natural $\omega_i / 2\pi$", ylabel=r"mean $\dot\theta_i / 2\pi$", xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    save(fig, "kuramoto_spectrum", title)


if __name__ == "__main__":
    main()
