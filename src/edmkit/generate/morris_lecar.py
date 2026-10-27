import numpy as np

__all__ = ["morris_lecar"]


def morris_lecar(
    C: float,
    g_L: float,
    g_Ca: float,
    g_K: float,
    V_L: float,
    V_Ca: float,
    V_K: float,
    V1: float,
    V2: float,
    V3: float,
    V4: float,
    phi: float,
    I: float,
):
    """Vector field of the Morris-Lecar neuron model.

    Parameters
    ----------
    C : float
        Membrane capacitance (typical: 20).
    g_L : float
        Leak conductance (typical: 2).
    g_Ca : float
        Calcium conductance (typical: 4.4).
    g_K : float
        Potassium conductance (typical: 8).
    V_L : float
        Leak reversal potential (typical: -60).
    V_Ca : float
        Calcium reversal potential (typical: 120).
    V_K : float
        Potassium reversal potential (typical: -84).
    V1 : float
        Half-activation voltage of the calcium channel (typical: -1.2).
    V2 : float
        Slope of the calcium activation (typical: 18).
    V3 : float
        Half-activation voltage of the potassium channel (typical: 2).
    V4 : float
        Slope of the potassium activation (typical: 30).
    phi : float
        Rate of the potassium activation (typical: 0.04).
    I : float
        Applied current (typical: 100 for spiking).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state ``(V, n)`` of shape ``(2,)``, or ``(B, 2)`` for a batch: membrane potential and potassium activation.
    """

    def f(t: float, x: np.ndarray):
        V, n = np.moveaxis(x, -1, 0)
        m_inf = 0.5 * (1 + np.tanh((V - V1) / V2))
        n_inf = 0.5 * (1 + np.tanh((V - V3) / V4))
        return np.stack(
            [
                (I - g_L * (V - V_L) - g_Ca * m_inf * (V - V_Ca) - g_K * n * (V - V_K)) / C,
                phi * (n_inf - n) * np.cosh((V - V3) / (2 * V4)),
            ],
            axis=-1,
        )

    return f
