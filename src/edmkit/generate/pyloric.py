import numpy as np

__all__ = ["pyloric"]

AREA = 0.628e-3  # membrane area in cm^2 (specific capacitance 1 uF/cm^2, i.e. C = 0.628 nF)
RT_2F = 12.2  # RT / 2F in mV at 10 C, for the Nernst potential of Ca
E_REV = np.array([50.0, 0.0, 0.0, -80.0, -80.0, -80.0, -20.0, -50.0])  # Na, CaT, CaS, A, KCa, Kd, H, leak; Ca is replaced by Nernst
POWER = np.array([3, 1, 3, 1, 3, 1, 3, 1, 4, 4, 1])[:, None]  # exponents of the 11 gates, see the state layout
E_SYN = np.array([-70.0, -80.0])[:, None]  # reversal potentials of the glutamatergic and cholinergic synapses
K_SYN = np.array([1 / 40, 1 / 100])[:, None]  # decay rates of the glutamatergic and cholinergic synapses


def pyloric(g_mem: np.ndarray, g_glu: np.ndarray, g_ach: np.ndarray):
    """Vector field of the three-neuron pyloric circuit of the crustacean stomatogastric ganglion (Prinz, Bucher & Marder 2004), in ms and mV.

    Parameters
    ----------
    g_mem : np.ndarray
        Maximal conductances in mS/cm^2 of shape ``(3, 8)``: rows are AB/PD, LP, PY and columns are Na, CaT, CaS, A, KCa, Kd, H, leak
        (typical: ``[[400, 2.5, 6, 50, 10, 100, 0.01, 0], [100, 0, 8, 40, 5, 75, 0.05, 0.02], [100, 2.5, 2, 50, 0, 125, 0.05, 0.01]]``).
    g_glu : np.ndarray
        Fast glutamatergic synapse strengths in nS of shape ``(3, 3)``; ``g_glu[i, j]`` is the synapse from neuron ``j`` onto neuron ``i``
        (typical: ``g_glu[1, 0] = 30`` (AB to LP), ``g_glu[2, 0] = 3`` (AB to PY), ``g_glu[0, 1] = 10`` (LP to AB/PD), ``g_glu[2, 1] = 3`` (LP to PY),
        ``g_glu[1, 2] = 3`` (PY to LP), the rest 0, for a PD, LP, PY rhythm of period about 1.7 s).
    g_ach : np.ndarray
        Slow cholinergic synapse strengths in nS of shape ``(3, 3)``, indexed like ``g_glu`` (typical: ``g_ach[1, 0] = 3`` (PD to LP), ``g_ach[2, 0] = 30`` (PD to PY), the rest 0).

    Returns
    -------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side ``f(t, x)`` for a state of shape ``(45,)``, or ``(B, 45)`` for a batch; ``x.reshape(15, 3)`` has one column per neuron and rows: membrane potential,
        intracellular [Ca] in uM, the gates m and h of Na, m and h of CaT, m and h of CaS, m and h of A, m of KCa, n of Kd, m of H, and the glutamatergic
        and cholinergic synaptic activations of the neuron as presynaptic cell
        (typical initial state: V = -50, [Ca] = 0.05, all gates and synaptic activations 0, discard the first 3 s as transient).
    """
    g_mem = np.asarray(g_mem, dtype=float)
    g_syn = np.stack([g_glu, g_ach]) * 1e-6 / AREA  # nS to uA/cm^2 per mV

    def f(t: float, x: np.ndarray):
        batch = x.shape[:-1]
        V, Ca = x[..., :3], x[..., 3:6]
        gate = x[..., 6:39].reshape(*batch, 11, 3)
        s = x[..., 39:].reshape(*batch, 2, 3)

        # steady states and time constants of the 11 gates
        def sig(a, b):
            return 1 / (1 + np.exp((V + a) / b))

        def tau(a, b, c, d):
            return a - b / (1 + np.exp((V + c) / -d))

        inf = np.stack(
            [
                sig(25.5, -5.29),
                sig(48.9, 5.18),
                sig(27.1, -7.2),
                sig(32.1, 5.5),
                sig(33.0, -8.1),
                sig(60.0, 6.2),
                sig(27.2, -8.7),
                sig(56.9, 4.9),
                Ca / (Ca + 3) * sig(28.3, -12.6),
                sig(12.3, -11.8),
                sig(75.0, 5.5),
            ],
            axis=-2,
        )
        tau_g = np.stack(
            [
                tau(2.64, 2.52, 120, 25),
                1.34 / (1 + np.exp((V + 62.9) / -10)) * (1.5 + 1 / (1 + np.exp((V + 34.9) / 3.6))),
                tau(43.4, 42.6, 68.1, 20.5),
                tau(210, 179.6, 55, 16.9),
                2.8 + 14 / (np.exp((V + 27) / 10) + np.exp((V + 70) / -13)),
                120 + 300 / (np.exp((V + 55) / 9) + np.exp((V + 65) / -16)),
                tau(23.2, 20.8, 32.9, 15.2),
                tau(77.2, 58.4, 38.9, 26.5),
                tau(180.6, 150.2, 46, 22.7),
                tau(14.4, 12.8, 28.3, 19.2),
                2 / (np.exp((V + 169.7) / -11.6) + np.exp((V - 26.7) / 14.3)),
            ],
            axis=-2,
        )

        # membrane currents in uA/cm^2
        m_Na, h_Na, m_CaT, h_CaT, m_CaS, h_CaS, m_A, h_A, m_KCa, n_Kd, m_H = np.moveaxis(gate**POWER, -2, 0)
        activation = np.stack([m_Na * h_Na, m_CaT * h_CaT, m_CaS * h_CaS, m_A * h_A, m_KCa, n_Kd, m_H, np.ones_like(V)], axis=-2)
        drive = V[..., None, :] - E_REV[:, None]
        drive[..., 1:3, :] = (V - RT_2F * np.log(3000 / Ca))[..., None, :]
        I_ion = g_mem.T * activation * drive
        # sum over presynaptic cells first: g_s[p, i] = sum_j g_syn[p, i, j] s[p, j]
        g_s = (g_syn @ s[..., None])[..., 0]
        I_syn = (g_s * (V[..., None, :] - E_SYN)).sum(axis=-2)

        # I_ion[1:3] in uA/cm^2 times area is uA, times 1e3 is nA, times f = 14.96 uM/nA gives the Ca influx
        dCa = (0.05 - Ca - 14.96 * 1e3 * AREA * I_ion[..., 1:3, :].sum(axis=-2)) / 200
        s_inf = sig(35.0, -5.0)[..., None, :]
        # tau_s = (1 - s_inf) / k, floored at 1e-4 / k to keep the spikes from stiffening the system
        ds = (s_inf - s) * K_SYN / np.maximum(1 - s_inf, 1e-4)

        dV = -(I_ion.sum(axis=-2) + I_syn)
        dgate = (inf - gate) / tau_g
        return np.concatenate([dV, dCa, dgate.reshape(*batch, -1), ds.reshape(*batch, -1)], axis=-1)

    return f
