import numpy as np

from edmkit.embedding import lagged_embed
from edmkit.tensor import Tensor, dtypes
from edmkit.util import pad, pairwise_distance, topk


def calculate_rho(observations: np.ndarray, predictions: np.ndarray):
    assert len(observations) == len(predictions), "observations and predictions must have the same length"
    return np.corrcoef(observations.T, predictions.T)[0, 1]


def search_best_embedding(
    x: np.ndarray, tau_list: list[int], e_list: list[int], Tp: int, max_L: int | None = None, rng: np.random.Generator | None = None
):
    assert all(tau > 0 for tau in tau_list), f"tau must be positive, got tau_list={tau_list}"
    assert all(e > 0 for e in e_list), f"e must be positive, got e_list={e_list}"
    assert max_L is None or max_L <= len(x), f"max_L must be less than or equal to len(x), got max_L={max_L} and len(x)={len(x)}"

    if rng is None:
        rng = np.random.default_rng()

    if max_L is not None:
        x = x[rng.choice(len(x), min(len(x), max_L), replace=False)]

    # lagged_embed(x, tau, e).shape[0] == len(x) - (e - 1) * tau
    min_L = len(x) - (max(e_list) - 1) * max(tau_list)
    assert min_L > 0, (
        f"Not enough data points to embed, got len(x)(={len(x)}) - max(e)(={max(e_list)}) * max(tau)(={max(tau_list)}) = min_L(={min_L})"
    )

    embeddings: list[np.ndarray] = []
    for tau in tau_list:
        for e in e_list:
            # align the time indices of the embeddings. note that the time index of the first embedding is `1 + (e - 1) * tau`
            embeddings.append(lagged_embed(x, tau, e)[-min_L:])  # (min_L, e)

    X = pad(embeddings)  # X.shape == (len(tau_list) * len(e_list), min_L, max(e_list))
    D = pairwise_distance(Tensor(X, dtype=dtypes.float32)).numpy()

    L = min_L
    seq = np.arange(L)
    lib_size = L // 2

    rho = np.zeros((len(tau_list), len(e_list)))

    for i, tau in enumerate(tau_list):
        for j, e in enumerate(e_list):
            batch = i * len(e_list) + j

            samples_indecies = np.arange(lib_size, L)

            observations = X[batch, samples_indecies, :e]
            predictions = np.zeros((len(samples_indecies), e), dtype=x.dtype)

            for k, t in enumerate(samples_indecies):
                # [0, 1, 2, 3, 4 | 5, 6, 7, 8, 9, 10]; initialize mask, `|` separates lib and test
                mask = np.ones(L, dtype=bool)

                # [0, 1, 2, 3, 4 | 5, F, 7, 8, 9, 10], t = 6; exclude self
                mask[t] = False

                # [0, 1, 2, 3, 4 | 5, F, 7, 8, F, F ], Tp = 2; exclude last Tp points to prevent out-of-bound indexing on predictions
                mask[-Tp:] = False

                # [0, 1, 2, 3, 4 | F, F, F, F, F, F ], lib_size = 5; exclude test points
                mask[lib_size:] = False

                # find k(=e+1) nearest neighbors in phase space for simplex projection
                indices_masked, _ = topk(D[batch, t, mask], e + 1, largest=False)
                indices = seq[mask][indices_masked]
                predictions[k] = X[batch, indices + Tp, :e].mean()

            rho[i, j] = calculate_rho(observations, predictions)

    return rho
