import numpy as np

from edmkit import generate
from edmkit.ccm import ccm
from edmkit.embedding import embed
from edmkit.metrics import pearson_correlation
from edmkit.simplex_projection import simplex_projection
from edmkit.smap import smap

if __name__ == "__main__":
    from tinygrad import Tensor

    x = np.sin(np.linspace(0.0, 6.0 * np.pi, 100))
    embedding, _ = embed(np.array([[0, 0], [0, 1]]), x)
    library, query = embedding[:50], embedding[50:-1]
    target = x[2:52]

    assert np.isfinite(simplex_projection(library, target, query)).all()
    assert np.isfinite(smap(library, target, query, theta=1.0)).all()
    tensor_prediction = smap(
        Tensor(library[:12].astype(np.float32)),
        Tensor(target[:12].astype(np.float32)),
        Tensor(query[:2].astype(np.float32)),
        theta=1.0,
    )
    assert np.isfinite(tensor_prediction.numpy()).all()

    n = len(embedding)
    correlations = ccm(
        embedding,
        x[1:],
        np.array([10, 20]),
        simplex_projection,
        pearson_correlation,
        n_samples=3,
        library_pool=np.arange(n // 2),
        prediction_pool=np.arange(n // 2, n),
    )
    assert correlations.shape == (2,) and np.isfinite(correlations).all()

    _, lorenz = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.01, 1)
    _, mackey_glass = generate.mackey_glass(17.0, 10, 0.2, 0.1, 0.9, 1.0, 50)
    _, double_pendulum = generate.rk45(generate.double_pendulum(1.0, 1.0, 1.0, 1.0, 9.81), np.array([0.5, 0.3, 0.0, 0.0]), 0.01, 1)
    assert all(np.isfinite(trajectory).all() for trajectory in (lorenz, mackey_glass, double_pendulum))
