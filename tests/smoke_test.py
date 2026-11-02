from concurrent.futures import ThreadPoolExecutor

import numpy as np

from edmkit import generate
from edmkit.ccm import bootstrap
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

    mid = len(embedding) // 2
    actual = x[mid + 1 :, None]
    lib_sizes = np.array([10, 20])
    n_samples = 3

    def score(samples):
        predictions = samples()
        return pearson_correlation(predictions, np.broadcast_to(actual, predictions.shape))

    scores = np.empty((len(lib_sizes), n_samples))
    tasks = bootstrap(embedding[:mid], x[1 : mid + 1], embedding[mid:], lib_sizes, predict_func=simplex_projection, n_samples=n_samples, batch_size=2)
    indices, samples = zip(*tasks)
    with ThreadPoolExecutor() as executor:
        for index, result in zip(indices, executor.map(score, samples)):
            scores[index] = result
    assert np.isfinite(scores).all()

    _, lorenz = generate.rk45(generate.lorenz(10.0, 28.0, 8.0 / 3.0), np.ones(3), 0.01, 1)
    _, mackey_glass = generate.mackey_glass(17.0, 10, 0.2, 0.1, 0.9, 1.0, 50)
    _, double_pendulum = generate.rk45(generate.double_pendulum(1.0, 1.0, 1.0, 1.0, 9.81), np.array([0.5, 0.3, 0.0, 0.0]), 0.01, 1)
    assert all(np.isfinite(trajectory).all() for trajectory in (lorenz, mackey_glass, double_pendulum))
