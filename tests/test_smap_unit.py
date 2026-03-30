import numpy as np
import pytest

from edmkit.smap import smap, weights


def ols_predict(x: np.ndarray, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((len(x), 1)), x])
    q_aug = np.hstack([np.ones((len(q), 1)), q])
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return q_aug @ coef


def weighted_lstsq_reference(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    *,
    theta: float,
    alpha: float,
) -> np.ndarray:
    """Reference S-Map via weighted least squares on an augmented system."""
    x_aug = np.hstack([np.ones((len(x), 1)), x])
    q_aug = np.hstack([np.ones((len(q), 1)), q])
    y_col = y[:, None] if y.ndim == 1 else y

    distances = np.linalg.norm(q[:, None, :] - x[None, :, :], axis=2)
    if theta == 0.0:
        w = np.ones_like(distances)
    else:
        d_mean = np.maximum(distances.mean(axis=1, keepdims=True), 1e-6)
        w = np.exp(-theta * distances / d_mean)

    eye = np.eye(x_aug.shape[1])
    eye[0, 0] = 0.0

    predictions = []
    for i in range(len(q)):
        sqrt_w = np.sqrt(w[i])[:, None]
        a = sqrt_w * x_aug
        b = sqrt_w * y_col
        trace = max(float(np.trace(a.T @ a)), 1e-12)
        reg_rows = np.sqrt(alpha * trace) * eye[1:]
        a_reg = np.vstack([a, reg_rows])
        b_reg = np.vstack([b, np.zeros((reg_rows.shape[0], y_col.shape[1]))])
        coef, *_ = np.linalg.lstsq(a_reg, b_reg, rcond=None)
        predictions.append((q_aug[i] @ coef).squeeze())

    return np.asarray(predictions).squeeze()


class TestWeightsExamples:
    def test_theta_zero_returns_binary_valid_mask(self):
        distances = np.array([[0.0, 1.0, np.inf]])
        actual = weights(distances, theta=0.0, min_points=2)
        np.testing.assert_array_equal(actual, np.array([[1.0, 1.0, 0.0]]))

    def test_rejects_too_few_valid_points(self):
        distances = np.array([[0.0, np.inf, np.inf]])
        with pytest.raises(ValueError, match="Not enough valid"):
            weights(distances, theta=2.0, min_points=2)


class TestSMapExamples:
    def test_theta_zero_matches_global_ols(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = 1.5 + 2.0 * x
        q = np.array([[1.5], [2.5]])
        expected = ols_predict(x, y, q)
        actual = smap(x, y, q, theta=0.0)
        np.testing.assert_allclose(actual, expected.squeeze(), atol=1e-9, rtol=1e-9)

    def test_constant_target_is_preserved_even_with_regularization(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.full(len(x), 4.2)
        q = np.array([[0.5], [2.5]])
        actual = smap(x, y, q, theta=0.0, alpha=100.0)
        np.testing.assert_allclose(actual, 4.2, atol=1e-12, rtol=1e-12)

    def test_matches_independent_weighted_lstsq_reference(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(40, 2))
        y = rng.normal(size=40)
        q = rng.normal(size=(6, 2))
        expected = weighted_lstsq_reference(x, y, q, theta=2.5, alpha=1e-4)
        actual = smap(x, y, q, theta=2.5, alpha=1e-4)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_larger_theta_downweights_distant_regime(self):
        rng = np.random.default_rng(42)
        x = np.vstack([rng.uniform(-0.1, 0.1, (15, 1)), rng.uniform(5.0, 6.0, (15, 1))])
        y = np.concatenate([np.zeros(15), np.full(15, 100.0)])
        q = np.array([[0.0]])
        global_prediction = smap(x, y, q, theta=0.0, alpha=0.0)
        local_prediction = smap(x, y, q, theta=4.0, alpha=0.0)
        assert float(local_prediction) < float(global_prediction)

    def test_larger_alpha_shrinks_prediction_variance(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, 2))
        y = rng.normal(size=30)
        q = rng.normal(size=(5, 2))
        weak_regularization = smap(x, y, q, theta=0.0, alpha=1e-10)
        strong_regularization = smap(x, y, q, theta=0.0, alpha=1.0)
        assert np.var(strong_regularization) < np.var(weak_regularization)

    def test_batch_path_matches_loop(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 18, 2))
        y = rng.normal(size=(2, 18, 1))
        q = rng.normal(size=(2, 5, 2))
        batched = smap(x, y, q, theta=1.5)
        for batch in range(x.shape[0]):
            expected = smap(x[batch], y[batch], q[batch], theta=1.5)
            np.testing.assert_allclose(batched[batch].squeeze(-1), expected.squeeze(), atol=1e-10, rtol=1e-10)

    def test_rejects_negative_theta(self):
        x = np.zeros((5, 2))
        y = np.zeros(5)
        q = np.zeros((2, 2))
        with pytest.raises(ValueError, match="non-negative"):
            smap(x, y, q, theta=-1.0)
