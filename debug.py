import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyEDM

from edmkit import generate, tensor
from edmkit.embedding import lagged_embed
from edmkit.simplex_projection import simplex_projection
from edmkit.util import pairwise_distance, topk


def logistic_map(n: int = 200):
    """Generate logistic map time series x."""
    x = np.zeros(n)
    x[0] = 0.1
    # Logistic map
    for i in range(1, n):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
    return x


def lorenz(n: int = 200):
    """Generate Lorenz attractor time series x."""
    sigma, rho, beta = 10, 28, 8 / 3
    X0 = np.array([1.0, 1.0, 1.0])
    t_max = 30
    dt = t_max / (n * 10)  # Generate enough points before subsampling

    t, X = generate.lorenz(sigma, rho, beta, X0, dt, t_max)
    return X[::10, 0][:n]  # Ensure we get exactly n points


def mackey_glass(n: int = 200):
    """Generate Mackey-Glass time series x."""
    tau, n_exponent = 17, 10
    beta, gamma = 0.2, 0.1
    x0 = 0.9
    t_max = 200
    dt = t_max / n

    t, x = generate.mackey_glass(tau, n_exponent, beta, gamma, x0, dt, t_max)
    return x


params = [
    ("logistic_map", 3, 2),
    ("lorenz", 3, 1),
    ("mackey_glass", 4, 2),
]


def main():
    for data, E, tau in params:
        if data == "logistic_map":
            x = logistic_map()
        elif data == "lorenz":
            x = lorenz()
        elif data == "mackey_glass":
            x = mackey_glass()
        else:
            raise ValueError(f"Unknown data type: {data}")

        # common parameters
        lib_size = 150
        Tp = 2

        # pyEDM
        df = pd.DataFrame({"time": np.arange(len(x)), "value": x})
        lib, pred = f"1 {lib_size}", f"{lib_size + 1} {len(x)}"
        pyedm_result = pyEDM.Simplex(dataFrame=df, lib=lib, pred=pred, E=E, tau=-tau, columns="value", target="value", Tp=Tp, verbose=False)
        pyedm_predictions = pyedm_result["Predictions"].values[Tp:-Tp]  # first Tp values are NaN, last Tp values are not in true x

        # edmkit
        embedding = lagged_embed(x, tau, E)
        shift = tau * (E - 1)  # embedding starts at this index (i.e. embedding[0][0] == x[shift])
        X = embedding[: lib_size - shift]
        Y = embedding[Tp : lib_size - shift + Tp, 0]  # shifted by Tp

        query_points = embedding[lib_size - shift :]
        edmkit_predictions = simplex_projection(X, Y, query_points)[:-Tp]  # last Tp values are not in true x

        ground_truth = x[lib_size + Tp :]
        pyedm_rmse = np.sqrt(np.mean((pyedm_predictions - ground_truth) ** 2))
        edmkit_rmse = np.sqrt(np.mean((edmkit_predictions - ground_truth) ** 2))

        diff = edmkit_predictions - pyedm_predictions

        fig1 = plt.figure(figsize=(10, 6))

        ax1 = fig1.add_subplot(2, 1, 1)

        ax1.plot(ground_truth, label="Ground Truth", color="black")
        ax1.plot(pyedm_predictions, label="pyEDM Predictions", linestyle="--", color="blue")
        ax1.plot(edmkit_predictions, label="edmkit Predictions", linestyle=":", color="red")

        print(pyedm_predictions.shape, edmkit_predictions.shape, ground_truth.shape, X.shape, Y.shape, query_points.shape)

        large_diff = np.abs(diff) > np.max(np.abs(diff)) * 0.1
        ax1.vlines(
            np.where(large_diff)[0], ymin=0, ymax=1, color="orange", alpha=0.3, label="Significant Difference", transform=ax1.get_xaxis_transform()
        )

        ax2 = fig1.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(diff, label="Difference (edmkit - pyEDM)", color="green")
        ax2.vlines(
            np.where(large_diff)[0], ymin=0, ymax=1, color="orange", alpha=0.3, label="Significant Difference", transform=ax2.get_xaxis_transform()
        )
        ax2.axhline(0, color="black", linestyle="--")

        fig1.tight_layout()
        fig1.legend()
        plt.show()

        # display embedding in 3D
        fig2 = plt.figure(figsize=(10, 6))
        ax3 = fig2.add_subplot(111, projection="3d")

        ax3.set_title(f"Embedding (E={E}, tau={tau})")

        ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.arange(len(X)), cmap="viridis", s=3)  # type: ignore
        ax3.set_xlabel("X(t)")
        ax3.set_ylabel(f"X(t-{tau})")
        ax3.set_zlabel(f"X(t-{2 * tau})")  # type: ignore

        ax3.scatter(
            query_points[:-Tp, 0],
            query_points[:-Tp, 1],
            query_points[:-Tp, 2],
            c="black",
            s=5,  # type: ignore
            label="Query Points",
        )
        ax3.scatter(
            query_points[:-Tp, 0][large_diff],
            query_points[:-Tp, 1][large_diff],
            query_points[:-Tp, 2][large_diff],
            c="red",
            s=5,  # type: ignore
            label="Significant Difference",
        )
        ax3.scatter(
            pyedm_predictions[large_diff],
            pyedm_predictions[large_diff],
            pyedm_predictions[large_diff],
            c="blue",
            s=5,  # type: ignore
            label="pyEDM Predictions",
            marker="x",
        )
        ax3.scatter(
            edmkit_predictions[large_diff],
            edmkit_predictions[large_diff],
            edmkit_predictions[large_diff],
            c="red",
            s=5,  # type: ignore
            label="edmkit Predictions",
            marker="x",
        )

        fig2.tight_layout()
        fig2.legend()
        plt.show()

        fig3 = plt.figure(figsize=(10, 6))
        ax4 = fig3.add_subplot(111, projection="3d")
        ax4.set_title("Sample #1 with Significant Difference (with neighbors)")

        pyedm_result_object = pyEDM.Simplex(
            dataFrame=df, lib=lib, pred=pred, E=E, tau=-tau, columns="value", target="value", Tp=Tp, verbose=False, returnObject=True
        )
        print(pyedm_result_object.libOverlap)

        query_point = query_points[:-Tp][large_diff][0]

        pyedm_neighbor_indices = pyedm_result_object.knn_neighbors[:-Tp][large_diff][0]

        D = pairwise_distance(tensor.Tensor(query_points, dtype=tensor.dtypes.float32), tensor.Tensor(X, dtype=tensor.dtypes.float32)).numpy()
        D = np.sqrt(D)

        k: int = X.shape[1] + 1
        neighbor_indices = np.zeros((len(query_points), k))

        for i in range(len(query_points)):
            # find k nearest neighbors
            indices, distances = topk(D[i], k, largest=False)
            neighbor_indices[i] = indices

        edmkit_neighbor_index = neighbor_indices[:-Tp][large_diff][0].astype(int)

        pyedm_neighbors = X[pyedm_neighbor_indices]
        edmkit_neighbors = X[edmkit_neighbor_index]

        ax4.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.arange(len(X)), cmap="viridis", s=3, zorder=1)  # type: ignore
        ax4.scatter(
            query_point[0],
            query_point[1],
            query_point[2],
            c="black",
            s=10,  # type: ignore
            label="Query Point",
            zorder=2,
        )
        ax4.scatter(
            edmkit_neighbors[:, 0],
            edmkit_neighbors[:, 1],
            edmkit_neighbors[:, 2],
            c="red",
            s=10,  # type: ignore
            label="edmkit Neighbors",
            marker="x",
            zorder=3,
        )
        ax4.scatter(
            pyedm_neighbors[:, 0],
            pyedm_neighbors[:, 1],
            pyedm_neighbors[:, 2],
            c="blue",
            s=10,  # type: ignore
            label="pyEDM Neighbors",
            marker="x",
            zorder=4,
        )

        fig3.tight_layout()
        fig3.legend()
        plt.show()

        try:
            assert np.abs(pyedm_rmse - edmkit_rmse) < 1e-6, f"RMSE: pyEDM {pyedm_rmse}, edmkit {edmkit_rmse}, diff {np.abs(pyedm_rmse - edmkit_rmse)}"
        except AssertionError as e:
            print(f"AssertionError: {e}")


if __name__ == "__main__":
    main()
