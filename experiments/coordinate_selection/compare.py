"""Evaluate the coordinates chosen by each method on the held-out half of every benchmark trajectory."""

import json
import sys
from pathlib import Path

import numpy as np
from evaluate import common_times, forecast_error, neighbour_inflation

METHODS = {
    "pecuzal": "pecuzal/{}.json",
    "uniform k=10": "selection10/{}.uniform.json",
    "defect k=10": "selection10/{}.defect.json",
    "uniform k=5": "selection5/{}.uniform.json",
    "defect k=5": "selection5/{}.defect.json",
    "defect k=10 @E": "selection10/{}.defect.json",
    "defect k=5 @E": "selection5/{}.defect.json",
}

if __name__ == "__main__":
    root = Path(sys.argv[1])
    rows = []
    for path in sorted((root / "data").glob("*.npz")):
        d = np.load(path)
        x, state, clean, width = d["test"], d["test_state"], d["test_clean"][:, 0], int(d["width"])
        t = common_times(x, 100, 4 * width)
        steps = np.array([width, 4 * width])
        for method, pattern in METHODS.items():
            file = root / pattern.format(path.stem)
            if not file.exists():
                continue
            coordinates = np.array(json.loads(file.read_text())["coordinates"])
            if method.endswith("@E"):  # the first coordinates selected, as many as PECUZAL chose
                reference = root / METHODS["pecuzal"].format(path.stem)
                if not reference.exists():
                    continue
                coordinates = coordinates[: len(json.loads(reference.read_text())["coordinates"])]
            # Compare with the state at whichever of the coordinates' own lags gives the smallest median ratio.
            ratios = [neighbour_inflation(coordinates, x, state, t, k=10, width=width, shift=int(s)) for s in np.unique(coordinates[:, 1])]
            ratio = min(ratios, key=np.median)
            error = forecast_error(coordinates, x, clean, t, steps, width=width)
            rows.append(
                {
                    "case": path.stem.rsplit("-", 1)[0],
                    "seed": int(path.stem.rsplit("-", 1)[1]),
                    "method": method,
                    "E": len(coordinates),
                    "coordinates": coordinates.tolist(),
                    "inflation_median": float(np.median(ratio)),
                    "inflation_p99": float(np.percentile(ratio, 99)),
                    "rmse_short": float(error[0]),
                    "rmse_long": float(error[1]),
                }
            )
    (root / "results.json").write_text(json.dumps(rows, indent=1))

    for row in rows:
        lags = " ".join(f"{i}:{lag}" for i, lag in row["coordinates"])
        print(
            f"{row['case']:16s} {row['seed']} {row['method']:14s} E={row['E']:2d}  infl {row['inflation_median']:.2f}/{row['inflation_p99']:5.2f}  rmse {row['rmse_short']:.4f}/{row['rmse_long']:.4f}  [{lags}]"
        )

    print("\nmean over seeds")
    keys = ("E", "inflation_median", "inflation_p99", "rmse_short", "rmse_long")
    for case in dict.fromkeys(r["case"] for r in rows):
        for method in METHODS:
            sub = [r for r in rows if r["case"] == case and r["method"] == method]
            if sub:
                E, median, p99, short, long = (np.mean([r[key] for r in sub]) for key in keys)
                print(f"{case:16s} {method:14s} n={len(sub)} E={E:.1f}  infl {median:.2f}/{p99:5.2f}  rmse {short:.4f}/{long:.4f}")
