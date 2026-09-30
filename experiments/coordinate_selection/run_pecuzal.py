"""Run PECUZAL (pecuzal-embedding) on every benchmark file. Runs in its own environment: the package pins numpy<2."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from pecuzal_embedding import pecuzal_embedding

MAX_LAG = 100

if __name__ == "__main__":
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(data.glob("*.npz"), reverse=len(sys.argv) > 3):
        target = out / f"{path.stem}.json"
        if target.exists():
            continue
        d = np.load(path)
        x = d["train"]
        start = time.time()
        _, taus, variables, Ls, _ = pecuzal_embedding(x[:, 0] if x.shape[1] == 1 else x, taus=range(MAX_LAG + 1), theiler=int(d["width"]), econ=True)
        # PECUZAL reads s(t + tau); the same vectors indexed by their latest time have lags max(tau) - tau.
        coordinates = [[int(i), int(max(taus) - tau)] for i, tau in zip(variables, taus)]
        target.write_text(
            json.dumps({"coordinates": coordinates, "taus": [int(v) for v in taus], "Ls": [float(v) for v in Ls], "seconds": time.time() - start})
        )
        print("DONE", path.stem, coordinates, f"{time.time() - start:.0f}s", flush=True)
