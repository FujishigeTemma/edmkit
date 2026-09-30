"""Run the profile-based selection with uniform and defect weighting on every benchmark file."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from selection import horizons, select

MAX_LAG = 100

if __name__ == "__main__":
    data, out, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(data.glob("*.npz")):
        d = np.load(path)
        x, width = d["train"], int(d["width"])
        candidates = np.array([[i, lag] for i in range(x.shape[1]) for lag in range(MAX_LAG + 1)])
        outputs = horizons(x.shape[1], 4 * width)
        for weighting in ("uniform", "defect"):
            start = time.time()
            selected, history = select(candidates, outputs, x, k=k, width=width, weighting=weighting, max_cycles=10)
            result = {"coordinates": selected.tolist(), "history": history, "seconds": time.time() - start}
            (out / f"{path.stem}.{weighting}.json").write_text(json.dumps(result))
            print(path.stem, weighting, selected.tolist(), f"{result['seconds']:.0f}s", flush=True)
