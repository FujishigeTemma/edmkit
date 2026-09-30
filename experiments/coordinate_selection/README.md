# Coordinate selection from pointwise spread profiles

Prototype of a greedy selection of input delay coordinates that keeps the spread of the outputs per query
point and per horizon (`edmkit.spread.spread`) and weights each point by the noise amplification of the
current coordinates there, compared with PECUZAL (`pecuzal-embedding`).

```bash
uv run python data.py DIR/data                       # trajectories: 2000 training and 5000 held-out samples
uv run python run_selection.py DIR/data DIR/selection10 10
uv run python run_selection.py DIR/data DIR/selection5 5
PY39/bin/python run_pecuzal.py DIR/data DIR/pecuzal  # separate environment: pecuzal-embedding pins numpy<2
uv run python compare.py DIR                         # evaluation on the held-out samples
```

Run the `uv` commands with `PYTHONPATH=experiments/coordinate_selection`. `results/` holds the outputs of one run;
`results/table.txt` is the output of `compare.py`.

PECUZAL ran with `econ=True` and otherwise default parameters, lags 0 to 100 and the same Theiler width as the prototype.
The training length is 2000 because one PECUZAL run on 5000 samples did not finish in 30 minutes.
