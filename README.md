# kausaali-benchmarks

Cross-framework speed and accuracy benchmarks for
[kausaali](https://github.com/vilhomaa/kausaali): the Scala forests are timed against
[grf](https://grf-labs.github.io/grf/) (R) and
[EconML](https://econml.azurewebsites.net/) (Python) on byte-identical synthetic
datasets.

This is a standalone project — it is **not** part of the published library.

## Benchmarks

Cross-framework comparison against R [`grf`](https://grf-labs.github.io/grf/) 2.3.2 and
Python [EconML](https://econml.azurewebsites.net/) 0.16.0 on byte-identical synthetic data
(RCT, `p = 10` Gaussian features, `τ(x) = −2·x₃`, additive Gaussian noise).

*500 trees · honest fraction 0.5 · subsample 0.5 · `mtry = p = 10` · min leaf 5 ·
median of 3 reps · Apple M4 Pro, JVM 23 / R 4.2.2 / Python 3.13.
`train` / `predict` in ms (predict over 5,000 rows); `RMSE` vs ground truth.*

### Summary (n_train = 20,000)

| Estimand | Implementation | train (ms) | RMSE | vs grf (train / RMSE) |
|---|---|--:|--:|--:|
| Causal, no centering | `GeneralizedRandomForest.causal` | 1,130 | 0.165 | 0.88× / 1.04× |
| | R `grf::causal_forest` | 1,278 | 0.158 | — |
| | EconML `grf.CausalForest` | 3,782 | 0.207 | 2.96× / 1.31× |
| Regression `E[Y\|X]` | `GeneralizedRandomForest.regression` | 856 | 0.280 | 0.63× / 1.05× |
| | R `grf::regression_forest` | 1,352 | 0.266 | — |
| | EconML `grf.RegressionForest` | 2,372 | 0.276 | 1.75× / 1.04× |
| Causal, local centering | `GeneralizedRandomForest.causal` (OOB cross-fit) | 1,986 | 0.121 | 0.99× / 1.06× |
| | R `grf::causal_forest` (Y.hat/W.hat) | 2,013 | 0.114 | — |
| | EconML `CausalForestDML` (5-fold) | 18,389 | 0.128 | 9.1× / 1.12× |

### Causal forest — no local centering

| Framework | Model | n_train | train | predict | RMSE |
|---|---|--:|--:|--:|--:|
| this project | `GeneralizedRandomForest.causal` (α-weighted) | 5,000 | 251 | 77 | 0.261 |
| this project | `GeneralizedRandomForest.causal` (per-tree avg) | 5,000 | 230 | 51 | 0.256 |
| this project | `CausalForest` (exact Athey–Wager 2018) | 5,000 | 274 | 35 | 0.344 |
| R grf | `causal_forest` | 5,000 | 208 | 40 | 0.271 |
| EconML | `grf.CausalForest` | 5,000 | 738 | 56 | 0.299 |
| this project | `GeneralizedRandomForest.causal` (α-weighted) | 20,000 | 1,130 | 149 | 0.165 |
| this project | `GeneralizedRandomForest.causal` (per-tree avg) | 20,000 | 1,456 | 95 | 0.159 |
| this project | `CausalForest` (exact Athey–Wager 2018) | 20,000 | 1,240 | 67 | 0.172 |
| R grf | `causal_forest` | 20,000 | 1,278 | 95 | 0.158 |
| EconML | `grf.CausalForest` | 20,000 | 3,782 | 58 | 0.207 |

### Regression forest — `E[Y|X]`

| Framework | Model | n_train | train | predict | RMSE |
|---|---|--:|--:|--:|--:|
| this project | `GeneralizedRandomForest.regression` | 5,000 | 189 | 30 | 0.373 |
| R grf | `regression_forest` | 5,000 | 313 | 84 | 0.354 |
| EconML | `grf.RegressionForest` | 5,000 | 494 | 44 | 0.368 |
| this project | `GeneralizedRandomForest.regression` | 20,000 | 856 | 65 | 0.280 |
| R grf | `regression_forest` | 20,000 | 1,352 | 255 | 0.266 |
| EconML | `grf.RegressionForest` | 20,000 | 2,372 | 56 | 0.276 |

### Causal forest — local centering (OOB cross-fit of `E[Y|X]`, `E[W|X]`)

| Framework | Model | n_train | train | predict | RMSE |
|---|---|--:|--:|--:|--:|
| this project | OOB cross-fit (default) | 5,000 | 353 | 56 | 0.218 |
| this project | OOB cross-fit (light nuisance) | 5,000 | 356 | 62 | 0.218 |
| R grf | `causal_forest` (centered) | 5,000 | 417 | 42 | 0.218 |
| EconML | `CausalForestDML` (5-fold) | 5,000 | 3,977 | 47 | 0.239 |
| this project | OOB cross-fit (default) | 20,000 | 1,986 | 135 | 0.121 |
| this project | OOB cross-fit (light nuisance) | 20,000 | 1,453 | 138 | 0.119 |
| R grf | `causal_forest` (centered) | 20,000 | 2,013 | 98 | 0.114 |
| EconML | `CausalForestDML` (5-fold) | 20,000 | 18,389 | 56 | 0.128 |

## Layout

| Path | What |
| --- | --- |
| `src/main/scala/benchmark/GrfBenchmark.scala` | Times every forest in kausaali; writes the shared datasets + `data/bench_results.csv`. |
| `src/main/scala/benchmark/ScalaBenchmark.scala` | Smaller single-forest run consumed by `compare.py`. |
| `bench_grf.R` | R `grf` side; appends rows to `bench_results.csv`. |
| `bench_econml.py` | Python EconML side; appends rows to `bench_results.csv`. |
| `compare.py` | Head-to-head Scala vs EconML table from `ScalaBenchmark` output. |
| `aggregate.py` | Median tables + speed ratios from `bench_results.csv`. |
| `run.sh` | End-to-end: runs all four steps in order. |
| `data/` | Generated datasets and results (git-ignored). |

## Dependency on the library

`build.sbt` uses an sbt **source dependency** on `../kausaali`, so the two
repositories must sit side by side:

```
~/workspace/omat/kausaali
~/workspace/omat/kausaali-benchmarks
```

No publishing step is needed — sbt compiles the library from source. To use a
binary artifact instead, run `sbt publishLocal` in the library and swap the
`.dependsOn(...)` in `build.sbt` for a `libraryDependencies` line (see the comment
there).

## Running

Everything at once:

```bash
bash run.sh
```

Or step by step:

```bash
# 1. Scala: generate datasets + time kausaali forests
sbt "runMain benchmark.GrfBenchmark"

# 2. Python / EconML (needs a venv with requirements.txt)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python bench_econml.py

# 3. R / grf  (needs the `grf` package installed)
Rscript bench_grf.R

# 4. Aggregate
.venv/bin/python aggregate.py
```

The single-forest Scala-vs-EconML comparison is separate:

```bash
sbt "runMain benchmark.ScalaBenchmark"
python3 compare.py
```
