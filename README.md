# kausaali-benchmarks

Cross-framework speed and accuracy benchmarks for
[kausaali](https://github.com/vilhomaa/kausaali): the Scala forests are timed against
[grf](https://grf-labs.github.io/grf/) (R) and
[EconML](https://econml.azurewebsites.net/) (Python) on byte-identical synthetic
datasets.

This is a standalone project — it is **not** part of the published library.

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
