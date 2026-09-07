#!/usr/bin/env python3
"""
EconML side of the cross-framework forest benchmark.

Reads the byte-identical datasets written by benchmark.GrfBenchmark
(train_<N>.csv / test.csv) and appends timing rows to bench_results.csv.

Models:
  econml.grf.CausalForest    -> group "raw-causal"       (no orthogonalization)
  econml.grf.RegressionForest -> group "regression"
  econml.dml.CausalForestDML  -> group "centered-causal"  (5-fold DML cross-fitting)

Run after the Scala step:  .venv/bin/python bench_econml.py
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from econml.grf import CausalForest, RegressionForest
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path(__file__).parent / "data"
N_TRAINS = [5_000, 20_000]
N_TREES  = 500
MAX_DEPTH = 30
MIN_LEAF = 5
MTRY = 10
SAMPLE_FRAC = 0.5
REPS = 3
SEED = 42

RESULTS = DATA_DIR / "bench_results.csv"


def load(path):
    df = pd.read_csv(path)
    xcols = [c for c in df.columns if c.startswith("x")]
    X = df[xcols].to_numpy(np.float64)
    T = df["treatment"].to_numpy(np.float64)
    Y = df["label"].to_numpy(np.float64)
    tau = df["true_hte"].to_numpy(np.float64)
    return X, T, Y, tau


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def append_row(w, model, group, n_train, rep, train_ms, predict_ms, err):
    w.writerow(["econml", model, group, n_train, N_TREES, rep,
                f"{train_ms:.3f}", f"{predict_ms:.3f}", f"{err:.5f}"])
    print(f"  [{model:<34s} n={n_train:6d} rep={rep}] "
          f"train={train_ms:10,.1f} ms  predict={predict_ms:8,.1f} ms  rmse={err:.4f}")


def run_grf_causal(Xtr, Ttr, Ytr, Xte):
    m = CausalForest(n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                     max_features=MTRY, max_samples=SAMPLE_FRAC, honest=True,
                     inference=False, n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter(); m.fit(Xtr, Ttr, Ytr); tr = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); pred = m.predict(Xte).ravel(); pr = (time.perf_counter() - t0) * 1e3
    return tr, pr, pred


def run_grf_reg(Xtr, Ytr, Xte):
    m = RegressionForest(n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                         max_features=MTRY, max_samples=SAMPLE_FRAC, honest=True,
                         inference=False, n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter(); m.fit(Xtr, Ytr); tr = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); pred = m.predict(Xte).ravel(); pr = (time.perf_counter() - t0) * 1e3
    return tr, pr, pred


def run_dml(Xtr, Ttr, Ytr, Xte):
    # Nuisance models = plain regression forests; 5-fold cross-fitting to mirror Scala CrossFit(5).
    ny = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    nt = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    m = CausalForestDML(model_y=ny, model_t=nt, cv=5, discrete_treatment=True,
                        n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                        max_features=MTRY, max_samples=SAMPLE_FRAC, honest=True,
                        inference=False, n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter(); m.fit(Ytr, Ttr, X=Xtr); tr = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); pred = m.effect(Xte).ravel(); pr = (time.perf_counter() - t0) * 1e3
    return tr, pr, pred


def main():
    Xte, Tte, Yte, tau_te = load(DATA_DIR / "test.csv")
    mean_te = Xte[:, 0] + 0.5 * Xte[:, 1] - 1.0 * Xte[:, 2]

    print(f"EconML benchmark  nTrees={N_TREES} maxDepth={MAX_DEPTH} minLeaf={MIN_LEAF} "
          f"mtry={MTRY} sampleFrac={SAMPLE_FRAC}\n")

    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        for n in N_TRAINS:
            print(f"════════ N_train = {n} ════════")
            Xtr, Ttr, Ytr, _ = load(DATA_DIR / f"train_{n}.csv")
            # warm-up (imports / numba / joblib pools)
            run_grf_causal(Xtr, Ttr, Ytr, Xte[:10])
            for rep in range(1, REPS + 1):
                tr, pr, pred = run_grf_causal(Xtr, Ttr, Ytr, Xte)
                append_row(w, "grf.CausalForest", "raw-causal", n, rep, tr, pr, rmse(pred, tau_te))
                f.flush()

                tr, pr, pred = run_grf_reg(Xtr, Ytr, Xte)
                append_row(w, "grf.RegressionForest", "regression", n, rep, tr, pr, rmse(pred, mean_te))
                f.flush()

                tr, pr, pred = run_dml(Xtr, Ttr, Ytr, Xte)
                append_row(w, "dml.CausalForestDML", "centered-causal", n, rep, tr, pr, rmse(pred, tau_te))
                f.flush()

    print(f"\nAppended to {RESULTS}")


if __name__ == "__main__":
    main()
