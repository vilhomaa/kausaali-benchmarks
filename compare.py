#!/usr/bin/env python3
"""
Compare Scala CausalForest vs EconML on the same benchmark dataset.

Run the Scala benchmark first:
    sbt "runMain benchmark.ScalaBenchmark"

Then run this script:
    python3 compare.py

Two EconML variants are evaluated:
  - econml.grf.CausalForest   : direct honest causal forest (Wager & Athey 2018),
                                 the closest algorithmic equivalent to the Scala impl.
  - econml.dml.CausalForestDML: causal forest with double-ML nuisance debiasing;
                                 EconML's flagship estimator, stronger on observational data.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# ── Benchmark parameters (must match ScalaBenchmark.scala) ───────────────────

N_TREES    = 200
MAX_DEPTH  = 15
MIN_LEAF   = 5
N_FEATURES = 10


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    mse   = float(np.mean((pred - true) ** 2))
    rmse  = float(np.sqrt(mse))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2    = float(1 - np.sum((true - pred) ** 2) / ss_tot)
    r     = float(np.corrcoef(pred, true)[0, 1])
    return {"MSE": mse, "RMSE": rmse, "R²": r2, "Pearson r": r}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(path: Path):
    df   = pd.read_csv(path)
    xcols = [c for c in df.columns if c.startswith("x")]
    X    = df[xcols].to_numpy(dtype=np.float64)
    T    = df["treatment"].to_numpy(dtype=np.float64)
    Y    = df["label"].to_numpy(dtype=np.float64)
    tau  = df["true_hte"].to_numpy(dtype=np.float64)
    return X, T, Y, tau


# ── EconML runners ────────────────────────────────────────────────────────────

def run_grf(X_tr, T_tr, Y_tr, X_te):
    """econml.grf.CausalForest — direct honest causal forest."""
    from econml.grf import CausalForest

    model = CausalForest(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        honest=True,          # matches Scala's honest splitting
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(X_tr, T_tr, Y_tr)
    train_ms = (time.perf_counter() - t0) * 1_000

    t0 = time.perf_counter()
    tau_hat = model.predict(X_te).ravel()
    pred_ms = (time.perf_counter() - t0) * 1_000

    return tau_hat, train_ms, pred_ms


def run_dml(X_tr, T_tr, Y_tr, X_te):
    """econml.dml.CausalForestDML — causal forest with double-ML debiasing."""
    from econml.dml import CausalForestDML

    model = CausalForestDML(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(Y_tr, T_tr, X=X_tr)
    train_ms = (time.perf_counter() - t0) * 1_000

    t0 = time.perf_counter()
    tau_hat = model.effect(X_te).ravel()
    pred_ms = (time.perf_counter() - t0) * 1_000

    return tau_hat, train_ms, pred_ms


# ── Pretty-printing ───────────────────────────────────────────────────────────

def print_table(rows: list[tuple], headers: list[str]) -> None:
    widths = [
        max(len(str(h)), max(len(str(row[i])) for row in rows))
        for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def fmt_ms(ms: float) -> str:
    return f"{ms:,.1f}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Guard: require Scala outputs
    missing = [p for p in [
        DATA_DIR / "train.csv",
        DATA_DIR / "test.csv",
        DATA_DIR / "scala_predictions.csv",
        DATA_DIR / "scala_timing.csv",
    ] if not p.exists()]
    if missing:
        print("Missing Scala outputs:")
        for p in missing:
            print(f"  {p}")
        print('\nRun:  sbt "runMain benchmark.ScalaBenchmark"')
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading datasets...")
    X_tr, T_tr, Y_tr, _        = load_dataset(DATA_DIR / "train.csv")
    X_te, T_te, Y_te, tau_true = load_dataset(DATA_DIR / "test.csv")

    scala_df  = pd.read_csv(DATA_DIR / "scala_predictions.csv")
    tau_scala = scala_df["scala_pred"].to_numpy(dtype=np.float64)

    timing_df     = pd.read_csv(DATA_DIR / "scala_timing.csv")
    timing        = dict(zip(timing_df["phase"], timing_df["milliseconds"]))
    scala_train_ms = timing.get("training",   float("nan"))
    scala_pred_ms  = timing.get("prediction", float("nan"))

    n_tr, n_feat = X_tr.shape
    n_te         = len(X_te)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  CAUSAL FOREST BENCHMARK")
    print(f"{'='*64}")
    print(f"  N train    : {n_tr:,}")
    print(f"  N test     : {n_te:,}")
    print(f"  N features : {n_feat}  (all continuous; features 3–{n_feat} are noise)")
    print(f"  N trees    : {N_TREES}")
    print(f"  Max depth  : {MAX_DEPTH}")
    print(f"  True HTE   : τ(x) = x₃ × −2  (heterogeneous in feature index 2)")
    print(f"{'='*64}\n")

    # ── Scala results (pre-computed) ───────────────────────────────────────────
    m_scala = compute_metrics(tau_scala, tau_true)

    acc_rows  = []
    time_rows = []

    acc_rows.append((
        "Scala CausalForest",
        f"{m_scala['RMSE']:.4f}",
        f"{m_scala['R²']:.4f}",
        f"{m_scala['Pearson r']:.4f}",
    ))
    time_rows.append((
        "Scala CausalForest",
        fmt_ms(scala_train_ms),
        fmt_ms(scala_pred_ms),
    ))

    # ── EconML GRF ────────────────────────────────────────────────────────────
    tau_grf = None
    try:
        print("Training econml.grf.CausalForest (honest=True) ...")
        tau_grf, grf_tr_ms, grf_pr_ms = run_grf(X_tr, T_tr, Y_tr, X_te)
        print(f"  Done. Train: {fmt_ms(grf_tr_ms)} ms  |  Predict: {fmt_ms(grf_pr_ms)} ms")

        m_grf = compute_metrics(tau_grf, tau_true)
        acc_rows.append((
            "EconML GRF CausalForest",
            f"{m_grf['RMSE']:.4f}",
            f"{m_grf['R²']:.4f}",
            f"{m_grf['Pearson r']:.4f}",
        ))
        time_rows.append((
            "EconML GRF CausalForest",
            fmt_ms(grf_tr_ms),
            fmt_ms(grf_pr_ms),
        ))
        pd.DataFrame({"econml_grf_pred": tau_grf, "true_hte": tau_true}).to_csv(
            DATA_DIR / "econml_grf_predictions.csv", index=False
        )
    except Exception as exc:
        print(f"  econml.grf.CausalForest failed: {exc}")

    # ── EconML DML ────────────────────────────────────────────────────────────
    tau_dml = None
    try:
        print("Training econml.dml.CausalForestDML ...")
        tau_dml, dml_tr_ms, dml_pr_ms = run_dml(X_tr, T_tr, Y_tr, X_te)
        print(f"  Done. Train: {fmt_ms(dml_tr_ms)} ms  |  Predict: {fmt_ms(dml_pr_ms)} ms")

        m_dml = compute_metrics(tau_dml, tau_true)
        acc_rows.append((
            "EconML CausalForestDML",
            f"{m_dml['RMSE']:.4f}",
            f"{m_dml['R²']:.4f}",
            f"{m_dml['Pearson r']:.4f}",
        ))
        time_rows.append((
            "EconML CausalForestDML",
            fmt_ms(dml_tr_ms),
            fmt_ms(dml_pr_ms),
        ))
        pd.DataFrame({"econml_dml_pred": tau_dml, "true_hte": tau_true}).to_csv(
            DATA_DIR / "econml_dml_predictions.csv", index=False
        )
    except Exception as exc:
        print(f"  CausalForestDML failed: {exc}")

    # ── Cross-model agreement ─────────────────────────────────────────────────
    cross_rows = []
    if tau_grf is not None:
        r_sg = float(np.corrcoef(tau_scala, tau_grf)[0, 1])
        cross_rows.append(("Scala vs EconML GRF",    f"{r_sg:.4f}"))
    if tau_dml is not None:
        r_sd = float(np.corrcoef(tau_scala, tau_dml)[0, 1])
        cross_rows.append(("Scala vs EconML DML",    f"{r_sd:.4f}"))
    if tau_grf is not None and tau_dml is not None:
        r_gd = float(np.corrcoef(tau_grf, tau_dml)[0, 1])
        cross_rows.append(("EconML GRF vs EconML DML", f"{r_gd:.4f}"))

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  PREDICTIVE ACCURACY  vs true HTE τ(x) = x₃ × −2")
    print("  (lower RMSE / higher R² and Pearson r = better)")
    print(f"{'─'*64}")
    print_table(acc_rows, ["Model", "RMSE", "R²", "Pearson r"])

    print(f"\n{'─'*64}")
    print("  SPEED  (wall-clock, single run — lower = faster)")
    print(f"{'─'*64}")
    print_table(time_rows, ["Model", "Train (ms)", "Predict (ms)"])

    if cross_rows:
        print(f"\n{'─'*64}")
        print("  CROSS-MODEL AGREEMENT  (Pearson r between predictions)")
        print(f"{'─'*64}")
        print_table(cross_rows, ["Pair", "Pearson r"])

    print()
    print("Notes:")
    print("  * Scala timing includes JVM warm-up; EconML timing is Python wall-clock.")
    print("  * EconML GRF: direct honest causal forest — closest to Scala's algorithm.")
    print("  * EconML DML: adds double-ML nuisance debiasing (cross-fitting overhead).")
    print("  * All models trained with identical n_trees / max_depth / min_leaf settings.")
    print()


if __name__ == "__main__":
    main()
