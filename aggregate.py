#!/usr/bin/env python3
"""Aggregate data/bench_results.csv into median tables + speed ratios."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data" / "bench_results.csv"
GROUPS = ["raw-causal", "centered-causal", "regression"]


def main():
    df = pd.read_csv(DATA)
    g = (df.groupby(["group", "framework", "model", "n_train"])
           .agg(train_ms=("train_ms", "median"),
                predict_ms=("predict_ms", "median"),
                rmse=("rmse", "median"))
           .reset_index())
    for grp in GROUPS:
        sub = g[g.group == grp].sort_values(["n_train", "train_ms"])
        if sub.empty:
            continue
        print(f"\n===== {grp} =====")
        print(sub[["framework", "model", "n_train", "train_ms", "predict_ms", "rmse"]]
              .to_string(index=False))

    print("\n\n### training time relative to this project's fastest model per group/size ###")
    for grp in GROUPS:
        for n in sorted(g.n_train.unique()):
            sub = g[(g.group == grp) & (g.n_train == n)]
            if sub.empty:
                continue
            mine = sub[sub.framework == "scala-thisproject"].train_ms.min()
            print(f"\n{grp}  n_train={n}   (this project baseline = {mine:,.0f} ms)")
            for r in sub.itertuples():
                tag = "  <-- this project" if r.framework == "scala-thisproject" else ""
                print(f"   {r.framework:18s} {r.model.split('(')[0].strip():34s} "
                      f"{r.train_ms:9,.0f} ms  ({r.train_ms/mine:4.2f}x){tag}")


if __name__ == "__main__":
    main()
