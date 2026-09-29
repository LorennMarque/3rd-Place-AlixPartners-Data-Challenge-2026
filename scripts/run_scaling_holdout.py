#!/usr/bin/env python3
"""Repeated holdout 15% — Actual vs Independiente (Mark17 minbox), 5 seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scaling_holdout import run_holdout_repetition  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--mark17-tl", type=float, default=30.0)
    ap.add_argument("--force-mark17", action="store_true")
    ap.add_argument("--seed0", type=int, default=100)
    args = ap.parse_args()

    tables = ROOT / "outputs" / "tables"
    cache = ROOT / "outputs" / "mark17" / "holdout_train"
    tables.mkdir(parents=True, exist_ok=True)

    pt = pd.read_csv(ROOT / "data/processed/products_truth.csv")
    ops = pd.read_csv(ROOT / "data/raw/operaciones_planta.csv")
    joined = pd.read_csv(ROOT / "data/processed/producto_joined_cajas_clean.csv")

    summaries, details = [], []
    for i in range(args.seeds):
        seed = args.seed0 + i
        print(f"=== seed={seed} holdout={args.holdout_frac:.0%} tl={args.mark17_tl}s ===", flush=True)
        out = run_holdout_repetition(
            seed=seed,
            products_truth=pt,
            ops=ops,
            joined=joined,
            holdout_frac=args.holdout_frac,
            mark17_tl=args.mark17_tl,
            mark17_cache_dir=cache,
            force_mark17=args.force_mark17,
        )
        summaries.append(out["summary_rows"])
        details.append(out["detail"])
        print(out["summary_rows"].to_string(index=False), flush=True)

    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)
    summary.to_csv(tables / "130_scaling_holdout_summary.csv", index=False)
    detail.to_csv(tables / "130_scaling_holdout_detail.csv", index=False)
    print("saved", tables / "130_scaling_holdout_summary.csv")
    print("saved", tables / "130_scaling_holdout_detail.csv")


if __name__ == "__main__":
    main()
