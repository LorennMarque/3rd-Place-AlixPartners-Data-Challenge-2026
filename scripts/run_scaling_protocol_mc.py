#!/usr/bin/env python3
"""Corre el MC de escalado (130) y cachea CSVs en outputs/tables/."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import SUBMISSION_COLS  # noqa: E402
from scaling_launch import run_simulation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-struct", type=int, default=60)
    ap.add_argument("--n-oracle", type=int, default=15)
    ap.add_argument("--n-sens", type=int, default=20)
    ap.add_argument("--max-launches", type=int, default=500)
    ap.add_argument("--checkpoints", default="50,100,250,500")
    ap.add_argument("--oracle-tl", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--resume-oracle",
        action="store_true",
        help="Reutiliza 130_scaling_checkpoints_struct.csv + opt_ind/growth y solo corre oracle+sens",
    )
    args = ap.parse_args()

    if args.smoke:
        args.n_struct = 12
        args.n_oracle = 4
        args.n_sens = 6
        args.max_launches = 100
        args.checkpoints = "50,100"
        args.oracle_tl = 15.0

    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    paths = {
        "ckpt": tables / "130_scaling_checkpoints.csv",
        "growth": tables / "130_scaling_growth.csv",
        "summary": tables / "130_scaling_mc_summary.csv",
        "regret": tables / "130_scaling_regret.csv",
        "structural": tables / "130_scaling_structural.csv",
        "opt_ind": tables / "130_scaling_opt_vs_ind.csv",
    }
    if paths["ckpt"].exists() and not args.force:
        print(f"Cache existe ({paths['ckpt']}). Usa --force para regenerar.")
        return

    ops = pd.read_csv(ROOT / "data/raw/operaciones_planta.csv")
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    products_truth = pd.read_csv(ROOT / "data/processed/products_truth.csv")
    products_truth["codigo_producto"] = products_truth["codigo_producto"].astype(str).str.strip()
    joined = pd.read_csv(ROOT / "data/processed/producto_joined_cajas_clean.csv")
    joined["codigo_producto"] = joined["codigo_producto"].astype(str).str.strip()
    cat_cols = [
        c
        for c in [
            "categoria",
            "subcategoria",
            "tipo_proyecto",
            "tamaño_corte",
            "ingrediente_forma",
            "cantidad_paquetes",
            "tamaño_paquete_peso",
        ]
        if c in joined.columns
    ]
    products_truth = products_truth.merge(
        joined[["codigo_producto"] + cat_cols], on="codigo_producto", how="left"
    )

    optima = pd.read_csv(tables / "04_mark16_best_+10.11098.csv")[SUBMISSION_COLS].copy()
    indep = pd.read_csv(ROOT / "outputs/mark17/mark17_from_minbox.csv")[SUBMISSION_COLS].copy()
    for df in (optima, indep):
        df["codigo_producto"] = df["codigo_producto"].astype(str).str.strip()

    t0 = time.time()
    ckpt_rows = []
    growth_rows = []
    struct_path = tables / "130_scaling_checkpoints_struct.csv"
    growth_path_tmp = tables / "130_scaling_growth_partial.csv"

    if args.resume_oracle and struct_path.exists():
        print(f"=== Resume desde {struct_path} ===", flush=True)
        ckpt_struct = pd.read_csv(struct_path)
        ckpt_rows.append(ckpt_struct)
        if paths["opt_ind"].exists():
            opt_ind = pd.read_csv(paths["opt_ind"])
        else:
            opt_ind = pd.DataFrame()
        if growth_path_tmp.exists():
            growth_rows.append(pd.read_csv(growth_path_tmp))
        elif paths["growth"].exists():
            growth_rows.append(pd.read_csv(paths["growth"]))
    else:
        # --- Comparación breve Óptima vs Indep (1 trayectoria estructural) ---
        print("=== Opt vs Ind (1 trayectoria M1, routine) ===", flush=True)
        opt_ind_rows = []
        for name, sol in [("optima", optima), ("independiente", indep)]:
            sim = run_simulation(
                sol,
                products_truth,
                ops,
                n_launches=args.max_launches,
                seed=42,
                family="routine",
                demand_mode="cannibal",
                policy="m1",
                checkpoints=checkpoints,
                run_oracle=False,
            )
            for _, r in sim["checkpoint_rows"].iterrows():
                opt_ind_rows.append({**r.to_dict(), "portfolio": name})
            g = sim["growth"].copy()
            g["portfolio"] = name
            growth_rows.append(g)
        opt_ind = pd.DataFrame(opt_ind_rows)
        opt_ind.to_csv(paths["opt_ind"], index=False)

        # --- Structural MC Independiente (m1 + pallet), routine ---
        print(f"=== Structural Indep n={args.n_struct} ===", flush=True)
        for i in range(args.n_struct):
            seed = 1000 + i
            for policy in ("m1", "pallet"):
                sim = run_simulation(
                    indep,
                    products_truth,
                    ops,
                    n_launches=args.max_launches,
                    seed=seed,
                    family="routine",
                    demand_mode="cannibal",
                    policy=policy,
                    checkpoints=checkpoints,
                    run_oracle=False,
                )
                ckpt_rows.append(sim["checkpoint_rows"])
                growth_rows.append(sim["growth"])
            if (i + 1) % 5 == 0:
                print(f"  struct {i+1}/{args.n_struct}", flush=True)

        # Guardar structural intermedio (por si el oracle falla a mitad)
        ckpt_struct = pd.concat(ckpt_rows, ignore_index=True)
        ckpt_struct.to_csv(struct_path, index=False)
        pd.concat(growth_rows, ignore_index=True).to_csv(growth_path_tmp, index=False)
        print(f"Structural parcial: {struct_path}", flush=True)

    # --- Oracle subsample ---
    print(f"=== Oracle Mark17 n={args.n_oracle} tl={args.oracle_tl}s ===", flush=True)
    for i in range(args.n_oracle):
        seed = 1000 + i  # overlap with first structural seeds
        for policy in ("m1", "pallet"):
            print(f"  oracle seed={seed} policy={policy}", flush=True)
            try:
                sim = run_simulation(
                    indep,
                    products_truth,
                    ops,
                    n_launches=args.max_launches,
                    seed=seed,
                    family="routine",
                    demand_mode="cannibal",
                    policy=policy,
                    checkpoints=checkpoints,
                    run_oracle=True,
                    oracle_tl=args.oracle_tl,
                    oracle_workers=args.workers,
                )
                ckpt_rows.append(sim["checkpoint_rows"])
                n_ok = int(sim["checkpoint_rows"]["oracle_ok"].sum())
                print(f"    oracle_ok={n_ok}/{len(sim['checkpoint_rows'])}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"    [skip] seed={seed} policy={policy}: {exc}", flush=True)
        print(f"  oracle done {i+1}/{args.n_oracle}", flush=True)

    # --- Sensibilidades (structural) ---
    print(f"=== Sensitivities n={args.n_sens} ===", flush=True)
    for family, demand_mode in [
        ("drift", "cannibal"),
        ("outlier", "cannibal"),
        ("routine", "additive"),
    ]:
        for i in range(args.n_sens):
            seed = 5000 + i
            sim = run_simulation(
                indep,
                products_truth,
                ops,
                n_launches=args.max_launches,
                seed=seed,
                family=family,
                demand_mode=demand_mode,
                policy="pallet",
                checkpoints=checkpoints,
                run_oracle=False,
            )
            ckpt_rows.append(sim["checkpoint_rows"])
            growth_rows.append(sim["growth"])

    ckpt = pd.concat(ckpt_rows, ignore_index=True)
    # Prefer oracle-enriched rows when available (dedupe keep last)
    ckpt = ckpt.sort_values(["family", "demand_mode", "policy", "seed", "t"])
    ckpt = ckpt.drop_duplicates(
        ["family", "demand_mode", "policy", "seed", "t"], keep="last"
    )
    growth = pd.concat(growth_rows, ignore_index=True)

    ckpt.to_csv(paths["ckpt"], index=False)
    growth.to_csv(paths["growth"], index=False)

    # Summaries
    def qstats(g: pd.DataFrame, col: str) -> dict:
        s = g[col].dropna()
        if s.empty:
            return {f"{col}_p50": np.nan, f"{col}_p10": np.nan, f"{col}_p90": np.nan}
        return {
            f"{col}_p50": float(s.median()),
            f"{col}_p10": float(s.quantile(0.10)),
            f"{col}_p90": float(s.quantile(0.90)),
        }

    sum_rows = []
    for keys, g in ckpt.groupby(["family", "demand_mode", "policy", "t"]):
        family, demand_mode, policy, t = keys
        row = {
            "family": family,
            "demand_mode": demand_mode,
            "policy": policy,
            "t": int(t),
            "n": len(g),
        }
        for col in [
            "tipos_local",
            "delta_tipos",
            "p_nueva",
            "tipos_per_100",
            "cost_local",
            "regret",
            "regret_pct",
            "util_local",
        ]:
            row.update(qstats(g, col))
        sum_rows.append(row)
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(paths["summary"], index=False)

    regret = ckpt.dropna(subset=["regret"]).copy()
    regret.to_csv(paths["regret"], index=False)

    structural = ckpt[
        [
            "seed",
            "family",
            "demand_mode",
            "policy",
            "t",
            "tipos_ini",
            "tipos_local",
            "delta_tipos",
            "p_nueva",
            "pct_catalogo_inicial",
            "tipos_per_100",
        ]
    ].copy()
    structural.to_csv(paths["structural"], index=False)

    print(f"Listo en {time.time()-t0:.1f}s")
    for k, p in paths.items():
        print(f"  {k}: {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
