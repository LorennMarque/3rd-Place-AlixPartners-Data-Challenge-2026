#!/usr/bin/env python3
"""Re-optimiza Mark17 (warm minbox) bajo alturas de pallet 1800+Δ.

Para cada Δ en {50,100,150,200}:
  - warm-start = mark17_from_minbox (óptimo a 1800 mm)
  - lim = (800, 1200, 1800+Δ)
  - corre hasta OPTIMAL (surrogate ≈ cota) o time-limit

Salida:
  outputs/mark17/altura/mark17_minbox_h{alto}.csv
  outputs/tables/210_mark17_altura_reopt.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate import SUBMISSION_COLS, evaluar_costos_solucion  # noqa: E402
from mark16 import costo_actual_oficial, score_publico  # noqa: E402
from mark17 import run_optimize  # noqa: E402
from settings import OPERACIONES_PLANTA_PATH  # noqa: E402

ALTO_HOY = 1800.0
DEFAULT_DELTAS = [50, 100, 150, 200]
DEFAULT_WARM = "outputs/mark17/mark17_from_minbox.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warm-start", default=DEFAULT_WARM)
    p.add_argument("--deltas", default=",".join(str(d) for d in DEFAULT_DELTAS))
    p.add_argument("--time-limit", type=float, default=900.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pool", choices=("fast", "full"), default="fast")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--grosores", default="3.0")
    p.add_argument("--out-dir", default="outputs/mark17/altura")
    p.add_argument(
        "--summary-out", default="outputs/tables/210_mark17_altura_reopt.csv"
    )
    args = p.parse_args()

    deltas = [int(x.strip()) for x in args.deltas.split(",") if x.strip()]
    grosores = [float(x.strip()) for x in args.grosores.split(",") if x.strip()]
    warm = REPO_ROOT / args.warm_start
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = REPO_ROOT / args.summary_out
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    baseline = costo_actual_oficial(ops)

    fixed = pd.read_csv(warm)[SUBMISSION_COLS].copy()
    fixed["codigo_producto"] = fixed["codigo_producto"].astype(str).str.strip()
    ref_cost = float(
        evaluar_costos_solucion(fixed, ops, lim=(800.0, 1200.0, ALTO_HOY))["resumen"][
            "costo_total_nuevo"
        ].iloc[0]
    )

    rows: list[dict] = []
    print(
        f"Mark17 altura sweep | warm={warm.name} | deltas={deltas} | "
        f"tl={args.time_limit:.0f}s workers={args.workers}",
        flush=True,
    )
    print(f"Ref 1800 mm (fixed minbox): USD {ref_cost:,.2f}", flush=True)

    for d in deltas:
        alto = ALTO_HOY + d
        lim = (800.0, 1200.0, alto)
        sol_path = out_dir / f"mark17_minbox_h{int(alto)}.csv"
        print(f"\n=== Δ=+{d} mm → alto={alto:.0f} mm ===", flush=True)
        t0 = time.time()

        def on_progress(e: dict) -> None:
            msg = e.get("message", "")
            phase = e.get("phase", "")
            if e.get("incumbent") or e.get("bound_update") or phase in {
                "pool",
                "pool_done",
                "solving",
                "done",
                "failed",
            }:
                extra = ""
                if "surrogate" in e and e["surrogate"] is not None:
                    extra += f" surr={e['surrogate']:,.1f}"
                if "bound" in e and e["bound"] is not None:
                    extra += f" bound={e['bound']:,.1f}"
                if "cost" in e and e["cost"] is not None:
                    extra += f" cost={e['cost']:,.0f}"
                print(f"  [{phase}] {msg}{extra}", flush=True)

        result = run_optimize(
            warm_start_path=warm,
            grosores=grosores,
            time_limit=args.time_limit,
            workers=args.workers,
            out_path=sol_path,
            pool_mode=args.pool,
            top_k=args.top_k,
            lim=lim,
            eval_official=True,
            on_progress=on_progress,
        )
        elapsed = time.time() - t0
        if not result.get("ok"):
            print(f"  FAIL en {elapsed:.1f}s", flush=True)
            rows.append(
                {
                    "delta_mm": d,
                    "alto_mm": alto,
                    "ok": False,
                    "elapsed_s": elapsed,
                    "solution_path": str(sol_path),
                }
            )
            continue

        surr = float(result["surrogate"])
        bound = float(result["bound"])
        gap = abs(surr - bound)
        optimal = gap <= 1e-3
        cost = float(result["cost"])
        score = float(result["score"])
        ahorro = ref_cost - cost

        # También eval fijo (mismas cajas minbox) a esta altura.
        fixed_cost = float(
            evaluar_costos_solucion(fixed, ops, lim=lim)["resumen"][
                "costo_total_nuevo"
            ].iloc[0]
        )

        row = {
            "delta_mm": d,
            "alto_mm": alto,
            "ok": True,
            "optimal_surrogate": optimal,
            "surrogate": surr,
            "bound": bound,
            "gap_surrogate": gap,
            "elapsed_s": elapsed,
            "n_tipos": int(result["n_tipos"]),
            "reopt_total": cost,
            "reopt_score": score,
            "ahorro_reopt": ahorro,
            "fixed_total": fixed_cost,
            "ahorro_fixed": ref_cost - fixed_cost,
            "ref_total_1800": ref_cost,
            "baseline": baseline,
            "solution_path": str(sol_path.relative_to(REPO_ROOT)),
        }
        rows.append(row)
        print(
            f"  DONE {elapsed:.1f}s | optimal={optimal} gap={gap:.4f} | "
            f"reopt=${cost:,.0f} (score {score:+.3f}) | "
            f"ahorro_reopt=${ahorro:,.0f} | ahorro_fixed=${row['ahorro_fixed']:,.0f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
