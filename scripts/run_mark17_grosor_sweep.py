#!/usr/bin/env python3
"""Mark17 from minbox — sweep por grosor único + mezcla (2.5, 2.7, 3.0).

Para cada política:
  - warm-start = caja mínima (producto + 2·g; en mezcla, g más fino factible)
  - Mark17 demand-free hasta OPTIMAL (surrogate ≈ cota) o time-limit
  - Eval Kaggle plant-tier (misma que truth table / notebook 200)

Salida:
  outputs/mark17/grosor/mark17_minbox_g{tag}.csv
  outputs/mark17/grosor/warm_minbox_g{tag}.csv
  outputs/tables/200_mark17_grosor_minbox.csv

Ejemplo:
  PYTHONPATH=src python scripts/run_mark17_grosor_sweep.py
  PYTHONPATH=src python scripts/run_mark17_grosor_sweep.py --skip-existing --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate import (  # noqa: E402
    SUBMISSION_COLS,
    calcular_cajas_por_pallet,
    costo_flete_eval,
    factor_precio_por_volumen,
    preparar_eval_planta,
    validate_solution_free,
)
from mark16 import costo_actual_oficial, score_publico  # noqa: E402
from mark17 import run_optimize  # noqa: E402
from scaling_launch import crear_caja_minima  # noqa: E402
from settings import (  # noqa: E402
    EPS,
    OPERACIONES_PLANTA_PATH,
    PLANTAS,
    PRECIO_BASE_GROSOR,
    PRODUCTS_TRUTH_PATH,
)

# Políticas a evaluar: uniques viables + mezcla protocolo fino.
DEFAULT_POLICIES = [
    "3.0",
    "4.1",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "5.0",
    "2.5,2.7,3.0",
]
DEFAULT_POLICIES_ARG = ";".join(DEFAULT_POLICIES)
KNOWN_3MM = "outputs/mark17/mark17_from_minbox.csv"


def parse_policy(text: str) -> list[float]:
    gs = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not gs:
        raise ValueError(f"Política vacía: {text!r}")
    unknown = [g for g in gs if g not in PRECIO_BASE_GROSOR]
    if unknown:
        raise ValueError(f"Grosores sin precio/ECT: {unknown}")
    return gs


def policy_tag(grosores: list[float]) -> str:
    return "-".join(f"{g:g}" for g in grosores)


def eval_kaggle_plant(sol: pd.DataFrame, ops: pd.DataFrame) -> dict:
    df = preparar_eval_planta(sol[SUBMISSION_COLS], ops)
    df = calcular_cajas_por_pallet(df)
    precio_base = (
        df.groupby("tipo")["caja_grosor_mm"].first().round(1).map(PRECIO_BASE_GROSOR)
    )
    if precio_base.isna().any():
        raise ValueError("Grosores sin precio base en evaluación")
    pack = 0.0
    for p in PLANTAS:
        vol = df.groupby("tipo")[f"volumen_producto_planta_{p}"].sum()
        fac = factor_precio_por_volumen(vol)
        pack += float((vol.values * precio_base.loc[vol.index].values * fac).sum())
    flete = float(costo_flete_eval(df))
    return {
        "total": pack + flete,
        "packaging": pack,
        "flete": flete,
        "n_tipos": int(df["tipo"].nunique()),
    }


def minbox_exterior(largo: float, ancho: float, alto: float, grosor: float) -> tuple[float, float, float]:
    """Exterior = producto + 2·g, redondeado a 0.01 y float-safe al volumen.

    `g=4.7` no es exacto en binario: `∏(dims)` puede quedar ~1e-8 por debajo del
    volumen del producto y el validador (EPS=1e-9) lo marca inválido. Si pasa,
    subimos 0.01 mm por eje hasta cumplir volumen.
    """
    g = float(grosor)
    dims = np.array([largo, ancho, alto], dtype=float)
    ext = np.round(dims + 2.0 * g, 2)
    vol = float(np.prod(dims))
    for _ in range(5):
        if float(np.prod(ext - 2.0 * g)) >= vol - EPS:
            break
        ext = np.round(ext + 0.01, 2)
    return float(ext[0]), float(ext[1]), float(ext[2])


def build_warm_minbox(pt: pd.DataFrame, grosores: list[float]) -> pd.DataFrame:
    """Caja mínima por SKU bajo la lista de grosores permitidos (orden = preferencia)."""
    rows = []
    for r in pt.itertuples(index=False):
        prod = {
            "codigo_producto": str(r.codigo_producto).strip(),
            "largo": float(r.largo),
            "ancho": float(r.ancho),
            "alto": float(r.alto),
            "peso_neto_caja": float(r.peso_neto_caja),
        }
        if len(grosores) == 1:
            g = float(grosores[0])
            el, ew, eh = minbox_exterior(prod["largo"], prod["ancho"], prod["alto"], g)
            caja = {
                "caja_grosor_mm": g,
                "caja_exterior_largo": el,
                "caja_exterior_ancho": ew,
                "caja_exterior_alto": eh,
            }
        else:
            # Preferir g fino factible; fallback float-safe si crear_caja_minima
            # tropieza con el mismo ruido binario (p.ej. único candidato 4.7).
            try:
                caja = crear_caja_minima(prod, grosores_candidatos=grosores)
            except ValueError:
                g = float(grosores[-1])
                el, ew, eh = minbox_exterior(
                    prod["largo"], prod["ancho"], prod["alto"], g
                )
                caja = {
                    "caja_grosor_mm": g,
                    "caja_exterior_largo": el,
                    "caja_exterior_ancho": ew,
                    "caja_exterior_alto": eh,
                }
        rows.append(
            {
                "codigo_producto": prod["codigo_producto"],
                "caja_grosor_mm": float(caja["caja_grosor_mm"]),
                "caja_exterior_largo": float(caja["caja_exterior_largo"]),
                "caja_exterior_ancho": float(caja["caja_exterior_ancho"]),
                "caja_exterior_alto": float(caja["caja_exterior_alto"]),
            }
        )
    return pd.DataFrame(rows)


def warm_is_feasible(warm: pd.DataFrame, pt: pd.DataFrame) -> tuple[bool, list[str]]:
    ok, errors, _, _ = validate_solution_free(warm[SUBMISSION_COLS], pt, verbose=False)
    return ok, list(errors or [])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policies",
        default=DEFAULT_POLICIES_ARG,
        help="Políticas separadas por ';'. Cada una: grosores separados por coma.",
    )
    p.add_argument(
        "--reuse-3mm",
        default=KNOWN_3MM,
        help="Si existe, reutiliza esta solución para política 3.0 (sin reoptimizar).",
    )
    p.add_argument("--time-limit", type=float, default=900.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pool", choices=("fast", "full"), default="fast")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-dir", default="outputs/mark17/grosor")
    p.add_argument(
        "--summary-out", default="outputs/tables/200_mark17_grosor_minbox.csv"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Si ya existe la solución CSV, solo re-evalúa y agrega al summary.",
    )
    args = p.parse_args()

    # policies: "3.0;4.1;2.5,2.7,3.0"
    policy_texts = [x.strip() for x in args.policies.split(";") if x.strip()]

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = REPO_ROOT / args.summary_out
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    baseline = float(costo_actual_oficial(ops))

    print(
        f"Mark17 grosor sweep | policies={policy_texts} | "
        f"tl={args.time_limit:.0f}s workers={args.workers}",
        flush=True,
    )
    print(f"Baseline oficial: USD {baseline:,.2f}", flush=True)

    rows: list[dict] = []
    for text in policy_texts:
        grosores = parse_policy(text)
        tag = policy_tag(grosores)
        label = (
            f"Único {grosores[0]:g} mm"
            if len(grosores) == 1
            else "Mezcla (" + ", ".join(f"{g:g}" for g in grosores) + " mm)"
        )
        warm_path = out_dir / f"warm_minbox_g{tag}.csv"
        sol_path = out_dir / f"mark17_minbox_g{tag}.csv"
        print(f"\n=== {label} | grosores={grosores} ===", flush=True)

        # Warm minbox
        try:
            warm = build_warm_minbox(pt, grosores)
        except ValueError as exc:
            print(f"  SKIP warm no factible: {exc}", flush=True)
            rows.append(
                {
                    "policy": text,
                    "label": label,
                    "grosores": ",".join(str(g) for g in grosores),
                    "ok": False,
                    "reason": f"warm_build: {exc}",
                    "solution_path": "",
                }
            )
            continue

        warm[SUBMISSION_COLS].to_csv(warm_path, index=False)
        ok_w, errs_w = warm_is_feasible(warm, pt)
        if not ok_w:
            print(f"  SKIP warm inválido: {errs_w[:3]}", flush=True)
            rows.append(
                {
                    "policy": text,
                    "label": label,
                    "grosores": ",".join(str(g) for g in grosores),
                    "ok": False,
                    "reason": f"warm_invalid: {errs_w[:2]}",
                    "solution_path": str(warm_path.relative_to(REPO_ROOT)),
                }
            )
            continue

        # Reutilizar óptimo 3 mm conocido
        reuse = REPO_ROOT / args.reuse_3mm
        if (
            len(grosores) == 1
            and abs(grosores[0] - 3.0) < 1e-9
            and reuse.exists()
        ):
            sol = pd.read_csv(reuse)[SUBMISSION_COLS].copy()
            sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
            sol[SUBMISSION_COLS].to_csv(sol_path, index=False)
            ev = eval_kaggle_plant(sol, ops)
            dist = (
                sol["caja_grosor_mm"]
                .round(1)
                .value_counts()
                .sort_index()
                .to_dict()
            )
            row = {
                "policy": text,
                "label": label,
                "grosores": "3.0",
                "ok": True,
                "reused": True,
                "optimal_surrogate": True,
                "surrogate": np.nan,
                "bound": np.nan,
                "gap_surrogate": 0.0,
                "elapsed_s": 0.0,
                "n_tipos": ev["n_tipos"],
                "total": ev["total"],
                "packaging": ev["packaging"],
                "flete": ev["flete"],
                "score": score_publico(ev["total"], baseline),
                "dist_grosor": dist,
                "solution_path": str(sol_path.relative_to(REPO_ROOT)),
                "source": str(reuse.relative_to(REPO_ROOT)),
            }
            rows.append(row)
            print(
                f"  REUSE {reuse.name} | USD {ev['total']:,.2f} "
                f"score {row['score']:+.5f} | tipos {ev['n_tipos']}",
                flush=True,
            )
            continue

        if args.skip_existing and sol_path.exists():
            sol = pd.read_csv(sol_path)[SUBMISSION_COLS].copy()
            sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
            ev = eval_kaggle_plant(sol, ops)
            dist = (
                sol["caja_grosor_mm"]
                .round(1)
                .value_counts()
                .sort_index()
                .to_dict()
            )
            row = {
                "policy": text,
                "label": label,
                "grosores": ",".join(str(g) for g in grosores),
                "ok": True,
                "reused": False,
                "optimal_surrogate": np.nan,
                "surrogate": np.nan,
                "bound": np.nan,
                "gap_surrogate": np.nan,
                "elapsed_s": 0.0,
                "n_tipos": ev["n_tipos"],
                "total": ev["total"],
                "packaging": ev["packaging"],
                "flete": ev["flete"],
                "score": score_publico(ev["total"], baseline),
                "dist_grosor": dist,
                "solution_path": str(sol_path.relative_to(REPO_ROOT)),
                "source": "skip_existing",
            }
            rows.append(row)
            print(
                f"  SKIP-EXISTING {sol_path.name} | USD {ev['total']:,.2f} "
                f"score {row['score']:+.5f}",
                flush=True,
            )
            continue

        t0 = time.time()

        def on_progress(e: dict) -> None:
            phase = e.get("phase", "")
            msg = e.get("message", "")
            if e.get("incumbent") or e.get("bound_update") or phase in {
                "pool",
                "pool_done",
                "solving",
                "done",
                "failed",
            }:
                extra = ""
                if e.get("surrogate") is not None:
                    extra += f" surr={e['surrogate']:,.1f}"
                if e.get("bound") is not None:
                    extra += f" bound={e['bound']:,.1f}"
                print(f"  [{phase}] {msg}{extra}", flush=True)

        result = run_optimize(
            warm_start_path=warm_path,
            grosores=grosores,
            time_limit=args.time_limit,
            workers=args.workers,
            out_path=sol_path,
            pool_mode=args.pool,
            top_k=args.top_k,
            eval_official=False,
            on_progress=on_progress,
        )
        elapsed = time.time() - t0
        if not result.get("ok"):
            print(f"  FAIL en {elapsed:.1f}s", flush=True)
            rows.append(
                {
                    "policy": text,
                    "label": label,
                    "grosores": ",".join(str(g) for g in grosores),
                    "ok": False,
                    "reason": "solver_failed",
                    "elapsed_s": elapsed,
                    "solution_path": str(sol_path.relative_to(REPO_ROOT)),
                }
            )
            continue

        sol = result["solution"][SUBMISSION_COLS].copy()
        sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
        sol[SUBMISSION_COLS].to_csv(sol_path, index=False)
        ev = eval_kaggle_plant(sol, ops)
        surr = float(result["surrogate"])
        bound = float(result["bound"])
        gap = abs(surr - bound)
        dist = (
            sol["caja_grosor_mm"].round(1).value_counts().sort_index().to_dict()
        )
        row = {
            "policy": text,
            "label": label,
            "grosores": ",".join(str(g) for g in grosores),
            "ok": True,
            "reused": False,
            "optimal_surrogate": gap <= 1e-3,
            "surrogate": surr,
            "bound": bound,
            "gap_surrogate": gap,
            "elapsed_s": elapsed,
            "n_tipos": int(result["n_tipos"]),
            "total": ev["total"],
            "packaging": ev["packaging"],
            "flete": ev["flete"],
            "score": score_publico(ev["total"], baseline),
            "dist_grosor": dist,
            "solution_path": str(sol_path.relative_to(REPO_ROOT)),
            "source": "mark17_from_minbox",
        }
        rows.append(row)
        print(
            f"  DONE {elapsed:.1f}s | optimal={row['optimal_surrogate']} "
            f"gap={gap:.4f} | USD {ev['total']:,.2f} score {row['score']:+.5f} "
            f"| tipos {row['n_tipos']} | dist={dist}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    # dist_grosor as string for CSV
    if "dist_grosor" in summary.columns:
        summary["dist_grosor"] = summary["dist_grosor"].map(
            lambda d: ""
            if not isinstance(d, dict)
            else ";".join(f"{g}:{n}" for g, n in d.items())
        )
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}", flush=True)
    cols = [
        c
        for c in [
            "policy",
            "ok",
            "optimal_surrogate",
            "total",
            "score",
            "n_tipos",
            "elapsed_s",
            "solution_path",
        ]
        if c in summary.columns
    ]
    print(summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
