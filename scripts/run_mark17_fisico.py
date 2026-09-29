#!/usr/bin/env python3
"""Mark 17 demand-free bajo fit físico por eje (producto rígido = Mark 19).

Parchea `mark16.intervalos` / `fits_matrix` como Mark 19 y corre Mark 17
(from warm físico minbox).

Ejemplos:
  PYTHONPATH=src python scripts/run_mark17_fisico.py --grosores 3.0 --time-limit 900
  PYTHONPATH=src python scripts/run_mark17_fisico.py --grosores 2.5,2.7,3.0 --time-limit 900
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

import mark16  # noqa: E402
from evaluate import (  # noqa: E402
    SUBMISSION_COLS,
    calcular_cajas_por_pallet,
    costo_flete_eval,
    factor_precio_por_volumen,
    preparar_eval_planta,
    validate_solution_free,
)
from mark16 import costo_actual_oficial, parse_grosores, score_publico  # noqa: E402
from mark17 import run_optimize  # noqa: E402
from mark19 import assert_axis_fit, intervalos_fisicos  # noqa: E402
from settings import (  # noqa: E402
    EPS,
    OPERACIONES_PLANTA_PATH,
    PLANTAS,
    PRECIO_BASE_GROSOR,
    PRODUCTS_TRUTH_PATH,
)

_FITS_LITERAL = mark16.fits_matrix


def _patch_fisico() -> tuple:
    prev_i, prev_f = mark16.intervalos, mark16.fits_matrix
    mark16.intervalos = intervalos_fisicos  # type: ignore[assignment]

    def _fits(boxes, lo, hi, vol_min, peso, grosor):
        fits = _FITS_LITERAL(boxes, lo, hi, vol_min, peso, grosor)
        axis_ok = (boxes[:, None, :] >= lo[None, :, :] - 1e-9).all(axis=2)
        return fits & axis_ok

    mark16.fits_matrix = _fits  # type: ignore[assignment]
    return prev_i, prev_f


def _restore(prev_i, prev_f) -> None:
    mark16.intervalos = prev_i
    mark16.fits_matrix = prev_f


def minbox_exterior(largo, ancho, alto, grosor) -> tuple[float, float, float]:
    g = float(grosor)
    dims = np.array([largo, ancho, alto], dtype=float)
    ext = np.round(dims + 2.0 * g, 2)
    vol = float(np.prod(dims))
    for _ in range(5):
        if float(np.prod(ext - 2.0 * g)) >= vol - EPS:
            break
        ext = np.round(ext + 0.01, 2)
    return float(ext[0]), float(ext[1]), float(ext[2])


def build_warm_fisico(pt: pd.DataFrame, grosores: list[float]) -> pd.DataFrame:
    """Warm minbox físico: interior ≥ producto en cada eje."""
    rows = []
    for r in pt.itertuples(index=False):
        L, W, H = float(r.largo), float(r.ancho), float(r.alto)
        if len(grosores) == 1:
            g = float(grosores[0])
            el, ew, eh = minbox_exterior(L, W, H, g)
        else:
            # g más fino factible con fit físico (caja justa)
            chosen = None
            for g in grosores:
                el, ew, eh = minbox_exterior(L, W, H, g)
                inter = np.array([el, ew, eh]) - 2.0 * g
                if (inter >= np.array([L, W, H]) - 1e-9).all():
                    chosen = (g, el, ew, eh)
                    break
            if chosen is None:
                g = float(grosores[-1])
                el, ew, eh = minbox_exterior(L, W, H, g)
                chosen = (g, el, ew, eh)
            g, el, ew, eh = chosen
        rows.append(
            {
                "codigo_producto": str(r.codigo_producto).strip(),
                "caja_grosor_mm": float(g),
                "caja_exterior_largo": el,
                "caja_exterior_ancho": ew,
                "caja_exterior_alto": eh,
            }
        )
    return pd.DataFrame(rows)


def eval_kaggle_plant(sol: pd.DataFrame, ops: pd.DataFrame) -> dict:
    df = preparar_eval_planta(sol[SUBMISSION_COLS], ops)
    df = calcular_cajas_por_pallet(df)
    precio_base = (
        df.groupby("tipo")["caja_grosor_mm"].first().round(1).map(PRECIO_BASE_GROSOR)
    )
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grosores", default="3.0")
    p.add_argument("--time-limit", type=float, default=900.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pool", choices=("fast", "full"), default="fast")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-dir", default="outputs/mark17/fisico")
    p.add_argument("--best-out", default=None)
    args = p.parse_args()

    grosores = parse_grosores(args.grosores)
    tag = "-".join(f"{g:g}" for g in grosores)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_path = out_dir / f"warm_fisico_g{tag}.csv"
    sol_path = (
        (REPO_ROOT / args.best_out).resolve()
        if args.best_out
        else out_dir / f"mark17_fisico_g{tag}.csv"
    )
    sol_path.parent.mkdir(parents=True, exist_ok=True)

    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    baseline = float(costo_actual_oficial(ops))

    warm = build_warm_fisico(pt, grosores)
    warm[SUBMISSION_COLS].to_csv(warm_path, index=False)
    ok_w, errs_w, _, _ = validate_solution_free(warm, pt, verbose=False)
    fis_w = assert_axis_fit(warm, pt)
    print(
        f"Mark17 FÍSICO | grosores={grosores} | tl={args.time_limit:.0f}s "
        f"workers={args.workers}",
        flush=True,
    )
    print(
        f"Warm: valid_free={ok_w} fisico_ok={fis_w['ok_fisico']} "
        f"path={warm_path.relative_to(REPO_ROOT)}",
        flush=True,
    )
    if errs_w:
        print(f"  warm warnings/errors: {errs_w[:3]}", flush=True)
    if not fis_w["ok_fisico"]:
        raise SystemExit(f"Warm no es físico: {fis_w}")

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

    prev = _patch_fisico()
    t0 = time.time()
    try:
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
    finally:
        _restore(*prev)
    elapsed = time.time() - t0

    if not result.get("ok"):
        print(f"FAIL en {elapsed:.1f}s", flush=True)
        raise SystemExit(1)

    sol = result["solution"][SUBMISSION_COLS].copy()
    sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
    sol[SUBMISSION_COLS].to_csv(sol_path, index=False)
    ev = eval_kaggle_plant(sol, ops)
    fis = assert_axis_fit(sol, pt)
    surr = float(result["surrogate"])
    bound = float(result["bound"])
    gap = abs(surr - bound)
    score = score_publico(ev["total"], baseline)
    dist = sol["caja_grosor_mm"].round(1).value_counts().sort_index().to_dict()

    print(
        f"\nDONE {elapsed:.1f}s | optimal={gap <= 1e-3} gap={gap:.4f} | "
        f"USD {ev['total']:,.2f} score {score:+.5f} | tipos {ev['n_tipos']} | "
        f"fisico_ok={fis['ok_fisico']} | dist={dist}",
        flush=True,
    )
    print(f"Guardado: {sol_path}", flush=True)

    meta = {
        "grosores": ",".join(str(g) for g in grosores),
        "ok": True,
        "optimal_surrogate": gap <= 1e-3,
        "surrogate": surr,
        "bound": bound,
        "gap_surrogate": gap,
        "elapsed_s": elapsed,
        "n_tipos": ev["n_tipos"],
        "total": ev["total"],
        "packaging": ev["packaging"],
        "flete": ev["flete"],
        "score": score,
        "fisico_ok": fis["ok_fisico"],
        "n_con_achique": fis["n_con_achique"],
        "dist_grosor": ";".join(f"{g}:{n}" for g, n in dist.items()),
        "solution_path": str(sol_path.relative_to(REPO_ROOT)),
        "fit_model": "physical_axis",
    }
    meta_path = out_dir / f"mark17_fisico_g{tag}_meta.csv"
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Meta: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
