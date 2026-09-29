"""Mark 19 — Mark 16 con fit físico por eje (sin “contenido líquido”).

Igual que Mark 16 (pool + CP-SAT + packaging por tiers / planta), pero la ventana
geométrica **no permite achicar** un eje por debajo del producto:

    interna_k ≥ producto_k   (además de volumen, ±10%, headspace, ECT)

Eso modela un producto rígido rectangular. El techo de score vuelve al régimen
~−12.5 (cota con fit por eje) en lugar de ~+10.5 (lectura literal por volumen).

Uso:
    PYTHONPATH=src .venv/bin/python src/mark19.py --time-limit 900
    PYTHONPATH=src .venv/bin/python src/mark19.py --make-warm-start
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import mark16
from evaluate import (
    SUBMISSION_COLS,
    evaluar_costos_solucion,
    validate_solution,
    validate_solution_free,
)
from mark16 import (
    construir_productos,
    costo_actual_oficial,
    parse_grosores,
    run_optimize as run_optimize_mark16,
    score_publico,
)
from settings import (
    DEFAULT_GROSORES,
    HEADSPACE_ABS_MAX,
    HEADSPACE_PCT,
    MAX_GROW,
    OPERACIONES_PLANTA_PATH,
    PRODUCTS_TRUTH_PATH,
    REPO_ROOT,
    SUBMISSION_COLS as SETTINGS_SUBMISSION_COLS,
)

assert list(SUBMISSION_COLS) == list(SETTINGS_SUBMISSION_COLS)


# ---------------------------------------------------------------------------
# Geometría física
# ---------------------------------------------------------------------------


def intervalos_fisicos(
    prod: pd.DataFrame, grosor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Como mark16.intervalos, pero lo = dims (no achique bajo el producto)."""
    dims = prod[["largo", "ancho", "alto"]].to_numpy(float)
    pct = HEADSPACE_PCT[grosor]
    lo = dims.copy()
    hi = np.minimum.reduce(
        [MAX_GROW * dims, dims / (1.0 - pct), dims + HEADSPACE_ABS_MAX]
    )
    hi = np.maximum(hi, lo)
    return lo, hi, dims.prod(axis=1)


# Implementación literal original de Mark 16 (antes de cualquier patch).
_FITS_LITERAL = mark16.fits_matrix


def assert_axis_fit(solution: pd.DataFrame, products_truth: pd.DataFrame) -> dict:
    """Chequeo post-hoc: interna_k ≥ producto_k en los tres ejes."""
    df = solution[SUBMISSION_COLS].copy()
    for c in ("caja_grosor_mm", "caja_exterior_largo", "caja_exterior_ancho", "caja_exterior_alto"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["codigo_producto"] = df["codigo_producto"].astype(str).str.strip()
    pt = products_truth[["codigo_producto", "largo", "ancho", "alto"]].copy()
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    m = df.merge(pt, on="codigo_producto", how="left", validate="one_to_one")
    g = m["caja_grosor_mm"].to_numpy(float)
    shrink = {}
    for ax, ecol, pcol in (
        ("L", "caja_exterior_largo", "largo"),
        ("W", "caja_exterior_ancho", "ancho"),
        ("H", "caja_exterior_alto", "alto"),
    ):
        inte = m[ecol].to_numpy(float) - 2.0 * g
        shrink[ax] = int((inte < m[pcol].to_numpy(float) - 1e-9).sum())
    n = len(m)
    n_bad = int(
        (
            (m["caja_exterior_largo"] - 2 * g < m["largo"] - 1e-9)
            | (m["caja_exterior_ancho"] - 2 * g < m["ancho"] - 1e-9)
            | (m["caja_exterior_alto"] - 2 * g < m["alto"] - 1e-9)
        ).sum()
    )
    return {
        "n_skus": n,
        "n_con_achique": n_bad,
        "ok_fisico": n_bad == 0,
        "shrink_L": shrink["L"],
        "shrink_W": shrink["W"],
        "shrink_H": shrink["H"],
    }


# ---------------------------------------------------------------------------
# Warm start físico (caja justa = producto + 2·grosor)
# ---------------------------------------------------------------------------


def make_physical_warm_start(
    grosor: float = 3.0,
    out_path: Path | str | None = None,
) -> pd.DataFrame:
    """Una caja por SKU: interior = dims del producto (fit justado, headspace 0)."""
    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    g = float(grosor)
    sol = pd.DataFrame(
        {
            "codigo_producto": pt["codigo_producto"],
            "caja_grosor_mm": g,
            "caja_exterior_largo": (pt["largo"].astype(float) + 2.0 * g).round(3),
            "caja_exterior_ancho": (pt["ancho"].astype(float) + 2.0 * g).round(3),
            "caja_exterior_alto": (pt["alto"].astype(float) + 2.0 * g).round(3),
        }
    )
    out = Path(out_path) if out_path else REPO_ROOT / "outputs/mark19/warm_fisico_g3.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sol[SUBMISSION_COLS].to_csv(out, index=False)
    return sol


def _patch_mark16_fisico():
    """Activa intervalos/fits físicos en el módulo mark16 (pool + solve)."""
    mark16.intervalos = intervalos_fisicos  # type: ignore[assignment]
    # fits_matrix ya usa lo; con lo=dims alcanza. Reforzamos igual.
    def _fits(boxes, lo, hi, vol_min, peso, grosor):
        fits = _FITS_LITERAL(boxes, lo, hi, vol_min, peso, grosor)
        axis_ok = (boxes[:, None, :] >= lo[None, :, :] - 1e-9).all(axis=2)
        return fits & axis_ok

    mark16.fits_matrix = _fits  # type: ignore[assignment]


def _restore_mark16(prev_intervalos, prev_fits):
    mark16.intervalos = prev_intervalos
    mark16.fits_matrix = prev_fits


def run_optimize(**kwargs) -> dict:
    """API = mark16.run_optimize bajo geometría física."""
    prev_i, prev_f = mark16.intervalos, mark16.fits_matrix
    _patch_mark16_fisico()
    try:
        result = run_optimize_mark16(**kwargs)
    finally:
        _restore_mark16(prev_i, prev_f)

    if result.get("ok") and result.get("solution") is not None:
        pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
        result["fisico"] = assert_axis_fit(result["solution"], pt)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark 19: Mark 16 + fit físico por eje (producto rígido)."
    )
    parser.add_argument("--grosores", default=None)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--warm-start",
        default="outputs/mark19/warm_fisico_g3.csv",
        help="CSV de partida (default: warm físico generado).",
    )
    parser.add_argument("--out-dir", default="outputs/mark19")
    parser.add_argument("--best-out", default="outputs/mark19/mark19_best.csv")
    parser.add_argument("--pool", choices=("fast", "full"), default="fast")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--make-warm-start",
        action="store_true",
        help="Solo genera warm_fisico_g3.csv y sale.",
    )
    args = parser.parse_args()

    grosores = parse_grosores(args.grosores)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    warm_path = REPO_ROOT / args.warm_start
    if args.make_warm_start or not warm_path.exists():
        g0 = grosores[0] if len(grosores) == 1 else 3.0
        make_physical_warm_start(grosor=g0, out_path=warm_path)
        print(f"Warm físico guardado: {warm_path}", flush=True)
        if args.make_warm_start:
            return

    print(
        f"Mark 19 | fit FÍSICO por eje | grosores={grosores} | "
        f"pool={args.pool} top_k={args.top_k} | t={args.time_limit:.0f}s",
        flush=True,
    )

    log_path = out_dir / f"mark19_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    events: list[dict] = []

    def on_progress(e: dict) -> None:
        events.append(dict(e))
        msg = f"[{e.get('phase')}] {e.get('message', '')}"
        if e.get("cost") is not None:
            msg += f" | cost={e['cost']:,.0f} score={e.get('score', float('nan')):+.4f}"
        if e.get("bound") is not None and e.get("bound_update"):
            msg += f" | bound={e['bound']:,.0f}"
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t0 = time.time()
    result = run_optimize(
        warm_start_path=warm_path,
        grosores=grosores,
        time_limit=args.time_limit,
        workers=args.workers,
        out_path=REPO_ROOT / args.best_out,
        pool_mode=args.pool,
        top_k=args.top_k,
        on_progress=on_progress,
    )
    elapsed = time.time() - t0

    if not result.get("ok"):
        print("Sin solución factible dentro del tiempo.", flush=True)
        return

    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    solution = result["solution"]
    baseline = result["baseline"]
    cost = float(result["cost"])
    bound = float(result["bound"])
    fisico = result.get("fisico") or assert_axis_fit(solution, pt)

    ok_k, err_k, _ = validate_solution(solution, pt, verbose=False)
    ev = evaluar_costos_solucion(solution, ops)
    pack = float(ev["resumen"]["costo_packaging_nuevo"].iloc[0])
    flete = float(ev["resumen"]["costo_flete_nuevo"].iloc[0])
    n_tipos = int(ev["resumen"]["n_tipos_caja"].iloc[0])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    g_tag = "-".join(str(g) for g in grosores)
    stamped = out_dir / f"mark19_g{g_tag}_{stamp}.csv"
    solution[SUBMISSION_COLS].to_csv(stamped, index=False)
    solution[SUBMISSION_COLS].to_csv(REPO_ROOT / args.best_out, index=False)

    report = out_dir / f"mark19_report_{stamp}.md"
    report.write_text(
        "\n".join(
            [
                "# Mark 19 — reporte de corrida (fit físico)",
                "",
                f"- Tiempo límite: {args.time_limit:.0f}s (wall {elapsed:.1f}s)",
                f"- Grosores: {grosores}",
                f"- Warm start: `{warm_path.relative_to(REPO_ROOT)}`",
                f"- Solución: `{stamped.relative_to(REPO_ROOT)}`",
                "",
                "## Resultados",
                "",
                f"| Métrica | Valor |",
                f"|---------|------:|",
                f"| Baseline oficial | USD {baseline:,.0f} |",
                f"| Costo solución | USD {cost:,.0f} |",
                f"| Score | {score_publico(cost, baseline):+.5f} |",
                f"| Cota CP-SAT | USD {bound:,.0f} |",
                f"| Score techo (cota) | {score_publico(bound, baseline):+.5f} |",
                f"| Packaging | USD {pack:,.0f} |",
                f"| Flete | USD {flete:,.0f} |",
                f"| Tipos de caja | {n_tipos} |",
                f"| Válida Kaggle | {ok_k} |",
                f"| Fit físico (sin achique) | {fisico['ok_fisico']} |",
                f"| SKUs con achique L/W/H | {fisico['shrink_L']}/{fisico['shrink_W']}/{fisico['shrink_H']} |",
                "",
                "## Lectura",
                "",
                "Mark 19 optimiza bajo **fit por eje** (producto rígido). El score queda en el",
                "régimen del techo ~−12.5 si la cota/solución son consistentes; no compite con",
                "el +10 de la lectura literal por volumen (Mark 16 / Best).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== RESUMEN MARK 19 ===", flush=True)
    print(f"Costo:  USD {cost:,.0f}  score {score_publico(cost, baseline):+.5f}", flush=True)
    print(f"Cota:   USD {bound:,.0f}  score techo {score_publico(bound, baseline):+.5f}", flush=True)
    print(f"Pack/Flete: {pack/1e6:.2f}M / {flete/1e6:.2f}M | tipos={n_tipos}", flush=True)
    print(f"Kaggle válida: {ok_k} | Físico OK: {fisico['ok_fisico']}", flush=True)
    if err_k:
        print(f"  errores Kaggle: {err_k[:3]}", flush=True)
    print(f"Guardado: {stamped}", flush=True)
    print(f"Reporte:  {report}", flush=True)


if __name__ == "__main__":
    main()
