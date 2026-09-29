"""Mark 18 — CP-SAT bajo half-pallet + double docking.

Arquitectura = Mark 16 (mismo pool / fits / packaging por tiers), pero:

1. Pallet de diseño: ``lim = (800, 1200, 900)`` (mitad de alto).
2. Flete: dos half-pallets comparten un slot de USD 150, mezclando SKUs
   dentro de cada planta::

       half_{i,p} = ceil(vol_{i,p} / cpp_half)
       slots_p    = ceil( (Σ_i half_{i,p}) / 2 )
       C_flete    = Σ_p slots_p × costo_unitario_p

Esto relaja la regla “1 SKU por pallet” solo en el sentido de apilar dos
medios pallets en el mismo cobro de flete (double docking).

Uso:
    PYTHONPATH=src python src/mark18.py --time-limit 300
    PYTHONPATH=src .venv/bin/python src/mark18.py --time-limit 300 \\
        --warm-start outputs/tables/04_mark16_best_+10.11098.csv
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from evaluate import (
    BOX_COLS,
    COSTO_PALLET,
    agregar_id_tipo_caja,
    calcular_cajas_por_pallet,
    factor_precio_por_volumen,
    inferir_costos_flete_por_planta,
    validate_solution,
    validate_solution_free,
)
from mark16 import (
    _LIM_CTX,
    assignment_to_solution,
    build_pool,
    construir_productos,
    costo_actual_oficial,
    normalize_lim,
    parse_grosores,
    score_publico,
)
from settings import (
    DEFAULT_GROSORES,
    OPERACIONES_PLANTA_PATH,
    PLANTAS,
    PRECIO_BASE_GROSOR,
    PRODUCTS_TRUTH_PATH,
    REPO_ROOT,
    SCALE,
    SUBMISSION_COLS,
    TIERS,
)

# Half-height design pallet (mitad de 1800 mm).
HALF_LIM = (800.0, 1200.0, 900.0)
# Dos half-pallets = 1 slot de flete.
DOCK_FACTOR = 2


# ---------------------------------------------------------------------------
# Evaluador double-dock
# ---------------------------------------------------------------------------


def evaluar_costos_double_dock(
    solution_df: pd.DataFrame,
    operaciones_planta: pd.DataFrame,
    *,
    half_lim: tuple[float, float, float] = HALF_LIM,
    dock_factor: int = DOCK_FACTOR,
    costos_flete_por_planta: dict | None = None,
) -> dict:
    """Packaging oficial + flete con double docking de half-pallets."""
    submission = solution_df[SUBMISSION_COLS].copy()
    for c in BOX_COLS:
        submission[c] = pd.to_numeric(submission[c], errors="coerce")
    submission["caja_grosor_mm"] = submission["caja_grosor_mm"].round(3)

    vol_cols = [
        "volumen_producto_total",
        *[f"volumen_producto_planta_{p}" for p in PLANTAS],
    ]
    df = submission.merge(
        operaciones_planta[["codigo_producto", *vol_cols]],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )
    if df["volumen_producto_total"].isna().any():
        raise ValueError("Productos sin match en operaciones_planta")

    df = calcular_cajas_por_pallet(agregar_id_tipo_caja(df), lim=half_lim)
    df["volumen_tipo_caja"] = df.groupby("caja_tipo_id_solucion")[
        "volumen_producto_total"
    ].transform("sum")
    df["precio_base_caja"] = df["caja_grosor_mm"].map(PRECIO_BASE_GROSOR)
    df["factor_descuento"] = factor_precio_por_volumen(df["volumen_tipo_caja"])
    df["precio_caja"] = df["precio_base_caja"] * df["factor_descuento"]
    df["costo_packaging_producto"] = df["precio_caja"] * df["volumen_producto_total"]

    if costos_flete_por_planta is None:
        costos_flete, diag = inferir_costos_flete_por_planta(operaciones_planta)
    else:
        costos_flete, diag = costos_flete_por_planta, pd.DataFrame()

    plant_rows = []
    half_total = 0
    slots_total = 0
    flete = 0.0
    for p in PLANTAS:
        vol_col = f"volumen_producto_planta_{p}"
        halves = np.ceil(df[vol_col] / df["cajas_por_pallet"]).astype(int)
        n_half = int(halves.sum())
        slots = int(np.ceil(n_half / float(dock_factor))) if n_half else 0
        costo = slots * float(costos_flete[p])
        df[f"half_pallets_planta_{p}"] = halves
        half_total += n_half
        slots_total += slots
        flete += costo
        plant_rows.append(
            {
                "planta": p,
                "volumen_total": float(df[vol_col].sum()),
                "half_pallets": n_half,
                "freight_slots": slots,
                "costo_unitario_flete": costos_flete[p],
                "costo_flete_nuevo": float(costo),
            }
        )

    pack = float(df["costo_packaging_producto"].sum())
    resumen = pd.DataFrame(
        [
            {
                "n_productos": df["codigo_producto"].nunique(),
                "n_tipos_caja": df["caja_tipo_id_solucion"].nunique(),
                "half_lim": list(half_lim),
                "dock_factor": int(dock_factor),
                "costo_packaging_nuevo": pack,
                "costo_flete_nuevo": flete,
                "costo_total_nuevo": pack + flete,
                "half_pallets_total": int(half_total),
                "freight_slots_total": int(slots_total),
                "costo_slot": float(COSTO_PALLET),
            }
        ]
    )
    return {
        "resumen": resumen,
        "detalle_productos": df,
        "detalle_plantas": pd.DataFrame(plant_rows),
        "diagnostico_costos_flete": diag,
        "submission_limpia": submission,
    }


# ---------------------------------------------------------------------------
# CP-SAT (packaging Mark16 + flete double-dock acoplado por planta)
# ---------------------------------------------------------------------------


class _IncumbentSaver(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        x: dict[tuple[int, int], cp_model.IntVar],
        n: int,
        boxes: np.ndarray,
        box_grosor: np.ndarray,
        codes: list[str],
        out_path: Path | None = None,
        on_incumbent=None,
        t0: float | None = None,
        initial_best: float = np.inf,
    ):
        super().__init__()
        self._x = x
        self._n = n
        self._boxes = boxes
        self._box_grosor = box_grosor
        self._codes = codes
        self._out_path = out_path
        self._on_incumbent = on_incumbent
        self._t0 = t0 if t0 is not None else time.time()
        self._best = float(initial_best)

    def on_solution_callback(self) -> None:
        cost = self.ObjectiveValue() / SCALE
        bound = self.BestObjectiveBound() / SCALE
        if cost >= self._best:
            return
        self._best = cost
        assign = np.full(self._n, -1, dtype=int)
        for (i, b), var in self._x.items():
            if self.BooleanValue(var):
                assign[i] = b
        solution = assignment_to_solution(
            self._codes, self._boxes, self._box_grosor, assign
        )
        if self._out_path is not None:
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._out_path.with_suffix(self._out_path.suffix + ".tmp")
            solution[SUBMISSION_COLS].to_csv(tmp, index=False)
            tmp.replace(self._out_path)
            print(f"  incumbent: {cost:,.2f} -> {self._out_path}", flush=True)
        if self._on_incumbent is not None:
            self._on_incumbent(
                {
                    "cost": float(cost),
                    "bound": float(bound),
                    "elapsed_s": float(time.time() - self._t0),
                    "solution": solution,
                }
            )


def solve_global_double_dock(
    fits: np.ndarray,
    cpp: np.ndarray,
    box_grosor: np.ndarray,
    vol_pl: np.ndarray,
    cvec: np.ndarray,
    hint: np.ndarray | None,
    time_limit: float,
    workers: int,
    *,
    dock_factor: int = DOCK_FACTOR,
    boxes: np.ndarray | None = None,
    codes: list[str] | None = None,
    incumbent_out: Path | None = None,
    on_incumbent=None,
    on_bound=None,
    initial_best: float = np.inf,
) -> tuple[np.ndarray | None, float, float]:
    """Asigna SKUs al pool. Packaging por tiers; flete = ceil(Σ half / dock) × $."""
    n_boxes, n = fits.shape
    n_plants = vol_pl.shape[1]
    model = cp_model.CpModel()

    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for b in range(n_boxes):
        for i in np.where(fits[b])[0]:
            x[int(i), b] = model.NewBoolVar(f"x_{i}_{b}")
    for i in range(n):
        model.AddExactlyOne(x[i, b] for b in range(n_boxes) if (i, b) in x)

    # halves[b, i, p] = ceil(vol_ip / cpp_b)  (0 si no cabe o vol=0)
    halves = np.zeros((n_boxes, n, n_plants), dtype=np.int64)
    for b in range(n_boxes):
        c = int(cpp[b])
        if c <= 0:
            continue
        members = np.where(fits[b])[0]
        for p in range(n_plants):
            v = vol_pl[members, p]
            halves[b, members, p] = np.ceil(v / float(c)).astype(np.int64)

    obj_vars: list = []
    obj_coefs: list[int] = []

    # --- Flete double-dock (acoplado por planta) ---
    for p in range(n_plants):
        terms = []
        max_h = 0
        for b in range(n_boxes):
            for i in np.where(fits[b])[0]:
                h = int(halves[b, i, p])
                if h <= 0:
                    continue
                terms.append(h * x[int(i), b])
                max_h += h
        if max_h == 0:
            continue
        H = model.NewIntVar(0, max_h, f"H_{p}")
        model.Add(H == sum(terms))
        max_slots = (max_h + dock_factor - 1) // dock_factor
        slots = model.NewIntVar(0, max_slots, f"slots_{p}")
        # slots >= ceil(H / dock_factor)  ≡  dock_factor * slots >= H
        model.Add(dock_factor * slots >= H)
        rate = int(round(float(cvec[p]) * SCALE))
        obj_vars.append(slots)
        obj_coefs.append(rate)

    # --- Packaging (igual Mark 16) ---
    for b in range(n_boxes):
        members = np.where(fits[b])[0]
        precio = PRECIO_BASE_GROSOR[float(box_grosor[b])]
        rates = [int(round(precio * factor * SCALE)) for _, _, factor in TIERS]

        for p in range(n_plants):
            max_vol = int(vol_pl[members, p].sum())
            if max_vol == 0:
                continue
            vol_bp = model.NewIntVar(0, max_vol, f"v_{b}_{p}")
            model.Add(
                vol_bp
                == sum(
                    int(vol_pl[int(i), p]) * x[int(i), b]
                    for i in members
                    if vol_pl[int(i), p] > 0
                )
            )
            tiers_pos = [k for k, (lo_t, _, _) in enumerate(TIERS) if lo_t <= max_vol]
            t_bools = {k: model.NewBoolVar(f"t_{b}_{p}_{k}") for k in tiers_pos}
            model.AddExactlyOne(t_bools.values())
            pack_bp = model.NewIntVar(0, max(rates) * max_vol, f"p_{b}_{p}")
            for k in tiers_pos:
                lo_t, hi_t, _ = TIERS[k]
                model.Add(vol_bp >= lo_t).OnlyEnforceIf(t_bools[k])
                if hi_t is not None:
                    model.Add(vol_bp <= min(hi_t, max_vol)).OnlyEnforceIf(t_bools[k])
                model.Add(pack_bp == rates[k] * vol_bp).OnlyEnforceIf(t_bools[k])
            model.Add(pack_bp >= min(rates[k] for k in tiers_pos) * vol_bp)
            obj_vars.append(pack_bp)
            obj_coefs.append(1)

    model.Minimize(cp_model.LinearExpr.weighted_sum(obj_vars, obj_coefs))

    if hint is not None:
        for i in range(n):
            b = int(hint[i])
            if b < 0:
                continue
            key = (i, b)
            if key in x:
                model.AddHint(x[key], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = max(1, workers)
    solver.parameters.log_search_progress = False
    if hint is not None:
        solver.parameters.repair_hint = True
        solver.parameters.use_optimization_hints = True

    t0_solve = time.time()
    last_bound_emit = {"t": -1.0, "bound": None}

    def _bound_cb(bound_scaled: float) -> None:
        if on_bound is None:
            return
        bound = float(bound_scaled) / SCALE
        if not np.isfinite(bound):
            return
        now = time.time() - t0_solve
        prev = last_bound_emit["bound"]
        improved = prev is None or bound > float(prev) + 1e-3
        timed = (now - float(last_bound_emit["t"])) >= 1.5
        if not (improved or timed):
            return
        last_bound_emit["t"] = now
        last_bound_emit["bound"] = bound
        on_bound({"bound": bound, "elapsed_s": float(now)})

    if on_bound is not None:
        solver.best_bound_callback = _bound_cb

    callback = None
    if (
        (incumbent_out is not None or on_incumbent is not None)
        and boxes is not None
        and codes is not None
    ):
        callback = _IncumbentSaver(
            x,
            n,
            boxes,
            box_grosor,
            codes,
            out_path=incumbent_out,
            on_incumbent=on_incumbent,
            initial_best=initial_best,
            t0=t0_solve,
        )

    status = solver.Solve(model, callback)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, np.inf, solver.BestObjectiveBound() / SCALE

    assign = np.full(n, -1, dtype=int)
    for (i, b), var in x.items():
        if solver.BooleanValue(var):
            assign[i] = b
    return assign, solver.ObjectiveValue() / SCALE, solver.BestObjectiveBound() / SCALE


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_optimize(
    warm_start_path: Path | str,
    grosores: list[float] | None = None,
    time_limit: float = 300.0,
    workers: int | None = None,
    out_path: Path | str | None = None,
    on_progress=None,
    pool_mode: str = "fast",
    top_k: int = 8,
    lim: tuple[float, float, float] | list[float] | None = None,
    dock_factor: int = DOCK_FACTOR,
) -> dict:
    """Optimiza bajo half-pallet + double docking. Default lim = HALF_LIM."""
    import os

    active_lim = normalize_lim(lim if lim is not None else HALF_LIM)
    lim_token = _LIM_CTX.set(active_lim)
    try:
        return _run_optimize_inner(
            warm_start_path=warm_start_path,
            grosores=grosores,
            time_limit=time_limit,
            workers=workers,
            out_path=out_path,
            on_progress=on_progress,
            pool_mode=pool_mode,
            top_k=top_k,
            active_lim=active_lim,
            dock_factor=dock_factor,
            cpu_count=os.cpu_count() or 1,
        )
    finally:
        _LIM_CTX.reset(lim_token)


def _run_optimize_inner(
    *,
    warm_start_path: Path | str,
    grosores: list[float] | None,
    time_limit: float,
    workers: int | None,
    out_path: Path | str | None,
    on_progress,
    pool_mode: str,
    top_k: int,
    active_lim: tuple[float, float, float],
    dock_factor: int,
    cpu_count: int,
) -> dict:
    grosores = list(grosores or DEFAULT_GROSORES)
    workers = workers or min(24, cpu_count)
    warm_start_path = Path(warm_start_path)
    out_path = Path(out_path) if out_path else None

    def progress(event: dict) -> None:
        if on_progress is not None:
            on_progress(event)

    progress(
        {
            "phase": "loading",
            "message": "Cargando datos (Mark 18 · double docking)…",
            "lim": list(active_lim),
            "dock_factor": dock_factor,
        }
    )
    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    prod = construir_productos(pt, ops)
    costs, _ = inferir_costos_flete_por_planta(ops)
    baseline = costo_actual_oficial(ops)

    peso = prod["peso_neto_caja"].to_numpy(float)
    vol_pl = prod[[f"volumen_producto_planta_{p}" for p in PLANTAS]].to_numpy(float)
    codes = prod["codigo_producto"].astype(str).str.strip().tolist()
    cvec = np.array([costs[p] for p in PLANTAS], dtype=float)

    warm = pd.read_csv(warm_start_path)
    warm["codigo_producto"] = warm["codigo_producto"].astype(str).str.strip()
    warm = warm.set_index("codigo_producto").loc[codes].reset_index()
    warm_g = warm["caja_grosor_mm"].to_numpy(float)
    warm_boxes = (
        warm[["caja_exterior_largo", "caja_exterior_ancho", "caja_exterior_alto"]].to_numpy(
            float
        )
        - 2.0 * warm_g[:, None]
    )

    extras: dict[float, np.ndarray] = {}
    for g in grosores:
        mask = np.isclose(warm_g, g)
        if mask.any():
            extras[g] = np.unique(np.round(warm_boxes[mask], 2), axis=0)

    progress(
        {
            "phase": "pool",
            "message": (
                f"Construyendo pool half-pallet ({pool_mode}, top_k={top_k}, "
                f"pallet {active_lim[0]:.0f}×{active_lim[1]:.0f}×{active_lim[2]:.0f}, "
                f"dock×{dock_factor})…"
            ),
            "baseline": baseline,
            "lim": list(active_lim),
        }
    )
    t0 = time.time()
    boxes, box_grosor, fits, cpp = build_pool(
        prod, peso, grosores, extras, mode=pool_mode, top_k=top_k
    )
    progress(
        {
            "phase": "pool_done",
            "message": f"Pool listo: {len(boxes)} cajas ({time.time() - t0:.1f}s)",
            "n_boxes": int(len(boxes)),
            "elapsed_s": time.time() - t0,
            "baseline": baseline,
        }
    )

    box_index = {
        (*np.round(boxes[b], 2), float(box_grosor[b])): b for b in range(len(boxes))
    }
    hint = np.full(len(codes), -1, dtype=int)
    for i in range(len(codes)):
        key = (*np.round(warm_boxes[i], 2), float(warm_g[i]))
        b = box_index.get(key)
        if b is not None and fits[b, i]:
            hint[i] = b
    n_hint = int((hint >= 0).sum())

    def on_incumbent(payload: dict) -> None:
        cost = float(payload["cost"])
        event = {
            "phase": "solving",
            "message": "Nueva mejora encontrada",
            "cost": cost,
            "baseline": baseline,
            "ahorro": baseline - cost,
            "score": score_publico(cost, baseline),
            "elapsed_s": float(payload["elapsed_s"]),
            "incumbent": True,
        }
        if "bound" in payload and np.isfinite(payload["bound"]):
            event["bound"] = float(payload["bound"])
        progress(event)

    def on_bound(payload: dict) -> None:
        bound = float(payload["bound"])
        if not np.isfinite(bound):
            return
        progress(
            {
                "phase": "solving",
                "message": "Cota CP-SAT actualizada",
                "bound": bound,
                "baseline": baseline,
                "ahorro_cota": baseline - bound,
                "score_cota": score_publico(bound, baseline),
                "elapsed_s": float(payload["elapsed_s"]),
                "bound_update": True,
            }
        )

    warm_cost: float | None = None
    warm_assign: np.ndarray | None = None
    if n_hint == len(codes):
        warm_assign = hint.copy()
        warm_sol = assignment_to_solution(codes, boxes, box_grosor, warm_assign)
        warm_cost = float(
            evaluar_costos_double_dock(
                warm_sol,
                ops,
                half_lim=active_lim,
                dock_factor=dock_factor,
            )["resumen"]["costo_total_nuevo"].iloc[0]
        )
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            warm_sol[SUBMISSION_COLS].to_csv(out_path, index=False)
        progress(
            {
                "phase": "solving",
                "message": "Warm start evaluado bajo double docking",
                "cost": warm_cost,
                "baseline": baseline,
                "ahorro": baseline - warm_cost,
                "score": score_publico(warm_cost, baseline),
                "elapsed_s": 0.0,
                "incumbent": True,
            }
        )
    elif n_hint > 0:
        progress(
            {
                "phase": "solving",
                "message": (
                    f"Warm parcial: {n_hint}/{len(codes)} cajas caben a "
                    f"{active_lim[2]:.0f} mm (sin seed de costo completo)"
                ),
            }
        )

    progress(
        {
            "phase": "solving",
            "message": (
                f"CP-SAT double-dock ({time_limit:.0f}s, {workers} workers, "
                f"hint {n_hint}/{len(codes)})…"
            ),
            "baseline": baseline,
            "time_limit": time_limit,
        }
    )
    assign, cost_obj, bound = solve_global_double_dock(
        fits,
        cpp,
        box_grosor,
        vol_pl,
        cvec,
        hint if (hint >= 0).any() else None,
        time_limit,
        workers,
        dock_factor=dock_factor,
        boxes=boxes,
        codes=codes,
        incumbent_out=out_path,
        on_incumbent=on_incumbent,
        on_bound=on_bound,
        initial_best=warm_cost if warm_cost is not None else np.inf,
    )

    if warm_assign is not None and warm_cost is not None:
        if assign is None:
            assign, cost = warm_assign, warm_cost
        else:
            sol_tmp = assignment_to_solution(codes, boxes, box_grosor, assign)
            cost = float(
                evaluar_costos_double_dock(
                    sol_tmp,
                    ops,
                    half_lim=active_lim,
                    dock_factor=dock_factor,
                )["resumen"]["costo_total_nuevo"].iloc[0]
            )
            if cost > warm_cost + 0.01:
                assign, cost = warm_assign, warm_cost
    else:
        if assign is None:
            progress(
                {
                    "phase": "failed",
                    "message": "Sin solución factible en el tiempo dado",
                    "bound": float(bound),
                    "baseline": baseline,
                }
            )
            return {
                "ok": False,
                "baseline": baseline,
                "bound": float(bound),
                "cost": None,
                "solution_path": None,
                "lim": list(active_lim),
                "dock_factor": dock_factor,
            }
        sol_tmp = assignment_to_solution(codes, boxes, box_grosor, assign)
        cost = float(
            evaluar_costos_double_dock(
                sol_tmp,
                ops,
                half_lim=active_lim,
                dock_factor=dock_factor,
            )["resumen"]["costo_total_nuevo"].iloc[0]
        )

    solution = assignment_to_solution(codes, boxes, box_grosor, assign)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        solution[SUBMISSION_COLS].to_csv(out_path, index=False)

    eval_dd = evaluar_costos_double_dock(
        solution, ops, half_lim=active_lim, dock_factor=dock_factor
    )
    resumen = eval_dd["resumen"].iloc[0]

    progress(
        {
            "phase": "done",
            "message": "Optimización Mark 18 terminada",
            "cost": float(cost),
            "bound": float(bound),
            "baseline": baseline,
            "ahorro": baseline - float(cost),
            "score": score_publico(float(cost), baseline),
            "packaging": float(resumen["costo_packaging_nuevo"]),
            "flete_dd": float(resumen["costo_flete_nuevo"]),
            "freight_slots": int(resumen["freight_slots_total"]),
            "half_pallets": int(resumen["half_pallets_total"]),
            "elapsed_s": float(time_limit),
            "n_tipos": int(
                solution[
                    [
                        "caja_grosor_mm",
                        "caja_exterior_largo",
                        "caja_exterior_ancho",
                        "caja_exterior_alto",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
            "solution_path": str(out_path) if out_path else None,
        }
    )
    return {
        "ok": True,
        "baseline": baseline,
        "bound": float(bound),
        "cost": float(cost),
        "cost_obj": float(cost_obj) if np.isfinite(cost_obj) else None,
        "score": score_publico(float(cost), baseline),
        "solution": solution,
        "solution_path": str(out_path) if out_path else None,
        "lim": list(active_lim),
        "dock_factor": dock_factor,
        "eval_dd": eval_dd,
        "warm_cost_dd": warm_cost,
        "n_hint": n_hint,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark 18: half-pallet + double docking (arquitectura Mark 16)."
    )
    parser.add_argument("--grosores", default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--warm-start",
        default="outputs/tables/04_mark16_best_+10.11098.csv",
    )
    parser.add_argument("--out-dir", default="outputs/mark18")
    parser.add_argument("--best-out", default="outputs/mark18/mark18_best.csv")
    parser.add_argument("--pool", choices=("fast", "full"), default="fast")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--half-alto",
        type=float,
        default=900.0,
        help="Altura útil del half-pallet en mm (default 900).",
    )
    parser.add_argument(
        "--dock-factor",
        type=int,
        default=DOCK_FACTOR,
        help="Half-pallets por slot de flete (default 2).",
    )
    args = parser.parse_args()

    grosores = parse_grosores(args.grosores)
    half_lim = (800.0, 1200.0, float(args.half_alto))
    print(
        f"Mark 18 | grosores={grosores} | lim={half_lim} | dock×{args.dock_factor} | "
        f"pool={args.pool} top_k={args.top_k} | t={args.time_limit:.0f}s",
        flush=True,
    )

    result = run_optimize(
        warm_start_path=REPO_ROOT / args.warm_start,
        grosores=grosores,
        time_limit=args.time_limit,
        workers=args.workers,
        out_path=REPO_ROOT / args.best_out,
        pool_mode=args.pool,
        top_k=args.top_k,
        lim=half_lim,
        dock_factor=args.dock_factor,
        on_progress=lambda e: print(
            f"[{e.get('phase')}] {e.get('message', '')}"
            + (
                f" | cost={e['cost']:,.0f} score={e['score']:+.3f}"
                if e.get("incumbent") and "cost" in e
                else ""
            ),
            flush=True,
        ),
    )
    if not result.get("ok"):
        print("Sin solución factible dentro del tiempo.", flush=True)
        return

    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    solution = result["solution"]
    baseline = result["baseline"]
    cost = result["cost"]
    bound = result["bound"]
    eval_dd = result["eval_dd"]
    r = eval_dd["resumen"].iloc[0]

    print(
        f"\nCota CP-SAT: {bound:,.2f} (score techo {score_publico(bound, baseline):+.5f})",
        flush=True,
    )
    print(
        f"Mejor DD: {cost:,.2f} (score vs baseline oficial {score_publico(cost, baseline):+.5f})",
        flush=True,
    )
    if result.get("warm_cost_dd") is not None:
        print(
            f"Warm DD:  {result['warm_cost_dd']:,.2f} "
            f"(Δ solver={cost - result['warm_cost_dd']:+,.2f})",
            flush=True,
        )
    print(
        f"  packaging={r['costo_packaging_nuevo']:,.0f}  "
        f"flete_dd={r['costo_flete_nuevo']:,.0f}  "
        f"slots={int(r['freight_slots_total']):,}  "
        f"halves={int(r['half_pallets_total']):,}",
        flush=True,
    )
    print("Detalle por planta:", flush=True)
    print(eval_dd["detalle_plantas"].to_string(index=False), flush=True)

    # Referencia: misma solución bajo reglas oficiales 1800 (sin DD) — solo diagnóstico.
    from evaluate import evaluar_costos_solucion

    off = evaluar_costos_solucion(solution, ops)["resumen"].iloc[0]
    print(
        f"\nDiagnóstico si se cobrara como pallet full 1800 (sin DD): "
        f"total={off['costo_total_nuevo']:,.0f}  flete={off['costo_flete_nuevo']:,.0f}",
        flush=True,
    )

    if len(grosores) == 1 and set(grosores).issubset({3.0, 4.5, 5.0}):
        ok, errors, _ = validate_solution(solution, pt, verbose=False)
        # Validación oficial usa lim 1800; puede fallar capas/compresión a 900.
        print(f"Válida (oficial @1800): {ok} {errors[:3] if not ok else ''}", flush=True)
    ok_f, errors_f, warnings, _ = validate_solution_free(solution, pt, verbose=False)
    print(f"Válida (free geom): {ok_f} {errors_f[:3] if not ok_f else ''}", flush=True)
    if warnings:
        print(f"  warnings: {warnings[:2]}", flush=True)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    g_tag = "-".join(str(g) for g in grosores)
    out = out_dir / f"mark18_g{g_tag}_h{int(args.half_alto)}_{stamp}.csv"
    solution[SUBMISSION_COLS].to_csv(out, index=False)
    print(f"Guardado: {out}", flush=True)
    print(f"Best-out: {REPO_ROOT / args.best_out}", flush=True)


if __name__ == "__main__":
    main()
