"""Mark 17 — CP-SAT demand-agnostic (sin magnitudes de demanda).

Basado en Mark 16 (mismo pool / factibilidad), pero el objetivo **no usa**
`volumen_producto_planta_*` ni tiers. Optimiza proxies robustos:

1. **Eficiencia** — minimizar `n_plantas(i) / cpp(b)` (proxy de flete sin vols).
2. **Costo unitario** — `precio_base(grosor)` por SKU (+ desperdicio geométrico opcional).
3. **Consolidación por planta** — minimizar tipos de caja distintos usados en cada planta.
4. **Stickiness** — penalizar tipos singleton por planta (frágiles ante shocks de volumen).

Pesos default: eff ≫ resto (calibrados con priors sintéticos, sin leakage de demanda).

Única señal de planta permitida: **presencia binaria** (`vol > 0`), no magnitudes.
Eso evita data leakage de la demanda observada en el objetivo.

El score Kaggle / costo oficial se calcula **solo post-hoc** como diagnóstico.

Uso:
    PYTHONPATH=src python src/mark17.py
    PYTHONPATH=src python src/mark17.py --time-limit 120 --warm-start outputs/tables/00_baseline_solution.csv
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
    evaluar_costos_solucion,
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
)

# Pesos default del surrogate (enteros).
# Calibrados para que eff domine (~flete 86% del costo real). Consolación/stickiness
# solo como tie-break — prior-score sintético (unit/prod_vol/lognormal), sin leakage.
# Config "lex_eff_cons" del grid 2026-07-27 → oracle Kaggle ~+9.8 en smoke.
DEFAULT_W_EFF = 200_000
DEFAULT_W_UNIT = 200
DEFAULT_W_WASTE = 0  # headspace: no mueve flete; apagado por defecto
DEFAULT_W_CONS = 1_000
DEFAULT_W_STICK = 500  # penalty por (tipo, planta) singleton


def plant_presence(prod: pd.DataFrame) -> np.ndarray:
    """(n, P) binario: SKU i aparece en planta p. Sin magnitudes."""
    cols = [f"volumen_producto_planta_{p}" for p in PLANTAS]
    return (prod[cols].to_numpy(float) > 0).astype(np.int8)


# ---------------------------------------------------------------------------
# CP-SAT demand-free
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
        # Objetivo surrogate (no USD oficiales).
        score = self.ObjectiveValue() / SCALE
        bound = self.BestObjectiveBound() / SCALE
        if score >= self._best:
            return
        self._best = score
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
            print(f"  incumbent surrogate: {score:,.4f} -> {self._out_path}", flush=True)
        if self._on_incumbent is not None:
            self._on_incumbent(
                {
                    "surrogate": float(score),
                    "bound": float(bound),
                    "elapsed_s": float(time.time() - self._t0),
                    "solution": solution,
                }
            )


def solve_demand_free(
    fits: np.ndarray,
    cpp: np.ndarray,
    box_grosor: np.ndarray,
    boxes: np.ndarray,
    presence: np.ndarray,
    prod_vol: np.ndarray,
    hint: np.ndarray | None,
    time_limit: float,
    workers: int,
    *,
    w_eff: int = DEFAULT_W_EFF,
    w_unit: int = DEFAULT_W_UNIT,
    w_waste: int = DEFAULT_W_WASTE,
    w_cons: int = DEFAULT_W_CONS,
    w_stick: int = DEFAULT_W_STICK,
    codes: list[str] | None = None,
    incumbent_out: Path | None = None,
    on_incumbent=None,
    on_bound=None,
    initial_best: float = np.inf,
) -> tuple[np.ndarray | None, float, float]:
    """Asigna SKU→caja minimizando surrogate demand-free.

    Returns
    -------
    assign, surrogate_obj, bound
        `surrogate_obj` / `bound` están en unidades del objetivo / SCALE (no USD).
    """
    n_boxes, n = fits.shape
    n_plants = presence.shape[1]
    model = cp_model.CpModel()

    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for b in range(n_boxes):
        for i in np.where(fits[b])[0]:
            x[int(i), b] = model.NewBoolVar(f"x_{i}_{b}")
    for i in range(n):
        model.AddExactlyOne(x[i, b] for b in range(n_boxes) if (i, b) in x)

    n_plants_i = presence.sum(axis=1).astype(int)
    box_vol = np.rint(boxes.prod(axis=1)).astype(np.int64)
    prod_vol_i = np.rint(prod_vol).astype(np.int64)

    # Precio base → enteros (milésimas de USD * SCALE ya es SCALE; usamos centavos*SCALE).
    precio_b = np.array(
        [int(round(PRECIO_BASE_GROSOR[float(g)] * SCALE)) for g in box_grosor],
        dtype=np.int64,
    )

    obj_vars: list = []
    obj_coefs: list[int] = []

    # --- 1+2: eficiencia + costo unitario + waste geométrico (por asignación) ---
    for b in range(n_boxes):
        members = np.where(fits[b])[0]
        cpp_b = max(int(cpp[b]), 1)
        for i in members:
            var = x[int(i), b]
            # Eficiencia: plantas presentes / cpp  (proxy pallets si vol≡1 por planta).
            # Multiplicamos por SCALE para granularidad al dividir.
            eff_coef = int(w_eff * n_plants_i[int(i)] * SCALE // cpp_b)
            if eff_coef:
                obj_vars.append(var)
                obj_coefs.append(eff_coef)

            # Costo unitario: precio_base (sin tiers / sin vol demanda).
            unit_coef = int(w_unit) * int(precio_b[b])
            if unit_coef:
                obj_vars.append(var)
                obj_coefs.append(unit_coef)

            # Waste geométrico (mm³ headspace). Demanda-free (atributo de producto).
            # Contribución a ObjectiveValue/SCALE ≈ w_waste * waste / 1000.
            waste = max(int(box_vol[b] - prod_vol_i[int(i)]), 0)
            waste_coef = int(w_waste) * (waste * SCALE // 1000)
            if waste_coef:
                obj_vars.append(var)
                obj_coefs.append(waste_coef)

    # --- 3+4: consolidación + stickiness por (caja, planta) ---
    for b in range(n_boxes):
        members = np.where(fits[b])[0]
        if len(members) == 0:
            continue
        for p in range(n_plants):
            local = [int(i) for i in members if presence[int(i), p] > 0]
            if not local:
                continue
            count = model.NewIntVar(0, len(local), f"n_{b}_{p}")
            model.Add(count == sum(x[i, b] for i in local if (i, b) in x))

            used = model.NewBoolVar(f"u_{b}_{p}")
            model.Add(count >= 1).OnlyEnforceIf(used)
            model.Add(count == 0).OnlyEnforceIf(used.Not())

            if w_cons:
                obj_vars.append(used)
                obj_coefs.append(int(w_cons) * SCALE)

            if w_stick:
                # singleton <=> count == 1
                singleton = model.NewBoolVar(f"s_{b}_{p}")
                model.Add(count == 1).OnlyEnforceIf(singleton)
                model.Add(count != 1).OnlyEnforceIf(singleton.Not())
                obj_vars.append(singleton)
                obj_coefs.append(int(w_stick) * SCALE)

    model.Minimize(cp_model.LinearExpr.weighted_sum(obj_vars, obj_coefs))

    # MARK17_NO_HINTS=1: no AddHint / repair_hint (workaround ortools abort
    # `heuristics.fixed_search != nullptr` en catálogos sintéticos).
    import os as _os

    _skip_hints = _os.environ.get("MARK17_NO_HINTS", "").strip() in {"1", "true", "True"}

    if hint is not None and not _skip_hints:
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
    if hint is not None and not _skip_hints:
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
        improved = prev is None or bound > float(prev) + 1e-6
        timed = (now - float(last_bound_emit["t"])) >= 1.5
        if not (improved or timed):
            return
        last_bound_emit["t"] = now
        last_bound_emit["bound"] = bound
        on_bound({"bound": bound, "elapsed_s": float(now)})

    if on_bound is not None:
        solver.best_bound_callback = _bound_cb

    callback = None
    if (incumbent_out is not None or on_incumbent is not None) and codes is not None:
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


def surrogate_of_assignment(
    assign: np.ndarray,
    cpp: np.ndarray,
    box_grosor: np.ndarray,
    boxes: np.ndarray,
    presence: np.ndarray,
    prod_vol: np.ndarray,
    *,
    w_eff: int = DEFAULT_W_EFF,
    w_unit: int = DEFAULT_W_UNIT,
    w_waste: int = DEFAULT_W_WASTE,
    w_cons: int = DEFAULT_W_CONS,
    w_stick: int = DEFAULT_W_STICK,
) -> float:
    """Evalúa el surrogate fuera del solver (alineado a ObjectiveValue/SCALE)."""
    n = len(assign)
    n_plants_i = presence.sum(axis=1).astype(int)
    box_vol = np.rint(boxes.prod(axis=1)).astype(np.int64)
    prod_vol_i = np.rint(prod_vol).astype(np.int64)
    # Acumulamos en escala del solver y dividimos por SCALE al final.
    total_scaled = 0

    for i in range(n):
        b = int(assign[i])
        if b < 0:
            return np.inf
        cpp_b = max(int(cpp[b]), 1)
        total_scaled += int(w_eff) * int(n_plants_i[i]) * SCALE // cpp_b
        precio_b = int(round(PRECIO_BASE_GROSOR[float(box_grosor[b])] * SCALE))
        total_scaled += int(w_unit) * precio_b
        waste = max(int(box_vol[b] - prod_vol_i[i]), 0)
        total_scaled += int(w_waste) * (waste * SCALE // 1000)

    n_boxes = len(cpp)
    n_plants = presence.shape[1]
    for b in range(n_boxes):
        for p in range(n_plants):
            members = [
                i
                for i in range(n)
                if int(assign[i]) == b and presence[i, p] > 0
            ]
            if not members:
                continue
            total_scaled += int(w_cons) * SCALE
            if len(members) == 1:
                total_scaled += int(w_stick) * SCALE
    return float(total_scaled) / SCALE


def run_optimize(
    warm_start_path: Path | str,
    grosores: list[float] | None = None,
    time_limit: float = 120.0,
    workers: int | None = None,
    out_path: Path | str | None = None,
    on_progress=None,
    pool_mode: str = "fast",
    top_k: int = 8,
    lim: tuple[float, float, float] | list[float] | None = None,
    *,
    pallet_ancho_mm: float | None = None,
    pallet_largo_mm: float | None = None,
    pallet_alto_mm: float | None = None,
    w_eff: int = DEFAULT_W_EFF,
    w_unit: int = DEFAULT_W_UNIT,
    w_waste: int = DEFAULT_W_WASTE,
    w_cons: int = DEFAULT_W_CONS,
    w_stick: int = DEFAULT_W_STICK,
    eval_official: bool = True,
) -> dict:
    """API programática. Optimiza sin magnitudes de demanda.

    Si `eval_official=True`, al final calcula costo/score Kaggle (diagnóstico;
    no entra al solver).
    """
    import os

    active_lim = normalize_lim(
        lim,
        pallet_ancho_mm=pallet_ancho_mm,
        pallet_largo_mm=pallet_largo_mm,
        pallet_alto_mm=pallet_alto_mm,
    )
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
            cpu_count=os.cpu_count() or 1,
            w_eff=w_eff,
            w_unit=w_unit,
            w_waste=w_waste,
            w_cons=w_cons,
            w_stick=w_stick,
            eval_official=eval_official,
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
    cpu_count: int,
    w_eff: int,
    w_unit: int,
    w_waste: int,
    w_cons: int,
    w_stick: int,
    eval_official: bool,
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
            "message": "Cargando datos (presencia binaria, sin vols en objetivo)…",
            "lim": list(active_lim),
        }
    )
    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    prod = construir_productos(pt, ops)
    baseline = costo_actual_oficial(ops)

    peso = prod["peso_neto_caja"].to_numpy(float)
    presence = plant_presence(prod)
    prod_vol = prod[["largo", "ancho", "alto"]].to_numpy(float).prod(axis=1)
    codes = prod["codigo_producto"].astype(str).str.strip().tolist()

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
                f"Construyendo pool ({pool_mode}, top_k={top_k}, "
                f"pallet {active_lim[0]:.0f}×{active_lim[1]:.0f}×{active_lim[2]:.0f})…"
            ),
            "baseline": baseline,
            "lim": list(active_lim),
            "weights": {
                "w_eff": w_eff,
                "w_unit": w_unit,
                "w_waste": w_waste,
                "w_cons": w_cons,
                "w_stick": w_stick,
            },
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
        if b is not None:
            hint[i] = b
    n_hint = int((hint >= 0).sum())

    def _official_of(sol: pd.DataFrame) -> float:
        return float(
            evaluar_costos_solucion(sol, ops, lim=active_lim)["resumen"][
                "costo_total_nuevo"
            ].iloc[0]
        )

    def on_incumbent(payload: dict) -> None:
        event = {
            "phase": "solving",
            "message": "Nueva mejora (surrogate)",
            "surrogate": float(payload["surrogate"]),
            "baseline": baseline,
            "elapsed_s": float(payload["elapsed_s"]),
            "incumbent": True,
            "demand_free": True,
        }
        if "bound" in payload and np.isfinite(payload["bound"]):
            # Surrogate ≠ USD; no usar como cota Kaggle.
            event["surrogate_bound"] = float(payload["bound"])
        if eval_official and payload.get("solution") is not None:
            cost = _official_of(payload["solution"])
            event["cost"] = cost
            event["ahorro"] = baseline - cost
            event["score"] = score_publico(cost, baseline)
            event["message"] = "Nueva mejora (diagnóstico oficial)"
        progress(event)

    def on_bound(payload: dict) -> None:
        bound = float(payload["bound"])
        if not np.isfinite(bound):
            return
        progress(
            {
                "phase": "solving",
                "message": "Cota CP-SAT (surrogate)",
                "surrogate_bound": bound,
                "baseline": baseline,
                "elapsed_s": float(payload["elapsed_s"]),
                "bound_update": True,
                "demand_free": True,
            }
        )

    warm_surrogate: float | None = None
    warm_assign: np.ndarray | None = None
    if n_hint == len(codes):
        warm_assign = hint.copy()
        warm_surrogate = surrogate_of_assignment(
            warm_assign,
            cpp,
            box_grosor,
            boxes,
            presence,
            prod_vol,
            w_eff=w_eff,
            w_unit=w_unit,
            w_waste=w_waste,
            w_cons=w_cons,
            w_stick=w_stick,
        )
        warm_sol = assignment_to_solution(codes, boxes, box_grosor, warm_assign)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            warm_sol[SUBMISSION_COLS].to_csv(out_path, index=False)
        warm_event = {
            "phase": "solving",
            "message": "Warm start cargado (surrogate)",
            "surrogate": warm_surrogate,
            "baseline": baseline,
            "elapsed_s": 0.0,
            "incumbent": True,
            "demand_free": True,
        }
        if eval_official:
            warm_cost = _official_of(warm_sol)
            warm_event["cost"] = warm_cost
            warm_event["ahorro"] = baseline - warm_cost
            warm_event["score"] = score_publico(warm_cost, baseline)
            warm_event["message"] = "Warm start cargado"
        progress(warm_event)

    progress(
        {
            "phase": "solving",
            "message": (
                f"CP-SAT demand-free ({time_limit:.0f}s, {workers} workers, "
                f"hint {n_hint}/{len(codes)})…"
            ),
            "baseline": baseline,
            "time_limit": time_limit,
        }
    )
    assign, surrogate, bound = solve_demand_free(
        fits,
        cpp,
        box_grosor,
        boxes,
        presence,
        prod_vol,
        hint if (hint >= 0).any() else None,
        time_limit,
        workers,
        w_eff=w_eff,
        w_unit=w_unit,
        w_waste=w_waste,
        w_cons=w_cons,
        w_stick=w_stick,
        codes=codes,
        incumbent_out=out_path,
        on_incumbent=on_incumbent,
        on_bound=on_bound,
        initial_best=warm_surrogate if warm_surrogate is not None else np.inf,
    )

    # Conservar warm si el solver no mejora el surrogate (sin mirar costo oficial).
    if warm_assign is not None and warm_surrogate is not None:
        if assign is None:
            assign, surrogate = warm_assign, warm_surrogate
        else:
            s_chk = surrogate_of_assignment(
                assign,
                cpp,
                box_grosor,
                boxes,
                presence,
                prod_vol,
                w_eff=w_eff,
                w_unit=w_unit,
                w_waste=w_waste,
                w_cons=w_cons,
                w_stick=w_stick,
            )
            if s_chk > warm_surrogate + 1e-6:
                assign, surrogate = warm_assign, warm_surrogate
            else:
                surrogate = s_chk

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
            "surrogate": None,
            "cost": None,
            "solution_path": None,
        }

    solution = assignment_to_solution(codes, boxes, box_grosor, assign)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        solution[SUBMISSION_COLS].to_csv(out_path, index=False)

    n_tipos = int(
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
    )

    official_cost = None
    official_score = None
    if eval_official:
        # Diagnóstico con demanda real — NO usado en la optimización.
        official_cost = float(
            evaluar_costos_solucion(solution, ops, lim=active_lim)["resumen"][
                "costo_total_nuevo"
            ].iloc[0]
        )
        official_score = score_publico(official_cost, baseline)

    done_event = {
        "phase": "done",
        "message": "Optimización demand-free terminada",
        "surrogate": float(surrogate),
        "surrogate_bound": float(bound),
        "baseline": baseline,
        "cost": official_cost,
        "score": official_score,
        "n_tipos": n_tipos,
        "solution_path": str(out_path) if out_path else None,
        "demand_free": True,
    }
    if official_cost is not None:
        done_event["ahorro"] = baseline - float(official_cost)
    progress(done_event)
    return {
        "ok": True,
        "baseline": baseline,
        "bound": float(bound),
        "surrogate": float(surrogate),
        "cost": official_cost,
        "score": official_score,
        "solution": solution,
        "solution_path": str(out_path) if out_path else None,
        "lim": list(active_lim),
        "n_tipos": n_tipos,
        "demand_free": True,
        "weights": {
            "w_eff": w_eff,
            "w_unit": w_unit,
            "w_waste": w_waste,
            "w_cons": w_cons,
            "w_stick": w_stick,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mark 17: CP-SAT demand-free (eficiencia, costo unitario, "
            "consolidación por planta, stickiness)."
        )
    )
    parser.add_argument(
        "--grosores",
        default=None,
        help="Lista separada por comas (default: settings.default_grosores).",
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--warm-start",
        default="outputs/tables/00_baseline_solution.csv",
        help="CSV de partida (default: baseline; evita warm Mark 15/16).",
    )
    parser.add_argument("--out-dir", default="outputs/mark17")
    parser.add_argument("--best-out", default="outputs/mark17/mark17_best.csv")
    parser.add_argument(
        "--pool",
        choices=("fast", "full"),
        default="fast",
        help="fast=top_k por SKU (default); full=meshgrid global.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--w-eff", type=int, default=DEFAULT_W_EFF)
    parser.add_argument("--w-unit", type=int, default=DEFAULT_W_UNIT)
    parser.add_argument("--w-waste", type=int, default=DEFAULT_W_WASTE)
    parser.add_argument("--w-cons", type=int, default=DEFAULT_W_CONS)
    parser.add_argument("--w-stick", type=int, default=DEFAULT_W_STICK)
    parser.add_argument(
        "--no-eval-official",
        action="store_true",
        help="No calcular costo/score Kaggle al final (puro demand-free).",
    )
    args = parser.parse_args()

    grosores = parse_grosores(args.grosores)
    print(
        f"Mark 17 demand-free | grosores={grosores} | pool={args.pool} "
        f"top_k={args.top_k}",
        flush=True,
    )
    print(
        f"Pesos: eff={args.w_eff} unit={args.w_unit} waste={args.w_waste} "
        f"cons={args.w_cons} stick={args.w_stick}",
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
        w_eff=args.w_eff,
        w_unit=args.w_unit,
        w_waste=args.w_waste,
        w_cons=args.w_cons,
        w_stick=args.w_stick,
        eval_official=not args.no_eval_official,
        on_progress=lambda e: print(
            f"[{e.get('phase')}] {e.get('message', '')}", flush=True
        ),
    )
    if not result.get("ok"):
        print("Sin solución factible dentro del tiempo.", flush=True)
        return

    pt = pd.read_csv(PRODUCTS_TRUTH_PATH)
    solution = result["solution"]
    baseline = result["baseline"]
    surrogate = result["surrogate"]
    bound = result["bound"]
    print(f"\nSurrogate: {surrogate:,.4f} | cota {bound:,.4f}", flush=True)
    print(f"Tipos de caja: {result['n_tipos']}", flush=True)
    if result.get("cost") is not None:
        print(
            f"Diagnóstico oficial (NO usado en opt): "
            f"{result['cost']:,.2f} (score {result['score']:+.5f})",
            flush=True,
        )
        print(f"Baseline oficial: {baseline:,.2f}", flush=True)

    if len(grosores) == 1 and set(grosores).issubset({3.0, 4.5, 5.0}):
        ok, errors, _ = validate_solution(solution, pt, verbose=False)
        print(f"Válida (oficial): {ok} {errors[:3] if not ok else ''}", flush=True)
    else:
        ok, errors, warnings, _ = validate_solution_free(solution, pt, verbose=False)
        print(f"Válida (free): {ok} {errors[:3] if not ok else ''}", flush=True)
        if warnings:
            print(f"  warnings: {warnings[:2]}", flush=True)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    g_tag = "-".join(str(g) for g in grosores)
    out = out_dir / f"mark17_g{g_tag}_{stamp}.csv"
    solution[SUBMISSION_COLS].to_csv(out, index=False)
    print(f"Guardado: {out}", flush=True)
    print(f"Best-out: {REPO_ROOT / args.best_out}", flush=True)


if __name__ == "__main__":
    main()
