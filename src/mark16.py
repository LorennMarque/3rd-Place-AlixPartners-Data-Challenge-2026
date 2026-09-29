"""Mark 16 — CP-SAT global con lista de grosores permitidos.

Equivalente a Mark 15 cuando `grosores=[3.0]` (default / Kaggle).
Con varios grosores (ej. 3.0,4.5,5.0) cada tipo de caja puede elegir el suyo
(modo tipo Prometeo).

Uso:
    PYTHONPATH=src python src/mark16.py
    PYTHONPATH=src python src/mark16.py --grosores 3.0,4.5,5.0 --time-limit 600
"""

from __future__ import annotations

import argparse
import math
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from evaluate import (
    evaluar_costos_solucion,
    inferir_costos_flete_por_planta,
    validate_solution,
    validate_solution_free,
)
from settings import (
    DEFAULT_GROSORES,
    ECT,
    EPS,
    GRAVITY,
    HEADSPACE_ABS_MAX,
    HEADSPACE_PCT,
    LIM,
    MAX_GROW,
    MAX_SHRINK,
    OPERACIONES_PLANTA_PATH,
    PLANTAS,
    PRECIO_BASE_GROSOR,
    PRODUCTS_TRUTH_PATH,
    REPO_ROOT,
    SCALE,
    SUBMISSION_COLS,
    TIERS,
)

# Override thread-safe de límites de pallet (L→ancho, W→largo, H→alto).
_LIM_CTX: ContextVar[tuple[float, float, float]] = ContextVar("mark16_lim", default=LIM)


def _lim() -> tuple[float, float, float]:
    return _LIM_CTX.get()


def normalize_lim(
    lim: tuple[float, float, float] | list[float] | None = None,
    *,
    pallet_ancho_mm: float | None = None,
    pallet_largo_mm: float | None = None,
    pallet_alto_mm: float | None = None,
) -> tuple[float, float, float]:
    """Convierte dims de pallet a LIM=(ancho, largo, alto) de orientación fija."""
    if lim is not None:
        vals = tuple(float(x) for x in lim)
        if len(vals) != 3:
            raise ValueError(f"lim debe tener 3 valores, got {vals}")
        if any(v <= 0 for v in vals):
            raise ValueError(f"lim debe ser > 0, got {vals}")
        return vals  # type: ignore[return-value]
    ancho = float(pallet_ancho_mm if pallet_ancho_mm is not None else LIM[0])
    largo = float(pallet_largo_mm if pallet_largo_mm is not None else LIM[1])
    alto = float(pallet_alto_mm if pallet_alto_mm is not None else LIM[2])
    if ancho <= 0 or largo <= 0 or alto <= 0:
        raise ValueError(f"Dims de pallet inválidas: ({ancho}, {largo}, {alto})")
    return (ancho, largo, alto)

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def construir_productos(
    products_truth: pd.DataFrame, operaciones_planta: pd.DataFrame
) -> pd.DataFrame:
    """Une dimensiones del producto con volúmenes por planta."""
    vol_cols = [f"volumen_producto_planta_{p}" for p in PLANTAS]
    prod = products_truth[
        ["codigo_producto", "largo", "ancho", "alto", "peso_neto_caja"]
    ].copy()
    prod["codigo_producto"] = prod["codigo_producto"].astype(str).str.strip()

    ops = operaciones_planta[["codigo_producto", "volumen_producto_total", *vol_cols]].copy()
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()

    df = prod.merge(ops, on="codigo_producto", how="left", validate="one_to_one")
    if df["volumen_producto_total"].isna().any():
        faltan = df.loc[df["volumen_producto_total"].isna(), "codigo_producto"].tolist()
        raise ValueError(f"Productos sin volúmenes en operaciones_planta: {faltan[:5]}")
    return df


def costo_actual_oficial(operaciones_planta: pd.DataFrame) -> float:
    return float(
        operaciones_planta["costo_total"].sum()
        + operaciones_planta["costo_pallets_total"].sum()
    )


def score_publico(costo: float, costo_actual: float) -> float:
    return 100.0 * (costo_actual - costo) / costo_actual


def parse_grosores(text: str | None) -> list[float]:
    """'3.0,4.5,5.0' → lista validada contra settings."""
    if not text:
        return list(DEFAULT_GROSORES)
    grosores = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not grosores:
        raise ValueError("Lista de grosores vacía")
    unknown = [g for g in grosores if g not in PRECIO_BASE_GROSOR]
    if unknown:
        raise ValueError(
            f"Grosores sin precio/ECT en settings.py: {unknown}. "
            f"Permitidos: {sorted(PRECIO_BASE_GROSOR)}"
        )
    return grosores


# ---------------------------------------------------------------------------
# Geometría (regla 9 literal)
# ---------------------------------------------------------------------------


def intervalos(prod: pd.DataFrame, grosor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lo, hi, vol_min por producto para un grosor."""
    dims = prod[["largo", "ancho", "alto"]].to_numpy(float)
    pct = HEADSPACE_PCT[grosor]
    lo = MAX_SHRINK * dims
    hi = np.minimum.reduce(
        [MAX_GROW * dims, dims / (1.0 - pct), dims + HEADSPACE_ABS_MAX]
    )
    return lo, hi, dims.prod(axis=1)


def cajas_por_pallet(boxes: np.ndarray, grosor: float) -> np.ndarray:
    lim = _lim()
    ext = boxes + 2.0 * grosor
    return (
        np.floor(lim[0] / ext[..., 0])
        * np.floor(lim[1] / ext[..., 1])
        * np.floor(lim[2] / ext[..., 2])
    ).astype(int)


def fits_matrix(
    boxes: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    vol_min: np.ndarray,
    peso: np.ndarray,
    grosor: float,
) -> np.ndarray:
    """(B, n): producto i cabe en caja interior b con este grosor."""
    lim = _lim()
    ge = (boxes[:, None, :] >= lo[None, :, :] - 1e-6).all(axis=2)
    le = (boxes[:, None, :] <= hi[None, :, :] + 1e-6).all(axis=2)
    vol_ok = boxes.prod(axis=1)[:, None] >= vol_min[None, :] - 1e-6

    ext = boxes + 2.0 * grosor
    capas = np.floor(lim[2] / ext[:, 2]).astype(int)
    carga = ECT[grosor] * 2 * (ext[:, 0] + ext[:, 1]) / 1000.0 / GRAVITY
    comp_ok = peso[None, :] * np.maximum(capas - 1, 0)[:, None] <= carga[:, None] + EPS
    return ge & le & vol_ok & comp_ok


# ---------------------------------------------------------------------------
# Pool de cajas
# ---------------------------------------------------------------------------


def _cands_eje(lo_k: float, hi_k: float, limite: float, grosor: float) -> list[float]:
    """Umbrales de pallet + extremos de la ventana en un eje."""
    cands = {np.ceil(lo_k * 100) / 100, np.floor(hi_k * 100) / 100}
    c_max = int(np.floor(limite / (lo_k + 2 * grosor)))
    c_min = max(1, int(np.floor(limite / (hi_k + 2 * grosor))))
    for c in range(c_min, c_max + 1):
        t = np.floor((limite / c - 2 * grosor) * 100) / 100
        if lo_k - EPS <= t <= hi_k + EPS:
            cands.add(float(t))
    return sorted(cands)


def _podar_dominadas(
    boxes: np.ndarray, fits: np.ndarray, cpp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dominancia: b dominada si existe k con fits[k] ⊇ fits[b] y cpp[k] ≥ cpp[b]."""
    order = np.argsort(-cpp, kind="stable")
    packed = np.packbits(fits, axis=1)
    kept: list[int] = []
    kept_pack: list[np.ndarray] = []
    kept_cpp: list[int] = []
    for b in order:
        sb = packed[b]
        cb = int(cpp[b])
        dominated = False
        for pk, ck in zip(kept_pack, kept_cpp):
            if ck >= cb and np.array_equal(np.bitwise_or(pk, sb), pk):
                dominated = True
                break
        if not dominated:
            kept.append(int(b))
            kept_pack.append(sb)
            kept_cpp.append(cb)
    kept = sorted(kept)
    return boxes[kept], fits[kept], cpp[kept]


def _pool_topk(
    lo: np.ndarray,
    hi: np.ndarray,
    vol_min: np.ndarray,
    peso: np.ndarray,
    grosor: float,
    top_k: int = 8,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool liviano: top_k cajas por SKU (umbrales de su ventana) + extras."""
    n = len(lo)
    pool: dict[tuple, int] = {}
    boxes_list: list[np.ndarray] = []

    lim = _lim()
    for i in range(n):
        cands = [_cands_eje(float(lo[i, k]), float(hi[i, k]), lim[k], grosor) for k in range(3)]
        factibles: list[tuple[int, float, tuple]] = []
        for length in cands[0]:
            a = int(np.floor(lim[0] / (length + 2 * grosor)))
            for width in cands[1]:
                b = int(np.floor(lim[1] / (width + 2 * grosor)))
                carga_max = ECT[grosor] * 2 * (length + width + 4 * grosor) / 1000.0 / GRAVITY
                for height in cands[2]:
                    vol = length * width * height
                    if vol < vol_min[i] - 1e-6:
                        continue
                    c = int(np.floor(lim[2] / (height + 2 * grosor)))
                    if a < 1 or b < 1 or c < 1:
                        continue
                    if peso[i] * (c - 1) > carga_max + EPS:
                        continue
                    factibles.append((a * b * c, vol, (length, width, height)))
        if not factibles:
            key = tuple(np.round(hi[i], 2))
            if key not in pool:
                pool[key] = len(boxes_list)
                boxes_list.append(np.array(key, dtype=float))
            continue
        factibles.sort(key=lambda t: (-t[0], -t[1]))
        for _, _, dims in factibles[:top_k]:
            key = tuple(np.round(dims, 2))
            if key not in pool:
                pool[key] = len(boxes_list)
                boxes_list.append(np.array(key, dtype=float))

    boxes = np.asarray(boxes_list, dtype=float)
    boxes, fits, cpp = _finalize_pool(boxes, lo, hi, vol_min, peso, grosor, prune=True)

    # Warm-start / extras: se fuerzan al pool (no se podan).
    if extra is not None and len(extra):
        extra = np.unique(np.round(extra, 2), axis=0)
        extra_fits = fits_matrix(extra, lo, hi, vol_min, peso, grosor)
        extra_cpp = cajas_por_pallet(extra, grosor)
        keep_extra = (extra_cpp > 0) & extra_fits.any(axis=1)
        extra, extra_fits, extra_cpp = extra[keep_extra], extra_fits[keep_extra], extra_cpp[keep_extra]
        if len(extra):
            # Deduplicar contra boxes existentes.
            existing = {tuple(row) for row in np.round(boxes, 2)}
            mask_new = np.array([tuple(row) not in existing for row in np.round(extra, 2)])
            if mask_new.any():
                boxes = np.vstack([boxes, extra[mask_new]])
                fits = np.vstack([fits, extra_fits[mask_new]])
                cpp = np.concatenate([cpp, extra_cpp[mask_new]])
    return boxes, fits, cpp


def _finalize_pool(
    boxes: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    vol_min: np.ndarray,
    peso: np.ndarray,
    grosor: float,
    *,
    prune: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.unique(np.round(boxes, 2), axis=0)
    fits = fits_matrix(boxes, lo, hi, vol_min, peso, grosor)
    cpp = cajas_por_pallet(boxes, grosor)
    valid = (cpp > 0) & fits.any(axis=1)
    boxes, fits, cpp = boxes[valid], fits[valid], cpp[valid]
    if prune and len(boxes):
        return _podar_dominadas(boxes, fits, cpp)
    return boxes, fits, cpp


def _pool_full(
    lo: np.ndarray,
    hi: np.ndarray,
    vol_min: np.ndarray,
    peso: np.ndarray,
    grosor: float,
    extra: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool completo (meshgrid global). Más lento; útil para cotas overnight."""
    lim = _lim()
    n = len(lo)
    axis_values: list[np.ndarray] = []
    for k in range(3):
        vals: set[float] = set()
        lo_min, hi_max = float(lo[:, k].min()), float(hi[:, k].max())
        c_max = int(np.floor(lim[k] / (lo_min + 2 * grosor)))
        c_min = max(1, int(np.floor(lim[k] / (hi_max + 2 * grosor))))
        for c in range(c_min, c_max + 1):
            t = np.floor((lim[k] / c - 2 * grosor) * 100) / 100
            if t >= lo_min - 1e-9:
                vals.add(float(t))
        for i in range(n):
            vals.add(float(np.floor(hi[i, k] * 100) / 100))
        arr = np.array(sorted(vals))
        in_window = (
            (arr[None, :] >= lo[:, k][:, None] - 1e-9)
            & (arr[None, :] <= hi[:, k][:, None] + 1e-9)
        ).any(axis=0)
        axis_values.append(arr[in_window])

    g1, g2, g3 = np.meshgrid(*axis_values, indexing="ij")
    boxes = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
    boxes, fits, cpp = _finalize_pool(boxes, lo, hi, vol_min, peso, grosor, prune=True)

    # Extras (warm start) fuera de la poda de dominancia.
    if extra is not None and len(extra):
        extra = np.unique(np.round(extra, 2), axis=0)
        extra_fits = fits_matrix(extra, lo, hi, vol_min, peso, grosor)
        extra_cpp = cajas_por_pallet(extra, grosor)
        keep_extra = (extra_cpp > 0) & extra_fits.any(axis=1)
        extra, extra_fits, extra_cpp = extra[keep_extra], extra_fits[keep_extra], extra_cpp[keep_extra]
        if len(extra):
            existing = {tuple(row) for row in np.round(boxes, 2)}
            mask_new = np.array([tuple(row) not in existing for row in np.round(extra, 2)])
            if mask_new.any():
                boxes = np.vstack([boxes, extra[mask_new]])
                fits = np.vstack([fits, extra_fits[mask_new]])
                cpp = np.concatenate([cpp, extra_cpp[mask_new]])
    return boxes, fits, cpp


def build_pool(
    prod: pd.DataFrame,
    peso: np.ndarray,
    grosores: list[float],
    extras_por_grosor: dict[float, np.ndarray] | None = None,
    *,
    mode: str = "fast",
    top_k: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool multi-grosor.

    mode='fast' (default): top_k por SKU — segundos o menos.
    mode='full': meshgrid global — minutos; mejor para cotas.
    """
    extras_por_grosor = extras_por_grosor or {}
    box_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    fit_parts: list[np.ndarray] = []
    cpp_parts: list[np.ndarray] = []
    builder = _pool_full if mode == "full" else _pool_topk

    for g in grosores:
        lo, hi, vol_min = intervalos(prod, g)
        extra = extras_por_grosor.get(g)
        if mode == "full":
            boxes, fits, cpp = builder(lo, hi, vol_min, peso, g, extra)
        else:
            boxes, fits, cpp = builder(lo, hi, vol_min, peso, g, top_k=top_k, extra=extra)
        box_parts.append(boxes)
        g_parts.append(np.full(len(boxes), g, dtype=float))
        fit_parts.append(fits)
        cpp_parts.append(cpp)

    return (
        np.vstack(box_parts),
        np.concatenate(g_parts),
        np.vstack(fit_parts),
        np.concatenate(cpp_parts),
    )


def assignment_to_solution(
    codes: list[str],
    boxes: np.ndarray,
    box_grosor: np.ndarray,
    assign: np.ndarray,
) -> pd.DataFrame:
    g = box_grosor[assign]
    ext = np.round(boxes[assign] + 2.0 * g[:, None], 2)
    return pd.DataFrame(
        {
            "codigo_producto": codes,
            "caja_grosor_mm": g,
            "caja_exterior_largo": ext[:, 0],
            "caja_exterior_ancho": ext[:, 1],
            "caja_exterior_alto": ext[:, 2],
        }
    )


# ---------------------------------------------------------------------------
# CP-SAT
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


def solve_global(
    fits: np.ndarray,
    cpp: np.ndarray,
    box_grosor: np.ndarray,
    flete_por_cpp: dict[int, np.ndarray],
    vol_pl: np.ndarray,
    hint: np.ndarray | None,
    time_limit: float,
    workers: int,
    boxes: np.ndarray | None = None,
    codes: list[str] | None = None,
    incumbent_out: Path | None = None,
    on_incumbent=None,
    on_bound=None,
    initial_best: float = np.inf,
) -> tuple[np.ndarray | None, float, float]:
    """Asigna cada SKU a una columna del pool. Packaging con precio del grosor de la caja."""
    n_boxes, n = fits.shape
    model = cp_model.CpModel()

    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for b in range(n_boxes):
        for i in np.where(fits[b])[0]:
            x[int(i), b] = model.NewBoolVar(f"x_{i}_{b}")
    for i in range(n):
        model.AddExactlyOne(x[i, b] for b in range(n_boxes) if (i, b) in x)

    obj_vars: list = []
    obj_coefs: list[int] = []

    for b in range(n_boxes):
        members = np.where(fits[b])[0]
        f_col = flete_por_cpp[int(cpp[b])]
        for i in members:
            obj_vars.append(x[int(i), b])
            obj_coefs.append(int(f_col[i]))

        precio = PRECIO_BASE_GROSOR[float(box_grosor[b])]
        rates = [int(round(precio * factor * SCALE)) for _, _, factor in TIERS]

        for p in range(vol_pl.shape[1]):
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
    # Completa variables de packaging a partir del hint de asignación.
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
        # Throttle: emitir si mejora o cada ~1.5s.
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
    if (incumbent_out is not None or on_incumbent is not None) and boxes is not None and codes is not None:
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
) -> dict:
    """API programática para el dashboard. `on_progress(event: dict)` es opcional.

    `lim` = (ancho, largo, alto) en mm con orientación fija L→ancho, W→largo, H→alto.
    Alternativa: `pallet_*_mm` con nombres de packaging (defaults = settings).
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
            "message": "Cargando datos…",
            "lim": list(active_lim),
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
    cvec = np.array([costs[p] for p in PLANTAS])

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

    flete_por_cpp: dict[int, np.ndarray] = {}
    for c in np.unique(cpp):
        pallets = np.ceil(vol_pl / float(c))
        flete_por_cpp[int(c)] = np.rint((pallets * cvec).sum(axis=1) * SCALE).astype(
            np.int64
        )

    # Hint exacto = caja del warm start (no max-cpp del grupo).
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

    # Sembrar incumbent con el warm start (UI + archivo) antes del solve.
    warm_cost: float | None = None
    warm_assign: np.ndarray | None = None
    if n_hint == len(codes):
        warm_assign = hint.copy()
        warm_sol = assignment_to_solution(codes, boxes, box_grosor, warm_assign)
        warm_cost = float(
            evaluar_costos_solucion(warm_sol, ops, lim=active_lim)["resumen"][
                "costo_total_nuevo"
            ].iloc[0]
        )
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            warm_sol[SUBMISSION_COLS].to_csv(out_path, index=False)
        progress(
            {
                "phase": "solving",
                "message": "Warm start cargado",
                "cost": warm_cost,
                "baseline": baseline,
                "ahorro": baseline - warm_cost,
                "score": score_publico(warm_cost, baseline),
                "elapsed_s": 0.0,
                "incumbent": True,
            }
        )

    progress(
        {
            "phase": "solving",
            "message": (
                "CP-SAT ("
                + (
                    "sin límite"
                    if (not math.isfinite(float(time_limit)) or float(time_limit) <= 0)
                    else f"{float(time_limit):.0f}s"
                )
                + f", {workers} workers, hint {n_hint}/{len(codes)})…"
            ),
            "baseline": baseline,
            "time_limit": time_limit,
        }
    )
    assign, cost, bound = solve_global(
        fits,
        cpp,
        box_grosor,
        flete_por_cpp,
        vol_pl,
        hint if (hint >= 0).any() else None,
        time_limit,
        workers,
        boxes=boxes,
        codes=codes,
        incumbent_out=out_path,
        on_incumbent=on_incumbent,
        on_bound=on_bound,
        initial_best=warm_cost if warm_cost is not None else np.inf,
    )

    # Si el solver no mejora el warm start (evaluación oficial), conservar el warm.
    if warm_assign is not None and warm_cost is not None:
        if assign is None:
            assign, cost = warm_assign, warm_cost
        else:
            sol_tmp = assignment_to_solution(codes, boxes, box_grosor, assign)
            cost = float(
                evaluar_costos_solucion(sol_tmp, ops, lim=active_lim)["resumen"][
                    "costo_total_nuevo"
                ].iloc[0]
            )
            if cost > warm_cost + 0.01:
                assign, cost = warm_assign, warm_cost

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
        }

    solution = assignment_to_solution(codes, boxes, box_grosor, assign)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        solution[SUBMISSION_COLS].to_csv(out_path, index=False)

    progress(
        {
            "phase": "done",
            "message": "Optimización terminada",
            "cost": float(cost),
            "bound": float(bound),
            "baseline": baseline,
            "ahorro": baseline - float(cost),
            "score": score_publico(float(cost), baseline),
            "elapsed_s": float(time_limit),
            "n_tipos": int(
                solution[
                    ["caja_grosor_mm", "caja_exterior_largo", "caja_exterior_ancho", "caja_exterior_alto"]
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
        "score": score_publico(float(cost), baseline),
        "solution": solution,
        "solution_path": str(out_path) if out_path else None,
        "lim": list(active_lim),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark 16: CP-SAT global con grosores configurables."
    )
    parser.add_argument(
        "--grosores",
        default=None,
        help="Lista separada por comas (default: settings.default_grosores, hoy 3.0).",
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--warm-start",
        default="outputs/mark15/mark15_best.csv",
        help="CSV de partida (p. ej. mejor Mark 15 o un Mark 16 previo).",
    )
    parser.add_argument("--out-dir", default="outputs/mark16")
    parser.add_argument("--best-out", default="outputs/mark16/mark16_best.csv")
    parser.add_argument(
        "--pool",
        choices=("fast", "full"),
        default="fast",
        help="fast=top_k por SKU (default); full=meshgrid global (lento).",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Cajas por SKU en pool fast.")
    args = parser.parse_args()

    grosores = parse_grosores(args.grosores)
    print(f"Grosores: {grosores} | pool={args.pool} top_k={args.top_k}", flush=True)

    result = run_optimize(
        warm_start_path=REPO_ROOT / args.warm_start,
        grosores=grosores,
        time_limit=args.time_limit,
        workers=args.workers,
        out_path=REPO_ROOT / args.best_out,
        pool_mode=args.pool,
        top_k=args.top_k,
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
    cost = result["cost"]
    bound = result["bound"]
    print(
        f"\nCota CP-SAT: {bound:,.2f} (score techo {score_publico(bound, baseline):+.5f})",
        flush=True,
    )
    print(f"Mejor: {cost:,.2f} (score {score_publico(cost, baseline):+.5f})", flush=True)

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
    out = out_dir / f"mark16_g{g_tag}_{stamp}.csv"
    solution[SUBMISSION_COLS].to_csv(out, index=False)
    print(f"Guardado: {out}", flush=True)


if __name__ == "__main__":
    main()
