"""Repeated holdout backtest (leave-p-out) para escalabilidad de catálogo.

Protocolo:
1. Hold-out estratificado (~15%) de productos reales.
2. Reconstruir Portafolio Actual y Mark17-minbox **solo** con el train.
3. Introducir hold-out secuencialmente con asignación M1.
4. Medir reuso, cajas nuevas, util. pallet y costo Kaggle (demanda real).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import PRECIO_BASE_GROSOR_FULL, SUBMISSION_COLS
from scaling_launch import (
    BOX_SPEC_COLS,
    PALLET_ANCHO,
    PALLET_LARGO,
    PALLET_MAX_ALTO,
    add_box_id,
    caja_cumple_restricciones_fisicas,
    crear_caja_minima,
    evaluar_kaggle,
)


def stratified_holdout_codes(
    meta: pd.DataFrame,
    *,
    holdout_frac: float = 0.15,
    seed: int,
    strata_col: str = "categoria",
    min_train_per_stratum: int = 3,
) -> tuple[list[str], list[str]]:
    """Split estratificado. Estratos chicos se quedan enteros en train."""
    rng = np.random.default_rng(seed)
    df = meta[["codigo_producto", strata_col]].copy()
    df["codigo_producto"] = df["codigo_producto"].astype(str).str.strip()
    df[strata_col] = df[strata_col].fillna("NA").astype(str)

    train, hold = [], []
    for _, g in df.groupby(strata_col, sort=False):
        codes = g["codigo_producto"].tolist()
        n = len(codes)
        if n <= min_train_per_stratum:
            train.extend(codes)
            continue
        n_hold = int(round(holdout_frac * n))
        n_hold = min(max(n_hold, 1), n - min_train_per_stratum)
        chosen = set(rng.choice(codes, size=n_hold, replace=False).tolist())
        hold.extend(chosen)
        train.extend([c for c in codes if c not in chosen])
    return sorted(train), sorted(hold)


def build_actual_from_joined(joined: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """Portafolio Actual = cajas históricas de los SKUs train (sin leakage del holdout)."""
    j = joined.copy()
    j["codigo_producto"] = j["codigo_producto"].astype(str).str.strip()
    sol = j.loc[j["codigo_producto"].isin(codes), SUBMISSION_COLS].copy()
    for c in SUBMISSION_COLS[1:]:
        sol[c] = pd.to_numeric(sol[c], errors="coerce")
    if sol.isna().any().any():
        raise ValueError("Actual train con NaNs")
    if sol["codigo_producto"].nunique() != len(set(codes)):
        missing = set(codes) - set(sol["codigo_producto"])
        raise ValueError(f"Faltan {len(missing)} códigos en joined para Actual train")
    return sol.sort_values("codigo_producto").reset_index(drop=True)


def optimize_mark17_minbox(
    products_truth_train: pd.DataFrame,
    ops_train: pd.DataFrame,
    warm_train: pd.DataFrame,
    *,
    out_path: Path,
    time_limit: float = 30.0,
    workers: int = 1,
    grosor: float = 3.0,
    top_k: int = 8,
) -> pd.DataFrame:
    """Mark17 g=3.0 sobre subset train (worker aislado)."""
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts" / "_mark17_ckpt_worker.py"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="holdout_m17_") as tmp:
        tmp = Path(tmp)
        pt_p = tmp / "pt.csv"
        ops_p = tmp / "ops.csv"
        warm_p = tmp / "warm.csv"
        products_truth_train.to_csv(pt_p, index=False)
        ops_train.to_csv(ops_p, index=False)
        warm_train[SUBMISSION_COLS].to_csv(warm_p, index=False)

        proc = subprocess.run(
            [
                sys.executable,
                str(worker),
                "--pt",
                str(pt_p),
                "--ops",
                str(ops_p),
                "--warm",
                str(warm_p),
                "--out",
                str(out_path),
                "--tl",
                str(float(time_limit)),
                "--workers",
                str(max(1, int(workers))),
                "--grosor",
                str(float(grosor)),
                "--top-k",
                str(int(top_k)),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"Mark17 holdout falló rc={proc.returncode}\n"
                f"stdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-1500:]}"
            )

    sol = pd.read_csv(out_path)[SUBMISSION_COLS].copy()
    sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
    return sol


def _catalogo_liviano(sol: pd.DataFrame) -> list[dict]:
    u = add_box_id(sol).drop_duplicates("box_id")
    out = []
    for r in u.itertuples(index=False):
        L, W, H = float(r.caja_exterior_largo), float(r.caja_exterior_ancho), float(r.caja_exterior_alto)
        g = float(r.caja_grosor_mm)
        out.append(
            {
                "box_id": r.box_id,
                "g": g,
                "L": L,
                "W": W,
                "H": H,
                "vol": L * W * H,
                "cpp": float(
                    np.floor(PALLET_ANCHO / L)
                    * np.floor(PALLET_LARGO / W)
                    * np.floor(PALLET_MAX_ALTO / H)
                ),
                "precio_base": float(PRECIO_BASE_GROSOR_FULL.get(round(g, 1), 0.0) or 0.0),
            }
        )
    return out


def assign_holdout(
    sol_train: pd.DataFrame,
    holdout_products: pd.DataFrame,
    policy: str = "m1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna holdout secuencialmente (M1 o pallet-aware). Devuelve (sol_final, detalle)."""
    if policy not in {"m1", "pallet"}:
        raise ValueError(f"policy inválida: {policy}")
    cat = _catalogo_liviano(sol_train)
    init_ids = {b["box_id"] for b in cat}
    rows = []
    assigned = []
    n_nuevas = 0

    for i, p in enumerate(holdout_products.itertuples(index=False), start=1):
        lo, an, al, pe = float(p.largo), float(p.ancho), float(p.alto), float(p.peso_neto_caja)
        compat = [
            b
            for b in cat
            if caja_cumple_restricciones_fisicas(
                lo, an, al, pe, b["g"], b["L"], b["W"], b["H"]
            )
        ]
        if compat:
            if policy == "m1":
                best = min(compat, key=lambda b: b["vol"])
            else:
                best = max(
                    compat,
                    key=lambda b: (b["cpp"], -b["precio_base"], -b["vol"]),
                )
            es_nueva = False
            caja = {
                "caja_grosor_mm": best["g"],
                "caja_exterior_largo": best["L"],
                "caja_exterior_ancho": best["W"],
                "caja_exterior_alto": best["H"],
                "box_id": best["box_id"],
            }
        else:
            es_nueva = True
            caja = crear_caja_minima(
                {
                    "largo": lo,
                    "ancho": an,
                    "alto": al,
                    "peso_neto_caja": pe,
                    "codigo_producto": str(p.codigo_producto),
                }
            )
            L, W, H = (
                caja["caja_exterior_largo"],
                caja["caja_exterior_ancho"],
                caja["caja_exterior_alto"],
            )
            g = caja["caja_grosor_mm"]
            cat.append(
                {
                    "box_id": caja["box_id"],
                    "g": g,
                    "L": L,
                    "W": W,
                    "H": H,
                    "vol": L * W * H,
                    "cpp": float(
                        np.floor(PALLET_ANCHO / L)
                        * np.floor(PALLET_LARGO / W)
                        * np.floor(PALLET_MAX_ALTO / H)
                    ),
                    "precio_base": float(
                        PRECIO_BASE_GROSOR_FULL.get(round(g, 1), 0.0) or 0.0
                    ),
                }
            )

        n_nuevas += int(es_nueva)
        destino = (
            "Caja nueva"
            if es_nueva
            else ("Catálogo train" if caja["box_id"] in init_ids else "Creada en holdout")
        )
        rows.append(
            {
                "orden": i,
                "codigo_producto": str(p.codigo_producto),
                "es_caja_nueva": bool(es_nueva),
                "box_id_asignado": caja["box_id"],
                "destino": destino,
                "p_reuse_acum": 1.0 - n_nuevas / i,
                "n_nuevas_acum": n_nuevas,
                "n_tipos_acum": len(cat),
            }
        )
        assigned.append(
            {
                "codigo_producto": str(p.codigo_producto),
                **{c: caja[c] for c in BOX_SPEC_COLS},
            }
        )

    sol_hold = pd.DataFrame(assigned)
    sol_final = pd.concat([sol_train[SUBMISSION_COLS], sol_hold[SUBMISSION_COLS]], ignore_index=True)
    return sol_final, pd.DataFrame(rows)


def assign_holdout_m1(sol_train, holdout_products):
    """Alias retrocompatible."""
    return assign_holdout(sol_train, holdout_products, policy="m1")


def run_actual_holdout_policies(
    *,
    seeds: list[int],
    products_truth: pd.DataFrame,
    ops: pd.DataFrame,
    joined: pd.DataFrame,
    holdout_frac: float = 0.15,
) -> pd.DataFrame:
    """Compara Actual completo vs Actual+holdout (M1 y pallet) en varias seeds."""
    baseline = build_actual_from_joined(
        joined, joined["codigo_producto"].astype(str).str.strip().unique().tolist()
    )
    ev_full = evaluar_kaggle(baseline, ops)
    n_tipos_full = int(add_box_id(baseline)["box_id"].nunique())

    rows = [
        {
            "seed": -1,
            "escenario": "Actual completo",
            "policy": "—",
            "n_train": len(baseline),
            "n_holdout": 0,
            "tipos_train": n_tipos_full,
            "tipos_final": ev_full["n_tipos"],
            "boxes_created": 0,
            "p_reuse": np.nan,
            "util_pallet": ev_full["util_pallet"],
            "cost_total": ev_full["total"],
            "cost_pack": ev_full["packaging"],
            "cost_flete": ev_full["flete"],
        }
    ]

    pt = products_truth.copy()
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    ops = ops.copy()
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    meta = joined[["codigo_producto", "categoria"]].drop_duplicates("codigo_producto")
    meta["codigo_producto"] = meta["codigo_producto"].astype(str).str.strip()

    for seed in seeds:
        train_codes, hold_codes = stratified_holdout_codes(
            meta, holdout_frac=holdout_frac, seed=seed
        )
        actual_train = build_actual_from_joined(joined, train_codes)
        rng = np.random.default_rng(seed + 17_000)
        hold_order = list(hold_codes)
        rng.shuffle(hold_order)
        hold_pt = pt.set_index("codigo_producto").loc[hold_order].reset_index()
        n_tipos_train = int(add_box_id(actual_train)["box_id"].nunique())

        for policy, label in [("m1", "M1"), ("pallet", "Pallet-aware")]:
            sol_final, detalle = assign_holdout(actual_train, hold_pt, policy=policy)
            ev = evaluar_kaggle(sol_final, ops)
            rows.append(
                {
                    "seed": seed,
                    "escenario": f"Actual holdout 15% · {label}",
                    "policy": policy,
                    "n_train": len(train_codes),
                    "n_holdout": len(hold_codes),
                    "tipos_train": n_tipos_train,
                    "tipos_final": ev["n_tipos"],
                    "boxes_created": int(detalle["es_caja_nueva"].sum()),
                    "p_reuse": float((~detalle["es_caja_nueva"]).mean()),
                    "util_pallet": ev["util_pallet"],
                    "cost_total": ev["total"],
                    "cost_pack": ev["packaging"],
                    "cost_flete": ev["flete"],
                }
            )
    return pd.DataFrame(rows)


def run_indep_holdout_policies(
    *,
    seeds: list[int],
    products_truth: pd.DataFrame,
    ops: pd.DataFrame,
    joined: pd.DataFrame,
    indep_full: pd.DataFrame,
    mark17_cache_dir: Path,
    holdout_frac: float = 0.15,
    mark17_tl: float = 30.0,
    force_mark17: bool = False,
) -> pd.DataFrame:
    """Compara Bajo protocolo completo vs train+holdout (M1 y pallet)."""
    indep_full = indep_full[SUBMISSION_COLS].copy()
    indep_full["codigo_producto"] = indep_full["codigo_producto"].astype(str).str.strip()
    ev_full = evaluar_kaggle(indep_full, ops)
    n_tipos_full = int(add_box_id(indep_full)["box_id"].nunique())

    rows = [
        {
            "seed": -1,
            "escenario": "Bajo protocolo completo",
            "policy": "—",
            "n_train": len(indep_full),
            "n_holdout": 0,
            "tipos_train": n_tipos_full,
            "tipos_final": ev_full["n_tipos"],
            "boxes_created": 0,
            "p_reuse": np.nan,
            "util_pallet": ev_full["util_pallet"],
            "cost_total": ev_full["total"],
            "cost_pack": ev_full["packaging"],
            "cost_flete": ev_full["flete"],
        }
    ]

    pt = products_truth.copy()
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    ops = ops.copy()
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    meta = joined[["codigo_producto", "categoria"]].drop_duplicates("codigo_producto")
    meta["codigo_producto"] = meta["codigo_producto"].astype(str).str.strip()
    cache = Path(mark17_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        train_codes, hold_codes = stratified_holdout_codes(
            meta, holdout_frac=holdout_frac, seed=seed
        )
        pt_train = pt[pt["codigo_producto"].isin(train_codes)].copy()
        ops_train = ops[ops["codigo_producto"].isin(train_codes)].copy()
        warm_train = build_actual_from_joined(joined, train_codes)
        m17_path = cache / f"mark17_train_seed{seed}.csv"
        if m17_path.exists() and not force_mark17:
            indep_train = pd.read_csv(m17_path)[SUBMISSION_COLS].copy()
            indep_train["codigo_producto"] = indep_train["codigo_producto"].astype(str).str.strip()
        else:
            indep_train = optimize_mark17_minbox(
                pt_train,
                ops_train,
                warm_train,
                out_path=m17_path,
                time_limit=mark17_tl,
                workers=1,
                grosor=3.0,
                top_k=8,
            )

        rng = np.random.default_rng(seed + 17_000)
        hold_order = list(hold_codes)
        rng.shuffle(hold_order)
        hold_pt = pt.set_index("codigo_producto").loc[hold_order].reset_index()
        n_tipos_train = int(add_box_id(indep_train)["box_id"].nunique())

        for policy, label in [("m1", "M1"), ("pallet", "Pallet-aware")]:
            sol_final, detalle = assign_holdout(indep_train, hold_pt, policy=policy)
            ev = evaluar_kaggle(sol_final, ops)
            rows.append(
                {
                    "seed": seed,
                    "escenario": f"Bajo protocolo holdout 15% · {label}",
                    "policy": policy,
                    "n_train": len(train_codes),
                    "n_holdout": len(hold_codes),
                    "tipos_train": n_tipos_train,
                    "tipos_final": ev["n_tipos"],
                    "boxes_created": int(detalle["es_caja_nueva"].sum()),
                    "p_reuse": float((~detalle["es_caja_nueva"]).mean()),
                    "util_pallet": ev["util_pallet"],
                    "cost_total": ev["total"],
                    "cost_pack": ev["packaging"],
                    "cost_flete": ev["flete"],
                }
            )
    return pd.DataFrame(rows)


def run_holdout_repetition(
    *,
    seed: int,
    products_truth: pd.DataFrame,
    ops: pd.DataFrame,
    joined: pd.DataFrame,
    holdout_frac: float = 0.15,
    mark17_tl: float = 30.0,
    mark17_cache_dir: Path | None = None,
    force_mark17: bool = False,
) -> dict:
    """Una repetición: split → Actual + Mark17 train → M1 holdout → métricas."""
    meta = joined[["codigo_producto", "categoria"]].drop_duplicates("codigo_producto")
    meta["codigo_producto"] = meta["codigo_producto"].astype(str).str.strip()
    train_codes, hold_codes = stratified_holdout_codes(
        meta, holdout_frac=holdout_frac, seed=seed
    )

    pt = products_truth.copy()
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    ops = ops.copy()
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()

    pt_train = pt[pt["codigo_producto"].isin(train_codes)].copy()
    ops_train = ops[ops["codigo_producto"].isin(train_codes)].copy()
    actual_train = build_actual_from_joined(joined, train_codes)

    cache_dir = Path(mark17_cache_dir) if mark17_cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        m17_path = cache_dir / f"mark17_train_seed{seed}.csv"
    else:
        m17_path = Path(tempfile.mkstemp(suffix=".csv")[1])

    if cache_dir is not None and m17_path.exists() and not force_mark17:
        indep_train = pd.read_csv(m17_path)[SUBMISSION_COLS].copy()
        indep_train["codigo_producto"] = indep_train["codigo_producto"].astype(str).str.strip()
    else:
        indep_train = optimize_mark17_minbox(
            pt_train,
            ops_train,
            actual_train,
            out_path=m17_path,
            time_limit=mark17_tl,
            workers=1,
            grosor=3.0,
            top_k=8,
        )

    # Orden de introducción del holdout (reproducible)
    rng = np.random.default_rng(seed + 17_000)
    hold_order = list(hold_codes)
    rng.shuffle(hold_order)
    hold_pt = (
        pt.set_index("codigo_producto")
        .loc[hold_order]
        .reset_index()
    )

    results = []
    details = []
    for name, sol_train in [("actual", actual_train), ("independiente", indep_train)]:
        sol_final, detalle = assign_holdout_m1(sol_train, hold_pt)
        detalle = detalle.assign(seed=seed, portfolio=name)
        details.append(detalle)

        # Demanda real de todos los productos (backtest honesto)
        ev = evaluar_kaggle(sol_final, ops)
        n_tipos_train = int(add_box_id(sol_train)["box_id"].nunique())
        results.append(
            {
                "seed": seed,
                "portfolio": name,
                "n_train": len(train_codes),
                "n_holdout": len(hold_codes),
                "holdout_frac": len(hold_codes) / (len(train_codes) + len(hold_codes)),
                "tipos_train": n_tipos_train,
                "tipos_final": ev["n_tipos"],
                "boxes_created": int(detalle["es_caja_nueva"].sum()),
                "p_reuse": float((~detalle["es_caja_nueva"]).mean()),
                "p_nueva": float(detalle["es_caja_nueva"].mean()),
                "util_pallet": ev["util_pallet"],
                "cost_total": ev["total"],
                "cost_pack": ev["packaging"],
                "cost_flete": ev["flete"],
            }
        )

    return {
        "summary_rows": pd.DataFrame(results),
        "detail": pd.concat(details, ignore_index=True),
        "train_codes": train_codes,
        "hold_codes": hold_codes,
    }
