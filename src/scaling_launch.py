"""Protocolo de lanzamientos: asignación local (M1 / pallet-aware) + oracle Mark17.

Usado por notebooks/130_scaling_protocol.ipynb.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from evaluate import (
    PRECIO_BASE_GROSOR_FULL,
    SUBMISSION_COLS,
    costo_flete_eval,
    factor_precio_por_volumen,
    headspace_pct,
    preparar_eval_planta,
)
from settings import (
    ECT,
    EPS,
    GRAVITY,
    HEADSPACE_ABS_MAX,
    MAX_GROW,
    MAX_SHRINK,
    PALLET_ALTO_MAX_MM,
    PALLET_ANCHO_MM,
    PALLET_LARGO_MM,
    PLANTAS,
)

G = GRAVITY
EPS_DIM = EPS
MAX_RESIZE = MAX_GROW
PALLET_LARGO = PALLET_LARGO_MM
PALLET_ANCHO = PALLET_ANCHO_MM
PALLET_MAX_ALTO = PALLET_ALTO_MAX_MM

BOX_SPEC_COLS = [
    "caja_grosor_mm",
    "caja_exterior_largo",
    "caja_exterior_ancho",
    "caja_exterior_alto",
]
DIM_COLS = ["largo", "ancho", "alto", "peso_neto_caja"]
PLANT_VOL_COLS = [f"volumen_producto_planta_{p}" for p in PLANTAS]
VOL_COLS = ["volumen_producto_total", *PLANT_VOL_COLS]


def make_box_id(grosor, largo, ancho, alto) -> str:
    return f"{grosor}x{largo}x{ancho}x{alto}"


def add_box_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["box_id"] = (
        out["caja_grosor_mm"].astype(str)
        + "x"
        + out["caja_exterior_largo"].astype(str)
        + "x"
        + out["caja_exterior_ancho"].astype(str)
        + "x"
        + out["caja_exterior_alto"].astype(str)
    )
    return out


def _producto_dims(producto) -> tuple[float, float, float, float]:
    return (
        float(producto["largo"]),
        float(producto["ancho"]),
        float(producto["alto"]),
        float(producto["peso_neto_caja"]),
    )


def caja_cumple_restricciones_fisicas(
    largo, ancho, alto, peso, grosor, ext_l, ext_w, ext_h,
) -> bool:
    grosor = float(grosor)
    if grosor not in ECT:
        return False
    interno = np.array([ext_l, ext_w, ext_h], dtype=float) - 2.0 * grosor
    prod = np.array([largo, ancho, alto], dtype=float)
    if (interno <= 0).any():
        return False
    lo = MAX_SHRINK * prod
    pct = headspace_pct(grosor)
    hi = np.minimum.reduce(
        [MAX_RESIZE * prod, prod / (1.0 - pct), prod + HEADSPACE_ABS_MAX]
    )
    if (interno < lo - EPS_DIM).any() or (interno > hi + EPS_DIM).any():
        return False
    if np.prod(interno) < np.prod(prod) - EPS_DIM:
        return False
    cpp = (
        np.floor(PALLET_ANCHO / ext_l)
        * np.floor(PALLET_LARGO / ext_w)
        * np.floor(PALLET_MAX_ALTO / ext_h)
    )
    if cpp <= 0:
        return False
    capas = np.floor(PALLET_MAX_ALTO / ext_h)
    perimetro_m = 2.0 * (ext_l + ext_w) / 1000.0
    carga_max = ECT[grosor] * perimetro_m / G
    if peso * max(capas - 1, 0) > carga_max + EPS_DIM:
        return False
    return True


def catalogo_cajas_unicas(solucion_df: pd.DataFrame) -> pd.DataFrame:
    df = add_box_id(solucion_df) if "box_id" not in solucion_df.columns else solucion_df.copy()
    cats = df[["box_id"] + BOX_SPEC_COLS].drop_duplicates("box_id").reset_index(drop=True)
    cats["volumen_externo"] = (
        cats["caja_exterior_largo"]
        * cats["caja_exterior_ancho"]
        * cats["caja_exterior_alto"]
    )
    cats["cajas_por_pallet"] = (
        np.floor(PALLET_ANCHO / cats["caja_exterior_largo"])
        * np.floor(PALLET_LARGO / cats["caja_exterior_ancho"])
        * np.floor(PALLET_MAX_ALTO / cats["caja_exterior_alto"])
    )
    cats["precio_base"] = cats["caja_grosor_mm"].round(1).map(PRECIO_BASE_GROSOR_FULL)
    return cats


def crear_caja_minima(producto, grosores_candidatos=None) -> dict:
    largo, ancho, alto, peso = _producto_dims(producto)
    if grosores_candidatos is None:
        grosores_candidatos = sorted(ECT.keys())
    for grosor in grosores_candidatos:
        ext_l, ext_w, ext_h = largo + 2 * grosor, ancho + 2 * grosor, alto + 2 * grosor
        if caja_cumple_restricciones_fisicas(
            largo, ancho, alto, peso, grosor, ext_l, ext_w, ext_h
        ):
            return {
                "caja_grosor_mm": float(grosor),
                "caja_exterior_largo": float(ext_l),
                "caja_exterior_ancho": float(ext_w),
                "caja_exterior_alto": float(ext_h),
                "box_id": make_box_id(grosor, ext_l, ext_w, ext_h),
            }
    raise ValueError(f"No hay grosor factible para {producto.get('codigo_producto', '?')}")


def _compatibles(producto, catalogo: pd.DataFrame) -> list[pd.Series]:
    largo, ancho, alto, peso = _producto_dims(producto)
    out = []
    for _, caja in catalogo.iterrows():
        ok = caja_cumple_restricciones_fisicas(
            largo,
            ancho,
            alto,
            peso,
            caja["caja_grosor_mm"],
            caja["caja_exterior_largo"],
            caja["caja_exterior_ancho"],
            caja["caja_exterior_alto"],
        )
        if ok:
            out.append(caja)
    return out


def _caja_dict(caja) -> dict:
    return {
        "caja_grosor_mm": float(caja["caja_grosor_mm"]),
        "caja_exterior_largo": float(caja["caja_exterior_largo"]),
        "caja_exterior_ancho": float(caja["caja_exterior_ancho"]),
        "caja_exterior_alto": float(caja["caja_exterior_alto"]),
        "box_id": caja["box_id"],
    }


def m1_asignar_producto_a_caja(producto, solucion_actual_df):
    """Menor volumen externo compatible; si no, caja mínima."""
    catalogo = catalogo_cajas_unicas(solucion_actual_df)
    compat = _compatibles(producto, catalogo)
    if compat:
        mejor = min(compat, key=lambda r: float(r["volumen_externo"]))
        return False, _caja_dict(mejor)
    grosores_catalogo = sorted(catalogo["caja_grosor_mm"].round(3).unique().tolist())
    resto = [g for g in sorted(ECT.keys()) if g not in grosores_catalogo]
    return True, crear_caja_minima(producto, grosores_candidatos=grosores_catalogo + resto)


def pallet_aware_asignar_producto(producto, solucion_actual_df):
    """Maximiza cajas/pallet; desempate por menor precio_base y volumen."""
    catalogo = catalogo_cajas_unicas(solucion_actual_df)
    compat = _compatibles(producto, catalogo)
    if compat:
        mejor = max(
            compat,
            key=lambda r: (
                float(r["cajas_por_pallet"]),
                -float(r["precio_base"]) if pd.notna(r["precio_base"]) else 0.0,
                -float(r["volumen_externo"]),
            ),
        )
        return False, _caja_dict(mejor)
    grosores_catalogo = sorted(catalogo["caja_grosor_mm"].round(3).unique().tolist())
    resto = [g for g in sorted(ECT.keys()) if g not in grosores_catalogo]
    return True, crear_caja_minima(producto, grosores_candidatos=grosores_catalogo + resto)


def leave_one_out_reassign(
    solucion_df: pd.DataFrame,
    products_truth_df: pd.DataFrame,
    policy: str = "m1",
) -> pd.DataFrame:
    """Saca un producto, reasigna con el catálogo restante y mide recuperación.

    Retorna una fila por producto con:
    - es_caja_nueva: no hay caja compatible en el catálogo residual
    - mismo_box_id: cae exactamente en su box_id original
    """
    if policy not in {"m1", "pallet"}:
        raise ValueError(f"policy inválida: {policy}")
    metodo = m1_asignar_producto_a_caja if policy == "m1" else pallet_aware_asignar_producto
    sol = add_box_id(solucion_df[SUBMISSION_COLS].copy())
    sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
    pt = products_truth_df.copy()
    pt["codigo_producto"] = pt["codigo_producto"].astype(str).str.strip()
    pt_idx = pt.set_index("codigo_producto", drop=False)
    counts = sol["box_id"].value_counts()
    rows = []
    for idx, row in sol.iterrows():
        code = row["codigo_producto"]
        if code not in pt_idx.index:
            raise KeyError(f"Producto {code} ausente en products_truth")
        prod = pt_idx.loc[code]
        if isinstance(prod, pd.DataFrame):
            prod = prod.iloc[0]
        es_nueva, caja = metodo(prod, sol.drop(index=idx))
        orig = str(row["box_id"])
        asg = str(caja["box_id"])
        rows.append(
            {
                "codigo_producto": code,
                "policy": policy,
                "es_caja_nueva": bool(es_nueva),
                "box_id_original": orig,
                "box_id_asignado": asg,
                "mismo_box_id": (not es_nueva) and (asg == orig),
                "n_usuarios_caja_original": int(counts[orig]),
                "caja_era_exclusiva": bool(int(counts[orig]) == 1),
            }
        )
    return pd.DataFrame(rows)


def evaluar_kaggle(sol: pd.DataFrame, ops_df: pd.DataFrame) -> dict:
    df = preparar_eval_planta(sol[SUBMISSION_COLS], ops_df)
    precio_base = (
        df.groupby("tipo")["caja_grosor_mm"].first().round(1).map(PRECIO_BASE_GROSOR_FULL)
    )
    if precio_base.isna().any():
        raise ValueError("Grosores sin precio base")
    pack = 0.0
    for planta in PLANTAS:
        vol_tp = df.groupby("tipo")[f"volumen_producto_planta_{planta}"].sum()
        factor = factor_precio_por_volumen(vol_tp)
        pack += float((vol_tp.values * precio_base.loc[vol_tp.index].values * factor).sum())
    flete = float(costo_flete_eval(df))
    vol_box = (
        df["caja_exterior_largo"]
        * df["caja_exterior_ancho"]
        * df["caja_exterior_alto"]
    ).to_numpy(float)
    vol_pallet = PALLET_LARGO * PALLET_ANCHO * PALLET_MAX_ALTO
    util = (df["cajas_por_pallet"].to_numpy(float) * vol_box) / vol_pallet
    vol_tot = df["volumen_producto_total"].to_numpy(float)
    util_w = float(np.average(util, weights=np.maximum(vol_tot, 1e-9)))
    return {
        "total": pack + flete,
        "packaging": pack,
        "flete": flete,
        "util_pallet": util_w,
        "n_tipos": int(df["tipo"].nunique()),
    }


# ---------------------------------------------------------------------------
# Generación de productos
# ---------------------------------------------------------------------------


def generar_lanzamientos(
    products_truth_df: pd.DataFrame,
    n: int,
    seed: int,
    family: str = "routine",
    noise_rel: float = 0.03,
    outlier_frac: float = 0.10,
    drift_shift: float = 0.10,
) -> pd.DataFrame:
    """Genera n productos sintéticos preservando attrs de categoría del padre."""
    rng = np.random.default_rng(seed)
    base = products_truth_df.copy().reset_index(drop=True)
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
        if c in base.columns
    ]

    if family == "drift" and "categoria" in base.columns:
        cats = base["categoria"].fillna("NA").value_counts()
        # reponderar: top categoría baja, cola sube
        weights = cats.astype(float).copy()
        order = weights.sort_values(ascending=False)
        weights.loc[order.index[: max(1, len(order) // 3)]] *= 0.55
        weights.loc[order.index[-(max(1, len(order) // 3)) :]] *= 1.8
        wmap = (weights / weights.sum()).to_dict()
        p = base["categoria"].fillna("NA").map(wmap).to_numpy(float)
        p = p / p.sum()
        idx = rng.choice(len(base), size=n, replace=True, p=p)
    else:
        idx = rng.integers(0, len(base), size=n)

    sampled = base.iloc[idx].reset_index(drop=True)
    parents = sampled["codigo_producto"].astype(str).tolist()

    # ruido correlacionado: un factor común + ruido por eje
    common = rng.normal(0.0, noise_rel * 0.7, size=n)
    for col in DIM_COLS:
        vals = sampled[col].to_numpy(dtype=float)
        axis = rng.normal(0.0, noise_rel * 0.5, size=n)
        noise = (common + axis) * vals
        if family == "drift":
            noise = noise + drift_shift * vals * rng.choice([-1.0, 1.0], size=n)
        lo = float(base[col].min()) * (0.85 if family == "outlier" else 0.97)
        hi = float(base[col].max()) * (1.25 if family == "outlier" else 1.03)
        sampled[col] = np.clip(vals + noise, lo, hi)

    if family == "outlier":
        n_out = max(1, int(round(outlier_frac * n)))
        out_idx = rng.choice(n, size=n_out, replace=False)
        for col in ["largo", "ancho", "alto"]:
            vals = sampled.loc[out_idx, col].to_numpy(float)
            scale = rng.uniform(1.15, 1.35, size=n_out)
            sampled.loc[out_idx, col] = vals * scale
        sampled.loc[out_idx, "peso_neto_caja"] = (
            sampled.loc[out_idx, "peso_neto_caja"].to_numpy(float)
            * rng.uniform(1.05, 1.25, size=n_out)
        )

    sampled["largo"] = sampled["largo"].round(1)
    sampled["ancho"] = sampled["ancho"].round(1)
    sampled["alto"] = sampled["alto"].round(1)
    sampled["peso_neto_caja"] = sampled["peso_neto_caja"].round(3)
    sampled["codigo_producto"] = [f"LN{seed % 10000:04d}_{i:04d}" for i in range(n)]
    sampled["parent_codigo"] = parents
    keep = ["codigo_producto", "parent_codigo"] + DIM_COLS + cat_cols
    return sampled[keep].copy()


# ---------------------------------------------------------------------------
# Demanda
# ---------------------------------------------------------------------------


def _shock_vector(rng: np.random.Generator, sigma: float = 0.10) -> float:
    return float(np.exp(rng.normal(0.0, sigma)))


def build_ops_after_launches(
    ops_base: pd.DataFrame,
    launches: pd.DataFrame,
    mode: str = "cannibal",
    seed: int = 0,
    shock_sigma: float = 0.10,
) -> pd.DataFrame:
    """Expande ops con demanda de lanzamientos (aditivo o canibalizante)."""
    rng = np.random.default_rng(seed)
    ops = ops_base.copy()
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    for c in VOL_COLS:
        if c in ops.columns:
            ops[c] = ops[c].astype(float)
    ops = ops.set_index("codigo_producto", drop=False)

    new_rows = []
    for _, row in launches.iterrows():
        parent = str(row["parent_codigo"])
        if parent not in ops.index:
            # fallback: SKU con demanda media
            parent = ops["volumen_producto_total"].astype(float).idxmax()
        parent_vols = ops.loc[parent, PLANT_VOL_COLS].to_numpy(float)
        shock = _shock_vector(rng, shock_sigma)
        child_vols = np.maximum(0.0, parent_vols * shock)
        # preservar ceros de planta del padre
        child_vols = np.where(parent_vols > 0, child_vols, 0.0)
        if child_vols.sum() <= 0:
            # si el padre no tiene volumen, tomar un SKU con volumen
            donor = ops.loc[ops["volumen_producto_total"].astype(float) > 0].sample(
                1, random_state=int(rng.integers(0, 1_000_000))
            ).iloc[0]
            parent_vols = donor[PLANT_VOL_COLS].to_numpy(float)
            child_vols = np.maximum(0.0, parent_vols * shock)
            child_vols = np.where(parent_vols > 0, child_vols, 0.0)
            parent = str(donor["codigo_producto"])

        if mode == "cannibal":
            # transferir desde el padre (misma planta)
            take = np.minimum(ops.loc[parent, PLANT_VOL_COLS].to_numpy(float), child_vols)
            ops.loc[parent, PLANT_VOL_COLS] = (
                ops.loc[parent, PLANT_VOL_COLS].to_numpy(float) - take
            )
            ops.loc[parent, "volumen_producto_total"] = float(
                ops.loc[parent, PLANT_VOL_COLS].to_numpy(float).sum()
            )
            child_vols = take

        new = {c: 0.0 for c in ops.columns if c != "codigo_producto"}
        new["codigo_producto"] = str(row["codigo_producto"])
        for c, v in zip(PLANT_VOL_COLS, child_vols):
            new[c] = float(v)
        new["volumen_producto_total"] = float(child_vols.sum())
        new_rows.append(new)

    if new_rows:
        ops = pd.concat([ops.reset_index(drop=True), pd.DataFrame(new_rows)], ignore_index=True)
    else:
        ops = ops.reset_index(drop=True)
    ops["codigo_producto"] = ops["codigo_producto"].astype(str).str.strip()
    return ops


def expand_products_truth(
    products_truth: pd.DataFrame, launches: pd.DataFrame
) -> pd.DataFrame:
    base_cols = [
        "codigo_producto",
        "largo",
        "ancho",
        "alto",
        "cantidad_paquetes",
        "peso_neto_paquete",
        "peso_neto_caja",
    ]
    pt = products_truth.copy()
    for c in base_cols:
        if c not in pt.columns:
            pt[c] = np.nan
    rows = []
    for _, r in launches.iterrows():
        parent = pt.loc[pt["codigo_producto"].astype(str) == str(r["parent_codigo"])]
        qty = float(parent["cantidad_paquetes"].iloc[0]) if len(parent) else 1.0
        peso = float(r["peso_neto_caja"])
        rows.append(
            {
                "codigo_producto": str(r["codigo_producto"]),
                "largo": float(r["largo"]),
                "ancho": float(r["ancho"]),
                "alto": float(r["alto"]),
                "cantidad_paquetes": qty,
                "peso_neto_paquete": peso / max(qty, 1.0),
                "peso_neto_caja": peso,
            }
        )
    return pd.concat([pt[base_cols], pd.DataFrame(rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Trayectoria local + oracle Mark17
# ---------------------------------------------------------------------------


def run_local_trajectory(
    sol_base: pd.DataFrame,
    launches: pd.DataFrame,
    metodo: Callable,
    checkpoints: list[int],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame]:
    """Asigna lanzamientos secuencialmente. Devuelve detalle, sol@checkpoint, crecimiento."""
    catalogo = add_box_id(sol_base[SUBMISSION_COLS].copy())
    init_ids = set(catalogo_cajas_unicas(catalogo)["box_id"])
    filas = []
    sols_at = {}
    growth = []

    for i, (_, producto) in enumerate(launches.iterrows(), start=1):
        es_nueva, caja = metodo(producto, catalogo)
        if es_nueva:
            destino = "Caja nueva"
        elif caja["box_id"] in init_ids:
            destino = "Catálogo inicial"
        else:
            destino = "Caja creada en MC"
        filas.append(
            {
                "orden": i,
                "codigo_producto": str(producto["codigo_producto"]),
                "es_caja_nueva": bool(es_nueva),
                "box_id_asignado": caja["box_id"],
                "destino": destino,
            }
        )
        catalogo = pd.concat(
            [
                catalogo,
                pd.DataFrame(
                    [
                        {
                            "codigo_producto": str(producto["codigo_producto"]),
                            **{c: caja[c] for c in BOX_SPEC_COLS},
                            "box_id": caja["box_id"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        n_tipos = int(catalogo_cajas_unicas(catalogo)["box_id"].nunique())
        growth.append(
            {
                "orden": i,
                "n_tipos": n_tipos,
                "n_nuevas_acum": int(sum(f["es_caja_nueva"] for f in filas)),
            }
        )
        if i in checkpoints:
            sols_at[i] = catalogo[SUBMISSION_COLS + (["box_id"] if "box_id" in catalogo.columns else [])].copy()
            if "box_id" not in sols_at[i].columns:
                sols_at[i] = add_box_id(sols_at[i])

    detalle = pd.DataFrame(filas)
    growth_df = pd.DataFrame(growth)
    return detalle, sols_at, growth_df


def mark17_reopt_checkpoint(
    products_truth_expanded: pd.DataFrame,
    ops_expanded: pd.DataFrame,
    warm_sol: pd.DataFrame,
    time_limit: float = 30.0,
    workers: int | None = None,
    grosores: list[float] | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Reoptimiza con Mark17 en un *subproceso* (aisla abortos de ortools).

    Nota: workers=1 por defecto — con hints + multi-worker ortools puede
    abortar (`heuristics.fixed_search != nullptr`) en catálogos ampliados.
    """
    import subprocess
    import sys

    warm = add_box_id(warm_sol) if "box_id" not in warm_sol.columns else warm_sol.copy()
    warm = warm.copy()
    warm["codigo_producto"] = warm["codigo_producto"].astype(str).str.strip()

    codes = products_truth_expanded["codigo_producto"].astype(str).str.strip().tolist()
    missing = sorted(set(codes) - set(warm["codigo_producto"]))
    if missing:
        raise ValueError(f"Warm start incompleto: faltan {len(missing)} códigos")

    n_workers = 1 if workers is None else max(1, int(workers))
    grosor = float((grosores or [3.0])[0])

    # Persistimos fuera de TemporaryDirectory para poder leer tras el worker
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts" / "_mark17_ckpt_worker.py"
    py = sys.executable

    with tempfile.TemporaryDirectory(prefix="m17_ckpt_") as tmp:
        tmp = Path(tmp)
        pt_path = tmp / "products_truth.csv"
        ops_path = tmp / "operaciones_planta.csv"
        warm_path = tmp / "warm.csv"
        out_path = tmp / "reopt.csv"

        ops_out = ops_expanded.copy()
        for col in ["costo_total", "costo_pallets_total"]:
            if col not in ops_out.columns:
                ops_out[col] = 0.0
            else:
                ops_out[col] = ops_out[col].fillna(0.0)
        products_truth_expanded.to_csv(pt_path, index=False)
        ops_out.to_csv(ops_path, index=False)
        warm[SUBMISSION_COLS].to_csv(warm_path, index=False)

        last_err: str | None = None
        # Un intento con hints; si falla, uno sin hints (no quemar 3× time_limit)
        attempts = [
            (float(time_limit), {}),
            (float(time_limit), {"MARK17_NO_HINTS": "1"}),
        ]
        for tl, extra_env in attempts:
            if out_path.exists():
                out_path.unlink()
            cmd = [
                py,
                str(worker),
                "--pt",
                str(pt_path),
                "--ops",
                str(ops_path),
                "--warm",
                str(warm_path),
                "--out",
                str(out_path),
                "--tl",
                str(tl),
                "--workers",
                str(n_workers),
                "--top-k",
                "12",
                "--grosor",
                str(grosor),
            ]
            env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), **extra_env}
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=tl + 120,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                last_err = f"timeout tl={tl}"
                continue
            if proc.returncode == 0 and out_path.exists():
                sol = pd.read_csv(out_path)
                if len(sol):
                    return add_box_id(sol[SUBMISSION_COLS].copy()), True
            last_err = (
                f"rc={proc.returncode} stderr={(proc.stderr or '')[-300:]}"
            )

    print(f"  [warn] Mark17 falló ({last_err}); omito oracle en este checkpoint")
    return add_box_id(warm[SUBMISSION_COLS].copy()), False


def run_simulation(
    sol_base: pd.DataFrame,
    products_truth: pd.DataFrame,
    ops_base: pd.DataFrame,
    *,
    n_launches: int,
    seed: int,
    family: str,
    demand_mode: str,
    policy: str,
    checkpoints: list[int],
    run_oracle: bool = False,
    oracle_tl: float = 30.0,
    oracle_workers: int | None = None,
) -> dict:
    """Una simulación completa. policy in {'m1','pallet'}."""
    metodo = m1_asignar_producto_a_caja if policy == "m1" else pallet_aware_asignar_producto
    launches = generar_lanzamientos(products_truth, n_launches, seed=seed, family=family)
    detalle, sols_at, growth = run_local_trajectory(
        sol_base, launches, metodo, checkpoints
    )

    rows = []
    n0 = int(add_box_id(sol_base)["box_id"].nunique())
    for t in checkpoints:
        if t not in sols_at:
            continue
        launch_t = launches.iloc[:t]
        ops_t = build_ops_after_launches(
            ops_base, launch_t, mode=demand_mode, seed=seed + 17_000 + t
        )
        sol_local = sols_at[t]
        # base + lanzamientos asignados (sols_at ya incluye ambos)
        sol_eval = sol_local[SUBMISSION_COLS].copy()
        sol_eval["codigo_producto"] = sol_eval["codigo_producto"].astype(str).str.strip()
        sol_eval = sol_eval.drop_duplicates("codigo_producto", keep="last")
        # alinear con ops (todos los códigos deben estar)
        codes_ops = set(ops_t["codigo_producto"].astype(str))
        missing = set(sol_eval["codigo_producto"]) - codes_ops
        if missing:
            raise ValueError(f"Ops incompleto en t={t}: faltan {len(missing)} códigos")

        ev_local = evaluar_kaggle(sol_eval, ops_t)
        det_t = detalle[detalle["orden"] <= t]
        row = {
            "seed": seed,
            "family": family,
            "demand_mode": demand_mode,
            "policy": policy,
            "t": t,
            "tipos_ini": n0,
            "tipos_local": ev_local["n_tipos"],
            "delta_tipos": ev_local["n_tipos"] - n0,
            "p_nueva": float(det_t["es_caja_nueva"].mean()) if len(det_t) else 0.0,
            "pct_catalogo_inicial": float((det_t["destino"] == "Catálogo inicial").mean())
            if len(det_t)
            else 0.0,
            "tipos_per_100": 100.0 * (ev_local["n_tipos"] - n0) / max(t, 1),
            "cost_local": ev_local["total"],
            "pack_local": ev_local["packaging"],
            "flete_local": ev_local["flete"],
            "util_local": ev_local["util_pallet"],
            "cost_oracle": np.nan,
            "tipos_oracle": np.nan,
            "regret": np.nan,
            "regret_pct": np.nan,
            "oracle_s": np.nan,
            "oracle_ok": False,
        }

        if run_oracle:
            pt_exp = expand_products_truth(products_truth, launch_t)
            t0 = time.time()
            sol_or, ok = mark17_reopt_checkpoint(
                pt_exp,
                ops_t,
                sol_eval,
                time_limit=oracle_tl,
                workers=oracle_workers,
                grosores=[3.0],
            )
            elapsed = time.time() - t0
            row["oracle_s"] = elapsed
            row["oracle_ok"] = bool(ok)
            if ok:
                ev_or = evaluar_kaggle(sol_or, ops_t)
                row["cost_oracle"] = ev_or["total"]
                row["tipos_oracle"] = ev_or["n_tipos"]
                row["regret"] = ev_local["total"] - ev_or["total"]
                row["regret_pct"] = 100.0 * row["regret"] / max(ev_or["total"], 1.0)

        rows.append(row)

    growth = growth.copy()
    growth["seed"] = seed
    growth["family"] = family
    growth["policy"] = policy
    return {
        "checkpoint_rows": pd.DataFrame(rows),
        "growth": growth,
        "detalle": detalle.assign(seed=seed, family=family, policy=policy),
    }
