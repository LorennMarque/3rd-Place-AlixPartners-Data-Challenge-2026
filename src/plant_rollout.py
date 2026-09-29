"""Evaluación por planta y rollout parcial (híbrido Actual/Propuesto)."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from evaluate import (
    BOX_COLS,
    COSTO_PALLET,
    SUBMISSION_COLS,
    _base_frame,
    _check_physics,
    factor_precio_por_volumen,
    preparar_eval_planta,
)
from settings import (
    ECT as ECT_POLICY,
    HEADSPACE_PCT,
    PLANTAS,
    PRECIO_BASE_GROSOR,
)

BOX_KEY_COLS = list(BOX_COLS)


def box_key_frame(df: pd.DataFrame) -> pd.Series:
    g = pd.to_numeric(df["caja_grosor_mm"], errors="coerce").round(1)
    L = pd.to_numeric(df["caja_exterior_largo"], errors="coerce").round(3)
    W = pd.to_numeric(df["caja_exterior_ancho"], errors="coerce").round(3)
    H = pd.to_numeric(df["caja_exterior_alto"], errors="coerce").round(3)
    return g.astype(str) + "|" + L.astype(str) + "|" + W.astype(str) + "|" + H.astype(str)


def evaluar_kaggle(sol: pd.DataFrame, ops_df: pd.DataFrame) -> dict:
    """Packaging con tier por planta + flete ($150/pallet)."""
    df = preparar_eval_planta(sol[SUBMISSION_COLS], ops_df)
    precio_base = (
        df.groupby("tipo")["caja_grosor_mm"].first().round(1).map(PRECIO_BASE_GROSOR)
    )
    if precio_base.isna().any():
        raise ValueError("Grosores sin precio base en la solución")

    pack = 0.0
    tier_rows = []
    plant_pack: dict[str, float] = {}
    plant_flete: dict[str, float] = {}
    plant_pallets: dict[str, float] = {}
    for planta in PLANTAS:
        vol_tp = df.groupby("tipo")[f"volumen_producto_planta_{planta}"].sum()
        factor = factor_precio_por_volumen(vol_tp)
        costo = vol_tp.values * precio_base.loc[vol_tp.index].values * factor
        pack_p = float(costo.sum())
        pack += pack_p
        plant_pack[planta] = pack_p
        vol = df[f"volumen_producto_planta_{planta}"]
        pallets = np.ceil(vol / df["cajas_por_pallet"])
        flete_p = float((pallets * COSTO_PALLET).sum())
        plant_flete[planta] = flete_p
        plant_pallets[planta] = float(pallets.sum())
        tier_rows.append(
            pd.DataFrame(
                {
                    "tipo": vol_tp.index,
                    "planta": planta,
                    "vol": vol_tp.values,
                    "factor": factor,
                    "costo": costo,
                }
            )
        )

    flete = float(sum(plant_flete.values()))
    return {
        "n_tipos": int(df["tipo"].nunique()),
        "packaging": pack,
        "flete": flete,
        "total": pack + flete,
        "df": df,
        "det_tp": pd.concat(tier_rows, ignore_index=True),
        "plant_pack": plant_pack,
        "plant_flete": plant_flete,
        "plant_pallets": plant_pallets,
    }


def metricas_por_planta(
    ev: dict,
    *,
    ops_oficial: pd.DataFrame | None = None,
) -> dict[str, dict]:
    df = ev["df"]
    det = ev["det_tp"]
    vol_pallet = 800.0 * 1200.0 * 1800.0
    vol_box = (
        df["caja_exterior_largo"] * df["caja_exterior_ancho"] * df["caja_exterior_alto"]
    )
    util_series = df["cajas_por_pallet"] * vol_box / vol_pallet

    out: dict[str, dict] = {}
    for p in PLANTAS:
        vol = df[f"volumen_producto_planta_{p}"]
        mask = vol > 0
        util = (
            float(np.average(util_series[mask], weights=vol[mask]) * 100) if mask.any() else 0.0
        )

        if ops_oficial is not None:
            pack = float(ops_oficial[f"costo_total_planta_{p}"].sum())
            envio = float(ops_oficial[f"costo_pallets_planta_{p}"].sum())
            pallets = float(ops_oficial[f"cantidad_pallets_planta_{p}"].sum())
        else:
            pack = float(det.loc[det["planta"] == p, "costo"].sum())
            pallets_s = np.ceil(vol / df["cajas_por_pallet"])
            envio = float((pallets_s * COSTO_PALLET).sum())
            pallets = float(pallets_s.sum())

        out[p] = {
            "costo_total": pack + envio,
            "costo_envio": envio,
            "costo_cajas": pack,
            "pallets": pallets,
            "util_pallet_pct": util,
        }
    return out


def _plant_cost_from_assignment(df: pd.DataFrame, planta: str) -> tuple[float, float, float]:
    """Pack + flete + pallets for one plant given prepared assignment frame."""
    vol_col = f"volumen_producto_planta_{planta}"
    vol_tp = df.groupby("tipo")[vol_col].sum()
    precio_base = (
        df.groupby("tipo")["caja_grosor_mm"].first().round(1).map(PRECIO_BASE_GROSOR)
    )
    factor = factor_precio_por_volumen(vol_tp)
    pack = float((vol_tp.values * precio_base.loc[vol_tp.index].values * factor).sum())
    vol = df[vol_col]
    pallets = np.ceil(vol / df["cajas_por_pallet"])
    flete = float((pallets * COSTO_PALLET).sum())
    return pack, flete, float(pallets.sum())


def evaluar_hybrid(
    actual: pd.DataFrame,
    proposed: pd.DataFrame,
    migrated: Iterable[str],
    ops: pd.DataFrame,
) -> dict:
    """Costo híbrido: plantas migradas usan propuesto; el resto Actual.

    El catálogo activo es la unión de tipos usados en cada planta bajo su
    asignación vigente (duplicación temporal durante la migración).
    """
    migrated_set = set(migrated)
    ev_act = preparar_eval_planta(actual[SUBMISSION_COLS], ops)
    ev_prop = preparar_eval_planta(proposed[SUBMISSION_COLS], ops)

    pack_total = 0.0
    flete_total = 0.0
    plant_pack: dict[str, float] = {}
    plant_flete: dict[str, float] = {}
    plant_pallets: dict[str, float] = {}
    box_keys: set[str] = set()

    for p in PLANTAS:
        df = ev_prop if p in migrated_set else ev_act
        pack_p, flete_p, pal_p = _plant_cost_from_assignment(df, p)
        plant_pack[p] = pack_p
        plant_flete[p] = flete_p
        plant_pallets[p] = pal_p
        pack_total += pack_p
        flete_total += flete_p
        # tipos con volumen > 0 en la planta
        vol = df[f"volumen_producto_planta_{p}"]
        keys = box_key_frame(df.loc[vol > 0])
        box_keys.update(keys.tolist())

    return {
        "packaging": pack_total,
        "flete": flete_total,
        "total": pack_total + flete_total,
        "n_tipos": len(box_keys),
        "plant_pack": plant_pack,
        "plant_flete": plant_flete,
        "plant_pallets": plant_pallets,
        "migrated": sorted(migrated_set),
    }


def product_fits_box(
    products_truth: pd.DataFrame,
    sol_with_boxes: pd.DataFrame,
) -> pd.Series:
    """True si el producto cabe en la caja asignada (reglas físicas Kaggle-like)."""
    df, errors = _base_frame(sol_with_boxes[SUBMISSION_COLS], products_truth)
    if errors:
        # fall back: all False if schema broken
        return pd.Series(False, index=sol_with_boxes["codigo_producto"].astype(str))

    g = df["caja_grosor_mm"].round(1).to_numpy(float)
    # map unknown grosor to nearest ECT key or fail
    ect = np.array([float(ECT_POLICY.get(float(x), ECT_POLICY[3.0])) for x in g])
    pct = np.array([float(HEADSPACE_PCT.get(float(x), 0.06)) for x in g])
    phys_errors = _check_physics(df, g, ect, pct)
    # per-row fail flags set by _check_physics
    fail_cols = [
        c
        for c in df.columns
        if c.startswith("fail_")
    ]
    ok = ~df[fail_cols].any(axis=1) if fail_cols else pd.Series(True, index=df.index)
    # if global errors about non-positive internals, those rows already flagged
    _ = phys_errors
    out = pd.Series(ok.to_numpy(), index=df["codigo_producto"].astype(str))
    return out.reindex(sol_with_boxes["codigo_producto"].astype(str).str.strip(), fill_value=False)


def migration_crosswalk(
    actual: pd.DataFrame,
    proposed: pd.DataFrame,
    ops: pd.DataFrame,
    products_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Métricas de migración por planta (overlap exacto / funcional / volumen)."""
    act = actual[SUBMISSION_COLS].copy()
    prop = proposed[SUBMISSION_COLS].copy()
    for df in (act, prop):
        df["codigo_producto"] = df["codigo_producto"].astype(str).str.strip()
        for c in BOX_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    act = act.set_index("codigo_producto")
    prop = prop.set_index("codigo_producto")
    common = act.index.intersection(prop.index)

    key_a = box_key_frame(act.loc[common].reset_index())
    key_a.index = common
    key_p = box_key_frame(prop.loc[common].reset_index())
    key_p.index = common
    exact = key_a == key_p

    # Functional: product fits in Actual box (can keep current box during migration)
    sol_actual_boxes = act.loc[common].reset_index()[SUBMISSION_COLS]
    fits_actual = product_fits_box(products_truth, sol_actual_boxes)
    functional = exact | fits_actual.reindex(common, fill_value=False).to_numpy()

    # Dimensional delta (exterior)
    dL = (prop.loc[common, "caja_exterior_largo"] - act.loc[common, "caja_exterior_largo"]).abs()
    dW = (prop.loc[common, "caja_exterior_ancho"] - act.loc[common, "caja_exterior_ancho"]).abs()
    dH = (prop.loc[common, "caja_exterior_alto"] - act.loc[common, "caja_exterior_alto"]).abs()
    d_max = pd.concat([dL, dW, dH], axis=1).max(axis=1)
    d_mean = (dL + dW + dH) / 3.0
    grosor_chg = (
        act.loc[common, "caja_grosor_mm"].round(1)
        != prop.loc[common, "caja_grosor_mm"].round(1)
    )

    ops_i = ops.copy()
    ops_i["codigo_producto"] = ops_i["codigo_producto"].astype(str).str.strip()
    ops_i = ops_i.set_index("codigo_producto").reindex(common)

    types_act = set(key_a.unique())
    types_prop = set(key_p.unique())
    retained_global = types_act & types_prop
    retired_global = types_act - types_prop
    new_global = types_prop - types_act

    rows = []
    for p in PLANTAS:
        vol = ops_i[f"volumen_producto_planta_{p}"].fillna(0.0)
        mask = vol > 0
        vol_p = float(vol[mask].sum())
        skus = int(mask.sum())
        skus_changed = int((mask & ~exact).sum())
        skus_exact = int((mask & exact).sum())
        skus_functional = int((mask & functional).sum())
        units_exact = float(vol[mask & exact].sum())
        units_functional = float(vol[mask & functional].sum())
        units_changed = float(vol[mask & ~exact].sum())

        # pack spend affected ≈ volume of changed SKUs (proxy; actual $ needs tiers)
        pack_spend_proxy = units_changed  # filled later with real $ if available

        keys_a_p = set(key_a[mask])
        keys_p_p = set(key_p[mask])
        retained = keys_a_p & keys_p_p
        retired = keys_a_p - keys_p_p
        new_boxes = keys_p_p - keys_a_p

        rows.append(
            {
                "planta": p,
                "skus_con_demanda": skus,
                "skus_exact_overlap": skus_exact,
                "skus_functional_overlap": skus_functional,
                "skus_changed": skus_changed,
                "pct_skus_changed": 100.0 * skus_changed / skus if skus else 0.0,
                "vol_total": vol_p,
                "vol_exact_overlap": units_exact,
                "vol_functional_overlap": units_functional,
                "vol_changed": units_changed,
                "pct_vol_exact": 100.0 * units_exact / vol_p if vol_p else 0.0,
                "pct_vol_functional": 100.0 * units_functional / vol_p if vol_p else 0.0,
                "pct_vol_changed": 100.0 * units_changed / vol_p if vol_p else 0.0,
                "box_types_actual": len(keys_a_p),
                "box_types_proposed": len(keys_p_p),
                "box_types_retained": len(retained),
                "box_types_retired": len(retired),
                "box_types_new": len(new_boxes),
                "dim_delta_mean_mm": float(d_mean[mask].mean()) if mask.any() else 0.0,
                "dim_delta_max_mm": float(d_max[mask].max()) if mask.any() else 0.0,
                "skus_grosor_changed": int((mask & grosor_chg).sum()),
                "units_affected_proxy": pack_spend_proxy,
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["global_retained"] = len(retained_global)
    out.attrs["global_retired"] = len(retired_global)
    out.attrs["global_new"] = len(new_global)
    return out


def all_subsets(plantas: list[str] = PLANTAS) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [tuple()]
    for k in range(1, len(plantas) + 1):
        out.extend(combinations(plantas, k))
    return out


def rollout_all_subsets(
    actual: pd.DataFrame,
    proposed: pd.DataFrame,
    ops: pd.DataFrame,
    *,
    baseline_total: float,
) -> pd.DataFrame:
    rows = []
    for subset in all_subsets():
        ev = evaluar_hybrid(actual, proposed, subset, ops)
        savings = baseline_total - ev["total"]
        rows.append(
            {
                "n_migrated": len(subset),
                "plants": ",".join(subset) if subset else "(none)",
                "plants_list": list(subset),
                "total": ev["total"],
                "packaging": ev["packaging"],
                "flete": ev["flete"],
                "n_tipos_activos": ev["n_tipos"],
                "savings_usd": savings,
                "pct_of_full": np.nan,  # filled after
            }
        )
    df = pd.DataFrame(rows)
    full = float(df.loc[df["n_migrated"] == len(PLANTAS), "savings_usd"].iloc[0])
    df["pct_of_full"] = np.where(full > 0, 100.0 * df["savings_usd"] / full, 0.0)
    return df


def greedy_wave_sequence(
    subsets_df: pd.DataFrame,
    plantas: list[str] = PLANTAS,
) -> pd.DataFrame:
    """Secuencia greedy: en cada ola agregar la planta con mayor valor marginal."""
    remaining = set(plantas)
    chosen: list[str] = []
    # index single-plant and lookup by frozenset
    lookup = {}
    for _, r in subsets_df.iterrows():
        key = frozenset(r["plants_list"])
        lookup[key] = r

    waves = []
    prev_savings = float(lookup[frozenset()]["savings_usd"])
    waves.append(
        {
            "wave": 0,
            "plant_added": "(none)",
            "plants_migrated": "",
            "annual_savings": prev_savings,
            "pct_total_value": float(lookup[frozenset()]["pct_of_full"]),
            "active_box_types": int(lookup[frozenset()]["n_tipos_activos"]),
            "incremental_value": 0.0,
        }
    )
    step = 1
    while remaining:
        best_p = None
        best_sav = -1e30
        best_row = None
        for p in remaining:
            key = frozenset(chosen + [p])
            row = lookup[key]
            if row["savings_usd"] > best_sav:
                best_sav = row["savings_usd"]
                best_p = p
                best_row = row
        assert best_p is not None and best_row is not None
        chosen.append(best_p)
        remaining.remove(best_p)
        inc = float(best_row["savings_usd"] - prev_savings)
        waves.append(
            {
                "wave": step,
                "plant_added": best_p,
                "plants_migrated": ",".join(chosen),
                "annual_savings": float(best_row["savings_usd"]),
                "pct_total_value": float(best_row["pct_of_full"]),
                "active_box_types": int(best_row["n_tipos_activos"]),
                "incremental_value": inc,
            }
        )
        prev_savings = float(best_row["savings_usd"])
        step += 1
    return pd.DataFrame(waves)


def pareto_mask(complexity: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Pareto: maximize value, minimize complexity."""
    n = len(complexity)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                complexity[j] <= complexity[i]
                and value[j] >= value[i]
                and (complexity[j] < complexity[i] or value[j] > value[i])
            ):
                mask[i] = False
                break
    return mask
