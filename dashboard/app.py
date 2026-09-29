"""Dashboard Flask — acceso por código de 4 letras."""

from __future__ import annotations

import os
import secrets
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

try:
    from .bootstrap import ROOT
except ImportError:
    from bootstrap import ROOT  # type: ignore[no-redef]

from settings import SUBMISSION_COLS, PRECIO_BASE_GROSOR
from evaluate import (
    BOX_COLS,
    COSTO_PALLET,
    PALLET_ALTO_MAX_MM,
    PALLET_ANCHO_MM,
    PALLET_LARGO_MM,
    PLANTAS,
    PRECIO_BASE_GROSOR_FULL,
    agregar_id_tipo_caja,
    calcular_cajas_por_pallet,
    costo_flete_eval as evaluar_flete_oficial,
    factor_precio_por_volumen,
    factor_tier,
    preparar_eval_planta,
    validate_solution_free,
    violates_grosor_policy,
)

try:
    from .optimize_jobs import EFFORT_SECONDS, MODELS, get_job, list_active_jobs, start_job
except ImportError:
    from optimize_jobs import EFFORT_SECONDS, MODELS, get_job, list_active_jobs, start_job  # type: ignore[no-redef]

from scaling_launch import (
    add_box_id,
    caja_cumple_restricciones_fisicas,
    catalogo_cajas_unicas,
    m1_asignar_producto_a_caja,
    pallet_aware_asignar_producto,
)

PLANT_LABELS = {
    "buenos_aires": "Buenos Aires",
    "curitiba": "Curitiba",
    "santiago": "Santiago",
    "monterrey": "Monterrey",
    "bakersfield": "Bakersfield",
}

PLANT_COORDS = {
    "buenos_aires": {"lat": -34.6037, "lng": -58.3816},
    "curitiba": {"lat": -25.4284, "lng": -49.2733},
    "santiago": {"lat": -33.4489, "lng": -70.6693},
    "monterrey": {"lat": 25.6866, "lng": -100.3161},
    "bakersfield": {"lat": 35.3733, "lng": -119.0187},
}

PLANT_COUNTRY = {
    "buenos_aires": "ARG",
    "curitiba": "BRA",
    "santiago": "CHL",
    "monterrey": "MEX",
    "bakersfield": "USA",
}

COUNTRY_NAMES = {
    "ARG": "Argentina",
    "BRA": "Brasil",
    "CHL": "Chile",
    "MEX": "Mexico",
    "USA": "Estados Unidos",
}

TIER_THRESHOLDS = [
    {"value": 20_000, "label": "20k · base", "color": "#C8C7C6"},
    {"value": 50_000, "label": "50k · -10%", "color": "#B8D892"},
    {"value": 100_000, "label": "100k · -20%", "color": "#78A844"},
    {"value": 500_000, "label": "500k · -30%", "color": "#5E8434"},
]

TIER_DEFS = [
    {"key": "tier_1", "label": "Tier 1 (+10%)", "short": "T1", "color": "#E8A598"},
    {"key": "tier_2", "label": "Tier 2 (base)", "short": "T2", "color": "#C8C7C6"},
    {"key": "tier_3", "label": "Tier 3 (-10%)", "short": "T3", "color": "#B8D892"},
    {"key": "tier_4", "label": "Tier 4 (-20%)", "short": "T4", "color": "#78A844"},
    {"key": "tier_5", "label": "Tier 5 (-30%)", "short": "T5", "color": "#5E8434"},
]

CHANNEL_DEFS = [
    {
        "id": "servicios_comida",
        "col": "volumen_producto_canal_servicios_comida",
        "label": "Servicios de comida",
        "color": "#78A844",
    },
    {
        "id": "cadenas_corporativas",
        "col": "volumen_producto_canal_cadenas_corporativas",
        "label": "Cadenas corporativas",
        "color": "#5B8DEF",
    },
    {
        "id": "minorista",
        "col": "volumen_producto_canal_minorista",
        "label": "Minorista",
        "color": "#E8A598",
    },
    {
        "id": "otros",
        "col": "volumen_producto_canal_otros",
        "label": "Otros",
        "color": "#C8C7C6",
    },
]

VOLUME_BUCKETS = [
    {"id": "lt_1k", "label": "< 1k", "min": 0, "max": 1_000},
    {"id": "1k_5k", "label": "1k – 5k", "min": 1_000, "max": 5_000},
    {"id": "5k_20k", "label": "5k – 20k", "min": 5_000, "max": 20_000},
    {"id": "20k_100k", "label": "20k – 100k", "min": 20_000, "max": 100_000},
    {"id": "gte_100k", "label": "≥ 100k", "min": 100_000, "max": None},
]

PARETO_TOP_N = (30, 97)

DASHBOARD_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(DASHBOARD_DIR / "templates"),
    static_folder=str(DASHBOARD_DIR / "assets"),
    static_url_path="/assets",
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

ACCESS_CODE = os.environ.get("DASHBOARD_CODE", "").strip().upper()
PRODUCTS_PATH = ROOT / "data/processed/products_truth.csv"
CATALOGO_PATH = ROOT / "data/raw/catalogo_productos.csv"
CAJAS_PATH = ROOT / "data/raw/especificaciones_cajas.csv"
OPERATIONS_PATH = ROOT / "data/raw/operaciones_planta.csv"
SOLUTIONS_DIR = DASHBOARD_DIR / "soluciones"
SOLUTIONS_DIR.mkdir(exist_ok=True)
# Cota inferior CP-SAT (Mark 8/9, tiers por planta) — ver logs/mark8_5h_planta.log
CPSAT_QUOTA = 206_093_063.92


def _load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    products_truth = pd.read_csv(PRODUCTS_PATH)
    operations = pd.read_csv(OPERATIONS_PATH)
    return products_truth, operations


def _list_solution_files() -> list[str]:
    return sorted(path.name for path in SOLUTIONS_DIR.glob("*.csv"))


def _safe_solution_path(filename: str) -> Path | None:
    candidate = SOLUTIONS_DIR / filename
    try:
        candidate.resolve().relative_to(SOLUTIONS_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def _resolve_selected_solution(available_solutions: list[str]) -> str | None:
    selected_solution = session.get("selected_solution")

    if selected_solution and selected_solution not in available_solutions:
        session.pop("selected_solution", None)
        selected_solution = None

    if not selected_solution and available_solutions:
        selected_solution = available_solutions[0]
        session["selected_solution"] = selected_solution

    return selected_solution


def _load_selected_solution_df(selected_solution: str | None) -> pd.DataFrame | None:
    if not selected_solution:
        return None

    solution_path = _safe_solution_path(selected_solution)
    if solution_path is None:
        return None

    return pd.read_csv(solution_path)


def _build_packaging_payload(solution_df: pd.DataFrame, products_truth: pd.DataFrame) -> dict:
    df = solution_df.merge(
        products_truth[["codigo_producto", "largo", "ancho", "alto"]],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )

    df["volumen_producto_mm3"] = df["largo"] * df["ancho"] * df["alto"]
    df["caja_interior_largo"] = df["caja_exterior_largo"] - 2 * df["caja_grosor_mm"]
    df["caja_interior_ancho"] = df["caja_exterior_ancho"] - 2 * df["caja_grosor_mm"]
    df["caja_interior_alto"] = df["caja_exterior_alto"] - 2 * df["caja_grosor_mm"]
    df["volumen_caja_interior_mm3"] = (
        df["caja_interior_largo"]
        * df["caja_interior_ancho"]
        * df["caja_interior_alto"]
    )
    df["ocupacion_caja_pct"] = (
        df["volumen_producto_mm3"] / df["volumen_caja_interior_mm3"] * 100
    )

    df = agregar_id_tipo_caja(df)
    df = calcular_cajas_por_pallet(df)

    box_types_df = (
        df.groupby("caja_tipo_id_solucion", as_index=False)
        .agg(
            caja_grosor_mm=("caja_grosor_mm", "first"),
            caja_exterior_largo=("caja_exterior_largo", "first"),
            caja_exterior_ancho=("caja_exterior_ancho", "first"),
            caja_exterior_alto=("caja_exterior_alto", "first"),
            n_productos=("codigo_producto", "count"),
            ocupacion_promedio_pct=("ocupacion_caja_pct", "mean"),
        )
        .sort_values(
            ["n_productos", "caja_exterior_largo", "caja_exterior_ancho", "caja_exterior_alto"],
            ascending=[False, True, True, True],
        )
    )

    box_types = []
    for row in box_types_df.itertuples(index=False):
        largo = float(row.caja_exterior_largo)
        ancho = float(row.caja_exterior_ancho)
        alto = float(row.caja_exterior_alto)
        max_dim = max(largo, ancho, alto)
        scale = 72 / max_dim if max_dim > 0 else 1

        box_types.append(
            {
                "id": row.caja_tipo_id_solucion,
                "grosor_mm": float(row.caja_grosor_mm),
                "largo": largo,
                "ancho": ancho,
                "alto": alto,
                "display_largo": round(largo * scale, 2),
                "display_ancho": round(ancho * scale, 2),
                "display_alto": round(alto * scale, 2),
                "n_productos": int(row.n_productos),
                "ocupacion_promedio_pct": float(row.ocupacion_promedio_pct),
            }
        )

    products = []
    for row in df.sort_values("codigo_producto").itertuples(index=False):
        util_pallet_pct = float(row.utilizacion_pallet_teorica * 100)
        products.append(
            {
                "codigo_producto": row.codigo_producto,
                "caja_id": row.caja_tipo_id_solucion,
                "caja_label": (
                    f"{row.caja_exterior_largo:.0f} x {row.caja_exterior_ancho:.0f} "
                    f"x {row.caja_exterior_alto:.0f} mm"
                ),
                "caja_largo": float(row.caja_exterior_largo),
                "caja_ancho": float(row.caja_exterior_ancho),
                "caja_alto": float(row.caja_exterior_alto),
                "producto_largo": float(row.largo),
                "producto_ancho": float(row.ancho),
                "producto_alto": float(row.alto),
                "ocupacion_caja_pct": float(row.ocupacion_caja_pct),
                "ocupacion_bar_pct": float(min(row.ocupacion_caja_pct, 100)),
                "utilizacion_pallet_teorica_pct": util_pallet_pct,
                "utilizacion_pallet_bar_pct": float(min(util_pallet_pct, 100)),
                "cajas_piso_largo": int(row.cajas_piso_largo),
                "cajas_piso_ancho": int(row.cajas_piso_ancho),
                "capas_alto": int(row.capas_alto),
                "cajas_por_pallet": int(row.cajas_por_pallet),
                "volumen_producto_mm3": float(row.volumen_producto_mm3),
                "volumen_caja_interior_mm3": float(row.volumen_caja_interior_mm3),
                "volumen_externo_caja_mm3": float(row.volumen_externo_caja_mm3),
            }
        )

    return {
        "box_types": box_types,
        "products": products,
        "n_tipos_caja": len(box_types),
        "n_productos": len(products),
        "pallet": {
            "largo_mm": PALLET_LARGO_MM,
            "ancho_mm": PALLET_ANCHO_MM,
            "alto_mm": PALLET_ALTO_MAX_MM,
        },
    }


def _solution_violates_grosor_policy(selected_solution: str | None) -> bool:
    if not selected_solution:
        return False
    solution_path = _safe_solution_path(selected_solution)
    if solution_path is None:
        return False
    return violates_grosor_policy(pd.read_csv(solution_path))


def _policy_toast_meta(selected_solution: str | None) -> dict:
    violates = _solution_violates_grosor_policy(selected_solution)
    dismissed = session.get("policy_toast_dismissed") == selected_solution
    return {
        "policy_violation": violates,
        "show_policy_toast": violates and not dismissed,
    }


def _should_show_policy_toast(selected_solution: str | None) -> bool:
    return _policy_toast_meta(selected_solution)["show_policy_toast"]


def _base_context() -> dict:
    available_solutions = _list_solution_files()
    selected_solution = _resolve_selected_solution(available_solutions)

    return {
        "available_solutions": available_solutions,
        "selected_solution": selected_solution,
        "solution_policy_violation": _should_show_policy_toast(selected_solution),
        "upload_error": session.pop("upload_error", None),
        "upload_warning": session.pop("upload_warning", None),
        "upload_success": session.pop("upload_success", None),
    }


def _redirect_back(fallback: str = "home"):
    return redirect(request.referrer or url_for(fallback))


def _costo_baseline(operations: pd.DataFrame) -> float:
    """Costo historico (packaging + flete actual). Denominador del score de Kaggle."""
    return float(operations["costo_total"].sum() + operations["costo_pallets_total"].sum())


def _ahorro_pct_sobre_baseline(costo: float, costo_baseline: float) -> float:
    """Porcentaje de ahorro vs baseline; coincide con el score publico de Kaggle."""
    if not costo_baseline:
        return 0.0
    return 100.0 * (costo_baseline - costo) / costo_baseline


def _costo_packaging_por_planta_extended(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Packaging con tiers por planta y precio base para toda la tabla ECT."""
    base = (
        df.groupby("tipo")["caja_grosor_mm"]
        .first()
        .round(1)
        .map(PRECIO_BASE_GROSOR_FULL)
    )
    if base.isna().any():
        missing = df.groupby("tipo")["caja_grosor_mm"].first()[base.isna()]
        raise ValueError(f"Grosores sin precio base: {missing.tolist()}")

    filas = []
    for planta in PLANTAS:
        vol_tp = df.groupby("tipo")[f"volumen_producto_planta_{planta}"].sum()
        factor = factor_tier(vol_tp)
        filas.append(
            pd.DataFrame(
                {
                    "tipo": vol_tp.index,
                    "planta": planta,
                    "vol": vol_tp.values,
                    "factor": factor,
                    "costo": vol_tp.values * base.loc[vol_tp.index].values * factor,
                }
            )
        )

    det = pd.concat(filas, ignore_index=True)
    return float(det["costo"].sum()), det


def _evaluar_costo_por_planta(
    solution_df: pd.DataFrame,
    operations: pd.DataFrame,
    *,
    lim: tuple[float, float, float] | None = None,
) -> dict:
    """Evaluador Mark 8/9: packaging con tier por planta + flete por pallet."""
    df = preparar_eval_planta(solution_df[SUBMISSION_COLS], operations, lim=lim)
    pack, det_tp = _costo_packaging_por_planta_extended(df)
    flete = evaluar_flete_oficial(df)
    costo_total = pack + flete
    costo_baseline = _costo_baseline(operations)
    return {
        "costo_total": costo_total,
        "costo_packaging": pack,
        "costo_envio": flete,
        "cajas_unicas": int(df["tipo"].nunique()),
        "costo_baseline": costo_baseline,
        "ahorro_pct": _ahorro_pct_sobre_baseline(costo_total, costo_baseline),
        "det_tp": det_tp,
        "df_eval": df,
    }


def _metrics_from_evaluacion(evaluacion: dict) -> dict:
    costo_total = float(evaluacion["costo_total"])
    diff_cpsat = costo_total - CPSAT_QUOTA
    return {
        "costo_total": costo_total,
        "costo_envio": float(evaluacion["costo_envio"]),
        "costo_packaging": float(evaluacion["costo_packaging"]),
        "cajas_unicas": int(evaluacion["cajas_unicas"]),
        "costo_baseline": float(evaluacion["costo_baseline"]),
        "ahorro_pct": float(evaluacion["ahorro_pct"]),
        "cpsat_quota": CPSAT_QUOTA,
        "diff_cpsat": diff_cpsat,
        "diff_cpsat_pct": (diff_cpsat / CPSAT_QUOTA) * 100 if CPSAT_QUOTA else 0.0,
    }


def _utilizacion_por_planta(
    solution_df: pd.DataFrame,
    operations: pd.DataFrame,
    *,
    lim: tuple[float, float, float] | None = None,
) -> dict[str, dict]:
    """Distribución de utilización teórica de pallet por planta (ponderada por volumen)."""
    products_truth, _ = _load_reference_data()
    df = solution_df.merge(
        products_truth[["codigo_producto", "largo", "ancho", "alto"]],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )
    vol_cols = [f"volumen_producto_planta_{p}" for p in PLANTAS]
    df = df.merge(
        operations[["codigo_producto"] + vol_cols],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )
    df = calcular_cajas_por_pallet(df, lim=lim)

    result: dict[str, dict] = {}
    bin_edges = list(range(0, 101, 10))
    bin_labels = [f"{lo}-{hi}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    for planta in PLANTAS:
        vol_col = f"volumen_producto_planta_{planta}"
        subset = df[df[vol_col] > 0].copy()
        if subset.empty:
            result[planta] = {
                "mean_pct": 0.0,
                "histogram_pct": [0.0] * len(bin_labels),
                "bin_labels": bin_labels,
            }
            continue

        util_pct = subset["utilizacion_pallet_teorica"] * 100
        weights = subset[vol_col]
        mean_pct = float(np.average(util_pct, weights=weights))

        hist = np.zeros(len(bin_labels), dtype=float)
        for util, weight in zip(util_pct, weights):
            idx = min(int(util // 10), len(bin_labels) - 1)
            if util >= 100:
                idx = len(bin_labels) - 1
            hist[idx] += float(weight)

        total_weight = hist.sum()
        histogram_pct = (hist / total_weight * 100).tolist() if total_weight > 0 else [0.0] * len(bin_labels)

        result[planta] = {
            "mean_pct": mean_pct,
            "histogram_pct": histogram_pct,
            "bin_labels": bin_labels,
        }

    return result


def _costo_por_planta(
    evaluacion: dict,
) -> list[dict]:
    det_tp = evaluacion["det_tp"]
    df = evaluacion["df_eval"]
    plants = []

    for planta in PLANTAS:
        pack = float(det_tp.loc[det_tp["planta"] == planta, "costo"].sum())
        pallets = np.ceil(df[f"volumen_producto_planta_{planta}"] / df["cajas_por_pallet"])
        flete = float((pallets * COSTO_PALLET).sum())
        plants.append(
            {
                "id": planta,
                "label": PLANT_LABELS[planta],
                "costo_total": pack + flete,
                "costo_packaging": pack,
                "costo_envio": flete,
            }
        )

    plants.sort(key=lambda item: item["costo_total"], reverse=True)
    return plants


def _tier_metrics_por_planta(
    solution_df: pd.DataFrame,
    operations: pd.DataFrame,
) -> dict[str, dict]:
    box_demand = _build_box_demand_data(solution_df, operations)
    result: dict[str, dict] = {}
    for chart in box_demand["tier_charts"]:
        tiers_enriched = _tiers_with_pct(chart["tiers"])
        result[chart["id"]] = {
            "tier_discount_pct": _tier_discount_pct(tiers_enriched),
            "tiers": tiers_enriched,
        }
    return result


def _empty_tiers() -> list[dict]:
    return [
        {
            "key": tier["key"],
            "label": tier["label"],
            "short": tier["short"],
            "color": tier["color"],
            "n_tipos_caja": 0,
            "volumen": 0.0,
            "pct_volumen": 0.0,
        }
        for tier in TIER_DEFS
    ]


def _tiers_with_pct(tiers: list[dict]) -> list[dict]:
    total_vol = sum(tier["volumen"] for tier in tiers)
    enriched: list[dict] = []
    for tier in tiers:
        enriched.append(
            {
                **tier,
                "pct_volumen": (tier["volumen"] / total_vol * 100) if total_vol else 0.0,
            }
        )
    return enriched


def _tier_discount_pct(tiers: list[dict]) -> float:
    total_vol = sum(tier["volumen"] for tier in tiers)
    if not total_vol:
        return 0.0
    discount_keys = {"tier_3", "tier_4", "tier_5"}
    discount_vol = sum(tier["volumen"] for tier in tiers if tier["key"] in discount_keys)
    return discount_vol / total_vol * 100


def _plant_assignment_counts(
    solution_df: pd.DataFrame,
    operations: pd.DataFrame,
    planta: str | None = None,
) -> dict[str, int]:
    vol_cols = [f"volumen_producto_planta_{p}" for p in PLANTAS]
    df = solution_df.merge(
        operations[["codigo_producto"] + vol_cols],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )
    if planta is not None:
        vol_col = f"volumen_producto_planta_{planta}"
        df = df[df[vol_col] > 0]
    else:
        active_mask = np.zeros(len(df), dtype=bool)
        for plant in PLANTAS:
            active_mask |= df[f"volumen_producto_planta_{plant}"] > 0
        df = df[active_mask]

    df = agregar_id_tipo_caja(df)
    return {
        "cajas_unicas": int(df["caja_tipo_id_solucion"].nunique()),
        "productos_unicos": int(df["codigo_producto"].nunique()),
    }


def _global_tiers_from_box_demand(box_demand: dict) -> list[dict]:
    tier_counts = {tier["key"]: {"n_tipos_caja": 0, "volumen": 0.0} for tier in TIER_DEFS}
    for box in box_demand["box_types"]:
        for plant_data in box["by_plant"].values():
            bucket = tier_counts[plant_data["tier_key"]]
            bucket["n_tipos_caja"] += 1
            bucket["volumen"] += plant_data["volumen"]

    tiers = [
        {
            "key": tier["key"],
            "label": tier["label"],
            "short": tier["short"],
            "color": tier["color"],
            "n_tipos_caja": tier_counts[tier["key"]]["n_tipos_caja"],
            "volumen": tier_counts[tier["key"]]["volumen"],
        }
        for tier in TIER_DEFS
    ]
    return _tiers_with_pct(tiers)


def _build_plant_summary_cards(
    solution_df: pd.DataFrame | None,
    operations: pd.DataFrame,
    demanda_plants: list[dict],
    costo_plants: list[dict],
) -> list[dict]:
    demanda_by_id = {plant["id"]: plant for plant in demanda_plants}
    costo_by_id = {plant["id"]: plant for plant in costo_plants}
    has_solution = solution_df is not None

    box_demand = _build_box_demand_data(solution_df, operations) if has_solution else None
    util_planta = _utilizacion_por_planta(solution_df, operations) if has_solution else {}
    tier_chart_by_id = (
        {chart["id"]: chart["tiers"] for chart in box_demand["tier_charts"]}
        if box_demand
        else {}
    )

    def _build_card(
        card_id: str,
        label: str,
        *,
        is_global: bool,
        volumen: float,
        volumen_pct: float,
        n_productos: int,
        costo: dict,
        counts: dict[str, int | None],
        tiers: list[dict],
        util_mean_pct: float | None,
    ) -> dict:
        tiers_enriched = _tiers_with_pct(tiers)
        return {
            "id": card_id,
            "label": label,
            "is_global": is_global,
            "has_solution": has_solution,
            "volumen": volumen,
            "volumen_pct": volumen_pct,
            "n_productos": n_productos,
            "cajas_unicas": counts.get("cajas_unicas"),
            "productos_unicos": counts.get("productos_unicos", n_productos),
            "costo_total": float(costo.get("costo_total", 0.0)),
            "costo_packaging": float(costo.get("costo_packaging", 0.0)),
            "costo_envio": float(costo.get("costo_envio", 0.0)),
            "util_mean_pct": util_mean_pct,
            "tier_discount_pct": _tier_discount_pct(tiers_enriched),
            "tiers": tiers_enriched,
        }

    total_volumen = sum(plant["volumen"] for plant in demanda_plants)
    global_costo = {
        "costo_total": sum(plant["costo_total"] for plant in costo_plants),
        "costo_packaging": sum(plant["costo_packaging"] for plant in costo_plants),
        "costo_envio": sum(plant["costo_envio"] for plant in costo_plants),
    }

    if has_solution:
        global_counts = _plant_assignment_counts(solution_df, operations)
        global_tiers = _global_tiers_from_box_demand(box_demand)
        global_util = float(
            np.average(
                [util_planta[planta]["mean_pct"] for planta in PLANTAS],
                weights=[max(demanda_by_id[planta]["volumen"], 1.0) for planta in PLANTAS],
            )
        )
        global_productos = global_counts["productos_unicos"]
    else:
        global_counts = {"cajas_unicas": None, "productos_unicos": int((operations["volumen_producto_total"] > 0).sum())}
        global_tiers = _empty_tiers()
        global_util = None
        global_productos = global_counts["productos_unicos"]

    cards: list[dict] = [
        _build_card(
            "global",
            "Todas las plantas",
            is_global=True,
            volumen=total_volumen,
            volumen_pct=100.0 if total_volumen else 0.0,
            n_productos=global_productos,
            costo=global_costo,
            counts=global_counts,
            tiers=global_tiers,
            util_mean_pct=global_util,
        )
    ]

    plant_cards: list[dict] = []
    for planta in PLANTAS:
        demand = demanda_by_id[planta]
        if demand["volumen"] <= 0:
            continue

        if has_solution:
            counts = _plant_assignment_counts(solution_df, operations, planta)
            tiers = tier_chart_by_id.get(planta, _empty_tiers())
            util_mean_pct = util_planta.get(planta, {}).get("mean_pct")
        else:
            counts = {"cajas_unicas": None, "productos_unicos": demand["n_productos"]}
            tiers = _empty_tiers()
            util_mean_pct = None

        plant_cards.append(
            _build_card(
                planta,
                demand["label"],
                is_global=False,
                volumen=demand["volumen"],
                volumen_pct=demand["pct"],
                n_productos=demand["n_productos"],
                costo=costo_by_id.get(planta, {}),
                counts=counts,
                tiers=tiers,
                util_mean_pct=util_mean_pct,
            )
        )

    plant_cards.sort(key=lambda item: item["volumen"], reverse=True)
    cards.extend(plant_cards)
    return cards


def _build_solution_snapshot(
    name: str | None,
    solution_df: pd.DataFrame | None,
    operations: pd.DataFrame,
    *,
    lim: tuple[float, float, float] | None = None,
) -> dict | None:
    if solution_df is None or not name:
        return None

    evaluacion = _evaluar_costo_por_planta(solution_df, operations, lim=lim)
    util_planta = _utilizacion_por_planta(solution_df, operations, lim=lim)
    tier_planta = _tier_metrics_por_planta(solution_df, operations)
    costo_planta = _costo_por_planta(evaluacion)

    for plant in costo_planta:
        util = util_planta.get(plant["id"], {})
        plant["util_mean_pct"] = util.get("mean_pct", 0.0)
        plant["util_histogram_pct"] = util.get("histogram_pct", [])
        plant["util_bin_labels"] = util.get("bin_labels", [])
        tier = tier_planta.get(plant["id"], {})
        plant["tier_discount_pct"] = tier.get("tier_discount_pct", 0.0)
        plant["tiers"] = tier.get("tiers", _empty_tiers())

    return {
        "name": name,
        **_metrics_from_evaluacion(evaluacion),
        "plants": costo_planta,
    }


def _resolve_compare_solutions(available_solutions: list[str]) -> tuple[str | None, str | None]:
    key_a = "compare_solution_a"
    key_b = "compare_solution_b"

    for key in (key_a, key_b):
        value = session.get(key)
        if value and value not in available_solutions:
            session.pop(key, None)

    solution_a = session.get(key_a)
    solution_b = session.get(key_b)

    if not solution_a and available_solutions:
        solution_a = available_solutions[0]
        session[key_a] = solution_a

    if not solution_b and len(available_solutions) > 1:
        solution_b = available_solutions[1]
        session[key_b] = solution_b
    elif not solution_b and available_solutions:
        solution_b = available_solutions[0]
        session[key_b] = solution_b

    return solution_a, solution_b


def _plant_cards_from_snapshot(snapshot: dict, operations: pd.DataFrame) -> list[dict]:
    total_volumen = float(operations["volumen_producto_total"].sum())
    cards: list[dict] = []
    for plant in snapshot.get("plants") or []:
        vol_col = f"volumen_producto_planta_{plant['id']}"
        volumen = float(operations[vol_col].sum())
        cards.append(
            {
                "id": plant["id"],
                "label": plant["label"],
                "volumen": volumen,
                "volumen_pct": _pct(volumen, total_volumen),
                "costo_total": plant["costo_total"],
                "costo_envio": plant["costo_envio"],
                "costo_packaging": plant["costo_packaging"],
                "util_mean_pct": plant["util_mean_pct"],
                "util_histogram_pct": plant.get("util_histogram_pct", []),
                "util_bin_labels": plant.get("util_bin_labels", []),
            }
        )
    return cards


def _build_home_context() -> dict:
    context = _base_context()
    metrics = None
    snapshot = None
    plant_cards: list[dict] = []
    selected_solution = context["selected_solution"]

    solution_df = _load_selected_solution_df(selected_solution)
    if solution_df is not None:
        _, operations = _load_reference_data()
        snapshot = _build_solution_snapshot(selected_solution, solution_df, operations)
        if snapshot is not None:
            metrics = {
                key: snapshot[key]
                for key in (
                    "costo_total",
                    "costo_envio",
                    "costo_packaging",
                    "cajas_unicas",
                    "costo_baseline",
                    "ahorro_pct",
                    "cpsat_quota",
                    "diff_cpsat",
                    "diff_cpsat_pct",
                )
            }
            plant_cards = _plant_cards_from_snapshot(snapshot, operations)
            plant_cards.sort(key=lambda item: item["volumen"], reverse=True)

    context["metrics"] = metrics
    context["snapshot"] = snapshot
    context["plant_cards"] = plant_cards
    return context


def _build_packaging_context() -> dict:
    context = _base_context()
    packaging = None
    selected_solution = context["selected_solution"]

    solution_df = _load_selected_solution_df(selected_solution)
    if solution_df is not None:
        products_truth, _ = _load_reference_data()
        packaging = _build_packaging_payload(solution_df, products_truth)

    context["packaging"] = packaging
    return context


def _tier_key(volumen: float) -> str:
    if volumen >= 500_000:
        return "tier_5"
    if volumen >= 100_000:
        return "tier_4"
    if volumen >= 50_000:
        return "tier_3"
    if volumen >= 20_000:
        return "tier_2"
    return "tier_1"


def _tier_meta(volumen: float) -> dict:
    key = _tier_key(volumen)
    tier = next(item for item in TIER_DEFS if item["key"] == key)
    factor = float(factor_precio_por_volumen(pd.Series([volumen]))[0])
    return {
        "tier_key": tier["key"],
        "tier_label": tier["label"],
        "tier_short": tier["short"],
        "tier_color": tier["color"],
        "factor_descuento": factor,
    }


def _build_box_demand_data(solution_df: pd.DataFrame, operations: pd.DataFrame) -> dict:
    """Tabla unificada de tipos de caja con volumen por planta y resumen por tier."""
    vol_cols = ["volumen_producto_total"] + [f"volumen_producto_planta_{p}" for p in PLANTAS]
    df = solution_df.merge(
        operations[["codigo_producto"] + vol_cols],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )
    df = agregar_id_tipo_caja(df)

    plant_columns = [{"id": planta, "label": PLANT_LABELS[planta]} for planta in PLANTAS]

    agg_cols = {f"volumen_{planta}": (f"volumen_producto_planta_{planta}", "sum") for planta in PLANTAS}
    grouped = (
        df.groupby(["caja_tipo_id_solucion"] + BOX_COLS, as_index=False)
        .agg(
            volumen_total=("volumen_producto_total", "sum"),
            n_productos=("codigo_producto", "nunique"),
            **agg_cols,
        )
        .sort_values("volumen_total", ascending=False)
    )

    box_types: list[dict] = []
    for row in grouped.itertuples(index=False):
        by_plant: dict[str, dict] = {}
        for planta in PLANTAS:
            volumen = float(getattr(row, f"volumen_{planta}"))
            if volumen > 0:
                meta = _tier_meta(volumen)
                by_plant[planta] = {
                    "volumen": volumen,
                    **meta,
                }

        box_df = df[df["caja_tipo_id_solucion"] == row.caja_tipo_id_solucion]
        products: list[dict] = []
        for prod in box_df.sort_values("volumen_producto_total", ascending=False).itertuples(index=False):
            prod_by_plant: dict[str, dict] = {}
            for planta in PLANTAS:
                vol = float(getattr(prod, f"volumen_producto_planta_{planta}"))
                if vol > 0:
                    prod_by_plant[planta] = {"volumen": vol}

            products.append(
                {
                    "codigo_producto": prod.codigo_producto,
                    "volumen_total": float(prod.volumen_producto_total),
                    "by_plant": prod_by_plant,
                }
            )

        box_types.append(
            {
                "id": row.caja_tipo_id_solucion,
                "label": (
                    f"{row.caja_exterior_largo:.0f} × {row.caja_exterior_ancho:.0f} "
                    f"× {row.caja_exterior_alto:.0f} mm"
                ),
                "volumen_total": float(row.volumen_total),
                "n_productos": int(row.n_productos),
                "by_plant": by_plant,
                "products": products,
            }
        )

    tier_charts: list[dict] = []
    for planta in PLANTAS:
        tier_counts = {tier["key"]: {"n_tipos_caja": 0, "volumen": 0.0} for tier in TIER_DEFS}
        for box in box_types:
            plant_data = box["by_plant"].get(planta)
            if plant_data is None:
                continue
            bucket = tier_counts[plant_data["tier_key"]]
            bucket["n_tipos_caja"] += 1
            bucket["volumen"] += plant_data["volumen"]

        tier_charts.append(
            {
                "id": planta,
                "label": PLANT_LABELS[planta],
                "tiers": [
                    {
                        "key": tier["key"],
                        "label": tier["label"],
                        "short": tier["short"],
                        "color": tier["color"],
                        "n_tipos_caja": tier_counts[tier["key"]]["n_tipos_caja"],
                        "volumen": tier_counts[tier["key"]]["volumen"],
                    }
                    for tier in TIER_DEFS
                ],
            }
        )

    tier_charts.sort(
        key=lambda item: sum(tier["volumen"] for tier in item["tiers"]),
        reverse=True,
    )

    return {
        "plant_columns": plant_columns,
        "box_types": box_types,
        "n_tipos_caja": len(box_types),
        "tier_charts": tier_charts,
    }


def _pct(part: float, total: float) -> float:
    return (part / total * 100.0) if total else 0.0


def _build_channel_mix(operations: pd.DataFrame, total_volumen: float) -> list[dict]:
    channels = []
    for channel in CHANNEL_DEFS:
        volumen = float(operations[channel["col"]].sum())
        channels.append(
            {
                "id": channel["id"],
                "label": channel["label"],
                "color": channel["color"],
                "volumen": volumen,
                "pct": _pct(volumen, total_volumen),
            }
        )
    channels.sort(key=lambda item: item["volumen"], reverse=True)
    return channels


def _build_sku_pareto(operations: pd.DataFrame, total_volumen: float) -> dict:
    volumes = (
        operations.loc[operations["volumen_producto_total"] > 0, "volumen_producto_total"]
        .sort_values(ascending=False)
        .to_numpy(dtype=float)
    )
    n_skus = int(len(volumes))
    if n_skus == 0 or total_volumen <= 0:
        return {
            "curve": [],
            "milestones": [],
            "top_30_pct": 0.0,
            "top_97_pct": 0.0,
        }

    cumulative = np.cumsum(volumes)
    step = max(1, n_skus // 80)
    sample_idx = list(range(0, n_skus, step))
    if sample_idx[-1] != n_skus - 1:
        sample_idx.append(n_skus - 1)

    curve = [
        {
            "rank": int(idx + 1),
            "pct_skus": _pct(float(idx + 1), float(n_skus)),
            "pct_volumen": _pct(float(cumulative[idx]), total_volumen),
        }
        for idx in sample_idx
    ]

    milestones = []
    for n in PARETO_TOP_N:
        capped = min(n, n_skus)
        pct_vol = _pct(float(cumulative[capped - 1]), total_volumen)
        milestones.append(
            {
                "n_skus": capped,
                "pct_volumen": pct_vol,
                "label": f"Top {capped}",
            }
        )

    return {
        "curve": curve,
        "milestones": milestones,
        "top_30_pct": milestones[0]["pct_volumen"] if milestones else 0.0,
        "top_97_pct": milestones[1]["pct_volumen"] if len(milestones) > 1 else 0.0,
    }


def _build_multiplant_fragmentation(operations: pd.DataFrame, total_volumen: float) -> dict:
    plant_cols = [f"volumen_producto_planta_{planta}" for planta in PLANTAS]
    active = operations.loc[operations["volumen_producto_total"] > 0].copy()
    n_plants = (active[plant_cols] > 0).sum(axis=1)

    buckets: list[dict] = []
    multiplant_skus = 0
    multiplant_volumen = 0.0

    for n in range(1, len(PLANTAS) + 1):
        mask = n_plants == n
        n_skus = int(mask.sum())
        volumen = float(active.loc[mask, "volumen_producto_total"].sum())
        if n > 1:
            multiplant_skus += n_skus
            multiplant_volumen += volumen
        buckets.append(
            {
                "n_plantas": n,
                "label": f"{n} planta{'s' if n != 1 else ''}",
                "n_skus": n_skus,
                "volumen": volumen,
                "pct_skus": _pct(float(n_skus), float(len(active))),
                "pct_volumen": _pct(volumen, total_volumen),
            }
        )

    return {
        "buckets": buckets,
        "multiplant_skus": multiplant_skus,
        "multiplant_pct_skus": _pct(float(multiplant_skus), float(len(active))),
        "multiplant_volumen": multiplant_volumen,
        "multiplant_pct_volumen": _pct(multiplant_volumen, total_volumen),
    }


def _build_volume_buckets(operations: pd.DataFrame, total_volumen: float) -> list[dict]:
    active = operations.loc[operations["volumen_producto_total"] > 0, "volumen_producto_total"]
    n_skus_total = int(len(active))
    rows: list[dict] = []

    for bucket in VOLUME_BUCKETS:
        if bucket["max"] is None:
            mask = active >= bucket["min"]
        else:
            mask = (active >= bucket["min"]) & (active < bucket["max"])
        n_skus = int(mask.sum())
        volumen = float(active.loc[mask].sum())
        rows.append(
            {
                "id": bucket["id"],
                "label": bucket["label"],
                "n_skus": n_skus,
                "volumen": volumen,
                "pct_skus": _pct(float(n_skus), float(n_skus_total)),
                "pct_volumen": _pct(volumen, total_volumen),
            }
        )
    return rows


def _build_category_mix(operations: pd.DataFrame, total_volumen: float) -> dict:
    catalogo = pd.read_csv(CATALOGO_PATH, usecols=["codigo_producto", "categoria", "tipo_proyecto"])
    merged = operations.merge(catalogo, on="codigo_producto", how="left", validate="one_to_one")
    active = merged.loc[merged["volumen_producto_total"] > 0].copy()
    active["categoria"] = active["categoria"].fillna("Sin categoría")
    active["tipo_proyecto"] = active["tipo_proyecto"].fillna("Sin tipo")

    def _group_rows(column: str) -> list[dict]:
        grouped = (
            active.groupby(column, as_index=False)
            .agg(volumen=("volumen_producto_total", "sum"), n_skus=("codigo_producto", "nunique"))
            .sort_values("volumen", ascending=False)
        )
        return [
            {
                "id": str(row[column]),
                "label": str(row[column]),
                "volumen": float(row["volumen"]),
                "pct": _pct(float(row["volumen"]), total_volumen),
                "n_skus": int(row["n_skus"]),
            }
            for _, row in grouped.iterrows()
        ]

    return {
        "categorias": _group_rows("categoria"),
        "tipos_proyecto": _group_rows("tipo_proyecto"),
    }


def _baseline_costo_por_planta(operations: pd.DataFrame) -> list[dict]:
    plants = []
    for planta in PLANTAS:
        packaging = float(operations[f"costo_total_planta_{planta}"].sum())
        flete = float(operations[f"costo_pallets_planta_{planta}"].sum())
        plants.append(
            {
                "id": planta,
                "label": PLANT_LABELS[planta],
                "costo_total": packaging + flete,
                "costo_packaging": packaging,
                "costo_envio": flete,
            }
        )
    plants.sort(key=lambda item: item["costo_total"], reverse=True)
    return plants


def _build_demanda_payload(
    solution_df: pd.DataFrame | None,
    operations: pd.DataFrame,
) -> dict:
    total_volumen = float(operations["volumen_producto_total"].sum())
    n_productos_total = int((operations["volumen_producto_total"] > 0).sum())

    plants = []
    for planta in PLANTAS:
        col = f"volumen_producto_planta_{planta}"
        volumen = float(operations[col].sum())
        n_productos = int((operations[col] > 0).sum())
        plants.append(
            {
                "id": planta,
                "label": PLANT_LABELS[planta],
                "volumen": volumen,
                "pct": _pct(volumen, total_volumen),
                "n_productos": n_productos,
            }
        )

    plants.sort(key=lambda item: item["volumen"], reverse=True)

    channels = _build_channel_mix(operations, total_volumen)
    pareto = _build_sku_pareto(operations, total_volumen)
    multiplant = _build_multiplant_fragmentation(operations, total_volumen)
    volume_buckets = _build_volume_buckets(operations, total_volumen)
    category_mix = _build_category_mix(operations, total_volumen)

    if solution_df is not None:
        evaluacion = _evaluar_costo_por_planta(solution_df, operations)
        costo_plants = _costo_por_planta(evaluacion)
        costo_source = "solution"
    else:
        costo_plants = _baseline_costo_por_planta(operations)
        costo_source = "baseline"

    costo_by_id = {item["id"]: item for item in costo_plants}
    plants_detail = []
    for plant in plants:
        costo = costo_by_id.get(plant["id"], {})
        plants_detail.append(
            {
                **plant,
                "costo_total": float(costo.get("costo_total", 0.0)),
                "costo_packaging": float(costo.get("costo_packaging", 0.0)),
                "costo_envio": float(costo.get("costo_envio", 0.0)),
            }
        )

    country_demand: dict[str, dict] = {}
    max_volumen = max((plant["volumen"] for plant in plants), default=0.0)
    for plant in plants:
        iso3 = PLANT_COUNTRY[plant["id"]]
        country_entry = country_demand.setdefault(
            iso3,
            {
                "iso3": iso3,
                "name": COUNTRY_NAMES.get(iso3, iso3),
                "volumen": 0.0,
                "pct": 0.0,
                "plants": [],
            },
        )
        country_entry["volumen"] += plant["volumen"]
        country_entry["plants"].append(
            {
                "id": plant["id"],
                "label": plant["label"],
                "volumen": plant["volumen"],
                "pct": plant["pct"],
            }
        )

    for entry in country_demand.values():
        entry["pct"] = _pct(entry["volumen"], total_volumen)
        entry["intensity"] = (entry["volumen"] / max_volumen) if max_volumen else 0.0

    summary_cards = _build_plant_summary_cards(
        solution_df,
        operations,
        plants,
        costo_plants,
    )

    box_demand: dict | None = None
    if solution_df is not None:
        box_demand = _build_box_demand_data(solution_df, operations)

    return {
        "plants": plants,
        "plants_detail": plants_detail,
        "total_volumen": total_volumen,
        "n_productos_total": n_productos_total,
        "top_30_pct": pareto["top_30_pct"],
        "multiplant_pct_skus": multiplant["multiplant_pct_skus"],
        "channels": channels,
        "pareto": pareto,
        "multiplant": multiplant,
        "volume_buckets": volume_buckets,
        "category_mix": category_mix,
        "costo_plants": costo_plants,
        "costo_source": costo_source,
        "country_demand": country_demand,
        "summary_cards": summary_cards,
        "box_demand": box_demand,
        "tier_thresholds": TIER_THRESHOLDS,
        "tier_defs": TIER_DEFS,
    }


def _build_demanda_context() -> dict:
    context = _base_context()
    _, operations = _load_reference_data()
    solution_df = _load_selected_solution_df(context["selected_solution"])
    context["demanda"] = _build_demanda_payload(solution_df, operations)
    return context


def _short_subcategoria_label(categoria: str, subcategoria: str) -> str:
    prefix = f"{categoria} - "
    if subcategoria.startswith(prefix):
        return subcategoria[len(prefix) :]
    return subcategoria


def _build_productos_payload() -> dict:
    catalogo = pd.read_csv(CATALOGO_PATH)
    operations = pd.read_csv(OPERATIONS_PATH)
    cajas = pd.read_csv(CAJAS_PATH)

    df = catalogo.merge(
        operations,
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )

    plants = [{"id": planta, "label": PLANT_LABELS[planta]} for planta in PLANTAS]
    categories = sorted(df["categoria"].dropna().unique())

    products: list[dict] = []
    for row in df.sort_values("codigo_producto").itertuples(index=False):
        volumen_por_planta: dict[str, float] = {}
        for planta in PLANTAS:
            col = f"volumen_producto_planta_{planta}"
            volumen_por_planta[planta] = float(getattr(row, col) or 0)

        categoria = str(row.categoria)
        subcategoria = str(row.subcategoria)
        products.append(
            {
                "codigo_producto": row.codigo_producto,
                "descripcion": str(row.descripcion_producto),
                "categoria": categoria,
                "subcategoria": subcategoria,
                "subcategoria_label": _short_subcategoria_label(categoria, subcategoria),
                "tipo_proyecto": str(row.tipo_proyecto),
                "ingrediente_forma": str(row.ingrediente_forma),
                "volumen_total": float(row.volumen_producto_total or 0),
                "volumen_por_planta": volumen_por_planta,
            }
        )

    return {
        "total_skus": int(df["codigo_producto"].nunique()),
        "cajas_disponibles": int(cajas["caja_tipo_id"].nunique()),
        "plants": plants,
        "categories": categories,
        "products": products,
    }


def _build_productos_context() -> dict:
    context = _base_context()
    context["productos"] = _build_productos_payload()

    solution_df = _load_selected_solution_df(context["selected_solution"])
    packaging = None
    if solution_df is not None:
        products_truth, _ = _load_reference_data()
        packaging = _build_packaging_payload(solution_df, products_truth)

    context["packaging"] = packaging
    return context


def _build_ordenes_compra_payload(solution_df: pd.DataFrame, operations: pd.DataFrame) -> dict:
    box_demand = _build_box_demand_data(solution_df, operations)
    catalogo = pd.read_csv(CATALOGO_PATH)
    catalog_index = catalogo.set_index("codigo_producto")

    enriched_df = agregar_id_tipo_caja(solution_df.copy())
    grosor_by_box = (
        enriched_df.groupby("caja_tipo_id_solucion")["caja_grosor_mm"].first().astype(float).to_dict()
    )

    plants_payload: list[dict] = []
    for planta in PLANTAS:
        boxes: list[dict] = []
        for box in box_demand["box_types"]:
            plant_data = box["by_plant"].get(planta)
            if plant_data is None:
                continue

            products: list[dict] = []
            for prod in box["products"]:
                plant_vol = prod["by_plant"].get(planta)
                if plant_vol is None:
                    continue

                if prod["codigo_producto"] in catalog_index.index:
                    cat_row = catalog_index.loc[prod["codigo_producto"]]
                    descripcion = str(cat_row["descripcion_producto"])
                    categoria = str(cat_row["categoria"])
                else:
                    descripcion = ""
                    categoria = ""

                products.append(
                    {
                        "codigo_producto": prod["codigo_producto"],
                        "descripcion": descripcion,
                        "categoria": categoria,
                        "volumen": float(plant_vol["volumen"]),
                    }
                )

            boxes.append(
                {
                    "id": box["id"],
                    "label": box["label"],
                    "grosor_mm": float(grosor_by_box.get(box["id"], 0.0)),
                    "volumen": float(plant_data["volumen"]),
                    "tier_label": plant_data["tier_label"],
                    "n_productos": len(products),
                    "products": products,
                }
            )

        boxes.sort(key=lambda item: item["volumen"], reverse=True)
        if boxes:
            product_codes = {
                product["codigo_producto"]
                for box in boxes
                for product in box["products"]
            }
            plants_payload.append(
                {
                    "id": planta,
                    "label": PLANT_LABELS[planta],
                    "boxes": boxes,
                    "n_cajas": len(boxes),
                    "n_productos": len(product_codes),
                    "volumen_total": float(sum(box["volumen"] for box in boxes)),
                }
            )

    solution_codes = set(solution_df["codigo_producto"])
    solution_catalog = catalogo[catalogo["codigo_producto"].isin(solution_codes)]
    category_counts = (
        solution_catalog.groupby("categoria")["codigo_producto"]
        .nunique()
        .sort_values(ascending=False)
    )
    category_chart = [
        {"label": str(label), "n_productos": int(value)}
        for label, value in category_counts.items()
    ]

    plant_chart: list[dict] = []
    solution_ops = operations[operations["codigo_producto"].isin(solution_codes)]
    for planta in PLANTAS:
        col = f"volumen_producto_planta_{planta}"
        n_productos = int((solution_ops[col] > 0).sum())
        if n_productos > 0:
            plant_chart.append(
                {
                    "id": planta,
                    "label": PLANT_LABELS[planta],
                    "n_productos": n_productos,
                }
            )
    plant_chart.sort(key=lambda item: item["n_productos"], reverse=True)

    return {
        "summary": {
            "n_productos": int(solution_df["codigo_producto"].nunique()),
            "n_plantas": len(plants_payload),
            "n_cajas": int(box_demand["n_tipos_caja"]),
        },
        "plants": plants_payload,
        "category_chart": category_chart,
        "plant_chart": plant_chart,
    }


def _build_ordenes_compra_context() -> dict:
    context = _base_context()
    ordenes = None
    selected_solution = context["selected_solution"]
    solution_df = _load_selected_solution_df(selected_solution)

    if solution_df is not None:
        _, operations = _load_reference_data()
        ordenes = _build_ordenes_compra_payload(solution_df, operations)

    context["ordenes"] = ordenes
    return context


def _require_ordenes_compra_payload(selected_solution: str) -> dict:
    solution_df = _load_selected_solution_df(selected_solution)
    if solution_df is None:
        raise ValueError("No hay solución seleccionada.")

    _, operations = _load_reference_data()
    payload = _build_ordenes_compra_payload(solution_df, operations)
    if not payload["plants"]:
        raise ValueError("La solución activa no tiene demanda asignada por planta.")

    return payload


def _build_ordenes_compra_document_context(
    selected_solution: str,
    *,
    embed: bool,
    plant_id: str | None,
) -> dict:
    payload = _require_ordenes_compra_payload(selected_solution)

    if plant_id is None:
        return {
            "ordenes": payload,
            "solution_name": selected_solution,
            "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "embed": embed,
            "plant_id": None,
            "plant_label": "Todas las plantas",
        }

    if plant_id not in PLANTAS:
        raise ValueError("Planta no válida.")

    plants = [plant for plant in payload["plants"] if plant["id"] == plant_id]
    if not plants:
        raise ValueError("La planta no tiene demanda en la solución activa.")

    return {
        "ordenes": {**payload, "plants": plants},
        "solution_name": selected_solution,
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "embed": embed,
        "plant_id": plant_id,
        "plant_label": plants[0]["label"],
    }


def _render_ordenes_document(
    template: str,
    selected_solution: str,
    plant_id: str | None,
) -> object:
    try:
        context = _build_ordenes_compra_document_context(
            selected_solution,
            embed=request.args.get("embed") == "1",
            plant_id=plant_id,
        )
    except ValueError as exc:
        session["upload_error"] = str(exc)
        return redirect(url_for("ordenes_compra"))

    response = make_response(render_template(template, **context))
    if request.args.get("download") == "1":
        stem = selected_solution.replace(".csv", "")
        suffix = "proveedor" if "proveedor" in template else "asignacion"
        scope = plant_id or "todas"
        response.headers["Content-Disposition"] = (
            f'attachment; filename="orden_cajas_{suffix}_{scope}_{stem}.html"'
        )
    return response


ASIGNACION_POLICIES = {
    "pallet": {
        "id": "pallet",
        "label": "Pallet-aware",
        "short": "Pallet",
        "description": "Maximiza cajas/pallet entre compatibles; desempate por menor precio y volumen.",
        "recommended": True,
        "metodo": pallet_aware_asignar_producto,
    },
    "m1": {
        "id": "m1",
        "label": "M1 (volumen)",
        "short": "M1",
        "description": "Asigna la caja existente de menor volumen externo compatible.",
        "recommended": False,
        "metodo": m1_asignar_producto_a_caja,
    },
}


def _parse_positive_float(raw, *, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} inválido.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{field} debe ser un número positivo.")
    return float(value)


def _box_occupancy_pct(producto: dict, caja: dict) -> float | None:
    grosor = float(caja["caja_grosor_mm"])
    int_l = float(caja["caja_exterior_largo"]) - 2.0 * grosor
    int_w = float(caja["caja_exterior_ancho"]) - 2.0 * grosor
    int_h = float(caja["caja_exterior_alto"]) - 2.0 * grosor
    if min(int_l, int_w, int_h) <= 0:
        return None
    vol_prod = float(producto["largo"]) * float(producto["ancho"]) * float(producto["alto"])
    vol_int = int_l * int_w * int_h
    if vol_int <= 0:
        return None
    return 100.0 * vol_prod / vol_int


def _compatibles_ranked(producto: dict, catalogo: pd.DataFrame, policy: str) -> list[dict]:
    rows: list[dict] = []
    for _, caja in catalogo.iterrows():
        ok = caja_cumple_restricciones_fisicas(
            float(producto["largo"]),
            float(producto["ancho"]),
            float(producto["alto"]),
            float(producto["peso_neto_caja"]),
            caja["caja_grosor_mm"],
            caja["caja_exterior_largo"],
            caja["caja_exterior_ancho"],
            caja["caja_exterior_alto"],
        )
        if not ok:
            continue
        rows.append(
            {
                "box_id": str(caja["box_id"]),
                "caja_grosor_mm": float(caja["caja_grosor_mm"]),
                "caja_exterior_largo": float(caja["caja_exterior_largo"]),
                "caja_exterior_ancho": float(caja["caja_exterior_ancho"]),
                "caja_exterior_alto": float(caja["caja_exterior_alto"]),
                "volumen_externo": float(caja["volumen_externo"]),
                "cajas_por_pallet": int(caja["cajas_por_pallet"]),
                "precio_base": float(caja["precio_base"]) if pd.notna(caja["precio_base"]) else None,
            }
        )

    if policy == "m1":
        rows.sort(key=lambda r: (r["volumen_externo"], -r["cajas_por_pallet"], r["caja_grosor_mm"]))
    else:
        rows.sort(
            key=lambda r: (
                -r["cajas_por_pallet"],
                r["precio_base"] if r["precio_base"] is not None else 1e9,
                r["volumen_externo"],
            )
        )
    return rows


def _assign_product_with_policy(
    producto: dict,
    solution_df: pd.DataFrame,
    policy: str,
) -> dict:
    if policy not in ASIGNACION_POLICIES:
        raise ValueError("Política inválida.")

    sol = add_box_id(solution_df[SUBMISSION_COLS].copy())
    sol = agregar_id_tipo_caja(sol)
    catalogo = catalogo_cajas_unicas(sol)
    counts = sol["box_id"].value_counts()
    id_map = (
        sol[["box_id", "caja_tipo_id_solucion"]]
        .drop_duplicates("box_id")
        .set_index("box_id")["caja_tipo_id_solucion"]
        .to_dict()
    )

    meta = ASIGNACION_POLICIES[policy]
    try:
        es_nueva, caja = meta["metodo"](producto, sol)
    except ValueError as exc:
        raise ValueError(
            "No hay tipología factible para esas dimensiones "
            "(restricciones físicas / pallet / ECT)."
        ) from exc
    box_id = str(caja["box_id"])
    ocupacion = _box_occupancy_pct(producto, caja)

    vol_ext = (
        float(caja["caja_exterior_largo"])
        * float(caja["caja_exterior_ancho"])
        * float(caja["caja_exterior_alto"])
    )
    cajas_pallet = int(
        np.floor(PALLET_ANCHO_MM / float(caja["caja_exterior_largo"]))
        * np.floor(PALLET_LARGO_MM / float(caja["caja_exterior_ancho"]))
        * np.floor(PALLET_ALTO_MAX_MM / float(caja["caja_exterior_alto"]))
    )
    vol_pallet = float(PALLET_ANCHO_MM * PALLET_LARGO_MM * PALLET_ALTO_MAX_MM)
    util_pallet = 100.0 * cajas_pallet * vol_ext / vol_pallet if vol_pallet > 0 else None

    n_skus = 0 if es_nueva else int(counts.get(box_id, 0))
    skus_ejemplo: list[str] = []
    if not es_nueva and n_skus > 0:
        skus_ejemplo = (
            sol.loc[sol["box_id"] == box_id, "codigo_producto"]
            .astype(str)
            .head(8)
            .tolist()
        )

    compat = _compatibles_ranked(producto, catalogo, policy)
    for row in compat:
        row["caja_tipo_id"] = id_map.get(row["box_id"])
        row["n_skus"] = int(counts.get(row["box_id"], 0))
        row["seleccionada"] = (not es_nueva) and row["box_id"] == box_id

    precio = PRECIO_BASE_GROSOR_FULL.get(round(float(caja["caja_grosor_mm"]), 1))

    return {
        "policy": policy,
        "policy_label": meta["label"],
        "policy_description": meta["description"],
        "recommended": meta["recommended"],
        "es_caja_nueva": bool(es_nueva),
        "destino": "Caja nueva" if es_nueva else "Catálogo existente",
        "box_id": box_id,
        "caja_tipo_id": None if es_nueva else id_map.get(box_id),
        "caja": {
            "grosor_mm": float(caja["caja_grosor_mm"]),
            "largo": float(caja["caja_exterior_largo"]),
            "ancho": float(caja["caja_exterior_ancho"]),
            "alto": float(caja["caja_exterior_alto"]),
            "label": (
                f"{caja['caja_exterior_largo']:.0f} × {caja['caja_exterior_ancho']:.0f} × "
                f"{caja['caja_exterior_alto']:.0f} mm"
            ),
            "volumen_externo": vol_ext,
            "cajas_por_pallet": cajas_pallet,
            "utilizacion_pallet_pct": util_pallet,
            "ocupacion_caja_pct": ocupacion,
            "precio_base": float(precio) if precio is not None else None,
        },
        "n_skus_en_caja": n_skus,
        "skus_ejemplo": skus_ejemplo,
        "n_compatibles": len(compat),
        "compatibles": compat[:12],
        "n_tipos_catalogo": int(len(catalogo)),
    }


def _build_asignacion_payload(
    solution_df: pd.DataFrame,
    *,
    largo: float,
    ancho: float,
    alto: float,
    peso_neto_caja: float,
) -> dict:
    producto = {
        "codigo_producto": "NUEVO",
        "largo": float(largo),
        "ancho": float(ancho),
        "alto": float(alto),
        "peso_neto_caja": float(peso_neto_caja),
    }
    results = {
        policy: _assign_product_with_policy(producto, solution_df, policy)
        for policy in ("pallet", "m1")
    }
    return {
        "producto": {
            "largo": float(largo),
            "ancho": float(ancho),
            "alto": float(alto),
            "peso_neto_caja": float(peso_neto_caja),
            "label": f"{largo:g} × {ancho:g} × {alto:g} mm",
        },
        "results": results,
        "same_decision": (
            results["pallet"]["es_caja_nueva"] == results["m1"]["es_caja_nueva"]
            and results["pallet"]["box_id"] == results["m1"]["box_id"]
        ),
    }


def _build_asignacion_context() -> dict:
    context = _base_context()
    selected = context["selected_solution"]
    solution_df = _load_selected_solution_df(selected)
    n_tipos = 0
    n_productos = 0
    if solution_df is not None and len(solution_df):
        sol = add_box_id(solution_df[SUBMISSION_COLS].copy())
        n_tipos = int(sol["box_id"].nunique())
        n_productos = int(len(sol))

    context.update(
        {
            "asignacion": {
                "n_tipos": n_tipos,
                "n_productos": n_productos,
                "has_solution": solution_df is not None and len(solution_df) > 0,
                "policies": [
                    {
                        "id": p["id"],
                        "label": p["label"],
                        "description": p["description"],
                        "recommended": p["recommended"],
                    }
                    for p in ASIGNACION_POLICIES.values()
                ],
            }
        }
    )
    return context


def _build_comparacion_context() -> dict:
    context = _base_context()
    available_solutions = context["available_solutions"]
    solution_a_name, solution_b_name = _resolve_compare_solutions(available_solutions)

    _, operations = _load_reference_data()
    snapshot_a = _build_solution_snapshot(
        solution_a_name,
        _load_selected_solution_df(solution_a_name),
        operations,
    )
    snapshot_b = _build_solution_snapshot(
        solution_b_name,
        _load_selected_solution_df(solution_b_name),
        operations,
    )

    ideal_winner_name = None
    ideal_winner_cost = None
    ideal_savings = None
    ideal_is_tie = False
    ideal_solution = None
    if snapshot_a and snapshot_b:
        if abs(snapshot_a["costo_total"] - snapshot_b["costo_total"]) < 0.01:
            ideal_is_tie = True
            ideal_winner_cost = snapshot_a["costo_total"]
        elif snapshot_a["costo_total"] < snapshot_b["costo_total"]:
            ideal_solution = "a"
            ideal_winner_name = snapshot_a["name"]
            ideal_winner_cost = snapshot_a["costo_total"]
            ideal_savings = snapshot_b["costo_total"] - snapshot_a["costo_total"]
        else:
            ideal_solution = "b"
            ideal_winner_name = snapshot_b["name"]
            ideal_winner_cost = snapshot_b["costo_total"]
            ideal_savings = snapshot_a["costo_total"] - snapshot_b["costo_total"]

    plant_cards: list[dict] = []
    if snapshot_a and snapshot_b:
        plants_b = {plant["id"]: plant for plant in snapshot_b["plants"]}
        for plant_a in snapshot_a["plants"]:
            plant_b = plants_b.get(plant_a["id"])
            if plant_b is None:
                continue
            plant_cards.append(
                {
                    "id": plant_a["id"],
                    "label": plant_a["label"],
                    "solution_a": {
                        "costo_total": plant_a["costo_total"],
                        "costo_envio": plant_a["costo_envio"],
                        "costo_packaging": plant_a["costo_packaging"],
                        "util_mean_pct": plant_a["util_mean_pct"],
                        "util_histogram_pct": plant_a["util_histogram_pct"],
                        "util_bin_labels": plant_a["util_bin_labels"],
                        "tier_discount_pct": plant_a["tier_discount_pct"],
                    },
                    "solution_b": {
                        "costo_total": plant_b["costo_total"],
                        "costo_envio": plant_b["costo_envio"],
                        "costo_packaging": plant_b["costo_packaging"],
                        "util_mean_pct": plant_b["util_mean_pct"],
                        "util_histogram_pct": plant_b["util_histogram_pct"],
                        "util_bin_labels": plant_b["util_bin_labels"],
                        "tier_discount_pct": plant_b["tier_discount_pct"],
                    },
                }
            )

    context.update(
        {
            "solution_a": solution_a_name,
            "solution_b": solution_b_name,
            "snapshot_a": snapshot_a,
            "snapshot_b": snapshot_b,
            "ideal_winner_name": ideal_winner_name,
            "ideal_winner_cost": ideal_winner_cost,
            "ideal_savings": ideal_savings,
            "ideal_is_tie": ideal_is_tie,
            "ideal_solution": ideal_solution,
            "plant_cards": plant_cards,
        }
    )
    return context


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("index"))
        return view(*args, **kwargs)

    return wrapped


@app.route("/")
def index():
    if session.get("authenticated"):
        return redirect(url_for("home"))
    return render_template("login.html", error=request.args.get("error"))


@app.route("/login", methods=["POST"])
def login():
    code = request.form.get("code", "").strip().upper()
    wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest"
    if ACCESS_CODE and code == ACCESS_CODE:
        session["authenticated"] = True
        if wants_json:
            return jsonify({"ok": True, "redirect": url_for("home")})
        return redirect(url_for("home"))
    if wants_json:
        return jsonify({"ok": False, "error": "Código incorrecto. Intentá de nuevo."}), 401
    return redirect(url_for("index", error="Código incorrecto. Intentá de nuevo."))


@app.route("/home")
@login_required
def home():
    return render_template("home.html", **_build_home_context())


@app.route("/packaging")
@login_required
def packaging():
    return render_template("packaging.html", **_build_packaging_context())


@app.route("/demanda")
@login_required
def demanda():
    return render_template("demanda.html", **_build_demanda_context())


@app.route("/productos")
@login_required
def productos():
    return render_template("productos.html", **_build_productos_context())


@app.route("/plantas")
@login_required
def plantas():
    return redirect(url_for("demanda"))


@app.route("/optimizar")
@login_required
def optimizar():
    return render_template(
        "optimizar.html",
        **_base_context(),
        available_grosores=sorted(PRECIO_BASE_GROSOR.keys()),
        effort_options=[
            {"id": "low", "label": "Low", "seconds": int(EFFORT_SECONDS["low"])},
            {"id": "mid", "label": "Mid", "seconds": int(EFFORT_SECONDS["mid"])},
            {"id": "high", "label": "High", "seconds": int(EFFORT_SECONDS["high"])},
        ],
        model_options=[
            {
                "id": m["id"],
                "label": m["label"],
                "short": m["short"],
                "description": (
                    "Minimiza costo real con vols/tiers por planta (Mark 16)."
                    if m["id"] == "mark16"
                    else "Proxies de utilización sin magnitudes de demanda (Mark 17 / protocolo)."
                ),
            }
            for m in MODELS.values()
        ],
        pallet_defaults={
            "ancho_mm": int(PALLET_ANCHO_MM),
            "largo_mm": int(PALLET_LARGO_MM),
            "alto_mm": int(PALLET_ALTO_MAX_MM),
        },
    )


@app.route("/asignacion")
@login_required
def asignacion():
    return render_template("asignacion.html", **_build_asignacion_context())


@app.route("/api/asignacion", methods=["POST"])
@login_required
def api_asignacion():
    context = _base_context()
    selected = context["selected_solution"]
    solution_df = _load_selected_solution_df(selected)
    if solution_df is None or not len(solution_df):
        return jsonify({"ok": False, "error": "Seleccioná una solución en el sidebar."}), 400

    payload = request.get_json(silent=True) or {}
    try:
        largo = _parse_positive_float(payload.get("largo"), field="Largo")
        ancho = _parse_positive_float(payload.get("ancho"), field="Ancho")
        alto = _parse_positive_float(payload.get("alto"), field="Alto")
        peso = _parse_positive_float(payload.get("peso_neto_caja"), field="Peso neto")
        result = _build_asignacion_payload(
            solution_df,
            largo=largo,
            ancho=ancho,
            alto=alto,
            peso_neto_caja=peso,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": f"No se pudo asignar: {exc}"}), 400

    return jsonify(
        {
            "ok": True,
            "solution_name": selected,
            **result,
        }
    )


def _analyze_optimize_solution(
    solution_df: pd.DataFrame, *, original_name: str, source: str
) -> tuple[dict, int]:
    """Valida, mide y materializa warm-start en .optimize_uploads."""
    try:
        products_truth, operations = _load_reference_data()
        is_valid, errors, warnings, _ = validate_solution_free(
            solution_df, products_truth, verbose=False
        )
        if not is_valid:
            return (
                {"ok": False, "error": errors[0] if errors else "Solución inválida."},
                400,
            )

        metrics = _metrics_from_evaluacion(
            _evaluar_costo_por_planta(solution_df, operations)
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 400

    safe_name = secure_filename(original_name) or "solution.csv"
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    upload_dir = DASHBOARD_DIR / "soluciones" / ".optimize_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + safe_name
    save_path = upload_dir / upload_id
    solution_df[SUBMISSION_COLS].to_csv(save_path, index=False)

    return (
        {
            "ok": True,
            "upload_id": upload_id,
            "filename": original_name,
            "source": source,
            "n_productos": int(solution_df["codigo_producto"].nunique()),
            "metrics": metrics,
            "warnings": warnings or [],
            "policy_violation": violates_grosor_policy(solution_df),
        },
        200,
    )


@app.route("/api/optimize/analyze", methods=["POST"])
@login_required
def api_optimize_analyze():
    """Acepta upload (multipart) o una solución ya cargada (`solution_name`)."""
    solution_name = None
    if request.is_json:
        solution_name = str((request.get_json(silent=True) or {}).get("solution_name") or "").strip()
    if not solution_name:
        solution_name = str(request.form.get("solution_name") or "").strip()

    if solution_name:
        if solution_name not in _list_solution_files():
            return jsonify({"ok": False, "error": "Solución no encontrada."}), 404
        path = _safe_solution_path(solution_name)
        if path is None:
            return jsonify({"ok": False, "error": "Solución no encontrada."}), 404
        try:
            solution_df = pd.read_csv(path)
        except Exception:
            return jsonify({"ok": False, "error": "No se pudo leer el CSV."}), 400
        payload, status = _analyze_optimize_solution(
            solution_df, original_name=solution_name, source="library"
        )
        return jsonify(payload), status

    file = request.files.get("solution_file")
    if file is None or not file.filename:
        return jsonify(
            {"ok": False, "error": "Subí un CSV o elegí una solución cargada."}
        ), 400

    original_name = secure_filename(file.filename)
    if not original_name.lower().endswith(".csv"):
        return jsonify({"ok": False, "error": "El archivo debe ser un CSV."}), 400

    try:
        solution_df = pd.read_csv(file)
    except Exception:
        return jsonify({"ok": False, "error": "No se pudo leer el CSV."}), 400

    payload, status = _analyze_optimize_solution(
        solution_df, original_name=original_name, source="upload"
    )
    return jsonify(payload), status


@app.route("/api/optimize/start", methods=["POST"])
@login_required
def api_optimize_start():
    payload = request.get_json(silent=True) or {}
    upload_id = str(payload.get("upload_id") or "").strip()
    effort = str(payload.get("effort") or "mid").strip().lower()
    model = str(payload.get("model") or "mark16").strip().lower()
    grosores_raw = payload.get("grosores") or [3.0]
    pallet = payload.get("pallet") or {}
    allow_transform = payload.get("allow_transform", True)
    if isinstance(allow_transform, str):
        allow_transform = allow_transform.strip().lower() in {"1", "true", "yes", "on"}
    else:
        allow_transform = bool(allow_transform)

    if not upload_id or "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        return jsonify({"ok": False, "error": "upload_id inválido."}), 400

    warm_path = DASHBOARD_DIR / "soluciones" / ".optimize_uploads" / upload_id
    if not warm_path.exists():
        return jsonify({"ok": False, "error": "No se encontró el archivo cargado. Volvé a subirlo."}), 404

    if model not in MODELS:
        return jsonify({"ok": False, "error": "Modelo inválido (mark16|mark17)."}), 400

    try:
        grosores = [float(g) for g in grosores_raw]
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Grosores inválidos."}), 400

    unknown = [g for g in grosores if g not in PRECIO_BASE_GROSOR]
    if not grosores or unknown:
        return jsonify(
            {
                "ok": False,
                "error": f"Grosores no permitidos: {unknown or 'vacío'}. Usá {sorted(PRECIO_BASE_GROSOR)}.",
            }
        ), 400

    if effort not in EFFORT_SECONDS:
        return jsonify({"ok": False, "error": "Effort inválido (low|mid|high)."}), 400

    try:
        ancho = float(pallet.get("ancho_mm", PALLET_ANCHO_MM))
        largo = float(pallet.get("largo_mm", PALLET_LARGO_MM))
        alto = float(pallet.get("alto_mm", PALLET_ALTO_MAX_MM))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Dims de pallet inválidas."}), 400

    if ancho <= 0 or largo <= 0 or alto <= 0:
        return jsonify({"ok": False, "error": "Dims de pallet deben ser > 0."}), 400
    if ancho > 5000 or largo > 5000 or alto > 5000:
        return jsonify({"ok": False, "error": "Dims de pallet fuera de rango (máx 5000 mm)."}), 400

    # Orientación fija: caja L→ancho pallet, W→largo pallet, H→alto.
    lim = (ancho, largo, alto)

    out_dir = ROOT / "outputs" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = min(24, os.cpu_count() or 1)

    job_id = start_job(
        warm_start_path=warm_path,
        grosores=grosores,
        effort=effort,
        workers=workers,
        out_dir=out_dir,
        lim=lim,
        model=model,
        allow_transform=allow_transform,
    )
    return jsonify(
        {
            "ok": True,
            "job_id": job_id,
            "model": model,
            "allow_transform": allow_transform,
            "time_limit": EFFORT_SECONDS[effort],
            "workers": workers,
            "pallet": {"ancho_mm": ancho, "largo_mm": largo, "alto_mm": alto},
        }
    )


@app.route("/api/optimize/jobs/<job_id>")
@login_required
def api_optimize_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job no encontrado."}), 404
    # No exponer traceback completo al cliente en producción, pero útil en local.
    public = {k: v for k, v in job.items() if k != "error"}
    if job.get("error"):
        public["error"] = job["message"]
    return jsonify({"ok": True, "job": public})


@app.route("/api/optimize/jobs/<job_id>/analysis")
@login_required
def api_optimize_job_analysis(job_id: str):
    """Snapshot tipo Home para el CSV resultante (solo cuando el job terminó)."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job no encontrado."}), 404
    if job.get("status") != "done":
        return jsonify({"ok": False, "error": "El job todavía no terminó."}), 400

    out_path = job.get("out_path") or job.get("solution_path")
    if not out_path or not Path(out_path).exists():
        return jsonify({"ok": False, "error": "No está el CSV de salida."}), 404

    try:
        solution_df = pd.read_csv(out_path)
        _, operations = _load_reference_data()
        name = Path(out_path).name
        job_lim = job.get("lim")
        lim = tuple(float(x) for x in job_lim) if job_lim and len(job_lim) == 3 else None
        snapshot = _build_solution_snapshot(name, solution_df, operations, lim=lim)
        if snapshot is None:
            return jsonify({"ok": False, "error": "No se pudo evaluar la solución."}), 400

        # Preferir cota USD de esta corrida CP-SAT (solo Mark 16; Mark 17 usa surrogate).
        bound = job.get("bound")
        if (
            job.get("model") != "mark17"
            and isinstance(bound, (int, float))
            and bound == bound
            and bound > 0
        ):
            snapshot = dict(snapshot)
            snapshot["cpsat_quota"] = float(bound)
            snapshot["diff_cpsat"] = float(snapshot["costo_total"]) - float(bound)
            snapshot["diff_cpsat_pct"] = (
                (snapshot["diff_cpsat"] / float(bound)) * 100.0 if bound else 0.0
            )
            snapshot["cota_label"] = "Cota CP-SAT (esta corrida)"
        else:
            snapshot["cota_label"] = "Cota teórica"

        plant_cards = _plant_cards_from_snapshot(snapshot, operations)
        plant_cards.sort(key=lambda item: item["volumen"], reverse=True)
        return jsonify(
            {
                "ok": True,
                "snapshot": snapshot,
                "plant_cards": plant_cards,
                "solution_name": name,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/optimize/jobs")
@login_required
def api_optimize_jobs():
    return jsonify({"ok": True, "jobs": list_active_jobs()})


@app.route("/api/optimize/save/<job_id>", methods=["POST"])
@login_required
def api_optimize_save(job_id: str):
    """Copia el resultado del job a soluciones/ y lo selecciona."""
    job = get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job no encontrado."}), 404
    if job.get("status") != "done" or not job.get("out_path"):
        return jsonify({"ok": False, "error": "El job todavía no terminó."}), 400

    src = Path(job["out_path"])
    if not src.exists():
        return jsonify({"ok": False, "error": "No está el CSV de salida."}), 404

    score = job.get("score")
    score_tag = f"{score:+.5f}" if isinstance(score, (int, float)) else "result"
    model_tag = job.get("model") if job.get("model") in MODELS else "mark16"
    geom_tag = "vol" if job.get("allow_transform", True) else "rigido"
    dest_name = f"{model_tag}_{geom_tag}_{job_id}_{score_tag}.csv"
    dest = SOLUTIONS_DIR / dest_name
    pd.read_csv(src)[SUBMISSION_COLS].to_csv(dest, index=False)
    session["selected_solution"] = dest_name
    return jsonify({"ok": True, "filename": dest_name})


@app.route("/ordenes-compra")
@login_required
def ordenes_compra():
    return render_template("ordenes_compra.html", **_build_ordenes_compra_context())


@app.route("/ordenes-compra/documento/proveedor", defaults={"plant_id": None})
@app.route("/ordenes-compra/documento/proveedor/<plant_id>")
@login_required
def ordenes_compra_proveedor(plant_id: str | None):
    selected_solution = session.get("selected_solution")
    if not selected_solution:
        session["upload_error"] = "Seleccioná una solución antes de generar las órdenes."
        return redirect(url_for("ordenes_compra"))

    return _render_ordenes_document("po/proveedor.html", selected_solution, plant_id)


@app.route("/ordenes-compra/documento/asignacion", defaults={"plant_id": None})
@app.route("/ordenes-compra/documento/asignacion/<plant_id>")
@login_required
def ordenes_compra_asignacion(plant_id: str | None):
    selected_solution = session.get("selected_solution")
    if not selected_solution:
        session["upload_error"] = "Seleccioná una solución antes de generar las órdenes."
        return redirect(url_for("ordenes_compra"))

    return _render_ordenes_document("po/asignacion.html", selected_solution, plant_id)


@app.route("/comparacion")
@login_required
def comparacion():
    return render_template("comparacion.html", **_build_comparacion_context())


@app.route("/comparacion/select", methods=["POST"])
@login_required
def select_compare_solution():
    slot = request.form.get("solution_slot", "").strip().lower()
    filename = request.form.get("solution_name", "").strip()
    if slot not in {"a", "b"}:
        return _redirect_back("comparacion")

    if filename in _list_solution_files():
        session_key = "compare_solution_a" if slot == "a" else "compare_solution_b"
        session[session_key] = filename

    return redirect(url_for("comparacion"))


@app.route("/comparacion/upload", methods=["POST"])
@login_required
def upload_compare_solution():
    slot = request.form.get("solution_slot", "").strip().lower()
    if slot not in {"a", "b"}:
        return _redirect_back("comparacion")

    file = request.files.get("solution_file")
    if file is None or not file.filename:
        session["upload_error"] = "Seleccioná un archivo CSV."
        return redirect(url_for("comparacion"))

    original_name = secure_filename(file.filename)
    if not original_name.lower().endswith(".csv"):
        session["upload_error"] = "El archivo debe ser un CSV."
        return redirect(url_for("comparacion"))

    try:
        solution_df = pd.read_csv(file)
    except Exception:
        session["upload_error"] = "No se pudo leer el CSV."
        return redirect(url_for("comparacion"))

    try:
        products_truth, operations = _load_reference_data()
        is_valid, errors, warnings, _ = validate_solution_free(
            solution_df, products_truth, verbose=False
        )
        if not is_valid:
            session["upload_error"] = errors[0]
            return redirect(url_for("comparacion"))

        if warnings:
            session["upload_warning"] = warnings[0]

        _evaluar_costo_por_planta(solution_df, operations)
    except Exception as exc:
        session["upload_error"] = str(exc)
        return redirect(url_for("comparacion"))

    target_path = SOLUTIONS_DIR / original_name
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 2
        while target_path.exists():
            target_path = SOLUTIONS_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    solution_df.to_csv(target_path, index=False)
    session_key = "compare_solution_a" if slot == "a" else "compare_solution_b"
    session[session_key] = target_path.name
    session["upload_success"] = f"Solucion cargada ({slot.upper()}): {target_path.name}"
    return redirect(url_for("comparacion"))


@app.route("/api/solutions/select", methods=["POST"])
@login_required
def api_select_solution():
    payload = request.get_json(silent=True) or {}
    filename = (payload.get("solution_name") or request.form.get("solution_name", "")).strip()
    if filename not in _list_solution_files():
        return jsonify({"ok": False, "error": "Solución no encontrada."}), 404

    session["selected_solution"] = filename
    meta = _policy_toast_meta(filename)
    return jsonify(
        {
            "ok": True,
            "selected_solution": filename,
            **meta,
        }
    )


@app.route("/policy-toast/dismiss", methods=["POST"])
@login_required
def dismiss_policy_toast():
    selected_solution = session.get("selected_solution")
    if selected_solution:
        session["policy_toast_dismissed"] = selected_solution
    return _redirect_back()


@app.route("/solutions/select", methods=["POST"])
@login_required
def select_solution():
    filename = request.form.get("solution_name", "").strip()
    if filename in _list_solution_files():
        session["selected_solution"] = filename
    return _redirect_back()


@app.route("/solutions/upload", methods=["POST"])
@login_required
def upload_solution():
    file = request.files.get("solution_file")
    if file is None or not file.filename:
        session["upload_error"] = "Seleccioná un archivo CSV."
        return _redirect_back()

    original_name = secure_filename(file.filename)
    if not original_name.lower().endswith(".csv"):
        session["upload_error"] = "El archivo debe ser un CSV."
        return _redirect_back()

    try:
        solution_df = pd.read_csv(file)
    except Exception:
        session["upload_error"] = "No se pudo leer el CSV."
        return _redirect_back()

    try:
        products_truth, operations = _load_reference_data()
        is_valid, errors, warnings, _ = validate_solution_free(
            solution_df, products_truth, verbose=False
        )
        if not is_valid:
            session["upload_error"] = errors[0]
            return _redirect_back()

        if warnings:
            session["upload_warning"] = warnings[0]

        _evaluar_costo_por_planta(solution_df, operations)
    except Exception as exc:
        session["upload_error"] = str(exc)
        return _redirect_back()

    target_path = SOLUTIONS_DIR / original_name
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 2
        while target_path.exists():
            target_path = SOLUTIONS_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    solution_df.to_csv(target_path, index=False)
    session["selected_solution"] = target_path.name
    session["upload_success"] = f"Solucion cargada: {target_path.name}"
    return _redirect_back()


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
