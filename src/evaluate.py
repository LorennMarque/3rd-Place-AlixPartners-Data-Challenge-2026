"""Validación física y evaluación de costos de una solución de cajas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from settings import (
    ECT as ECT_POLICY,
    EPS,
    GRAVITY,
    GROSORES_KAGGLE,
    HEADSPACE_ABS_MAX,
    HEADSPACE_PCT,
    MAX_GROW,
    MAX_SHRINK,
    PALLET_ALTO_MAX_MM,
    PALLET_ANCHO_MM,
    PALLET_LARGO_MM,
    PLANTAS,
    PRECIO_BASE_GROSOR,
    SUBMISSION_COLS,
)

BOX_COLS = [
    "caja_grosor_mm",
    "caja_exterior_largo",
    "caja_exterior_ancho",
    "caja_exterior_alto",
]

# Alias: tabla completa = settings (instrucciones). Política Kaggle = subset.
ECT_FULL = dict(ECT_POLICY)
PRECIO_BASE_GROSOR_FULL = dict(PRECIO_BASE_GROSOR)
VALID_GROSORES_POLICY = set(GROSORES_KAGGLE)
GROSOR_POLICY_WARNING = (
    "Solución seleccionada no cumple políticas de grosor único entre 3, 4.5 mm y 5"
)
COSTO_PALLET = 150.0
VOLUMEN_PALLET = PALLET_LARGO_MM * PALLET_ANCHO_MM * PALLET_ALTO_MAX_MM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_numeric_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.lower()
        .str.replace("mm", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


def headspace_pct(grosor: float) -> float:
    if grosor <= 3.0:
        return 0.06
    if grosor <= 4.5:
        return 0.08
    return 0.10


def factor_precio_por_volumen(volumen: pd.Series | np.ndarray) -> np.ndarray:
    vol = pd.Series(volumen, dtype=float)
    return np.select(
        [vol >= 500_000, vol >= 100_000, vol >= 50_000, vol >= 20_000],
        [0.70, 0.80, 0.90, 1.00],
        default=1.10,
    )


factor_tier = factor_precio_por_volumen


def violates_grosor_policy(solution_df: pd.DataFrame) -> bool:
    g = clean_numeric_col(solution_df["caja_grosor_mm"]).round(1)
    unique = set(g.dropna().unique())
    return len(unique) != 1 or not unique.issubset(VALID_GROSORES_POLICY)


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------


def _base_frame(solution_df: pd.DataFrame, products_truth: pd.DataFrame):
    """Schema + merge con productos. Devuelve (df|sol, errors)."""
    errors: list[str] = []
    sol = solution_df.copy()

    if list(sol.columns) != list(SUBMISSION_COLS):
        errors.append(
            f"Invalid columns. Expected exactly: {list(SUBMISSION_COLS)}. "
            f"Got: {list(sol.columns)}"
        )
        return sol, errors

    sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
    for col in BOX_COLS:
        sol[col] = clean_numeric_col(sol[col])

    if sol[SUBMISSION_COLS].isna().any().any():
        errors.append("There are missing or non-numeric values.")
    if not np.isfinite(sol[BOX_COLS]).all().all():
        errors.append("There are non-finite numeric values.")
    if (sol[["caja_exterior_largo", "caja_exterior_ancho", "caja_exterior_alto"]] <= 0).any().any():
        errors.append("There are non-positive box dimensions.")

    expected = set(products_truth["codigo_producto"].astype(str).str.strip())
    actual = set(sol["codigo_producto"])
    missing, extra = expected - actual, actual - expected
    if missing:
        errors.append(f"Missing products: {len(missing)}. Example: {list(missing)[:5]}")
    if extra:
        errors.append(f"Extra unknown products: {len(extra)}. Example: {list(extra)[:5]}")
    if sol["codigo_producto"].duplicated().any():
        dupes = sol.loc[sol["codigo_producto"].duplicated(), "codigo_producto"].head().tolist()
        errors.append(f"Duplicated codigo_producto values. Example: {dupes}")

    if errors:
        return sol, errors

    truth = products_truth[
        ["codigo_producto", "largo", "ancho", "alto", "peso_neto_caja"]
    ].copy()
    truth["codigo_producto"] = truth["codigo_producto"].astype(str).str.strip()
    return sol.merge(truth, on="codigo_producto", how="left", validate="one_to_one"), errors


def _check_physics(df: pd.DataFrame, gros: np.ndarray, ect: np.ndarray, pct: np.ndarray) -> list[str]:
    """Regla 9 literal: ±10%, volumen, headspace, pallet, compresión."""
    errors: list[str] = []

    for axis, dim_col in (
        ("largo", "caja_exterior_largo"),
        ("ancho", "caja_exterior_ancho"),
        ("alto", "caja_exterior_alto"),
    ):
        interno = df[dim_col].to_numpy() - 2.0 * gros
        prod_dim = df[axis].to_numpy()
        if (interno <= 0).any():
            errors.append(f"Some internal dimensions are non-positive ({axis}).")
        df[f"internal_{axis}"] = interno
        df[f"fail_10pct_{axis}"] = (interno < prod_dim * MAX_SHRINK - EPS) | (
            interno > prod_dim * MAX_GROW + EPS
        )
        df[f"headspace_{axis}"] = interno - prod_dim
        df[f"headspace_lim_{axis}"] = np.minimum(interno * pct, HEADSPACE_ABS_MAX)
        df[f"fail_headspace_{axis}"] = (
            df[f"headspace_{axis}"] > df[f"headspace_lim_{axis}"] + EPS
        )

    resize = ["fail_10pct_largo", "fail_10pct_ancho", "fail_10pct_alto"]
    if df[resize].any().any():
        errors.append(f"10% resizing limit failed for {df[resize].any(axis=1).sum()} products.")

    df["vol_internal"] = df["internal_largo"] * df["internal_ancho"] * df["internal_alto"]
    df["vol_original"] = df["largo"] * df["ancho"] * df["alto"]
    df["fail_volume"] = df["vol_internal"] < df["vol_original"] - EPS
    if df["fail_volume"].any():
        errors.append(f"Internal volume failed for {int(df['fail_volume'].sum())} products.")

    hs = ["fail_headspace_largo", "fail_headspace_ancho", "fail_headspace_alto"]
    if df[hs].any().any():
        errors.append(f"Headspace failed for {df[hs].any(axis=1).sum()} products.")

    df["cajas_largo_pallet"] = np.floor(PALLET_ANCHO_MM / df["caja_exterior_largo"]).astype(int)
    df["cajas_ancho_pallet"] = np.floor(PALLET_LARGO_MM / df["caja_exterior_ancho"]).astype(int)
    df["capas_pallet"] = np.floor(PALLET_ALTO_MAX_MM / df["caja_exterior_alto"]).astype(int)
    df["cajas_por_pallet"] = (
        df["cajas_largo_pallet"] * df["cajas_ancho_pallet"] * df["capas_pallet"]
    )
    df["fail_pallet_fit"] = df["cajas_por_pallet"] <= 0
    if df["fail_pallet_fit"].any():
        errors.append(f"Pallet fit failed for {int(df['fail_pallet_fit'].sum())} products.")

    df["perimetro_m"] = 2 * (df["caja_exterior_largo"] + df["caja_exterior_ancho"]) / 1000
    df["carga_max_kg"] = ect * df["perimetro_m"] / GRAVITY
    df["peso_encima_kg"] = df["peso_neto_caja"] * np.maximum(df["capas_pallet"] - 1, 0)
    df["fail_compression"] = df["peso_encima_kg"] > df["carga_max_kg"] + EPS
    if df["fail_compression"].any():
        errors.append(f"Compression failed for {int(df['fail_compression'].sum())} products.")

    return errors


def validate_solution(solution_df, products_truth, verbose=True):
    """Validador oficial Kaggle: un solo grosor en {3.0, 4.5, 5.0}."""
    df, errors = _base_frame(solution_df, products_truth)
    if errors:
        if verbose:
            print("INVALID:", *errors, sep="\n- ")
        return False, errors, df

    unique = sorted(df["caja_grosor_mm"].dropna().unique())
    if len(unique) != 1:
        errors.append(f"Thickness is not unique. Found: {unique}")
        return False, errors, df
    grosor = float(unique[0])
    if grosor not in VALID_GROSORES_POLICY:
        errors.append(f"Invalid thickness {grosor}. Valid: {sorted(VALID_GROSORES_POLICY)}")
        return False, errors, df

    n = len(df)
    gros = np.full(n, grosor)
    ect = np.full(n, float(ECT_POLICY[grosor]))
    pct = np.full(n, float(HEADSPACE_PCT[grosor]))
    errors.extend(_check_physics(df, gros, ect, pct))

    ok = not errors
    if verbose:
        print("VALID" if ok else "INVALID")
        for e in errors:
            print("-", e)
    return ok, errors, df


def validate_solution_free(solution_df, products_truth, verbose=True):
    """Validador flexible: cualquier grosor de ECT_FULL; política → warning."""
    warnings: list[str] = []
    df, errors = _base_frame(solution_df, products_truth)
    if errors:
        return False, errors, warnings, df

    g = clean_numeric_col(df["caja_grosor_mm"]).round(1)
    invalid = sorted(set(g.unique()) - set(ECT_FULL))
    if invalid:
        errors.append(f"Grosores fuera de la tabla ECT: {invalid}")
        return False, errors, warnings, df

    if violates_grosor_policy(df):
        warnings.append(GROSOR_POLICY_WARNING)

    gros = g.to_numpy(float)
    pct = g.map(headspace_pct).to_numpy(float)
    ect = g.map(ECT_FULL).to_numpy(float)
    errors.extend(_check_physics(df, gros, ect, pct))

    ok = not errors
    if verbose:
        print("VALID" if ok else "INVALID", f"({len(warnings)} warnings)" if warnings else "")
        for e in errors:
            print("-", e)
    return ok, errors, warnings, df


# ---------------------------------------------------------------------------
# Costos
# ---------------------------------------------------------------------------


def agregar_id_tipo_caja(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    df = df.copy()
    for c in BOX_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(decimals)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(df[BOX_COLS]))
    df["caja_tipo_id_solucion"] = [f"BOX_{i + 1:04d}" for i in codes]
    return df


def calcular_cajas_por_pallet(
    df: pd.DataFrame,
    *,
    lim: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """`lim` = (ancho, largo, alto) mm; default = pallet oficial."""
    df = df.copy()
    ancho, largo, alto = lim or (PALLET_ANCHO_MM, PALLET_LARGO_MM, PALLET_ALTO_MAX_MM)
    vol_pallet = float(ancho * largo * alto)
    df["cajas_piso_largo"] = np.floor(ancho / df["caja_exterior_largo"]).astype(int)
    df["cajas_piso_ancho"] = np.floor(largo / df["caja_exterior_ancho"]).astype(int)
    df["capas_alto"] = np.floor(alto / df["caja_exterior_alto"]).astype(int)
    df["cajas_por_pallet"] = (
        df["cajas_piso_largo"] * df["cajas_piso_ancho"] * df["capas_alto"]
    ).astype(int)
    if (df["cajas_por_pallet"] <= 0).any():
        bad = df.loc[df["cajas_por_pallet"] <= 0, ["codigo_producto", *BOX_COLS]]
        raise ValueError(f"Cajas que no entran en el pallet:\n{bad.head(10)}")
    df["volumen_externo_caja_mm3"] = (
        df["caja_exterior_largo"] * df["caja_exterior_ancho"] * df["caja_exterior_alto"]
    )
    df["utilizacion_pallet_teorica"] = (
        df["cajas_por_pallet"] * df["volumen_externo_caja_mm3"] / vol_pallet
    )
    return df


def preparar_eval_planta(
    sol: pd.DataFrame,
    ops: pd.DataFrame,
    *,
    lim: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    df = sol.copy()
    for c in BOX_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
    vol_cols = ["volumen_producto_total", *[f"volumen_producto_planta_{p}" for p in PLANTAS]]
    df = df.merge(ops[["codigo_producto", *vol_cols]], on="codigo_producto", how="left")
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(df[BOX_COLS]))
    df["tipo"] = codes
    ancho, largo, alto = lim or (PALLET_ANCHO_MM, PALLET_LARGO_MM, PALLET_ALTO_MAX_MM)
    df["cajas_por_pallet"] = (
        np.floor(ancho / df["caja_exterior_largo"])
        * np.floor(largo / df["caja_exterior_ancho"])
        * np.floor(alto / df["caja_exterior_alto"])
    ).astype(int)
    return df


def costo_flete_eval(df: pd.DataFrame) -> float:
    total = 0.0
    for p in PLANTAS:
        pallets = np.ceil(df[f"volumen_producto_planta_{p}"] / df["cajas_por_pallet"])
        total += float((pallets * COSTO_PALLET).sum())
    return total


def inferir_costos_flete_por_planta(
    operaciones_planta: pd.DataFrame,
    plantas: list[str] = PLANTAS,
) -> tuple[dict, pd.DataFrame]:
    costos = {}
    filas = []
    for planta in plantas:
        cant_col = f"cantidad_pallets_planta_{planta}"
        costo_col = f"costo_pallets_planta_{planta}"
        tmp = operaciones_planta[[cant_col, costo_col]]
        tmp = tmp[tmp[cant_col] > 0]
        if tmp.empty:
            raise ValueError(f"No hay pallets positivos para planta: {planta}")
        total_pallets = float(tmp[cant_col].sum())
        total_costo = float(tmp[costo_col].sum())
        unit = total_costo / total_pallets
        costos[planta] = unit
        unitarios = tmp[costo_col] / tmp[cant_col]
        filas.append(
            {
                "planta": planta,
                "pallets_actuales": total_pallets,
                "costo_flete_actual": total_costo,
                "costo_unitario_usado": unit,
                "min_unitario_observado": float(unitarios.min()),
                "max_unitario_observado": float(unitarios.max()),
            }
        )
    return costos, pd.DataFrame(filas)


def evaluar_costos_solucion(
    solution_df: pd.DataFrame,
    operaciones_planta: pd.DataFrame,
    costos_flete_por_planta: dict | None = None,
    *,
    lim: tuple[float, float, float] | None = None,
) -> dict:
    """Evaluador completo (packaging global + flete). Útil para notebooks.

    `lim` opcional = (ancho, largo, alto) mm para what-if de pallet.
    """
    submission = solution_df[SUBMISSION_COLS].copy()
    for c in BOX_COLS:
        submission[c] = pd.to_numeric(submission[c], errors="coerce")
    submission["caja_grosor_mm"] = submission["caja_grosor_mm"].round(3)

    vol_cols = ["volumen_producto_total", *[f"volumen_producto_planta_{p}" for p in PLANTAS]]
    df = submission.merge(
        operaciones_planta[["codigo_producto", *vol_cols]],
        on="codigo_producto",
        how="left",
        validate="one_to_one",
    )
    if df["volumen_producto_total"].isna().any():
        raise ValueError("Productos sin match en operaciones_planta")

    df = calcular_cajas_por_pallet(agregar_id_tipo_caja(df), lim=lim)
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

    df["pallets_total_nuevo"] = 0
    df["costo_flete_total_producto"] = 0.0
    plant_rows = []
    for p in PLANTAS:
        vol_col = f"volumen_producto_planta_{p}"
        pallets = np.ceil(df[vol_col] / df["cajas_por_pallet"]).astype(int)
        costo = pallets * costos_flete[p]
        df[f"pallets_nuevo_planta_{p}"] = pallets
        df[f"costo_flete_nuevo_planta_{p}"] = costo
        df["pallets_total_nuevo"] += pallets
        df["costo_flete_total_producto"] += costo
        plant_rows.append(
            {
                "planta": p,
                "volumen_total": float(df[vol_col].sum()),
                "pallets_nuevo": int(pallets.sum()),
                "costo_unitario_flete": costos_flete[p],
                "costo_flete_nuevo": float(costo.sum()),
            }
        )

    pack = float(df["costo_packaging_producto"].sum())
    flete = float(df["costo_flete_total_producto"].sum())
    resumen = pd.DataFrame(
        [
            {
                "n_productos": df["codigo_producto"].nunique(),
                "n_tipos_caja": df["caja_tipo_id_solucion"].nunique(),
                "grosor_unico": df["caja_grosor_mm"].unique()[0],
                "costo_packaging_nuevo": pack,
                "costo_flete_nuevo": flete,
                "costo_total_nuevo": pack + flete,
                "pallets_total_nuevo": int(df["pallets_total_nuevo"].sum()),
            }
        ]
    )
    return {
        "resumen": resumen,
        "detalle_productos": df,
        "detalle_tipos_caja": df.groupby("caja_tipo_id_solucion", as_index=False).first(),
        "detalle_plantas": pd.DataFrame(plant_rows),
        "diagnostico_costos_flete": diag,
        "submission_limpia": submission,
    }
