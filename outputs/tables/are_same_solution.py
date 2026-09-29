import hashlib
import sys
from pathlib import Path

import pandas as pd

TABLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = TABLES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.solution_value_evaluator import evaluar_costos_solucion

# Paths used in notebooks/90_notebook_colapse.ipynb (Improvement Oportunity analisis)
MARK2_PATH = TABLES_DIR / "07_submission_optimzer_mark_2.csv"
MARK5_PATH = TABLES_DIR / "09_submission_optimizer_mark4_gridbest_20260708_154509.csv"


def solution_hash(solution_df: pd.DataFrame) -> str:
    cols = [
        "codigo_producto",
        "caja_grosor_mm",
        "caja_exterior_largo",
        "caja_exterior_ancho",
        "caja_exterior_alto",
    ]

    tmp = solution_df[cols].copy()
    tmp = tmp.sort_values("codigo_producto").reset_index(drop=True)

    dim_cols = [
        "caja_grosor_mm",
        "caja_exterior_largo",
        "caja_exterior_ancho",
        "caja_exterior_alto",
    ]
    tmp[dim_cols] = tmp[dim_cols].round(6)

    return hashlib.md5(tmp.to_csv(index=False).encode()).hexdigest()


def count_differing_rows(a: pd.DataFrame, b: pd.DataFrame) -> int:
    cols = [
        "codigo_producto",
        "caja_grosor_mm",
        "caja_exterior_largo",
        "caja_exterior_ancho",
        "caja_exterior_alto",
    ]
    merged = a[cols].merge(
        b[cols],
        on="codigo_producto",
        how="outer",
        suffixes=("_mark2", "_mark5"),
        indicator=True,
    )
    only_one = (merged["_merge"] != "both").sum()
    if only_one:
        return int(only_one)

    dim_cols = [c for c in cols if c != "codigo_producto"]
    for col in dim_cols:
        merged[f"{col}_mark2"] = merged[f"{col}_mark2"].round(6)
        merged[f"{col}_mark5"] = merged[f"{col}_mark5"].round(6)

    return int(
        (merged[[f"{c}_mark2" for c in dim_cols]].values
         != merged[[f"{c}_mark5" for c in dim_cols]].values).any(axis=1).sum()
    )


def print_evaluation(label: str, evaluation: dict) -> pd.Series:
    resumen = evaluation["resumen"].iloc[0]

    print(f"--- {label} ---")
    print(f"Productos:        {int(resumen['n_productos'])}")
    print(f"Tipos de caja:    {int(resumen['n_tipos_caja'])}")
    print(f"Grosor único:     {resumen['grosor_unico']} mm")
    print(f"Pallets totales:  {resumen['pallets_total_nuevo']:,.2f}")
    print(f"Costo packaging:  {resumen['costo_packaging_nuevo']:,.2f}")
    print(f"Costo flete:      {resumen['costo_flete_nuevo']:,.2f}")
    print(f"Costo total:      {resumen['costo_total_nuevo']:,.2f}")

    if "costo_total_actual" in resumen.index:
        print(f"Costo actual:     {resumen['costo_total_actual']:,.2f}")
        print(
            f"Ahorro vs actual: {resumen['ahorro_total']:,.2f} "
            f"({resumen['ahorro_pct']:.2%})"
        )
    print()

    return resumen


def main() -> None:
    mark2 = pd.read_csv(MARK2_PATH)
    mark5 = pd.read_csv(MARK5_PATH)

    hash_mark2 = solution_hash(mark2)
    hash_mark5 = solution_hash(mark5)
    same = hash_mark2 == hash_mark5
    n_diff = count_differing_rows(mark2, mark5)

    print("Notebook: 90_notebook_colapse.ipynb (Improvement Oportunity analisis)")
    print(f"Mark 2: {MARK2_PATH.name}")
    print(f"Mark 5: {MARK5_PATH.name}")
    print()
    print(f"Mark 2 hash: {hash_mark2}")
    print(f"Mark 5 hash: {hash_mark5}")
    print(f"Same solution: {same}")
    print(f"Differing products: {n_diff} / {len(mark2)}")
    print()

    operaciones_planta = pd.read_csv(REPO_ROOT / "data/raw/operaciones_planta.csv")

    eval_mark2 = evaluar_costos_solucion(mark2, operaciones_planta)
    eval_mark5 = evaluar_costos_solucion(mark5, operaciones_planta)

    resumen_mark2 = print_evaluation("Mark 2", eval_mark2)
    resumen_mark5 = print_evaluation("Mark 5", eval_mark5)

    delta_total = resumen_mark5["costo_total_nuevo"] - resumen_mark2["costo_total_nuevo"]
    delta_packaging = (
        resumen_mark5["costo_packaging_nuevo"] - resumen_mark2["costo_packaging_nuevo"]
    )
    delta_flete = resumen_mark5["costo_flete_nuevo"] - resumen_mark2["costo_flete_nuevo"]

    print("--- Mark 5 vs Mark 2 ---")
    print(f"Delta packaging:  {delta_packaging:+,.2f}")
    print(f"Delta flete:      {delta_flete:+,.2f}")
    print(f"Delta total:      {delta_total:+,.2f}")
    if delta_total < 0:
        print("Mark 5 is cheaper.")
    elif delta_total > 0:
        print("Mark 2 is cheaper.")
    else:
        print("Same total cost.")


if __name__ == "__main__":
    main()
