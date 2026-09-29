#!/usr/bin/env python3
"""Mark 16 · warm configurable · log de score + certificado vivo.

Registra cada mejora de incumbent con score Kaggle (tier/descuento por planta)
para detectar cuándo se alcanza +10.11098 y cuánto tardó.

Actualiza (sobrescribe) certificate.md/json cada vez que se acota el problema
o mejora el incumbent — un solo certificado por corrida.

Ejemplos:
    PYTHONPATH=src python scripts/run_mark16_12h_minbox.py
    PYTHONPATH=src python scripts/run_mark16_12h_minbox.py \\
        --warm-start outputs/tables/04_mark16_best_+10.11098.csv \\
        --pool full --time-limit 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate import (  # noqa: E402
    SUBMISSION_COLS,
    calcular_cajas_por_pallet,
    costo_flete_eval,
    factor_precio_por_volumen,
    preparar_eval_planta,
)
from mark16 import costo_actual_oficial, run_optimize, score_publico  # noqa: E402
from settings import OPERACIONES_PLANTA_PATH, PLANTAS, PRECIO_BASE_GROSOR  # noqa: E402

try:
    import ortools

    ORTOOLS_VERSION = ortools.__version__
except Exception:  # noqa: BLE001
    ORTOOLS_VERSION = "unknown"

# Target histórico (04_mark16_best_+10.11098.csv) bajo evaluador plant-tier.
TARGET_SCORE = 10.11098
DEFAULT_TIME_LIMIT_S = 12 * 3600  # 12 h; 0 = sin límite
DEFAULT_WARM_START = "outputs/tables/04_mark16_best_+10.11098.csv"
GROSORES = [3.0]


def eval_kaggle_plant(sol: pd.DataFrame, ops: pd.DataFrame, baseline: float) -> dict:
    """Evaluador alineado a Kaggle: packaging con tier por planta + flete."""
    df = preparar_eval_planta(sol[SUBMISSION_COLS], ops)
    df = calcular_cajas_por_pallet(df)
    precio_base = (
        df.groupby("tipo")["caja_grosor_mm"].first().round(1).map(PRECIO_BASE_GROSOR)
    )
    pack = 0.0
    for p in PLANTAS:
        vol = df.groupby("tipo")[f"volumen_producto_planta_{p}"].sum()
        fac = factor_precio_por_volumen(vol)
        pack += float((vol.values * precio_base.loc[vol.index].values * fac).sum())
    flete = float(costo_flete_eval(df))
    total = pack + flete
    return {
        "cost_kaggle": total,
        "score_kaggle": score_publico(total, baseline),
        "n_tipos": int(df["tipo"].nunique()),
        "pack": pack,
        "flete": flete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark 16 · score history + certificado vivo (se actualiza al acotar)"
    )
    parser.add_argument(
        "--warm-start",
        default=DEFAULT_WARM_START,
        help="CSV de partida (default: mejor Mark 16 conocida).",
    )
    parser.add_argument("--pool", choices=("fast", "full"), default="full")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT_S,
        help="Segundos de CP-SAT. 0 = sin límite.",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--tag",
        default=None,
        help="Prefijo de carpeta de salida (default: run_{12h|nolimit}_{pool}).",
    )
    args = parser.parse_args()

    warm_start = Path(args.warm_start)
    if not warm_start.is_absolute():
        warm_start = REPO_ROOT / warm_start
    # OR-Tools: default max_time es +inf; 0 lo mapeamos a sin límite.
    raw_limit = float(args.time_limit)
    unlimited = (not math.isfinite(raw_limit)) or raw_limit <= 0
    time_limit_s = float("inf") if unlimited else raw_limit
    pool_mode = args.pool
    top_k = int(args.top_k)
    if args.tag:
        tag = args.tag
    elif unlimited:
        tag = f"run_nolimit_{pool_mode}"
    else:
        tag = f"run_12h_{pool_mode}"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "mark16" / f"{tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_out = out_dir / "mark16_incumbent.csv"
    score_log = out_dir / "score_history.csv"
    run_log = out_dir / "run.log"
    cert_json = out_dir / "certificate.json"
    cert_md = out_dir / "certificate.md"
    meta_path = out_dir / "run_meta.json"

    ops = pd.read_csv(OPERACIONES_PLANTA_PATH)
    baseline = float(costo_actual_oficial(ops))

    meta = {
        "stamp": stamp,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "time_limit_s": None if unlimited else time_limit_s,
        "unlimited": unlimited,
        "grosores": GROSORES,
        "pool_mode": pool_mode,
        "top_k": top_k,
        "warm_start": str(warm_start.relative_to(REPO_ROOT)),
        "baseline": baseline,
        "target_score": TARGET_SCORE,
        "ortools": ORTOOLS_VERSION,
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    score_fields = [
        "event_idx",
        "timestamp",
        "timestamp_utc",
        "elapsed_s",
        "elapsed_hms",
        "event",
        "cost_cpsat",
        "bound",
        "score_cpsat",
        "cost_kaggle",
        "score_kaggle",
        "n_tipos",
        "gap_pct",
        "hit_target_10_11098",
        "first_hit_target",
    ]

    def log_line(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def hms(seconds: float) -> str:
        s = max(0, int(seconds))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    with score_log.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=score_fields).writeheader()

    state = {
        "event_idx": 0,
        "best_score_kaggle": -1e18,
        "first_hit": None,  # dict when target first reached
        "t0": time.time(),
        "last_bound": None,
        "cost_cpsat": None,
        "bound": None,
        "kaggle": None,
        "finished": False,
    }

    def append_score_row(row: dict) -> None:
        with score_log.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=score_fields).writerow(row)

    def gap_of(cost: float | None, bound: float | None) -> float | None:
        if cost is None or bound is None or abs(float(cost)) <= 1e-9:
            return None
        return abs(float(cost) - float(bound)) / abs(float(cost)) * 100.0

    def solver_status(ok: bool | None = None) -> str:
        gap_pct = gap_of(state["cost_cpsat"], state["bound"])
        if ok is False:
            return "INFEASIBLE_OR_UNKNOWN"
        if state["cost_cpsat"] is None:
            return "RUNNING"
        if gap_pct is not None and gap_pct <= 1e-4:
            return "OPTIMAL"
        if state["finished"]:
            return "FEASIBLE"
        return "FEASIBLE"

    def write_certificate(*, reason: str) -> None:
        """Sobrescribe el mismo certificate.md/json (no crea archivos nuevos)."""
        wall = time.time() - state["t0"]
        cost = state["cost_cpsat"]
        bound = state["bound"]
        gap_pct = gap_of(cost, bound)
        kg = state["kaggle"]
        status = solver_status()
        updated_at = datetime.now().isoformat(timespec="seconds")

        certificate = {
            "titulo": "Certificado de optimalidad — solución reportada",
            "estado_solver": status,
            "valor_objetivo": None if cost is None else float(cost),
            "mejor_cota": None if bound is None else float(bound),
            "gap_relativo_pct": gap_pct,
            "tiempo_ejecucion_s": round(wall, 2),
            "tiempo_ejecucion_hms": hms(wall),
            "solver": f"Google OR-Tools {ORTOOLS_VERSION}",
            "grosores_mm": GROSORES,
            "warm_start": str(warm_start.relative_to(REPO_ROOT)),
            "pool_mode": pool_mode,
            "top_k": top_k,
            "baseline": baseline,
            "score_kaggle_final": None if kg is None else kg["score_kaggle"],
            "cost_kaggle_final": None if kg is None else kg["cost_kaggle"],
            "n_tipos_final": None if kg is None else kg["n_tipos"],
            "target_score": TARGET_SCORE,
            "milestone_target_10_11098": state["first_hit"],
            "score_history_csv": str(score_log.relative_to(REPO_ROOT)),
            "incumbent_csv": str(best_out.relative_to(REPO_ROOT)),
            "updated_at": updated_at,
            "update_reason": reason,
            "finished": bool(state["finished"]),
        }
        if state["finished"]:
            certificate["finished_at"] = updated_at
        cert_json.write_text(json.dumps(certificate, indent=2), encoding="utf-8")

        hit = state["first_hit"]
        hit_block = (
            (
                f"- **Primera vez ≥ +{TARGET_SCORE:.5f}:** {hit['elapsed_hms']} "
                f"({hit['timestamp']}) · score={hit['score_kaggle']:+.5f} · "
                f"tipos={hit['n_tipos']}"
            )
            if hit
            else f"- **Primera vez ≥ +{TARGET_SCORE:.5f}:** no alcanzado en esta corrida"
        )
        gap_txt = "n/d" if gap_pct is None else f"{gap_pct:.2f}%".replace(".", ",")
        obj_txt = "n/d" if cost is None else f"{float(cost):,.2f}"
        bnd_txt = "n/d" if bound is None else f"{float(bound):,.2f}"
        sc_txt = (
            "n/d" if kg is None else f"{kg['score_kaggle']:+.5f}".replace(".", ",")
        )

        cert_md.write_text(
            "\n".join(
                [
                    "# Certificado de optimalidad — solución reportada",
                    "",
                    f"**Estado del solver:** {status} · "
                    f"**Valor objetivo:** {obj_txt} · "
                    f"**Mejor cota:** {bnd_txt} · "
                    f"**Gap relativo:** {gap_txt} · "
                    f"**Tiempo de ejecución:** {hms(wall)} ({wall:.2f} s) · "
                    f"**Solver:** Google OR-Tools {ORTOOLS_VERSION}.",
                    "",
                    "## Score Kaggle (tier por planta)",
                    f"- **Score final:** {sc_txt}",
                    (
                        f"- **Costo final:** ${kg['cost_kaggle']:,.2f}"
                        if kg
                        else "- **Costo final:** n/d"
                    ),
                    (f"- **Tipos:** {kg['n_tipos']}" if kg else "- **Tipos:** n/d"),
                    hit_block,
                    "",
                    "## Artefactos",
                    f"- Log: `{run_log.relative_to(REPO_ROOT)}`",
                    f"- Score history: `{score_log.relative_to(REPO_ROOT)}`",
                    f"- Incumbent: `{best_out.relative_to(REPO_ROOT)}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def record_incumbent(event_name: str, cost_cpsat: float | None, bound: float | None, elapsed_s: float) -> None:
        if not best_out.exists():
            log_line(f"SKIP incumbent ({event_name}): aún no hay CSV en {best_out.name}")
            return
        sol = pd.read_csv(best_out)
        sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
        kg = eval_kaggle_plant(sol, ops, baseline)

        if cost_cpsat is not None:
            state["cost_cpsat"] = float(cost_cpsat)
        if bound is not None:
            state["bound"] = float(bound)
            state["last_bound"] = float(bound)
        state["kaggle"] = kg

        score_cpsat = (
            score_publico(float(cost_cpsat), baseline) if cost_cpsat is not None else None
        )
        gap_pct = gap_of(cost_cpsat, bound)

        # Comparar a 5 decimales (el target histórico se reporta así).
        hit = round(float(kg["score_kaggle"]), 5) + 1e-12 >= TARGET_SCORE
        first_hit = False
        if hit and state["first_hit"] is None:
            state["first_hit"] = {
                "elapsed_s": float(elapsed_s),
                "elapsed_hms": hms(elapsed_s),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "score_kaggle": kg["score_kaggle"],
                "cost_kaggle": kg["cost_kaggle"],
                "cost_cpsat": cost_cpsat,
                "n_tipos": kg["n_tipos"],
                "event_idx": state["event_idx"] + 1,
            }
            first_hit = True
            log_line(
                f"🎯 TARGET +{TARGET_SCORE:.5f} alcanzado en {hms(elapsed_s)} "
                f"(score_kaggle={kg['score_kaggle']:+.5f}, tipos={kg['n_tipos']})"
            )

        improved = kg["score_kaggle"] > state["best_score_kaggle"] + 1e-9
        if improved:
            state["best_score_kaggle"] = kg["score_kaggle"]

        state["event_idx"] += 1
        row = {
            "event_idx": state["event_idx"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(float(elapsed_s), 3),
            "elapsed_hms": hms(elapsed_s),
            "event": event_name,
            "cost_cpsat": None if cost_cpsat is None else round(float(cost_cpsat), 4),
            "bound": None if bound is None else round(float(bound), 4),
            "score_cpsat": None if score_cpsat is None else round(float(score_cpsat), 5),
            "cost_kaggle": round(kg["cost_kaggle"], 4),
            "score_kaggle": round(kg["score_kaggle"], 5),
            "n_tipos": kg["n_tipos"],
            "gap_pct": None if gap_pct is None else round(gap_pct, 5),
            "hit_target_10_11098": hit,
            "first_hit_target": first_hit,
        }
        append_score_row(row)
        log_line(
            f"SCORE#{state['event_idx']} {event_name} | t={hms(elapsed_s)} | "
            f"kaggle={kg['score_kaggle']:+.5f} (${kg['cost_kaggle']:,.2f}) | "
            f"cpsat={'' if score_cpsat is None else f'{score_cpsat:+.5f}'} | "
            f"tipos={kg['n_tipos']}"
            + (" | FIRST_HIT_TARGET" if first_hit else "")
        )
        write_certificate(reason=f"incumbent:{event_name}")

    def on_progress(e: dict) -> None:
        phase = e.get("phase", "")
        msg = e.get("message", "")
        if e.get("incumbent"):
            record_incumbent(
                event_name=str(msg or "incumbent"),
                cost_cpsat=e.get("cost"),
                bound=e.get("bound"),
                elapsed_s=float(e.get("elapsed_s", time.time() - state["t0"])),
            )
            return
        if e.get("bound_update"):
            b = e.get("bound")
            if b is None:
                return
            prev = state["last_bound"]
            # Minimización: acotar = subir la cota dual.
            if prev is not None and float(b) <= float(prev) + 1.0:
                return
            state["last_bound"] = float(b)
            state["bound"] = float(b)
            elapsed = float(e.get("elapsed_s", time.time() - state["t0"]))
            gap_pct = gap_of(state["cost_cpsat"], state["bound"])
            gap_txt = "n/d" if gap_pct is None else f"{gap_pct:.3f}%"
            log_line(
                f"BOUND t={hms(elapsed)} | cota=${float(b):,.2f} "
                f"(score_techo={score_publico(float(b), baseline):+.5f}, gap={gap_txt})"
            )
            write_certificate(reason="bound_update")
            return
        log_line(f"[{phase}] {msg}")

    limit_label = "sin límite" if unlimited else f"{time_limit_s/3600:.0f}h"
    log_line(
        f"START Mark16 {limit_label} | warm={warm_start.relative_to(REPO_ROOT)} | "
        f"pool={pool_mode} top_k={top_k} | grosores={GROSORES} | "
        f"target_score=+{TARGET_SCORE:.5f}"
    )
    log_line(f"out_dir={out_dir.relative_to(REPO_ROOT)}")
    state["t0"] = time.time()
    write_certificate(reason="start")

    result = run_optimize(
        warm_start_path=warm_start,
        grosores=GROSORES,
        time_limit=time_limit_s,
        workers=int(args.workers),
        out_path=best_out,
        pool_mode=pool_mode,
        top_k=top_k,
        on_progress=on_progress,
    )

    wall = time.time() - state["t0"]
    ok = bool(result.get("ok"))
    if result.get("cost") is not None:
        state["cost_cpsat"] = float(result["cost"])
    if result.get("bound") is not None:
        state["bound"] = float(result["bound"])
        state["last_bound"] = float(result["bound"])
    state["finished"] = True

    if best_out.exists():
        sol = pd.read_csv(best_out)
        sol["codigo_producto"] = sol["codigo_producto"].astype(str).str.strip()
        state["kaggle"] = eval_kaggle_plant(sol, ops, baseline)
        final_csv = out_dir / f"mark16_g3.0_final_{stamp}.csv"
        sol[SUBMISSION_COLS].to_csv(final_csv, index=False)

    write_certificate(reason="finished" if ok else "failed")

    kg = state["kaggle"]
    gap_pct = gap_of(state["cost_cpsat"], state["bound"])
    gap_txt = "n/d" if gap_pct is None else f"{gap_pct:.2f}%".replace(".", ",")
    sc_txt = (
        "n/d" if kg is None else f"{kg['score_kaggle']:+.5f}".replace(".", ",")
    )
    status = solver_status(ok=ok)
    hit = state["first_hit"]

    log_line(
        f"DONE status={status} wall={hms(wall)} "
        f"score_kaggle={sc_txt} gap={gap_txt}"
    )
    if hit:
        log_line(
            f"MILESTONE +{TARGET_SCORE:.5f} @ {hit['elapsed_hms']} "
            f"({hit['timestamp']})"
        )
    else:
        log_line(f"MILESTONE +{TARGET_SCORE:.5f} NOT REACHED")
    log_line(f"certificate -> {cert_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
