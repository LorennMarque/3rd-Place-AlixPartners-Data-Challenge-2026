"""Jobs de optimización (Mark 16 / Mark 17) en background para el dashboard."""

from __future__ import annotations

import threading
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

import mark16
from mark16 import run_optimize as run_optimize_mark16
from mark17 import run_optimize as run_optimize_mark17
from mark19 import intervalos_fisicos

ModelId = Literal["mark16", "mark17"]

EFFORT_SECONDS = {
    "low": 30.0,
    "mid": 120.0,
    "high": 300.0,
}

# Pool liviano: más top_k = más columnas CP-SAT (un poco más lento de resolver).
EFFORT_TOP_K = {
    "low": 4,
    "mid": 8,
    "high": 12,
}

MODELS = {
    "mark16": {
        "id": "mark16",
        "label": "Optimizado con demanda conocida",
        "short": "Demanda conocida",
    },
    "mark17": {
        "id": "mark17",
        "label": "Optimizando utilización (protocolo)",
        "short": "Protocolo",
    },
}

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}

_FITS_LITERAL = mark16.fits_matrix


@contextmanager
def _geometry_mode(allow_transform: bool) -> Iterator[None]:
    """allow_transform=True: lectura literal (volumen / ±10%).

    allow_transform=False: producto rígido — interna_k ≥ producto_k en cada eje.
    """
    if allow_transform:
        yield
        return

    prev_i, prev_f = mark16.intervalos, mark16.fits_matrix
    mark16.intervalos = intervalos_fisicos  # type: ignore[assignment]

    def _fits(boxes, lo, hi, vol_min, peso, grosor):
        fits = _FITS_LITERAL(boxes, lo, hi, vol_min, peso, grosor)
        axis_ok = (boxes[:, None, :] >= lo[None, :, :] - 1e-9).all(axis=2)
        return fits & axis_ok

    mark16.fits_matrix = _fits  # type: ignore[assignment]
    try:
        yield
    finally:
        mark16.intervalos = prev_i
        mark16.fits_matrix = prev_f


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_active_jobs() -> list[dict[str, Any]]:
    with _LOCK:
        return [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "phase": j["phase"],
                "model": j.get("model"),
                "created_at": j["created_at"],
            }
            for j in _JOBS.values()
            if j["status"] in {"queued", "running"}
        ]


def _update(job_id: str, **fields: Any) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"


def _append_history(job_id: str, point: dict[str, Any]) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        history = job.setdefault("history", [])
        # Evitar duplicar el mismo costo consecutivo.
        if history and abs(history[-1].get("cost", 0) - point.get("cost", -1)) < 1e-6:
            history[-1] = point
        else:
            history.append(point)
        job["best_cost"] = point.get("cost", job.get("best_cost"))
        job["ahorro"] = point.get("ahorro", job.get("ahorro"))
        if "score" in point:
            job["score"] = point["score"]
        if "bound" in point:
            job["bound"] = point["bound"]
        job["elapsed_s"] = point.get("elapsed_s", job.get("elapsed_s"))
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"


def _append_bound_history(job_id: str, point: dict[str, Any]) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        history = job.setdefault("bound_history", [])
        bound = point.get("bound")
        if bound is None or not isinstance(bound, (int, float)):
            return
        if history and abs(history[-1].get("bound", 0) - bound) < 1e-3:
            history[-1] = point
        else:
            history.append(point)
        job["bound"] = bound
        if "ahorro_cota" in point:
            job["ahorro_cota"] = point["ahorro_cota"]
        if "score_cota" in point:
            job["score_cota"] = point["score_cota"]
        job["elapsed_s"] = point.get("elapsed_s", job.get("elapsed_s"))
        job["updated_at"] = datetime.utcnow().isoformat() + "Z"


def start_job(
    *,
    warm_start_path: Path,
    grosores: list[float],
    effort: str,
    workers: int,
    out_dir: Path,
    lim: tuple[float, float, float] | None = None,
    model: str = "mark16",
    allow_transform: bool = True,
) -> str:
    effort_key = effort if effort in EFFORT_SECONDS else "mid"
    time_limit = EFFORT_SECONDS[effort_key]
    top_k = EFFORT_TOP_K.get(effort_key, 8)
    model_key: ModelId = "mark17" if model == "mark17" else "mark16"
    job_id = uuid.uuid4().hex[:12]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{model_key}_job_{job_id}_{stamp}.csv"
    active_lim = tuple(float(x) for x in lim) if lim is not None else None
    transform = bool(allow_transform)

    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "message": "En cola…",
            "model": model_key,
            "model_label": MODELS[model_key]["label"],
            "allow_transform": transform,
            "grosores": grosores,
            "effort": effort_key,
            "time_limit": time_limit,
            "top_k": top_k,
            "pool_mode": "fast",
            "workers": workers,
            "lim": list(active_lim) if active_lim else None,
            "warm_start": str(warm_start_path),
            "out_path": str(out_path),
            "baseline": None,
            "best_cost": None,
            "ahorro": None,
            "bound": None,
            "ahorro_cota": None,
            "score": None,
            "score_cota": None,
            "surrogate": None,
            "surrogate_bound": None,
            "history": [],
            "bound_history": [],
            "error": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "elapsed_s": 0.0,
            "progress_pct": 0.0,
        }

    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            warm_start_path,
            grosores,
            time_limit,
            workers,
            out_path,
            top_k,
            active_lim,
            model_key,
            transform,
        ),
        daemon=True,
        name=f"{model_key}-{job_id}",
    )
    thread.start()
    return job_id


def _run_job(
    job_id: str,
    warm_start_path: Path,
    grosores: list[float],
    time_limit: float,
    workers: int,
    out_path: Path,
    top_k: int,
    lim: tuple[float, float, float] | None = None,
    model: ModelId = "mark16",
    allow_transform: bool = True,
) -> None:
    demand_free = model == "mark17"
    geom_label = "volumen (±10%)" if allow_transform else "rígido (sin compensar)"
    _update(
        job_id,
        status="running",
        phase="loading",
        message=f"Iniciando {MODELS[model]['short']} · {geom_label}…",
        progress_pct=2.0,
    )

    def on_progress(event: dict) -> None:
        phase = event.get("phase", "solving")
        message = event.get("message", "")
        patch: dict[str, Any] = {
            "phase": phase,
            "message": message,
            "status": "running" if phase not in {"done", "failed"} else phase,
        }
        if "baseline" in event:
            patch["baseline"] = event["baseline"]
        # Solo cotas en USD (Mark 16). Mark 17 manda surrogate_bound.
        if "bound" in event and not event.get("demand_free"):
            patch["bound"] = event["bound"]
        if "surrogate" in event:
            patch["surrogate"] = event["surrogate"]
        if "surrogate_bound" in event:
            patch["surrogate_bound"] = event["surrogate_bound"]
        if "score" in event:
            patch["score"] = event["score"]
        if "n_boxes" in event:
            patch["n_boxes"] = event["n_boxes"]
        if "n_tipos" in event:
            patch["n_tipos"] = event["n_tipos"]
        if "solution_path" in event:
            patch["solution_path"] = event["solution_path"]
        if "time_limit" in event:
            patch["time_limit"] = event["time_limit"]

        elapsed = float(event.get("elapsed_s") or 0.0)
        # No pisar elapsed con 0 si el evento no trae tiempo (p.ej. phase=done).
        if elapsed > 0 or phase in {"loading", "pool", "pool_done", "queued"}:
            patch["elapsed_s"] = elapsed
        if phase in {"loading"}:
            patch["progress_pct"] = 5.0
        elif phase in {"pool", "pool_done"}:
            patch["progress_pct"] = 15.0 if phase == "pool" else 25.0
        elif phase == "solving":
            with _LOCK:
                prev_elapsed = float((_JOBS.get(job_id) or {}).get("elapsed_s") or 0.0)
            use_elapsed = elapsed if elapsed > 0 else prev_elapsed
            patch["elapsed_s"] = use_elapsed
            patch["progress_pct"] = min(95.0, 25.0 + 70.0 * (use_elapsed / max(time_limit, 1.0)))
        elif phase == "done":
            patch["progress_pct"] = 100.0
            patch["status"] = "done"
            with _LOCK:
                prev_elapsed = float((_JOBS.get(job_id) or {}).get("elapsed_s") or 0.0)
            if elapsed > 0:
                patch["elapsed_s"] = elapsed
            elif prev_elapsed > 0:
                patch["elapsed_s"] = prev_elapsed
            else:
                patch["elapsed_s"] = float(time_limit)
        elif phase == "failed":
            patch["progress_pct"] = 100.0
            patch["status"] = "failed"

        _update(job_id, **patch)

        # Cotas USD solo para Mark 16 (objetivo = costo oficial).
        if event.get("bound_update") and "bound" in event and not event.get("demand_free"):
            with _LOCK:
                job = _JOBS.get(job_id) or {}
                baseline = float(
                    event.get("baseline") or job.get("baseline") or 0.0
                )
                t_point = elapsed if elapsed > 0 else float(job.get("elapsed_s") or 0.0)
            bound = float(event["bound"])
            _append_bound_history(
                job_id,
                {
                    "t": t_point,
                    "bound": bound,
                    "ahorro_cota": float(event.get("ahorro_cota", baseline - bound)),
                    "score_cota": float(
                        event.get(
                            "score_cota",
                            100.0 * (baseline - bound) / baseline if baseline else 0.0,
                        )
                    ),
                    "elapsed_s": t_point,
                },
            )

        if event.get("incumbent") and "cost" in event:
            with _LOCK:
                job = _JOBS.get(job_id) or {}
                baseline = float(event.get("baseline") or job.get("baseline") or 0.0)
                t_point = elapsed if elapsed > 0 else float(job.get("elapsed_s") or 0.0)
            cost = float(event["cost"])
            ahorro = float(event.get("ahorro") or (baseline - cost))
            score = float(event.get("score") if event.get("score") is not None else (
                100.0 * ahorro / baseline if baseline else 0.0
            ))
            point = {
                "t": t_point,
                "cost": cost,
                "ahorro": ahorro,
                "score": score,
                "baseline": baseline,
                "elapsed_s": t_point,
            }
            if "bound" in event and not event.get("demand_free"):
                point["bound"] = float(event["bound"])
            _append_history(job_id, point)
            if "bound" in event and not event.get("demand_free"):
                bound = float(event["bound"])
                _append_bound_history(
                    job_id,
                    {
                        "t": t_point,
                        "bound": bound,
                        "ahorro_cota": baseline - bound,
                        "score_cota": 100.0 * (baseline - bound) / baseline if baseline else 0.0,
                        "elapsed_s": t_point,
                    },
                )
        elif phase == "done" and "cost" in event and event.get("cost") is not None:
            with _LOCK:
                job = _JOBS.get(job_id) or {}
                baseline = float(event.get("baseline") or job.get("baseline") or 0.0)
                t_point = elapsed if elapsed > 0 else float(job.get("elapsed_s") or time_limit)
            cost = float(event["cost"])
            ahorro = float(event.get("ahorro") or (baseline - cost))
            score = float(event.get("score") if event.get("score") is not None else (
                100.0 * ahorro / baseline if baseline else 0.0
            ))
            # Solo agregar punto final si el tiempo avanza o el costo mejora;
            # evita t=0 que dibuja una línea horizontal a lo ancho del chart.
            history = job.get("history") or []
            last = history[-1] if history else None
            should_append = (
                last is None
                or abs(float(last.get("cost", 0)) - cost) >= 1e-6
                or float(last.get("t", -1)) < t_point - 1e-9
            )
            if should_append:
                _append_history(
                    job_id,
                    {
                        "t": t_point,
                        "cost": cost,
                        "ahorro": ahorro,
                        "score": score,
                        "baseline": baseline,
                        "elapsed_s": t_point,
                    },
                )
            if "bound" in event and not event.get("demand_free"):
                bound = float(event["bound"])
                _append_bound_history(
                    job_id,
                    {
                        "t": t_point,
                        "bound": bound,
                        "ahorro_cota": baseline - bound,
                        "score_cota": 100.0 * (baseline - bound) / baseline if baseline else 0.0,
                        "elapsed_s": t_point,
                    },
                )

    try:
        run_fn = run_optimize_mark17 if model == "mark17" else run_optimize_mark16
        with _geometry_mode(allow_transform):
            kwargs: dict[str, Any] = dict(
                warm_start_path=warm_start_path,
                grosores=grosores,
                time_limit=time_limit,
                workers=workers,
                out_path=out_path,
                on_progress=on_progress,
                pool_mode="fast",
                top_k=top_k,
                lim=lim,
            )
            if model == "mark17":
                kwargs["eval_official"] = True
            result = run_fn(**kwargs)
        if not result.get("ok"):
            _update(
                job_id,
                status="failed",
                phase="failed",
                message="Sin solución factible",
                progress_pct=100.0,
            )
            return
        patch_done: dict[str, Any] = {
            "status": "done",
            "phase": "done",
            "message": "Listo",
            "best_cost": result.get("cost"),
            "score": result.get("score"),
            "ahorro": (
                (result["baseline"] - result["cost"])
                if result.get("baseline") is not None and result.get("cost") is not None
                else None
            ),
            "solution_path": result.get("solution_path"),
            "progress_pct": 100.0,
        }
        if demand_free:
            patch_done["surrogate"] = result.get("surrogate")
            patch_done["surrogate_bound"] = result.get("bound")
            # No pisar bound USD con surrogate.
        else:
            patch_done["bound"] = result.get("bound")
        _update(job_id, **patch_done)
    except Exception as exc:
        _update(
            job_id,
            status="failed",
            phase="failed",
            message=str(exc),
            error=traceback.format_exc(),
            progress_pct=100.0,
        )
