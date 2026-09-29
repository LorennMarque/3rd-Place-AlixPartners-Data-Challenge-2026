"""Constantes del problema Bonsai Corp / AlixPartners 2026."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PRODUCTS_TRUTH_PATH = REPO_ROOT / "data/processed/products_truth.csv"
OPERACIONES_PLANTA_PATH = REPO_ROOT / "data/raw/operaciones_planta.csv"

# Mark 16: un solo valor (3.0) = Kaggle; varios = modo tipo Prometeo.
DEFAULT_GROSORES = [3.0]

SUBMISSION_COLS = [
    "codigo_producto",
    "caja_grosor_mm",
    "caja_exterior_largo",
    "caja_exterior_ancho",
    "caja_exterior_alto",
]

PLANTAS = [
    "buenos_aires",
    "curitiba",
    "santiago",
    "monterrey",
    "bakersfield",
]

PRECIO_BASE_GROSOR = {
    2.5: 0.60,
    2.7: 0.60,
    3.0: 0.60,
    4.1: 0.65,
    4.5: 0.65,
    4.6: 0.70,
    4.7: 0.70,
    4.8: 0.70,
    5.0: 0.70,
}

# Subset homologado Kaggle (política oficial).
GROSORES_KAGGLE = {3.0, 4.5, 5.0}

PALLET_LARGO_MM = 1200
PALLET_ANCHO_MM = 800
PALLET_ALTO_MAX_MM = 1800
# Orientación fija de caja en pallet: L→800, W→1200, H→1800
LIM = (800.0, 1200.0, 1800.0)

SCALE = 1000  # milésimas de USD para CP-SAT

# (volumen_lo, volumen_hi inclusive o None, factor sobre precio base)
TIERS = [
    (0, 19_999, 1.10),
    (20_000, 49_999, 1.00),
    (50_000, 99_999, 0.90),
    (100_000, 499_999, 0.80),
    (500_000, None, 0.70),
]

# Headspace máx. según instrucciones: ≤3.0 → 6%, ≤4.5 → 8%, >4.5 → 10%.
HEADSPACE_PCT = {
    2.5: 0.06,
    2.7: 0.06,
    3.0: 0.06,
    4.1: 0.08,
    4.5: 0.08,
    4.6: 0.10,
    4.7: 0.10,
    4.8: 0.10,
    5.0: 0.10,
}

ECT = {
    2.5: 600,
    2.7: 730,
    3.0: 1000,
    4.1: 1200,
    4.5: 1400,
    4.6: 1450,
    4.7: 1500,
    4.8: 1550,
    5.0: 1650,
}

MAX_SHRINK = 0.90
MAX_GROW = 1.10
HEADSPACE_ABS_MAX = 40.0

GRAVITY = 9.81
EPS = 1e-9
