# Análisis de tie-breakers del public score

_Generado por `src/score_tiebreaker_analysis.py`_

## TL;DR

1. **El public score de Kaggle es el ahorro porcentual vs. la situación actual** (`score = 100 x (costo_actual - costo_solucion) / costo_actual`).
2. **El 'empate' de costo era un artefacto de nuestro evaluador interno**: aplica el descuento por volumen sobre el volumen GLOBAL consolidado de cada tipo de caja, y todas las soluciones caen en el tier máximo (-30%), por eso da idéntico costo.
3. **El evaluador oficial aplica los tiers por volumen POR PLANTA de cada tipo de caja** (verificado contra `procurement_cajas.csv`: el factor de descuento real coincide con el tier del volumen por planta en ~87% de las celdas, y con el global en solo ~30%).
4. **El tie-breaker es entonces el packaging por planta**: al consolidar en menos tipos, el volumen de cada tipo en cada planta cruza más bandas de descuento. Menos tipos y menos celdas tipo x planta con poco volumen implican mejor score.

## Datos de ajuste del modelo de score

- Costo actual (de `operaciones_planta`): **USD 209,235,094**
- Ajuste con evaluador interno (tier global), solo empatadas: Pearson 0.297, Spearman 0.300, RMSE 0.6556 puntos de score.
- Ajuste con tier por planta, solo empatadas: Pearson 1.000, Spearman 1.000, RMSE 0.0000 puntos de score.

![Predicho vs real](outputs/figures/score_tiebreaker_pred_vs_real.png)

## Tabla comparativa

| submission | public_score | n_tipos | score_pred_cost_global | score_pred_cost_planta | pack_planta | flete | celdas_tipo_planta_activas | factor_medio_ponderado | pct_vol_tier_max |
|---|---|---|---|---|---|---|---|---|---|
| 12_submission_optimizer_mark7_20260709_131329 | -13.14 | 53 | -12.60 | -13.14 | 27,502,499.34 | 209,222,850.00 | 192 | 0.73 | 0.73 |
| 12_submission_optimizer_mark7_20260709_151847 | -13.15 | 53 | -12.60 | -13.15 | 27,523,134.66 | 209,222,850.00 | 191 | 0.73 | 0.72 |
| PALETO_MIN_BOXES | -13.19 | 56 | -12.60 | -13.19 | 27,603,980.94 | 209,222,850.00 | 206 | 0.74 | 0.71 |
| PALETO_candidato_06_frac0.35_seed5 | -13.20 | 59 | -12.60 | -13.20 | 27,639,819.90 | 209,222,850.00 | 214 | 0.74 | 0.69 |
| MEJOR_FIX_candidato_06_frac0.35_seed5 | -13.25 | 65 | -12.60 | -13.25 | 27,744,768.72 | 209,222,850.00 | 228 | 0.74 | 0.68 |
| PALETO_MAX_candidato_0197_frac0.45_seed568 | -13.29 | 67 | -12.60 | -13.29 | 27,816,907.98 | 209,222,850.00 | 235 | 0.74 | 0.66 |
| PEOR_candidato_02_frac0.35_seed1 | -13.30 | 69 | -12.60 | -13.30 | 27,846,557.52 | 209,222,850.00 | 247 | 0.74 | 0.66 |
| 09_submission_optimizer_mark_4_gridbest (1) | -13.32 | 67 | -12.60 | -13.32 | 27,875,691.60 | 209,222,850.00 | 233 | 0.74 | 0.65 |
| 05_submission_optimizer_mark_1 (1) | -13.33 | 79 | -12.66 | -13.33 | 27,898,640.64 | 209,222,850.00 | 257 | 0.74 | 0.67 |
| MEJOR_candidato_05_frac0.35_seed4 | -13.34 | 70 | -12.60 | -13.34 | 27,919,007.70 | 209,222,850.00 | 243 | 0.75 | 0.64 |
| 07_submission_optimzer_mark_2 (1) | -13.34 | 67 | -12.60 | -13.34 | 27,930,589.68 | 209,222,850.00 | 230 | 0.75 | 0.63 |
| 03_submission_valid_baseline | -46.11 | 200 | -45.52 | -46.11 | 31,141,927.77 | 274,564,350.00 | 426 | 0.77 | 0.55 |

Notas: `pack_planta` es el costo de packaging con tiers por planta; `celdas_tipo_planta_activas` cuenta combinaciones tipo x planta con volumen > 0; `factor_medio_ponderado` es el factor de descuento promedio ponderado por volumen (más bajo = más descuento); `pct_vol_tier_max` es la fracción del volumen que accede al tier máximo (-30%).

## Qué variable correlaciona con el score (solo soluciones 'empatadas')

| feature | pearson | spearman | nota |
|---|---|---|---|
| pack_planta | -1.000 | -1.000 |  |
| factor_medio_ponderado | -1.000 | -1.000 |  |
| pct_vol_tier_max | 0.964 | 0.936 |  |
| n_tipos | -0.911 | -0.878 |  |
| celdas_tipo_planta_activas | -0.922 | -0.791 |  |
| headspace_medio_mm | 0.812 | 0.727 |  |
| pack_global | -0.297 | -0.300 |  |
| pallets_total | — | — | constante entre submissions |
| flete | — | — | constante entre submissions |

![Score vs n_tipos](outputs/figures/score_tiebreaker_ntipos.png)

## Decomposición mejor vs peor (entre empatadas)

- Mejor: **12_submission_optimizer_mark7_20260709_131329** (score -13.13845)
- Peor: **07_submission_optimzer_mark_2 (1)** (score -13.34305)
- Diferencia de costo estimado (tier por planta): USD 428,090 (packaging: 428,090, flete: 0)
- Productos con asignación distinta entre ambas: **427 de 427**

Distribución del volumen por tier de descuento (fracción del volumen total):

| tier | 12_submission_optimizer_mark7_20260709_131329 | 07_submission_optimzer_mark_2 (1) |
|---|---|---|
| T5 (>=500k, -30%) | 0.7264 | 0.6306 |
| T4 (100-500k, -20%) | 0.2306 | 0.3123 |
| T3 (50-100k, -10%) | 0.0226 | 0.0328 |
| T2 (20-50k) | 0.0144 | 0.0174 |
| T1 (<20k, +10%) | 0.0061 | 0.0069 |

## Implicancias para optimizar

1. **Cambiar la función objetivo del optimizador**: usar packaging con tiers por volumen por planta (no global). Es lo que el evaluador oficial parece premiar.
2. **Consolidar pensando por planta**: un tipo de caja con 600k unidades globales puede estar repartido en 5 plantas con 120k cada una y pagar -20% en vez de -30%. Conviene que cada tipo concentre volumen dentro de cada planta.
3. **El flete no está rompiendo el empate** entre estas soluciones (idéntico en todas las empatadas), pero sigue siendo ~89% del costo: cualquier mejora real de cajas por pallet domina sobre el packaging.
4. **La fórmula quedó identificada de forma exacta** (el RMSE del modelo por planta es prácticamente cero): ya no hace falta gastar submissions para probar el score, se puede calcular offline con este mismo script antes de subir.
