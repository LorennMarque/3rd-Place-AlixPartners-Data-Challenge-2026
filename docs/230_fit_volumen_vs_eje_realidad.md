# Fit por volumen vs fit por eje — ¿es lógico en la realidad?

**Bonsai Corp / AlixPartners 2026**  
Evaluación de la interpretación de la regla 9 que destrabó el score (~−13 → ~+10).

---

## 1. Pregunta

El enunciado permite reajustar cada dimensión interna ±10% y exige:

1. **volumen interno ≥ volumen del producto**, y  
2. headspace máximo por eje.

**No** exige explícitamente que cada eje interno sea ≥ al del producto.

¿Esa lectura es válida para ganar la competencia? Sí.  
¿Es lógica en operación real de empaque rígido? Eso es lo que evalúa este reporte.

---

## 2. Las dos interpretaciones

| | Interpretación vieja (fit por eje) | Interpretación literal (fit por volumen) |
|--|-----------------------------------|------------------------------------------|
| Contención | `interna_k ≥ producto_k` en L, W y H | Solo `L·W·H ≥ vol_producto` |
| Reajuste | ±10% por eje | ±10% por eje |
| Headspace | Por eje, según grosor | Igual |
| ¿Puede la caja ser más chica que el producto en un eje? | No | Sí, si se compensa volumen en otro |
| Techo de score (cota g=3,0) | **≈ −12,5** (USD ~235,4 M) | **≈ +10,5** (USD ~187,3 M) |
| Mejor solución conocida | Quedó trabada ~−13 | Best ≈ **+10,11 / +10,35** |

El gap entre cotas es del orden de **USD 48 M / año** (~22 pts de score). No es un detalle numérico: define si el problema parece “imposible de ganar” o “ya casi resuelto”.

---

## 3. Qué dice el enunciado (regla 9)

> …cada dimensión interna puede modificarse hasta un 10%… El reajuste debe **(a) mantener el volumen interno ≥ al volumen del producto** y **(b) respetar el headspace máximo por eje**.

Lectura literal:

- El ±10% aplica **en cualquier dirección** (agrandar o achicar).
- La condición de “que entre” está escrita como **volumen**, no como tres desigualdades por eje.
- Si se exigiera fit por eje, la condición (a) sería casi redundante (con holgura mínima en tres ejes el volumen ya sobra).

Además, el enunciado **prohíbe rotar** la caja en el pallet: H es siempre vertical. El desbloqueo **no** es rotación; es **redimensionar** (a veces achicar un eje).

---

## 4. Evidencia en la solución Best

Fuente: `outputs/tables/04_mark16_best_+10.11098.csv` + `products_truth`.

| Métrica | Valor |
|---------|------:|
| SKUs totales | 427 |
| SKUs con achique en ≥1 eje (`interna < producto`) | **315 (73,8%)** |
| SKUs que cumplen fit por eje estricto | **112 (26,2%)** |
| % del volumen anual en SKUs con achique | **~79,6%** |
| Achique en L / W / H (conteo de SKUs) | 70 / 140 / 171 |

### Magnitud del achique

| Eje | SKUs | Achique típico | Achique máximo |
|-----|-----:|----------------|----------------|
| Largo (L) | 70 | **1,0 mm** fijo (~0,25% del producto) | 1,0 mm |
| Ancho (W) | 140 | ~14 mm (~5%) | ~22 mm (~7,8%) |
| Alto (H) | 171 | ~6,6 mm (~3%) | ~25 mm (~8,3%) |

Notas:

- En **L**, el achique de 1 mm es el “cliff” del pallet: con grosor 3,0, exterior 400 mm → interior 394 mm; muchos productos están ~1 mm arriba de eso. Achicar 1 mm permite **2 columnas** en el lado de 800 mm del pallet (`⌊800/400⌋=2` vs `⌊800/401⌋=1`).
- En **W** y **H**, el achique no es simbólico: decenas de milímetros. Un producto rígido **no entra** en esa caja.

Todas las cajas del Best quedan con `L_ext ≈ 400 mm`: la densificación del pallet está anclada a ese umbral.

---

## 5. ¿Es lógico en la realidad?

### 5.1 Física del empaque rígido — **No**

Si la caja primaria (producto) es un prisma rígido:

- debe cumplir **fit por eje** (o una rotación de ejes, aquí prohibida en el pallet);
- si `producto_k > interna_k` en cualquier eje, **no cabe**, aunque el volumen total de la caja sea mayor;
- “compensar volumen hinchando otro eje” no crea espacio en el eje corto.

Conclusión operativa: la interpretación literal del brief **no describe un empaque físicamente realizable** para productos rígidos rectangulares.

### 5.2 Cuándo podría acercarse a algo realista — con matices

Solo bajo supuestos poco aplicables a brócoli en caja primaria rígida:

| Supuesto | ¿Aplica aquí? |
|----------|----------------|
| Producto deformable / acomodación menor | No (caja primaria con dims fijas) |
| Holgura de manufactura / tolerancia de 1 mm | Parcial: explica el cliff de L, no el achique de 15–25 mm en W/H |
| Error de wording del brief (quisieron decir fit por eje + volumen) | Plausible |
| Evaluador Kaggle implementó volumen, no eje | Confirmado por scores positivos del leaderboard |

El caso L (−1 mm) podría discutirse como tolerancia o redondeo; el de W/H **no**.

### 5.3 Implicancia de negocio

Si en la planta se exigiera fit por eje real:

- el techo volvería cerca de **−12,5** (o un poco mejor con optimización, pero lejos de +10);
- gran parte del ahorro del Best (~USD 48 M vs cota con fit por eje) **desaparecería**;
- habría que reportar el +10 como **óptimo bajo reglas del concurso**, no como plan de implementación sin rediseñar producto o permitir rotación.

---

## 6. Síntesis para el informe

| Pregunta | Respuesta |
|----------|-----------|
| ¿La lectura literal destrabó el score? | **Sí** (−13 → +10) |
| ¿Es consistente con el texto del enunciado? | **Sí** |
| ¿La acepta el evaluador / leaderboard? | **Sí** |
| ¿Es lógica para empaque rígido real? | **No** |
| ¿Qué parte es más discutible? | Achiques de **cm** en W/H; el de **1 mm** en L es el cliff del pallet |
| ¿Qué haría operaciones? | Exigir fit por eje (y/o rotación), y recalcular costo |

**Mensaje recomendado (1 párrafo):**  
El salto de score se explica por interpretar la regla 9 como contención por **volumen** y no por **eje**. Eso es fiel al brief y al evaluador, y habilita achicar un lado de la caja por debajo del producto para ganar columnas en el pallet. En la realidad, un producto rígido no puede “entrar por volumen”: hace falta fit dimensional (o rotar). Por eso el +10 se presenta como óptimo **bajo las reglas de la competencia**; bajo un criterio físico estricto, el piso de costo vuelve a un régimen cercano al −13.

---

## 7. Fuentes

- Enunciado: `docs/00_instructions_competition.md` (regla 9)
- Diagnóstico del error: `notebooks/cota_optimización.ipynb`
- Cotas actuales: lógica `notebooks/160_cota_baseline_restricciones.ipynb` / Mark 16
- Best: `outputs/tables/04_mark16_best_+10.11098.csv`
- Submissions históricas ~−13: `docs/80_kaggle_submissions.md`
- Orientación fija / sin rotación: enunciado § apilado; `src/settings.py` (`LIM`)
