# Hipótesis de limitación teórica y contexto de datos

Informe breve sobre por qué, bajo las reglas actuales de la competencia, el costo mínimo alcanzable parece estar **por encima** del baseline histórico de Bonsai Corp — y qué implica eso para la optimización.

---

## 1. Contexto del problema

Bonsai Corp distribuye brócoli congelado desde 5 plantas hacia 5 regiones. Cada uno de los **427 productos** (`codigo_producto`) se empaqueta en una caja de cartón secundaria y se apila en pallets estándar (1200 × 800 mm, altura máxima de carga 1800 mm).

La función objetivo oficial es:

```
C_total = C_packaging + C_flete
```

| Componente | Cómo se calcula |
|------------|-----------------|
| **C_packaging** | precio_base(grosor) × factor_descuento(volumen consolidado del tipo de caja) × volumen anual |
| **C_flete** | pallets necesarios por producto y planta × costo unitario (USD 150 intra-región / USD 500 inter-región) |

Los descuentos de packaging dependen del **volumen anual consolidado por tipo de caja**, no por producto:

| Volumen anual del tipo | Factor de precio |
|------------------------|------------------|
| < 20.000 | +10 % (markup) |
| 20.000 – 49.999 | base (0 %) |
| 50.000 – 99.999 | −10 % |
| 100.000 – 499.999 | −20 % |
| ≥ 500.000 | −30 % |

El número de pallets por producto depende de cuántas cajas entran por pallet (`cajas_por_pallet`), calculado con column stacking: lado largo de la caja paralelo al lado corto del pallet (800 mm).

---

## 2. Punto de partida: datos históricos

### Archivos relevantes

| Archivo | Contenido |
|---------|-----------|
| `data/processed/products_truth.csv` | Dimensiones del producto (largo, ancho, alto) y peso por caja |
| `data/processed/producto_joined_cajas_clean.csv` | Catálogo actual: caja asignada a cada producto |
| `data/raw/operaciones_planta.csv` | Volúmenes anuales por planta y **costos históricos** |

### Situación actual (histórico en `operaciones_planta`)

| Métrica | Valor |
|---------|-------|
| Productos | 427 |
| Tipos de caja distintos (por dimensiones exteriores) | ~204 |
| Grosores de cartón en uso | 2.5, 2.7, 4.1, 4.5, 4.6, 4.7, 4.8 mm |
| Volumen anual total | ~62,4 millones de cajas/año |
| Costo packaging histórico | USD 30,2 M |
| Costo flete histórico | USD 179,1 M |
| **Costo total histórico** | **USD 209,2 M** |

El flete representa aproximadamente el **85 %** del costo total. Cualquier estrategia que empeore la utilización del pallet tiene un impacto grande.

### Referencia interna: baseline válido (`03_submission_valid_baseline.csv`)

Solución válida con grosor uniforme 4.5 mm y las cajas históricas fragmentadas:

- Costo estimado: **USD 304,5 M** (−45,5 % vs histórico)
- Sirve como punto de partida para iterar, pero **no replica** la situación histórica (grosor único + reglas nuevas).

---

## 3. Reglas nuevas que cambian el juego

A partir de la reorganización de Bonsai Corp, solo se permiten tres grosores homogéneos en toda la solución:

| Grosor (mm) | Precio base (USD/caja) | ECT (N/m) | Headspace máx. |
|-------------|------------------------|-----------|----------------|
| 3.0 | 0,60 | 1000 | 6 % |
| 4.5 | 0,65 | 1400 | 8 % |
| 5.0 | 0,70 | 1650 | 10 % |

Otras restricciones relevantes:

- Dimensiones internas mínimas = dimensiones del producto.
- Reajuste máximo del 10 % por eje respecto a las dimensiones del producto.
- Headspace máximo por eje (con tope absoluto de 40 mm).
- Un solo producto por pallet; la misma caja puede compartirse entre productos compatibles.
- Resistencia a compresión según ECT y número de capas apiladas.

**Consecuencia clave:** el baseline histórico (USD 209 M) fue calculado con grosores más finos y un catálogo fragmentado que **ya no es válido** bajo las reglas actuales.

---

## 4. Hipótesis de la limitación

### 4.1 Enunciado

> Bajo las reglas nuevas de la competencia, existe un **piso de costo teórico** por encima del gasto histórico. La comparación `pct_saved` vs `operaciones_planta` puede ser negativa incluso para la mejor solución posible.

### 4.2 Mecanismo principal: migración de grosor y borde del pallet

El caso más ilustrativo son **70 productos** que hoy usan cartón fino (2.5 o 2.7 mm) con caja exterior de **exactamente 400,0 mm de largo**:

```
Con cartón 2.7 mm:  exterior_largo = 400 mm  →  floor(800 / 400) = 2 columnas en pallet
Con cartón 3.0 mm:  exterior_largo ≈ 401 mm  →  floor(800 / 401) = 1 columna en pallet
```

Al pasar al grosor mínimo permitido (3.0 mm), esos productos **pierden la mitad de la densidad de pallet en el eje largo**. El efecto no se puede evitar: las dimensiones internas no pueden ser menores que las del producto, y el exterior mínimo con 3 mm supera el umbral de 400 mm.

Impacto estimado solo de este grupo: **~USD 31 M de flete adicional**, superior a toda la brecha contra el histórico (~USD 26 M).

### 4.3 Mecanismo secundario: precio base vs descuentos

El grosor más barato (3.0 mm) tiene menor precio base, pero obliga a cajas exteriores más grandes que reducen `cajas_por_pallet`. Los grosores 4.5 y 5.0 empeoran aún más el flete:

| Grosor | Cota inferior teórica | vs histórico |
|--------|----------------------|--------------|
| 3.0 mm | USD 235,4 M | −12,5 % |
| 4.5 mm | USD 297,4 M | −42,1 % |
| 5.0 mm | USD 305,1 M | −45,8 % |

### 4.4 ¿Y el trade-off “más pallets a cambio de descuentos”?

Los documentos de la competencia lo contemplan explícitamente: consolidar tipos de caja puede reducir la utilización del pallet si el descuento de packaging lo compensa.

En nuestros datos:

- El ahorro máximo de packaging (todo el volumen al tier 5) es del orden de **USD 15 M**.
- Eso equivaldría a “pagar” hasta ~100.000 pallets extra a USD 150/pallet.
- Existen **~60 pares de productos** donde fusionar empeora `cajas_por_pallet` pero el costo total baja (ej.: +USD 12.750 flete, −USD 34.288 packaging).
- Sin embargo, esas ganancias son **locales y pequeñas** frente a los USD 31 M perdidos por la migración de grosor.

**Conclusión:** el trade-off pallet ↔ descuento es real y explotable, pero en este dataset **no es suficiente** para revertir la brecha estructural contra el histórico.

---

## 5. Cota inferior teórica

Definimos la **cota inferior** como el costo mínimo que ninguna solución válida puede bajar:

| Componente | Supuesto optimista |
|------------|-------------------|
| Flete mínimo | Cada producto con su caja más chica posible (máximo `cajas_por_pallet`) |
| Packaging mínimo | Todo el volumen anual al descuento máximo (factor 0,70, tier 5) |

Para grosor 3.0 mm (el mejor de los tres):

| | Monto (USD) |
|---|------------|
| Flete mínimo | 209,2 M |
| Packaging mínimo | 26,2 M |
| **Total mínimo teórico** | **235,4 M** |
| vs histórico (209,2 M) | **−12,5 %** |

Ninguna solución válida puede costar menos que esto. No es un fallo del optimizador: es una propiedad del espacio factible bajo las reglas nuevas.

---

## 6. Resultado del optimizador (fable5 + ILS)

| Solución | Costo total | vs histórico | Gap vs cota inferior |
|----------|-------------|--------------|----------------------|
| Histórico (`operaciones_planta`) | 209,2 M | 0 % | — (configuración no válida hoy) |
| Cajas mínimas (g 3.0, sin consolidar) | 236,8 M | −13,2 % | +0,6 % |
| **fable5 + ILS (g 3.0)** | **235,6 M** | **−12,6 %** | **+0,07 %** |
| Cota inferior teórica | 235,4 M | −12,5 % | 0 % |

El optimizador está a **~USD 163.000** del piso teórico (principalmente packaging de tipos de caja pequeños que no alcanzan tier 5). El flete ya coincide con el mínimo teórico.

---

## 7. Puntos ciegos revisados

| Hipótesis | Veredicto |
|-----------|-----------|
| “El optimizador no explora fusiones que empeoran pallet” | Descartado: fable5 minimiza Δflete + Δpackaging sin restricción de layout |
| “Con más consolidación se llega al histórico” | Descartado: mark3 sin restricción de pallet empeoró a −65 % |
| “El baseline 03 es el histórico” | Descartado: cuesta USD 304 M (−45 % vs real) |
| “El modelo de pallets no reproduce la data” | Descartado: column-stacking reproduce exactamente los 1.193.792 pallets históricos |
| “El histórico mezcla reglas distintas” | **Confirmado**: grosores obsoletos + cajas que no pasan el validador con g 3.0 |
| “Queda margen grande de mejora” | Parcial: ~0,07 % vs cota inferior; no vs histórico |

---

## 8. Implicancias para el informe final

1. **No presentar el −12 % como fracaso del modelo.** Es la distancia entre dos regímenes distintos (histórico vs reglas nuevas).
2. **Mostrar la cota inferior** como evidencia de que la solución está cerca del óptimo bajo las reglas actuales.
3. **Cuantificar el trade-off packaging ↔ flete** con al menos un ejemplo numérico (los 70 productos en el borde de 400 mm son un caso de negocio claro).
4. **Separar métricas de seguimiento del objetivo:** U_pallet y N_tipos informan, pero el KPI es `C_total`.
5. **Pregunta abierta para el workshop (15/7):** confirmar si la comparación oficial en Kaggle usa el mismo baseline histórico o una referencia recalculada bajo reglas nuevas.

---

## 9. Fuentes en el repositorio

| Recurso | Ubicación |
|---------|-----------|
| Instrucciones oficiales | `docs/00_instructions_competition.md` |
| Restricciones resumidas | `docs/20_constraints.md` |
| Descomposición de costos | `docs/30_money_value.md` |
| Validador de restricciones | `src/solution_validation.py` |
| Evaluador de costos | `src/solution_value_evaluator.py` |
| Optimizador principal | `src/optimizer_fable5.py` |
| Mejor solución actual | `outputs/tables/05_submission_optimizer_fable5_ils_best.csv` |

---

*Generado: julio 2026. Números redondeados; detalle exacto en notebooks `70_solution_check.ipynb` y corridas de `optimizer_fable5.py`.*
