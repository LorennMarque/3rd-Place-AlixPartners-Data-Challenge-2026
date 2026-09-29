# Precio de las políticas de la empresa

**Bonsai Corp — What-if de grosores, mezcla y altura de pallet**  
AlixPartners 2026

---

## Lectura ejecutiva

| Política | Valor / precio |
|----------|----------------|
| Mezclar grosores 2,5/2,7/3,0 vs Best único 3,0 | **~USD 1,6 M / año** (solución) · **~USD 1,8 M** (cota) |
| Elegir único 4,5 o 5,0 vs único 3,0 | **+USD 7–10 M** de costo (cotas) |
| Mín. altura extra que mueve Best | **+54 mm** (solo USD 9.750) |
| Best con +100 / +150 mm de altura | **−USD 2,0 M / −USD 3,6 M** |

> Precios de 2,5 y 2,7 mm se asumen **USD 0,60/caja** (igual que 3,0), inferidos del baseline. Validar con el proveedor.

---

## 1. Política de grosor único

Hoy: un solo grosor ∈ {3,0 · 4,5 · 5,0} mm para todo el catálogo.

### Cotas analíticas (piso de costo)

| Política | Cota total | Flete piso | Pack piso | Score techo | vs cota 3,0 |
|----------|----------:|----------:|----------:|------------:|------------:|
| **Único 3,0 mm** | **187,3 M** | 161,1 M | 26,2 M | **+10,47** | — |
| Único 4,5 mm | 194,4 M | 166,0 M | 28,4 M | +7,10 | **+7,0 M** |
| Único 5,0 mm | 196,9 M | 166,3 M | 30,6 M | +5,88 | **+9,6 M** |
| Único 2,7 mm | No viable | — | — | — | 13 SKUs fallan ECT |
| Único 2,5 mm | No viable | — | — | — | 69 SKUs fallan ECT |

**Lectura:** 2,5/2,7 no pueden ser política única (resistencia insuficiente en parte del portafolio). Dentro de los homologados, **3,0 mm es el mejor**. Elegir 4,5/5,0 “por seguridad” sin necesidad de ECT es pagar USD 7–10 M extra al año.

---

## 2. Valor de permitir mezclar grosores

### Cotas (piso teórico)

| Escenario | Cota | vs único 3,0 |
|-----------|-----:|-------------:|
| Único 3,0 | 187,3 M | — |
| Mezcla 2,7/3,0 | 185,6 M | **−1,7 M** |
| Mezcla 2,5/2,7/3,0 | 185,6 M | **−1,8 M** |
| Mezcla todos ECT | 184,8 M | **−2,5 M** |

### Mejor solución conocida

| Plan | Tipos | Grosores | Packaging | Flete | Total | Score |
|------|------:|----------|----------:|------:|------:|------:|
| **Best** (política actual) | 56 | solo 3,0 | 26,4 M | 161,2 M | **187,6 M** | +10,35 |
| **MixBest** | 84 | 2,5 / 2,7 / 3,0 | 26,6 M | 159,4 M | **186,0 M** | +11,10 |

**Valor logrado de la mezcla: USD 1,56 M / año.**

Distribución MixBest: 51 SKUs @2,5 · 117 @2,7 · 259 @3,0. El packaging sube un poco; el ahorro viene del flete (menos pallets).

### Trade-off operativo

Mezclar grosores suma complejidad de compras e inventario. El precio que se paga hoy por “un solo grosor fácil” es del orden de **USD 1,6–1,8 M/año** — contrastarlo con el costo de fragmentar el cartón.

---

## 3. Altura de pallet (buffer de 200 mm)

Política actual: carga máx. **1800 mm** (~200 mm libres vs contenedor ~2000 mm).

### Mínimo que mejora el Best actual

Con las **mismas cajas** de Best, el primer ahorro aparece a **+54 mm** y es pequeño (**USD 9.750**). Con +1 / +10 / +20 / +50 mm el Best **no cambia**: ninguna caja gana una capa.

### Barrido pedido

| Altura extra | Costo Best | Δ vs Best | Cota g=3,0 | Δ cota | Cota mezcla 2,5/2,7/3,0 |
|-------------:|----------:|----------:|----------:|-------:|------------------------:|
| +0 mm (hoy) | 187,585 M | — | 187,321 M | — | 185,559 M |
| +1 mm | 187,585 M | 0 | 187,321 M | 0 | 185,559 M |
| +10 mm | 187,585 M | 0 | 186,735 M | **−0,59 M** | 184,152 M |
| +20 mm | 187,585 M | 0 | 185,245 M | **−2,08 M** | 183,481 M |
| +50 mm | 187,585 M | 0 | 182,952 M | **−4,37 M** | 181,616 M |
| **+54 mm** | 187,575 M | **−0,01 M** | 182,882 M | −4,44 M | 181,558 M |
| **+100 mm** | 185,610 M | **−1,97 M** | 179,721 M | **−7,60 M** | 176,835 M |
| **+150 mm** | 184,015 M | **−3,57 M** | 173,131 M | **−14,19 M** | 172,178 M |

- **Best** = mismas cajas, solo más altura.  
- **Cota** = techo si se rediseñan cajas con esa altura.

**Lectura:** entre +10 y +50 mm el upside está en rediseñar (cota), no en el Best actual. Con +100 / +150 mm ya hay ahorro material **sin** cambiar el catálogo Best (~USD 2,0 M / 3,6 M).

---

## 4. Mapa de valor de políticas

| Política | Estado hoy | Precio / valor | Notas |
|----------|------------|----------------|-------|
| Grosor único 3,0 | Vigente (Best) | Referencia | Mejor de los tres homologados |
| Grosor único 4,5 / 5,0 | Permitido | Cuesta **+7 / +10 M** vs 3,0 | Solo si ECT lo exige |
| Único 2,5 o 2,7 | No homologado | No viable solo | Falla ECT en 69 / 13 SKUs |
| Mezcla 2,5/2,7/3,0 | Fuera de política Kaggle | Ahorra **~1,6 M** (solución) / **~1,8 M** (cota) | Más tipos de cartón (84 vs 56) |
| Buffer 200 mm (altura 1800) | Vigente | +100 mm → **~2,0 M**; +150 → **~3,6 M** en Best | Validar vs reefers / seguridad |
| 1 SKU por pallet | Vigente | Relajar: **−USD 24–51 k** techo; half+DD: **+USD 14 M** | No priorizar (Mark 18 / what-if) |

---

## 5. Recomendaciones

1. **Mantener 3,0 mm** como ancla si la política sigue siendo grosor único.  
2. **Evaluar mezcla fina 2,5/2,7/3,0** si Compras tolera más SKUs de cartón a cambio de ~USD 1,6 M/año (validar precio real con el proveedor).  
3. **Cuestionar el buffer de 200 mm:** ya con +100 mm hay ~USD 2 M sin rediseñar Best; con rediseño la cota supera USD 7 M.  
4. No promover 4,5/5,0 únicos salvo requisito documentado de resistencia.  
5. **Mantener 1 SKU por pallet / no double deck:** el techo de mezclar es ~USD 50 k; convertir a half-pallets empeora ~USD 14 M (Mark 18).

---

## Fuentes

- Cotas: lógica notebook `160_cota_baseline_restricciones` (extendida a 2,5/2,7 y mezcla)  
- Best: `04_mark15_best_+10.11098csv`  
- MixBest: `outputs/mark16/mark16_job_8ebc73743e77_20260727_151017.csv`  
- Altura: evaluador con `lim=(800, 1200, 1800+Δ)`
