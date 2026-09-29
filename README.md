# Packaging the broccoli

Bonsai Corp ships frozen broccoli from five plants to five regions. Each product has its own carton. The bill is packaging plus freight, and freight is about 85% of it: USD 150 per pallet inside a plant's home region, USD 500 everywhere else.

A catalog of hundreds of box types means every launch adds a mold, a production run, and stock. The score is the percent saved against today's cost. Positive means the plan is cheaper than the current operation.

I wrote a checker that prices a full assignment of products to boxes, then a CP-SAT search in OR-Tools over which products share a carton, how thick the board is, and how the pallet fills. The same model sits behind a dashboard for comparing solutions: [alix.lorenn.ai](https://alix.lorenn.ai/).

The first file the grader accepted scored about −46. A feasible search then sat near −13, close to the ceiling you get if the box also has to clear the product on every axis.

The brief lets each internal dimension move by up to 10%. The box has to keep at least the product's volume, and it has to respect a headspace cap on each axis. Under that reading, the solution in this repo scores **+10.11** (`dashboard/soluciones/Optimal_+10.11098.csv`). The gap between the two ceilings is about USD 48 million a year. Notes on that reading are in `docs/230_fit_volumen_vs_eje_realidad.md`.

## Run it

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Price a solution locally:

```bash
PYTHONPATH=src python src/mark16.py --help
```

The search entry points are `src/mark16.py` through `src/mark19.py`. Shared limits live in `src/settings.py`. The cost checker is `src/evaluate.py`.

Dashboard, on port 5001:

```bash
cp .env.example .env
# set DASHBOARD_CODE and FLASK_SECRET_KEY
set -a && source .env && set +a
python -m dashboard
```

The public copy is [alix.lorenn.ai](https://alix.lorenn.ai/).

## Where the story is

The notebooks are in Spanish, in the order I worked:

| Notebook | What it covers |
|---|---|
| `notebooks/00_data_cleaning.ipynb` | Cleaning the case files |
| `notebooks/10_exploration.ipynb` | What the catalog actually looks like |
| `notebooks/150_contexto_negocio.ipynb` | Cost, plants, and the current operation |
| `notebooks/160_cota_baseline_restricciones.ipynb` | How low the cost can go |
| `notebooks/170_baseline_vs_best.ipynb` | Current operation against the best plan |
| `notebooks/180_demand_stress_test.ipynb` | What happens if demand moves |
| `notebooks/200_politicas_grosor.ipynb` | Board thickness |
| `notebooks/210_politicas_altura_pallet.ipynb` | Pallet height |
| `notebooks/260_piloto_por_planta.ipynb` | Rolling the plan out plant by plant |

Longer notes, also in Spanish: `docs/40_hipotesis_limitacion.md`, `docs/60_score_tiebreaker_report.md`, `docs/140_precio_politicas.md`.

Case data is in `data/raw` and `data/processed`, so the checker can run without an extra download.
