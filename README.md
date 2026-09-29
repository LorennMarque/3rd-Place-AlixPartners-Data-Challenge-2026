# Packaging the broccoli

Bonsai Corp ships frozen broccoli from five plants to five regions. It sells 427 products and buys 204 different cartons to do it. The annual bill is about **USD 209.2 million**. Freight is most of that bill: USD 150 per pallet inside a plant's home region, and USD 500 anywhere else.

Sharing a carton across similar products cuts the catalog, the price per box, and the number of pallets. The score is the percent saved against today's cost. Positive means the plan is cheaper than the current operation.

The live dashboard is [alix.lorenn.ai](https://alix.lorenn.ai/). This is the walkthrough that plays on the home screen:

![Dashboard walkthrough](docs/dashboard.gif)

The written recommendation is in [the report (PDF)](docs/report.pdf).

## What the work produced

- A local cost checker, so a full assignment of products to boxes can be priced without waiting on an outside grader.
- A CP-SAT search (OR-Tools) that picks box dimensions, board thickness, and which products share a carton.
- A four-step way of working: rebuild today's cost, generate boxes that physically fit, optimize the catalog, then test what happens if demand moves.
- A dashboard for exploring a solution, launching a new product, buying the cartons, and comparing two plans.
- The report above, written as the decision document.

Two results sit next to each other on purpose.

- **Under today's rules, the recommended protocol saves about USD 20.0 million a year (9.58%).** It uses 41 carton types instead of 204. It does not need each plant's demand by SKU, and in 1,000 simulated demand shocks it kept about 95% of the available savings.
- **Knowing demand buys a bit more, not a different business.** A portfolio tuned to historical demand uses 55 carton types and is about USD 1.1 million a year cheaper. The protocol gives up that slice in exchange for fewer carton types and a rule that still works when demand is unknown.
- **The file in this repo, `dashboard/soluciones/Optimal_+10.11098.csv`, scores +10.11.** That number comes from reading the fit rule as a volume test: each internal side may move by up to 10%, the box must cover the product's volume, and headspace is capped per axis. If the box also has to clear the product on every axis, the same 3.0 mm board goes from about **+9.6% to about −13.5%**. The gap is the implementation risk called out in the report. A 5 × 5 × 5 product does not fit in a 4 × 5 × 6.25 box.

Two policy changes, outside today's rules, are priced in the report:

- Mixing boards thinner than 3.0 mm is worth about **USD 1.6 million** more per year.
- Raising the pallet by 50 to 150 mm, then reoptimizing, is worth about **USD 4.1 to 14.4 million** more per year.

## The report

[docs/report.pdf](docs/report.pdf) is the decision document. The sections are:

- Executive summary: the recommendation, the USD 20.0 million, and what to do next.
- Business case: current cost, the protocol, and the proposed catalog.
- Assumptions and risks: pallet rules, a single board thickness, and the volume-fit assumption.
- Operating policies: thickness and pallet height, including the options that break today's rules.
- Implementation roadmap: validate the physical fit first, then migrate plant by plant.
- Robustness: demand shocks, and what happens when new products show up.
- The decision tool: how the dashboard is meant to be used.
- Technical annex: the model, the checks, and the sensitivity work.

## The dashboard

Open [alix.lorenn.ai](https://alix.lorenn.ai/). Pick a solution in the sidebar. Its score sits on the picker. You can upload another CSV, and the app checks the file before it keeps it. A short tour, the GIF above, runs from Inicio.

The sidebar is split into two jobs.

### Understand the current plan

- **Inicio.** Total cost, the split between freight and cartons, and cost by plant for the solution you selected. The four tools below are shortcuts into the actions.
- **Productos.** The SKU catalog and the carton types in the active solution: how many products, how many box types, and a way to open a product or assign a new one.
- **Demanda.** Where volume sits across plants, how much of the savings you still get if you ignore SKU-level demand, which plant and which SKU actually move that gap, and a stress test (base, global +10%, and random shocks).

### Act on it

- **Optimizar.** Four steps. Set the rules (which thicknesses are allowed, and whether a product may change shape inside the box as long as volume holds). Run the search. Watch savings over time. Export the result.
- **Asignar.** Type the length, width, height, and net weight of a new product. The rule picks the compatible carton that fits the most units on a pallet, and the cheaper one if two tie. You also see the second-best existing carton and what a brand-new minimum box would cost. On held-out products this reused an existing carton **96.9%** of the time.
- **Órdenes de compra.** Download the active solution as CSV, and open printable purchase orders and plant assignment sheets.
- **Comparar.** Put two solutions side by side: total cost, freight against cartons, distance to the theoretical floor, a metric summary, and the same split by plant.

## The model

Freight dominates, so a smaller box is not automatically a cheaper box. A tighter carton can mean fewer boxes per pallet, and that can cost more than the packaging you saved.

The search in `src/mark16.py` through `src/mark19.py` chooses dimensions and assigns each product to a carton type, minimizing packaging plus freight together. Volume discounts are in the objective. `src/evaluate.py` prices a solution and checks the physical rules. Shared limits (pallet 1200 × 800 mm, load height 1800 mm, thicknesses, headspace) live in `src/settings.py`.

The protocol version of the search does not use demand by SKU and plant. It optimizes pallet fit and catalog size, then prices the result on the real demand afterwards. That is why it stays useful for the next launch, when the volume is still a guess.

## Notebooks

These are in Spanish, in the order the analysis moved:

- `notebooks/00_data_cleaning.ipynb` — clean the case files.
- `notebooks/10_exploration.ipynb` — what the catalog looks like.
- `notebooks/150_contexto_negocio.ipynb` — cost, plants, and today's operation.
- `notebooks/160_cota_baseline_restricciones.ipynb` — how low the cost can go.
- `notebooks/170_baseline_vs_best.ipynb` — today's operation against the best plan.
- `notebooks/180_demand_stress_test.ipynb` — demand shocks.
- `notebooks/200_politicas_grosor.ipynb` — board thickness.
- `notebooks/210_politicas_altura_pallet.ipynb` — pallet height.
- `notebooks/260_piloto_por_planta.ipynb` — rolling the plan out one plant at a time.

Longer notes, also in Spanish:

- `docs/40_hipotesis_limitacion.md` — why the cost looked stuck above the historical bill.
- `docs/60_score_tiebreaker_report.md` — why two solutions with the same cost did not get the same score.
- `docs/140_precio_politicas.md` — what thickness and pallet height are worth.
- `docs/230_fit_volumen_vs_eje_realidad.md` — volume fit against a rigid product, and whether that reading is operable.

Case data is in `data/raw` and `data/processed`, so the checker runs without a separate download.

## Run it

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Price or search locally:

```bash
PYTHONPATH=src python src/mark16.py --help
```

Dashboard, on port 5001:

```bash
cp .env.example .env
# set DASHBOARD_CODE and FLASK_SECRET_KEY
set -a && source .env && set +a
python -m dashboard
```

The public copy is [alix.lorenn.ai](https://alix.lorenn.ai/).

## What is in the repo

- `src/` — checker, CP-SAT search, scaling, and plant rollout.
- `dashboard/` — the decision tool. The solution CSV scored +10.11 is in `dashboard/soluciones/`.
- `notebooks/` — the analysis, in order.
- `docs/report.pdf` — the decision document.
- `docs/*.md` — the shorter technical notes.
- `data/` — the case files.
- `outputs/` — tables and figures the notebooks write.
- `scripts/` — longer search and sweep runs.
