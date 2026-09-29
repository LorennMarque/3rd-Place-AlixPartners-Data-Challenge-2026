#!/usr/bin/env python3
"""Worker aislado: Mark17 reopt sobre CSVs temporales. Exit 0 si escribió out_path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--ops", type=Path, required=True)
    ap.add_argument("--warm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tl", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--grosor", type=float, default=3.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    import mark17 as m17
    from evaluate import SUBMISSION_COLS

    m17.PRODUCTS_TRUTH_PATH = args.pt
    m17.OPERACIONES_PLANTA_PATH = args.ops

    result = m17.run_optimize(
        warm_start_path=args.warm,
        grosores=[args.grosor],
        time_limit=float(args.tl),
        workers=max(1, int(args.workers)),
        out_path=args.out,
        pool_mode="fast",
        top_k=int(args.top_k),
        eval_official=False,
    )
    sol = result.get("solution")
    if sol is None and args.out.exists():
        return 0
    if sol is not None and len(sol):
        sol[SUBMISSION_COLS].to_csv(args.out, index=False)
        return 0
    print(f"worker: no solution ok={result.get('ok')}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
