"""Run every rule-based benchmark on the fixed validation weeks and print the
headline table (kickoff slide 18). Saves per-week rows to results/.

    python scripts/run_benchmarks.py            # all 300 validation weeks
    python scripts/run_benchmarks.py --n 50     # quicker
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wsa import EnvConfig, generate_company  # noqa: E402
from wsa.benchmarks import BENCHMARKS  # noqa: E402
from wsa.evaluate import evaluate, results_table, rows_to_csv  # noqa: E402
from wsa.render import plot_policy_costs  # noqa: E402
from wsa.splits import MAIN_COMPANY_SEED, VALIDATION_SEEDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(VALIDATION_SEEDS), help="number of validation weeks")
    ap.add_argument("--company-seed", type=int, default=MAIN_COMPANY_SEED)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = EnvConfig(company_seed=args.company_seed)
    company = generate_company(cfg)
    seeds = VALIDATION_SEEDS[: args.n]
    print(company.describe())
    print(f"Evaluating {len(BENCHMARKS)} benchmarks on {len(seeds)} validation weeks...\n")

    results = {name: evaluate(cls(), seeds, cfg, company) for name, cls in BENCHMARKS.items()}
    print(results_table(results))

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    rows_to_csv(results, out / f"benchmarks_validation_{len(seeds)}w.csv")
    plot_policy_costs(results, out / "benchmarks_validation.png")
    print(f"\nSaved {out / f'benchmarks_validation_{len(seeds)}w.csv'} and {out / 'benchmarks_validation.png'}")


if __name__ == "__main__":
    main()
