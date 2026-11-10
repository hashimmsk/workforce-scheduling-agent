"""Regenerate the figures used in the progress-presentation slides.

    python scripts/make_figures.py          # writes figures/*.png
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wsa import WorkforceSchedulingEnv, generate_week  # noqa: E402
from wsa.benchmarks import BENCHMARKS, GreedyPolicy, ManagerRulePolicy  # noqa: E402
from wsa.evaluate import evaluate, run_episode  # noqa: E402
from wsa.render import (plot_env_loop, plot_observation_layout, plot_policy_costs,  # noqa: E402
                        plot_week_schedule)
from wsa.splits import VALIDATION_SEEDS  # noqa: E402


def main():
    out = pathlib.Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    env = WorkforceSchedulingEnv()

    plot_env_loop(out / "env_loop.png")
    plot_observation_layout(env, out / "observation_layout.png")

    week = generate_week(env.cfg, VALIDATION_SEEDS[0])
    run_episode(env, GreedyPolicy(), week=week)
    plot_week_schedule(env, out / "week_schedule_greedy.png", title="Greedy benchmark · validation week 0")
    run_episode(env, ManagerRulePolicy(), week=week)
    plot_week_schedule(env, out / "week_schedule_manager_rule.png",
                       title="Manager-rule benchmark · validation week 0")

    results = {n: evaluate(cls(), VALIDATION_SEEDS[:100]) for n, cls in BENCHMARKS.items()}
    plot_policy_costs(results, out / "benchmark_costs.png", title="Rule-based benchmarks on validation weeks")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
