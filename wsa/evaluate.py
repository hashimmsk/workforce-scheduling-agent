"""Run policies over fixed weeks and summarise cost, coverage and overtime
(kickoff slides 17-18). Every policy faces identical weeks and absences.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .company import Company
from .config import EnvConfig
from .env import WorkforceSchedulingEnv
from .week import Week, generate_week


def run_episode(env: WorkforceSchedulingEnv, policy, week: Optional[Week] = None,
                seed: Optional[int] = None) -> dict:
    options = {"week": week} if week is not None else None
    obs, info = env.reset(seed=seed, options=options)
    policy.reset(env)
    total, done = 0.0, False
    while not done:
        action = policy.act(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    out = env.unwrapped.summary()
    out["return"] = float(total)
    return out


def evaluate(policy, seeds: Iterable[int], cfg: Optional[EnvConfig] = None,
             company: Optional[Company] = None) -> list[dict]:
    env = WorkforceSchedulingEnv(cfg, company)
    rows = []
    for sd in seeds:
        row = run_episode(env, policy, week=generate_week(env.cfg, sd))
        row["seed"] = int(sd)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    cost = np.array([r["cost"] for r in rows])
    worst = np.sort(cost)[-max(1, int(np.ceil(0.05 * len(cost)))):]
    return {
        "n_weeks": len(rows),
        "cost_mean": float(cost.mean()),
        "cost_std": float(cost.std(ddof=1)) if len(cost) > 1 else 0.0,
        "cost_ci95": float(1.96 * cost.std(ddof=1) / np.sqrt(len(cost))) if len(cost) > 1 else 0.0,
        "worst5_mean": float(worst.mean()),
        "payroll_mean": float(np.mean([r["payroll"] for r in rows])),
        "unfilled_mean": float(np.mean([r["unfilled"] for r in rows])),
        "overtime_hours_mean": float(np.mean([r["overtime_hours"] for r in rows])),
        "overrun_penalty_mean": float(np.mean([r["budget_overrun_penalty"] for r in rows])),
    }


def results_table(results: dict[str, list[dict]], reference: Optional[str] = None) -> str:
    """Plain-text table. ``reference`` names the policy the cost gap is measured against
    (default: the lowest-cost policy in the table)."""
    summ = {name: summarize(rows) for name, rows in results.items()}
    if reference is None:
        reference = min(summ, key=lambda n: summ[n]["cost_mean"])
    ref = summ[reference]["cost_mean"]
    header = (f"{'policy':<14}{'mean cost':>12}{'±95% CI':>10}{'gap vs ' + reference:>18}"
              f"{'worst 5%':>12}{'unfilled':>10}{'OT hours':>10}{'overrun $':>11}")
    lines = [header, "-" * len(header)]
    for name, s in sorted(summ.items(), key=lambda kv: kv[1]["cost_mean"]):
        gap = 100.0 * (s["cost_mean"] - ref) / ref
        lines.append(
            f"{name:<14}{s['cost_mean']:>12,.0f}{s['cost_ci95']:>10,.0f}{gap:>17.1f}%"
            f"{s['worst5_mean']:>12,.0f}{s['unfilled_mean']:>10.2f}"
            f"{s['overtime_hours_mean']:>10.1f}{s['overrun_penalty_mean']:>11,.0f}"
        )
    return "\n".join(lines)


def rows_to_csv(rows_by_policy: dict[str, list[dict]], path) -> None:
    import csv
    keys = ["policy", "seed", "cost", "payroll", "unfilled", "unfilled_cost",
            "overtime_hours", "budget", "budget_overrun_penalty", "n_slots", "invalid_actions"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for name, rows in rows_by_policy.items():
            for r in rows:
                w.writerow([name] + [r.get(k, "") for k in keys[1:]])


class SB3Policy:
    """Adapter so a trained Stable-Baselines3 model plugs into ``evaluate``.
    Set ``masked=True`` for sb3_contrib.MaskablePPO."""

    def __init__(self, model, masked: bool = False, deterministic: bool = True, name: str = "agent"):
        self.model, self.masked, self.deterministic, self.name = model, masked, deterministic, name

    def reset(self, env) -> None:
        pass

    def act(self, env, obs, info) -> int:
        if self.masked:
            action, _ = self.model.predict(obs, action_masks=info["action_mask"],
                                           deterministic=self.deterministic)
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)
