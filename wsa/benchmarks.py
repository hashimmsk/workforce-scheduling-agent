"""Rule-based benchmarks the agents must beat (kickoff slide 16).

Every policy exposes ``reset(env)`` and ``act(env, obs, info) -> int``.
Rule-based policies are allowed to read the environment's internal state
(hours, wages, mask); learned agents only see ``obs`` and the mask.

* Greedy       : cheapest marginal cost right now, even into overtime.
* Manager rule : eligible employee with the fewest hours so far (common practice).
* Fixed plan   : the Sunday plan, executed blind. Absent/ineligible -> slot stays open.
                 Planner is greedy for now; Ryan's MILP plan should replace it.
* Random       : uniform over eligible actions (sanity floor).
"""
from __future__ import annotations

import numpy as np

from .planning import marginal_costs


class Policy:
    name = "policy"

    def reset(self, env) -> None:  # noqa: D401 - hook
        pass

    def act(self, env, obs, info) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, info) -> int:
        mask = info["action_mask"]
        return int(self.rng.choice(np.flatnonzero(mask)))


class GreedyPolicy(Policy):
    """Cheapest eligible employee for this slot (slide 16, 'Greedy')."""
    name = "greedy"

    def act(self, env, obs, info) -> int:
        u = env.unwrapped
        mask = info["action_mask"][: u.cfg.n_employees]
        if not mask.any():
            return u.UNFILLED
        mc, _, _ = marginal_costs(u.cfg, u.company.wages, u.hours)
        mc = np.where(mask, mc, np.inf)
        return int(np.argmin(mc))


class ManagerRulePolicy(Policy):
    """Eligible employee with the fewest hours; ties go to the cheaper wage."""
    name = "manager_rule"

    def act(self, env, obs, info) -> int:
        u = env.unwrapped
        mask = info["action_mask"][: u.cfg.n_employees]
        if not mask.any():
            return u.UNFILLED
        key = u.hours * 1000.0 + u.company.wages
        key = np.where(mask, key, np.inf)
        return int(np.argmin(key))


class FixedPlanPolicy(Policy):
    """Sunday plan from the forecast, never revised (slide 16, 'MILP plan')."""
    name = "fixed_plan"

    def act(self, env, obs, info) -> int:
        u = env.unwrapped
        planned = u.plan[u.day][u.shift]
        if u.slot >= len(planned):            # demand came in above forecast
            return u.UNFILLED
        e = planned[u.slot]
        if e == u.UNFILLED or not info["action_mask"][e]:
            return u.UNFILLED                 # planned person is out: the plan cannot react
        return int(e)


BENCHMARKS = {
    "greedy": GreedyPolicy,
    "manager_rule": ManagerRulePolicy,
    "fixed_plan": FixedPlanPolicy,
    "random": RandomPolicy,
}
