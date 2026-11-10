"""Shared scheduling arithmetic: eligibility rules, marginal cost of a fill,
and the Sunday "forecast plan" that sets the weekly budget and serves as the
fixed-plan benchmark.

The planner here is a cheap greedy one. Ryan's MILP plan (kickoff slide 16)
should replace ``greedy_forecast_plan`` for the benchmark; the environment only
needs *a* deterministic plan cost to define the budget.
"""
from __future__ import annotations

import numpy as np

from .company import Company
from .config import EnvConfig
from .week import Week


def marginal_costs(cfg: EnvConfig, wages: np.ndarray, hours_before: np.ndarray):
    """Vectorised cost of giving each employee one more shift.

    Returns (cost, regular_hours, overtime_hours), each shape (E,).
    """
    regular = np.clip(cfg.overtime_threshold - hours_before, 0.0, cfg.shift_hours)
    overtime = cfg.shift_hours - regular
    cost = wages * (regular + cfg.overtime_multiplier * overtime)
    return cost, regular, overtime


def eligible(
    cfg: EnvConfig,
    company: Company,
    available_slot: np.ndarray,
    absent_today: np.ndarray,
    worked_today: np.ndarray,
    streak: np.ndarray,
    hours: np.ndarray,
) -> np.ndarray:
    """Who may legally take the current slot (all five frictions, slide 9)."""
    return (
        available_slot
        & ~absent_today
        & ~worked_today                                   # no double shifts
        & (streak < cfg.max_consecutive_days)             # at most 6 consecutive days
        & (hours + cfg.shift_hours <= company.max_hours + 1e-9)  # weekly hour cap
    )


def greedy_forecast_plan(cfg: EnvConfig, company: Company, week: Week):
    """Build the Sunday plan: fill every *forecast* slot with the cheapest
    eligible employee, assuming nobody is absent. Returns (plan, cost) where
    ``plan[d][s]`` is the ordered list of employee ids (or the unfilled action).
    """
    E, D, S = cfg.n_employees, cfg.n_days, cfg.shifts_per_day
    hours = np.zeros(E)
    streak = np.zeros(E, dtype=int)
    nobody_absent = np.zeros(E, dtype=bool)
    plan = [[[] for _ in range(S)] for _ in range(D)]
    cost = 0.0
    for d in range(D):
        worked_today = np.zeros(E, dtype=bool)
        for s in range(S):
            for _ in range(int(week.forecast[d, s])):
                elig = eligible(cfg, company, company.availability[:, d, s],
                                nobody_absent, worked_today, streak, hours)
                if not elig.any():
                    plan[d][s].append(cfg.unfilled_action)
                    cost += cfg.unfilled_penalty
                    continue
                mc, _, _ = marginal_costs(cfg, company.wages, hours)
                mc = np.where(elig, mc, np.inf)
                e = int(np.argmin(mc))
                plan[d][s].append(e)
                cost += float(mc[e])
                hours[e] += cfg.shift_hours
                worked_today[e] = True
        streak = np.where(worked_today, streak + 1, 0)
    return plan, float(cost)
