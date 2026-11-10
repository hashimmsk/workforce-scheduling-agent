"""Synthetic company generator: 50 hourly employees with wages, hour caps,
availability and shift preferences (kickoff slide 10, "Inside the simulator").

A company is fixed for a whole training run. A different ``seed`` gives the
held-out company used in the final test (slide 14).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvConfig

SHIFT_NAMES = ("Morning", "Afternoon", "Evening")
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class Company:
    seed: int
    wages: np.ndarray          # (E,) dollars per hour, $15-$22 in $0.50 steps
    full_time: np.ndarray      # (E,) bool
    max_hours: np.ndarray      # (E,) hard weekly cap
    availability: np.ndarray   # (E, D, S) bool: can the employee work this slot at all
    preference: np.ndarray     # (E, S) in {0, 0.5, 1}: how much they like each shift type

    @property
    def n_employees(self) -> int:
        return self.wages.shape[0]

    def describe(self) -> str:
        ft = int(self.full_time.sum())
        avail = self.availability.mean()
        return (
            f"Company seed {self.seed}: {self.n_employees} employees "
            f"({ft} full-time, {self.n_employees - ft} part-time), "
            f"wages ${self.wages.min():.2f}-${self.wages.max():.2f} "
            f"(mean ${self.wages.mean():.2f}), availability {avail:.0%} of slots"
        )


def generate_company(cfg: EnvConfig, seed: int | None = None) -> Company:
    """Deterministically build a company from ``seed`` (defaults to cfg.company_seed)."""
    seed = cfg.company_seed if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    E, D, S = cfg.n_employees, cfg.n_days, cfg.shifts_per_day

    full_time = rng.random(E) < cfg.full_time_share
    wages = np.round(rng.uniform(cfg.wage_min, cfg.wage_max, E) * 2) / 2
    max_hours = np.where(full_time, cfg.full_time_max_hours, cfg.part_time_max_hours).astype(float)

    # Each employee prefers one shift type; adjacent shift types are half-preferred.
    preferred = rng.integers(0, S, E)
    preference = np.zeros((E, S))
    for s in range(S):
        dist = np.abs(preferred - s)
        preference[:, s] = np.where(dist == 0, 1.0, np.where(dist == 1, 0.5, 0.0))

    # Availability: full-timers are around more often, and people are more often
    # free for the shifts they prefer.
    base = np.where(full_time, 0.80, 0.55)
    prob = base[:, None, None] * (0.7 + 0.5 * preference[:, None, :])
    prob = np.clip(prob, 0.0, 0.97)
    availability = rng.random((E, D, S)) < prob

    return Company(
        seed=seed,
        wages=wages,
        full_time=full_time,
        max_hours=max_hours,
        availability=availability,
        preference=preference,
    )
