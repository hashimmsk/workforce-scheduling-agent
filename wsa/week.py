"""Week generator: forecast demand (known on Sunday), actual demand and
absences (revealed each morning). Kickoff slides 8-10.

A ``Week`` is fully determined by its seed, so the validation and test sets
(``wsa.splits``) are frozen lists of seeds that every policy faces identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvConfig

# Busier Fridays and Saturdays, quieter Sundays; evenings a little busier.
DAY_FACTOR = np.array([1.0, 1.0, 1.0, 1.0, 1.15, 1.25, 0.90])
SHIFT_FACTOR = np.array([0.9, 1.0, 1.1])


@dataclass
class Week:
    seed: int
    forecast: np.ndarray   # (D, S) int: staff the plan expects to need, known up front
    actual: np.ndarray     # (D, S) int: staff really needed, revealed the morning of day d
    absent: np.ndarray     # (E, D) bool: who calls out, revealed the morning of day d

    @property
    def n_slots(self) -> int:
        return int(self.actual.sum())

    def copy(self) -> "Week":
        return Week(self.seed, self.forecast.copy(), self.actual.copy(), self.absent.copy())


def generate_week(cfg: EnvConfig, seed: int) -> Week:
    rng = np.random.default_rng(int(seed))
    E, D, S = cfg.n_employees, cfg.n_days, cfg.shifts_per_day

    day_f = DAY_FACTOR[:D] if D <= len(DAY_FACTOR) else np.resize(DAY_FACTOR, D)
    shift_f = SHIFT_FACTOR[:S] if S <= len(SHIFT_FACTOR) else np.resize(SHIFT_FACTOR, S)
    mean = cfg.base_demand * day_f[:, None] * shift_f[None, :]

    forecast = np.rint(mean * rng.normal(1.0, 0.08, (D, S)))
    forecast = np.clip(forecast, cfg.min_staff, cfg.max_demand).astype(int)

    noise = rng.uniform(1.0 - cfg.demand_noise, 1.0 + cfg.demand_noise, (D, S))
    actual = np.rint(forecast * noise)
    actual = np.clip(actual, cfg.min_staff, cfg.max_demand).astype(int)

    absent = rng.random((E, D)) < cfg.absence_rate
    return Week(seed=int(seed), forecast=forecast, actual=actual, absent=absent)
