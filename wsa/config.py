"""Environment configuration for the Workforce Scheduling Agent.

Every number here comes from the BTE 422 kickoff deck (slides 4, 6, 9, 10, 15):
50 hourly employees, 7 days x 3 shifts, overtime at 1.5x beyond 40 hours,
$200 per unfilled slot, 25% penalty on payroll above budget, demand within
+/-20% of forecast, ~5% absences, budget = 105% of the forecast plan cost.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # Company / calendar
    n_employees: int = 50
    n_days: int = 7
    shifts_per_day: int = 3
    shift_hours: float = 8.0

    # Wages and hours
    wage_min: float = 15.0
    wage_max: float = 22.0
    overtime_threshold: float = 40.0      # hours per week before 1.5x applies
    overtime_multiplier: float = 1.5
    full_time_share: float = 0.6          # fraction of employees who are full-time
    full_time_max_hours: float = 48.0     # hard weekly cap (6 shifts)
    part_time_max_hours: float = 24.0     # hard weekly cap (3 shifts)
    max_consecutive_days: int = 6         # rest rule: never 7 days in a row

    # Penalties
    unfilled_penalty: float = 200.0       # $ per slot left open
    min_staff: int = 2                    # two-person minimum per shift (demand floor)
    budget_multiplier: float = 1.05       # budget = 1.05 x forecast plan cost
    budget_overrun_penalty: float = 0.25  # 25% of payroll above budget

    # Uncertainty (revealed each morning)
    demand_noise: float = 0.20            # actual demand within +/-20% of forecast
    absence_rate: float = 0.05            # ~5% chance an employee is out on a given day
    base_demand: float = 7.0              # mean staff per shift (~150 slots a week)
    max_demand: int = 14                  # generator cap; also used to normalise the state

    # Reproducibility / scaling
    company_seed: int = 0                 # which synthetic company to generate
    reward_scale: float = 1.0             # training scripts use 0.01 (dollars -> ~[-3, 0])

    # Derived helpers -------------------------------------------------------
    @property
    def n_shifts(self) -> int:
        return self.n_days * self.shifts_per_day

    @property
    def n_actions(self) -> int:
        """One action per employee plus 'leave the slot unfilled'."""
        return self.n_employees + 1

    @property
    def unfilled_action(self) -> int:
        return self.n_employees
