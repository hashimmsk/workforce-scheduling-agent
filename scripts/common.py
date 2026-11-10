"""Shared helpers for the training scripts (Megan: DQN, Shaz: PPO)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from wsa import EnvConfig, WorkforceSchedulingEnv, generate_company  # noqa: E402
from wsa.benchmarks import GreedyPolicy, ManagerRulePolicy  # noqa: E402
from wsa.evaluate import SB3Policy, evaluate, results_table  # noqa: E402
from wsa.splits import MAIN_COMPANY_SEED, VALIDATION_SEEDS  # noqa: E402

REWARD_SCALE = 0.01   # dollars -> roughly [-2.6, 0] per step; keeps value targets well-scaled


def training_env(seed: int, company_seed: int = MAIN_COMPANY_SEED) -> WorkforceSchedulingEnv:
    """Fresh random weeks every episode; seeded so runs are reproducible."""
    cfg = EnvConfig(company_seed=company_seed, reward_scale=REWARD_SCALE)
    env = WorkforceSchedulingEnv(cfg, generate_company(cfg))
    env.reset(seed=seed)
    return env


def eval_against_benchmarks(model, masked: bool, n_weeks: int, company_seed: int = MAIN_COMPANY_SEED) -> str:
    cfg = EnvConfig(company_seed=company_seed)           # unscaled dollars for reporting
    company = generate_company(cfg)
    seeds = VALIDATION_SEEDS[:n_weeks]
    results = {
        "agent": evaluate(SB3Policy(model, masked=masked), seeds, cfg, company),
        "greedy": evaluate(GreedyPolicy(), seeds, cfg, company),
        "manager_rule": evaluate(ManagerRulePolicy(), seeds, cfg, company),
    }
    return results_table(results)


def set_global_seeds(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
