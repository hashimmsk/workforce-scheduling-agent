"""Workforce Scheduling Agent — BTE 422 Group 5 (University of Miami, Fall 2026).

Quick start::

    import gymnasium as gym
    import wsa                                   # registers "WorkforceScheduling-v0"
    env = gym.make("WorkforceScheduling-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from gymnasium.envs.registration import register

from .config import EnvConfig
from .company import Company, generate_company
from .week import Week, generate_week
from .env import WorkforceSchedulingEnv

__version__ = "0.1.0"
__all__ = [
    "EnvConfig", "Company", "generate_company", "Week", "generate_week",
    "WorkforceSchedulingEnv", "make_env",
]

register(
    id="WorkforceScheduling-v0",
    entry_point="wsa.env:WorkforceSchedulingEnv",
    # ~150 slots a week; a generous ceiling so TimeLimit never cuts a week short.
    max_episode_steps=EnvConfig().n_shifts * EnvConfig().max_demand,
)


def make_env(**kwargs) -> WorkforceSchedulingEnv:
    """Plain constructor (no TimeLimit wrapper) for scripts and tests."""
    return WorkforceSchedulingEnv(**kwargs)
