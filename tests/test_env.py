"""Environment tests: Gymnasium's checker plus the project's own success
indicators (charter §3, kickoff slide 19): no lookahead, every rule enforced,
costs add up, runs are reproducible."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import wsa
from wsa import EnvConfig, WorkforceSchedulingEnv, generate_week
from wsa.benchmarks import BENCHMARKS, GreedyPolicy, RandomPolicy
from wsa.evaluate import evaluate, run_episode


@pytest.fixture
def env():
    return WorkforceSchedulingEnv()


def rollout(env, policy, seed=0, week=None):
    """Run one week, returning (observations, rewards, infos)."""
    options = {"week": week} if week is not None else None
    obs, info = env.reset(seed=seed, options=options)
    policy.reset(env)
    obs_list, rew_list, info_list = [obs], [], [info]
    done = False
    while not done:
        a = policy.act(env, obs, info)
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(obs); rew_list.append(r); info_list.append(info)
        done = term or trunc
    return obs_list, rew_list, info_list


# --------------------------------------------------------------- Gymnasium API
def test_passes_gymnasium_check_env(env):
    check_env(env, skip_render_check=True)


def test_registered_id_works():
    e = gym.make("WorkforceScheduling-v0")
    obs, info = e.reset(seed=1)
    assert obs.shape == e.observation_space.shape
    assert e.action_space.n == 51
    assert "action_mask" in info and info["action_mask"].shape == (51,)


def test_spaces_match_deck(env):
    assert env.action_space.n == 51                      # 50 employees + unfilled
    assert env.observation_space.shape == (260,)         # "about 300 inputs"
    assert env.observation_space.dtype == np.float32


# ---------------------------------------------------------------- episode shape
def test_episode_length_equals_actual_demand(env):
    obs, info = env.reset(seed=3)
    n_steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.UNFILLED)
        n_steps += 1
        done = term or trunc
    assert n_steps == env.week.actual.sum()
    assert 100 <= n_steps <= 220                         # "about 150 slots"


def test_reset_with_seed_is_reproducible():
    a, b = WorkforceSchedulingEnv(), WorkforceSchedulingEnv()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.array_equal(oa, ob)
    assert np.array_equal(a.week.actual, b.week.actual)
    assert np.array_equal(a.week.absent, b.week.absent)
    assert a.budget == b.budget


def test_week_seeds_cycle_in_order():
    e = WorkforceSchedulingEnv(week_seeds=[7, 8])
    e.reset(); assert e.week.seed == 7
    e.reset(); assert e.week.seed == 8
    e.reset(); assert e.week.seed == 7


# ------------------------------------------------------------- the five rules
@pytest.mark.parametrize("policy_cls", [GreedyPolicy, RandomPolicy])
def test_schedule_respects_all_constraints(env, policy_cls):
    rollout(env, policy_cls(), seed=11)
    cfg, c, w, sched = env.cfg, env.company, env.week, env.schedule
    # availability and absences
    assert not (sched & ~np.transpose(c.availability, (1, 2, 0))).any()
    assert not (sched.any(axis=1) & w.absent.T).any()
    # no double shifts
    assert (sched.sum(axis=1) <= 1).all()
    # weekly hour cap
    assert (env.hours <= c.max_hours + 1e-9).all()
    # never more than 6 consecutive days
    worked = sched.any(axis=1).T                          # (E, D)
    for row in worked:
        run = best = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        assert best <= cfg.max_consecutive_days


def test_mask_marks_only_eligible(env):
    env.reset(seed=5)
    mask = env.action_masks()
    assert mask[env.UNFILLED]
    d, s = env.day, env.shift
    expected = env.company.availability[:, d, s] & ~env.week.absent[:, d]
    assert np.array_equal(mask[:-1], expected)           # nothing booked yet, so only these two rules bite


def test_invalid_action_is_treated_as_unfilled(env):
    env.reset(seed=5)
    mask = env.action_masks()
    bad = int(np.flatnonzero(~mask[:-1])[0])
    _, r, _, _, info = env.step(bad)
    assert r == -env.cfg.unfilled_penalty
    assert info["unfilled"] == 1 and env.invalid_actions == 1


# ------------------------------------------------------------------ accounting
def test_rewards_add_up_to_total_cost(env):
    _, rewards, _ = rollout(env, GreedyPolicy(), seed=9)
    s = env.summary()
    assert np.isclose(-sum(rewards), s["cost"])
    assert np.isclose(s["cost"], s["payroll"] + s["unfilled_cost"] + s["budget_overrun_penalty"])


def test_overtime_is_charged_at_1_5x():
    cfg = EnvConfig(full_time_share=1.0, absence_rate=0.0)
    e = WorkforceSchedulingEnv(cfg)
    e.reset(seed=1)
    emp = int(np.argmax(e.company.wages))                # pick someone and force 6 shifts
    e.company.availability[emp] = True
    wage, costs = e.company.wages[emp], []
    while e.hours[emp] < 48 and e.day < cfg.n_days:
        if e.action_masks()[emp]:
            _, r, *_ = e.step(emp); costs.append(-r)
        else:
            e.step(e.UNFILLED)
    assert costs[:5] == pytest.approx([wage * 8] * 5)
    assert costs[5] == pytest.approx(wage * 8 * 1.5)     # 6th shift = 8 overtime hours


def test_unfilled_costs_more_than_a_regular_fill(env):
    """Reward-hacking guard (slide 19). NOTE: an overtime fill at the top wage is
    $22 x 8 x 1.5 = $264 > $200, so the guard only holds for regular hours; see README."""
    cfg = env.cfg
    assert cfg.unfilled_penalty > cfg.wage_max * cfg.shift_hours


# ------------------------------------------------------------- information rule
def test_no_lookahead():
    """Observations and rewards through day 3 must not change when days 4-6 change."""
    cfg = EnvConfig()
    base = generate_week(cfg, 123)
    changed = base.copy()
    rng = np.random.default_rng(0)
    changed.actual[4:] = np.clip(changed.actual[4:] + rng.integers(-2, 3, changed.actual[4:].shape),
                                 cfg.min_staff, cfg.max_demand)
    changed.absent[:, 4:] = rng.random(changed.absent[:, 4:].shape) < 0.3
    assert not np.array_equal(base.actual, changed.actual)

    e1, e2 = WorkforceSchedulingEnv(cfg), WorkforceSchedulingEnv(cfg)
    o1, r1, i1 = rollout(e1, GreedyPolicy(), week=base)
    o2, r2, i2 = rollout(e2, GreedyPolicy(), week=changed)
    n_common = int(base.actual[:4].sum())                # steps that happen in days 0-3
    for k in range(n_common):
        assert np.array_equal(o1[k], o2[k]), f"obs differs at step {k} (day {i1[k]['day']})"
        assert r1[k] == r2[k]
    assert e1.budget == e2.budget                        # budget uses the forecast only


def test_obs_within_bounds_over_many_steps(env):
    obs_list, _, _ = rollout(env, RandomPolicy(1), seed=21)
    arr = np.stack(obs_list)
    assert np.isfinite(arr).all() and arr.min() >= 0.0 and arr.max() <= 2.0


# ------------------------------------------------------------------- benchmarks
def test_benchmarks_run_and_rank_sensibly():
    seeds = range(5)
    res = {n: evaluate(cls(), seeds) for n, cls in BENCHMARKS.items()}
    means = {n: np.mean([r["cost"] for r in rows]) for n, rows in res.items()}
    assert all(np.isfinite(v) and v > 0 for v in means.values())
    assert means["greedy"] < means["random"]
    assert means["manager_rule"] < means["random"]


def test_run_episode_returns_summary(env):
    out = run_episode(env, GreedyPolicy(), seed=2)
    assert {"cost", "payroll", "unfilled", "overtime_hours", "return"} <= set(out)
