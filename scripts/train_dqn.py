"""DQN training skeleton (owner: Megan). Hyper-parameters from kickoff slide 15:
lr 1e-4, replay buffer 200k, batch 128, epsilon 1.0 -> 0.05, two 128-unit layers.

Stable-Baselines3's DQN has no action masking, so an ineligible pick is scored
as an unfilled slot (-$200) by the environment. The mask is still in the
observation (the "eligible" block), so the network can learn to avoid it.

    python scripts/train_dqn.py --timesteps 1000000 --seed 0
    python scripts/train_dqn.py --timesteps 20000 --seed 0 --eval-weeks 10   # smoke test
"""
import argparse
import pathlib

from stable_baselines3 import DQN

from common import eval_against_benchmarks, set_global_seeds, training_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-weeks", type=int, default=50)
    ap.add_argument("--out", default="models")
    ap.add_argument("--tensorboard", action="store_true", help="log to runs/dqn (pip install tensorboard)")
    args = ap.parse_args()

    set_global_seeds(args.seed)
    env = training_env(args.seed)
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=128,
        learning_starts=5_000,
        gamma=1.0,                              # finite week, undiscounted cost
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.5,
        target_update_interval=2_000,
        train_freq=4,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=1,
        tensorboard_log="runs/dqn" if args.tensorboard else None,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"dqn_seed{args.seed}"
    model.save(path)
    print(f"\nsaved {path}.zip\n")
    print(eval_against_benchmarks(model, masked=False, n_weeks=args.eval_weeks))


if __name__ == "__main__":
    main()
