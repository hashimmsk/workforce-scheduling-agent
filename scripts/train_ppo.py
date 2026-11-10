"""Maskable PPO training skeleton (owner: Shaz). Hyper-parameters from kickoff
slide 15: lr 3e-4, 4,096 steps per update, 10 epochs, clip 0.2, two 128-unit layers.

Uses sb3_contrib.MaskablePPO with the environment's ``action_masks()`` so the
policy only ever samples eligible employees (or "unfilled").

    python scripts/train_ppo.py --timesteps 2000000 --seed 0
    python scripts/train_ppo.py --timesteps 8192 --seed 0 --eval-weeks 10     # smoke test
"""
import argparse
import pathlib

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from common import eval_against_benchmarks, set_global_seeds, training_env


def mask_fn(env):
    return env.action_masks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-weeks", type=int, default=50)
    ap.add_argument("--out", default="models")
    ap.add_argument("--tensorboard", action="store_true", help="log to runs/ppo (pip install tensorboard)")
    args = ap.parse_args()

    set_global_seeds(args.seed)
    env = ActionMasker(training_env(args.seed), mask_fn)
    model = MaskablePPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=4_096,
        batch_size=256,
        n_epochs=10,
        gamma=1.0,                              # finite week, undiscounted cost
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=1,
        tensorboard_log="runs/ppo" if args.tensorboard else None,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"ppo_seed{args.seed}"
    model.save(path)
    print(f"\nsaved {path}.zip\n")
    print(eval_against_benchmarks(model, masked=True, n_weeks=args.eval_weeks))


if __name__ == "__main__":
    main()
