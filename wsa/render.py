"""Figures: the weekly schedule grid, policy cost comparison, and two
explanatory diagrams (environment loop, observation layout) for the slides.
All functions return the matplotlib Figure and optionally save it to ``path``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from .company import DAY_NAMES, SHIFT_NAMES  # noqa: E402
from .evaluate import summarize  # noqa: E402

NAVY, ORANGE, RED, GREY, GREEN, TEAL = "#005030", "#F47321", "#C0392B", "#D9D9D9", "#2E8B57", "#1F77B4"


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig


def plot_week_schedule(env, path=None, title=None):
    """Employees (rows, sorted by wage) x 21 shifts (columns)."""
    u = env.unwrapped
    cfg, E = u.cfg, u.cfg.n_employees
    D, S = cfg.n_days, cfg.shifts_per_day
    order = np.argsort(u.company.wages)

    # 0 unavailable, 1 available & idle, 2 absent, 3 regular shift, 4 overtime shift
    grid = np.zeros((E, D * S), dtype=int)
    for d in range(D):
        for s in range(S):
            col = d * S + s
            avail = u.company.availability[:, d, s]
            grid[:, col] = np.where(avail, 1, 0)
            grid[u.week.absent[:, d], col] = 2
            assigned = u.schedule[d, s]
            grid[assigned, col] = np.where(u.overtime_booked[d, s, assigned] > 0, 4, 3)
    grid = grid[order]

    cmap = ListedColormap(["#F4F4F4", "#FFFFFF", "#F5B7B1", TEAL, ORANGE])
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    for d in range(1, D):
        ax.axvline(d * S - 0.5, color="black", lw=1.2)
    ax.set_xticks(range(D * S))
    ax.set_xticklabels([SHIFT_NAMES[s][0] for _ in range(D) for s in range(S)], fontsize=8)
    for d in range(D):
        ax.text(d * S + (S - 1) / 2, -1.6, DAY_NAMES[d], ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_yticks(range(E))
    ax.set_yticklabels([f"E{e:02d}  ${u.company.wages[e]:.2f}  {int(u.hours[e]):>2}h"
                        for e in order], fontsize=6.5, family="monospace")
    ax.set_ylabel("Employee (sorted by wage)  ·  hours worked this week")
    ax.tick_params(axis="x", length=0)
    sm = u.summary()
    ttl = title or "One simulated week"
    ax.set_title(f"{ttl}\npayroll \\${sm['payroll']:,.0f} · unfilled slots {sm['unfilled']} "
                 f"(\\${sm['unfilled_cost']:,.0f}) · overtime {sm['overtime_hours']:.0f} h · "
                 f"budget \\${sm['budget']:,.0f} · total cost \\${sm['cost']:,.0f}", fontsize=11, pad=28)
    legend = [Patch(color=TEAL, label="assigned (regular)"), Patch(color=ORANGE, label="assigned (overtime)"),
              Patch(color="#F5B7B1", label="absent"), Patch(color="#FFFFFF", ec="#BBBBBB", label="available, idle"),
              Patch(color="#F4F4F4", ec="#BBBBBB", label="unavailable")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_policy_costs(results: dict[str, list[dict]], path=None, title="Weekly cost by policy"):
    """Mean weekly cost (bars, 95% CI) with worst-5% marker, stacked payroll / unfilled / overrun."""
    summ = {k: summarize(v) for k, v in results.items()}
    names = sorted(summ, key=lambda n: summ[n]["cost_mean"])
    payroll = [summ[n]["payroll_mean"] for n in names]
    unfilled = [summ[n]["cost_mean"] - summ[n]["payroll_mean"] - summ[n]["overrun_penalty_mean"] for n in names]
    overrun = [summ[n]["overrun_penalty_mean"] for n in names]
    ci = [summ[n]["cost_ci95"] for n in names]
    worst = [summ[n]["worst5_mean"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, payroll, color=TEAL, label="payroll")
    ax.bar(x, unfilled, bottom=payroll, color=RED, label="unfilled-slot penalties")
    ax.bar(x, overrun, bottom=np.array(payroll) + np.array(unfilled), color=ORANGE, label="budget-overrun penalty")
    tot = np.array(payroll) + np.array(unfilled) + np.array(overrun)
    ax.errorbar(x, tot, yerr=ci, fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.scatter(x, worst, marker="_", s=600, color="black", label="worst 5% of weeks (mean)", zorder=3)
    for i, t in enumerate(tot):
        ax.text(i, t + max(tot) * 0.015, f"${t:,.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names])
    ax.set_ylabel("Mean weekly cost ($)")
    ax.set_title(f"{title}  (n = {summ[names[0]]['n_weeks']} identical weeks)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_env_loop(path=None):
    """Diagram: what happens inside one environment step."""
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=10, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if tc == "white" else "normal", linespacing=1.4)

    def arrow(p, q, color="black", style="-|>", cs="arc3,rad=0.0"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color,
                                     lw=1.6, connectionstyle=cs))

    box(0.3, 1.6, 2.4, 2.0, "AGENT\nDQN  /  PPO\n\npolicy π(a | s)", NAVY, 11)
    box(3.6, 0.35, 11.1, 4.5, "", "#F1F5F3")
    ax.text(9.15, 4.55, "ENVIRONMENT  ·  WorkforceScheduling-v0  ·  one episode = one week",
            ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)

    steps = [("1  Observe", "260 floats: clock, budget,\nper-employee hours, wage,\neligibility, preference, streak"),
             ("2  Mask", "hide ineligible staff:\nunavailable · absent\ndouble shift · 7th day\nin a row · hour cap"),
             ("3  Assign", "51 actions:\nemployee 0–49\nor leave slot unfilled"),
             ("4  Book", "hours += 8\ncost = wage × 8\n(×1.5 past 40 h)\nor \\$200 if unfilled"),
             ("5  Advance", "next slot → next shift\n→ next morning reveals\ndemand ±20%\nand absences")]
    x0, w, gap = 3.85, 2.05, 0.12
    for i, (t, sub) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(x, 2.55, w, 1.3, t, TEAL if i != 2 else ORANGE, 11)
        ax.text(x + w / 2, 2.35, sub, ha="center", va="top", fontsize=8.2, color="#222222", linespacing=1.35)
        if i < len(steps) - 1:
            arrow((x + w + 0.01, 3.2), (x + w + gap - 0.01, 3.2))

    arrow((2.7, 3.1), (3.85, 3.2), NAVY, cs="arc3,rad=-0.15")
    ax.text(3.2, 3.65, "action aₜ", ha="center", fontsize=9, color=NAVY, fontweight="bold")
    arrow((3.85, 1.0), (2.7, 2.0), RED, cs="arc3,rad=-0.15")
    ax.text(3.15, 1.05, "reward rₜ = −cost\nnext state sₜ₊₁", ha="center", va="top", fontsize=9,
            color=RED, fontweight="bold")
    ax.text(9.15, 0.62, "7 days × 3 shifts ≈ 150 slots per week  ·  end of week: −25% × (payroll − budget)⁺",
            ha="center", fontsize=9.5, color="#333333", style="italic")
    return _save(fig, path)


def plot_observation_layout(env, path=None):
    """Horizontal bar showing the blocks of the observation vector."""
    u = env.unwrapped
    layout = u.observation_layout()
    total = sum(n for _, n in layout)
    colors = [NAVY, TEAL, GREEN, ORANGE, "#8E44AD", RED]
    desc = {"global": "",
            "eligible": "1 if the employee may take\nthis slot (after all 5 rules)",
            "hours": "hours worked so far ÷ 48",
            "wage": "(wage − \\$15) ÷ 7",
            "preference": "likes this shift type\n0 / 0.5 / 1",
            "streak": "consecutive days worked ÷ 6"}
    fig, ax = plt.subplots(figsize=(12, 3.6))
    left = 0
    for (name, n), c in zip(layout, colors):
        ax.barh(0, n, left=left, color=c, edgecolor="white", height=0.6)
        ax.text(left + n / 2, 0, f"{name}\n{n}", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        ax.text(left + n / 2, -0.55, desc[name], ha="center", va="top", fontsize=7.8, color="#222222")
        left += n
    ax.text(0, -1.05, "global (10): day · shift · slot · demand left in shift · filled in shift · "
            "week progress · payroll ÷ budget · budget left · unfilled so far · days left",
            ha="left", va="top", fontsize=7.8, color=NAVY)
    ax.set_xlim(0, total)
    ax.set_ylim(-1.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel(f"observation index  (total {total} floats in [0, 2], one Box space)")
    ax.set_title(f"What the agent sees at every slot: {total} numbers = 10 global + 5 × {u.cfg.n_employees} employees",
                 fontsize=11, color=NAVY, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curve(timesteps, values, path=None, label="validation cost", title="Learning curve"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timesteps, values, color=TEAL, lw=2, label=label)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean weekly cost on validation weeks ($)")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)
