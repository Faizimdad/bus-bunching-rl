"""Generate all figures in results/figures from the outputs of train/evaluate/ablation.

Usage:  python plots.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from busrl import BusConfig  # noqa: E402
from busrl.agent import GAP_EDGES, n_gap_bins, n_load_bins  # noqa: E402

R = "results"
F = os.path.join(R, "figures")
os.makedirs(F, exist_ok=True)

# Fixed colour per controller (validated categorical palette; colour follows entity).
COLORS = {"Q-learning (ours)": "#2a78d6", "No holding": "#eb6834",
          "Forward-headway rule": "#1baf7a", "Two-way rule": "#eda100"}
SHORT = {"Q-learning (ours)": "Q-learning", "No holding": "No hold",
         "Forward-headway rule": "Fwd rule", "Two-way rule": "Two-way"}
ORDER = ["No holding", "Forward-headway rule", "Two-way rule", "Q-learning (ours)"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(os.path.join(F, name + ".pdf"))
    fig.savefig(os.path.join(F, name + ".png"))
    plt.close(fig)


metrics = json.load(open(os.path.join(R, "metrics.json")))
raw = pickle.load(open(os.path.join(R, "eval_raw.pkl"), "rb"))
cfg = BusConfig()

# ---------------------------------------------------------------- 1. time-space
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, name in zip(axes, ["No holding", "Q-learning (ours)"]):
    tr = np.array(raw["trajectories"][name])
    for b in range(cfg.n_buses):
        sel = tr[tr[:, 1] == b]
        t, x = sel[:, 0], sel[:, 2] % cfg.n_stops
        x = np.where(np.abs(np.diff(x, prepend=x[0])) > cfg.n_stops / 2, np.nan, x)
        mask = (t >= 60) & (t <= 180)
        ax.plot(t[mask], x[mask], lw=1.2, label=f"bus {b + 1}",
                color=["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"][b])  # colour = bus identity
    ax.set_title(f"{SHORT[name]}: bus trajectories", color=INK)
    ax.set_xlabel("time (min)")
    ax.set_ylim(0, cfg.n_stops)
axes[0].set_ylabel("stop index on loop")
fig.tight_layout(rect=(0, 0.08, 1, 1))
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=4, loc="lower center")
save(fig, "time_space")

# ------------------------------------------------------------ 2. learning curve
rows = list(csv.DictReader(open(os.path.join(R, "training_log.csv"))))
ep = np.array([int(r["episode"]) + 1 for r in rows])
tw = np.array([float(r["avg_wait_min"]) for r in rows])
gi = [i for i, r in enumerate(rows) if r["greedy_wait_min"]]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
k = 50
smooth = np.convolve(tw, np.ones(k) / k, mode="valid")
axes[0].plot(ep[k - 1:], smooth, color=MUTED, lw=1.2, label="training (ε-greedy, 50-ep avg)")
axes[0].plot(ep[gi], [float(rows[i]["greedy_wait_min"]) for i in gi], color=COLORS["Q-learning (ours)"],
             lw=2, marker="o", ms=3, label="greedy policy (validation)")
for n in ["No holding", "Two-way rule"]:
    axes[0].axhline(metrics[n]["avg_wait_min"][0], color=COLORS[n], ls="--", lw=1.2, label=SHORT[n])
axes[0].set_xlabel("episode"); axes[0].set_ylabel("avg passenger wait (min)")
axes[0].set_title("Learning curve: waiting time")
axes[0].set_ylim(4, 21)
axes[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5)
rew = np.array([float(r["mean_reward"]) for r in rows])
axes[1].plot(ep[k - 1:], np.convolve(rew, np.ones(k) / k, mode="valid"), color=COLORS["Q-learning (ours)"], lw=1.5)
ax2 = axes[1]
ax2.set_xlabel("episode"); ax2.set_ylabel("mean reward per decision")
ax2.set_title("Learning curve: reward (50-ep avg)")
fig.tight_layout()
save(fig, "learning_curve")

# ---------------------------------------------------------- 3. comparison bars
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
x = np.arange(len(ORDER))
wait = [metrics[n]["avg_wait_min"][0] for n in ORDER]
hold = [metrics[n]["hold_delay_per_pax_min"][0] for n in ORDER]
cols = [COLORS[n] for n in ORDER]
axes[0].bar(x, wait, color=cols, width=0.62, edgecolor="white", linewidth=1)
axes[0].bar(x, hold, bottom=wait, color=cols, alpha=0.4, width=0.62, hatch="////",
            edgecolor="white", linewidth=1)
for xi, w, h in zip(x, wait, hold):
    axes[0].text(xi, w + h + 0.2, f"{w + h:.1f}", ha="center", fontsize=7, color=INK)
axes[0].set_title("Cost per passenger (min)\nsolid = wait, hatched = hold delay")
axes[0].set_ylim(0, max(np.add(wait, hold)) * 1.15)
for ax, key, title, fmt in [(axes[1], "headway_cv", "Headway CV (lower = even)", "{:.2f}"),
                            (axes[2], "bunching_pct", "Bunched arrivals (%)", "{:.1f}")]:
    v = [metrics[n][key][0] for n in ORDER]
    ax.bar(x, v, color=cols, width=0.62, edgecolor="white", linewidth=1)
    for xi, vi in zip(x, v):
        ax.text(xi, vi + max(v) * 0.02, fmt.format(vi), ha="center", fontsize=7, color=INK)
    ax.set_title(title)
    ax.set_ylim(0, max(v) * 1.15)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in ORDER], rotation=20)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "comparison")

# ------------------------------------------------------ 4. headway distribution
fig, ax = plt.subplots(figsize=(3.4, 2.2))
bins = np.linspace(0, 3.5, 36)
for n in ORDER:
    h = np.array(raw["headways"][n])
    ax.hist(h / h.mean(), bins=bins, histtype="step", lw=1.6, density=True,
            color=COLORS[n], label=SHORT[n])
ax.axvline(1, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("headway / mean headway"); ax.set_ylabel("density")
ax.set_title("Headway distribution at stops")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "headway_dist")

# ------------------------------------------------------------ 5. policy heatmap
Q = np.load(os.path.join(R, "q_table.npz"))
Qt, visits = Q["Q"], Q["visits"]
G, L = n_gap_bins(), n_load_bins()
labels = ["<.3", ".3-.6", ".6-.85", ".85-1.15", "1.15-1.4", "1.4-1.7", ">1.7"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, li, tl in zip(axes, [0, 2], ["low load (<30%)", "high load (>60%)"]):
    pol = np.full((G, G), np.nan)
    for f in range(G):
        for b in range(G):
            s = (f * G + b) * L + li
            if visits[s].sum() > 0:
                pol[f, b] = cfg.hold_options[int(np.argmax(Qt[s]))]
    im = ax.imshow(pol, origin="lower", cmap="Blues", vmin=-0.5, vmax=4.5)
    for f in range(G):
        for b in range(G):
            if not np.isnan(pol[f, b]):
                ax.text(b, f, f"{pol[f, b]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if pol[f, b] >= 3 else INK)
    ax.set_xticks(range(G)); ax.set_xticklabels(labels, rotation=40)
    ax.set_yticks(range(G)); ax.set_yticklabels(labels)
    ax.set_xlabel("gap behind / ideal"); ax.set_ylabel("gap ahead / ideal")
    ax.set_title(f"Learned hold (min), {tl}")
    ax.grid(False)
fig.tight_layout()
save(fig, "policy")

# ------------------------------------------------------ 6. demand sensitivity
sens = json.load(open(os.path.join(R, "sensitivity.json")))
lams = sorted(sens, key=float)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for n in ORDER:
    ax.plot([float(l) for l in lams], [sens[l][n]["total_cost_per_pax_min"][0] for l in lams],
            marker="o", ms=4, lw=1.8, color=COLORS[n], label=SHORT[n])
ax.axvline(0.6, color=MUTED, lw=0.8, ls=":")
ax.text(0.607, 10.6, "trained\nhere", fontsize=6.5, color=MUTED, va="top")
ax.set_xlabel("passenger arrival rate (pax/min/stop)")
ax.set_ylabel("total cost per pax (min)")
ax.set_title("Generalisation to unseen demand")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
save(fig, "sensitivity")

# ---------------------------------------------------------------- 7. ablation
ab_path = os.path.join(R, "ablation.json")
if os.path.exists(ab_path):
    ab = json.load(open(ab_path))
    sweep = sorted([r for r in ab if r["seed"] == 1], key=lambda r: r["w_hold"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = [r["w_hold"] for r in sweep]
    ax.plot(w, [r["avg_wait_min"] for r in sweep], marker="o", ms=4, color=COLORS["Q-learning (ours)"], label="wait")
    ax.plot(w, [r["hold_delay_per_pax_min"] for r in sweep], marker="s", ms=4, color=COLORS["No holding"], label="hold delay")
    ax.plot(w, [r["total_cost_per_pax_min"] for r in sweep], marker="^", ms=4, color=INK, label="total")
    ax.set_xscale("symlog", linthresh=0.02)
    ax.set_xlabel("holding-cost weight $w_h$"); ax.set_ylabel("min per passenger")
    ax.set_title("Reward-weight ablation")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "ablation")

print("figures written to", F)
