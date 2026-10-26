"""Evaluate the trained agent against the baselines on held-out seeds.

Usage:  python evaluate.py [--qtable results/q_table.npz] [--n_seeds 100]
Outputs results/metrics.json, results/metrics_table.md, results/sensitivity.json
and pickled trajectories/headways used by plots.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from busrl import BASELINES, BusConfig, BusRouteEnv, QLearningAgent, summarize

TEST_SEED_BASE = 100_000          # disjoint from training (0..) and validation (50_000..)
METRICS = ["avg_wait_min", "hold_delay_per_pax_min", "total_cost_per_pax_min",
           "headway_cv", "bunching_pct", "mean_hold_min", "boardings_per_hour"]


def controllers(cfg: BusConfig, agent: QLearningAgent):
    ctrls = {name: make(cfg) for name, make in BASELINES.items()}
    ctrls["Q-learning (ours)"] = agent.act
    return ctrls


def evaluate(cfg, ctrl, seeds, keep=False):
    rows, headways = [], []
    for s in seeds:
        st = BusRouteEnv(cfg, seed=s).run(ctrl)
        rows.append(summarize(st, cfg))
        if keep:
            headways.extend(st.headways)
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
           for k in METRICS}
    return agg, headways


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qtable", default="results/q_table.npz")
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    agent = QLearningAgent()
    agent.load(args.qtable)
    agent.training = False
    seeds = range(TEST_SEED_BASE, TEST_SEED_BASE + args.n_seeds)

    # ---- main comparison at the training demand level
    cfg = BusConfig()
    results, headways = {}, {}
    for name, ctrl in controllers(cfg, agent).items():
        results[name], headways[name] = evaluate(cfg, ctrl, seeds, keep=True)
        m = results[name]
        print(f"{name:22s} wait={m['avg_wait_min'][0]:.2f}  hold-delay={m['hold_delay_per_pax_min'][0]:.2f}"
              f"  total={m['total_cost_per_pax_min'][0]:.2f}  CV={m['headway_cv'][0]:.3f}"
              f"  bunched={m['bunching_pct'][0]:.1f}%")

    # ---- generalisation: same policy, different demand (never seen in training)
    sens = {}
    for lam in [0.4, 0.5, 0.6, 0.7, 0.8]:
        c = BusConfig(arrival_rate=lam)
        sens[lam] = {name: evaluate(c, ctrl, range(TEST_SEED_BASE, TEST_SEED_BASE + 40))[0]
                     for name, ctrl in controllers(c, agent).items()}

    # ---- one representative episode with trajectories for the time-space plot
    traj = {}
    for name in ["No holding", "Q-learning (ours)"]:
        ctrl = controllers(cfg, agent)[name]
        st = BusRouteEnv(cfg, seed=TEST_SEED_BASE + 7, record_trajectory=True).run(ctrl)
        traj[name] = st.trajectory

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "sensitivity.json"), "w") as f:
        json.dump({str(k): v for k, v in sens.items()}, f, indent=2)
    with open(os.path.join(args.out, "eval_raw.pkl"), "wb") as f:
        pickle.dump({"headways": headways, "trajectories": traj}, f)

    hdr = ("| Controller | Avg wait (min) | Hold delay/pax (min) | Total cost/pax (min) "
           "| Headway CV | Bunched (%) | Mean hold (min) |\n|---|---|---|---|---|---|---|\n")
    lines = [f"| {n} | {m['avg_wait_min'][0]:.2f} ± {m['avg_wait_min'][1]:.2f} | "
             f"{m['hold_delay_per_pax_min'][0]:.2f} | {m['total_cost_per_pax_min'][0]:.2f} | "
             f"{m['headway_cv'][0]:.3f} | {m['bunching_pct'][0]:.1f} | {m['mean_hold_min'][0]:.2f} |"
             for n, m in results.items()]
    with open(os.path.join(args.out, "metrics_table.md"), "w") as f:
        f.write(hdr + "\n".join(lines) + "\n")
    print(f"\nwrote metrics to {args.out}/")


if __name__ == "__main__":
    main()
