"""Ablations: (a) robustness across training seeds, (b) holding-cost weight w_hold.

Usage:  python ablation.py [--episodes 3000] [--workers 4]
Outputs results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from busrl import BusConfig, BusRouteEnv, QLearningAgent, summarize
from evaluate import TEST_SEED_BASE

N_TEST = 50


def train_and_test(job):
    seed, w_hold, episodes = job
    cfg = BusConfig()
    agent = QLearningAgent(w_hold=w_hold, eps_decay_eps=int(0.7 * episodes), seed=seed)
    for ep in range(episodes):
        BusRouteEnv(cfg, seed=seed * 1_000_000 + ep).run(agent.act, learner=agent)
    agent.training = False
    rows = [summarize(BusRouteEnv(cfg, seed=s).run(agent.act), cfg)
            for s in range(TEST_SEED_BASE, TEST_SEED_BASE + N_TEST)]
    return {"seed": seed, "w_hold": w_hold,
            **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    jobs = [(s, 0.05, args.episodes) for s in [1, 2, 3, 4]]                 # seed robustness
    jobs += [(1, w, args.episodes) for w in [0.0, 0.02, 0.1, 0.2, 0.5]]    # cost weight sweep
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(train_and_test, jobs))
    for r in res:
        print(f"seed={r['seed']} w_hold={r['w_hold']:<5} wait={r['avg_wait_min']:.2f} "
              f"hold={r['hold_delay_per_pax_min']:.2f} total={r['total_cost_per_pax_min']:.2f} "
              f"CV={r['headway_cv']:.3f}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
