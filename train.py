"""Train the Q-learning holding agent.

Usage:  python train.py [--episodes 3000] [--seed 0]
Outputs results/q_table.npz and results/training_log.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from busrl import BusConfig, BusRouteEnv, QLearningAgent, summarize

TRAIN_SEED_BASE = 0          # training episodes use seeds 0, 1, 2, ...
VALID_SEEDS = range(50_000, 50_010)   # held-out seeds for greedy checkpoints


def greedy_eval(agent: QLearningAgent, cfg: BusConfig, seeds) -> dict:
    agent.training = False
    rows = []
    for s in seeds:
        env = BusRouteEnv(cfg, seed=s)
        agent.start_episode()
        st = env.run(agent.act, learner=agent)
        rows.append(summarize(st, cfg))
    agent.training = True
    agent.episode -= len(seeds)   # evaluation episodes do not advance epsilon
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--w_hold", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = BusConfig()
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, w_hold=args.w_hold,
                           eps_decay_eps=int(0.7 * args.episodes), seed=args.seed)

    log = []
    for ep in range(args.episodes):
        env = BusRouteEnv(cfg, seed=TRAIN_SEED_BASE + args.seed * 1_000_000 + ep)
        agent.start_episode()
        st = env.run(agent.act, learner=agent)
        m = summarize(st, cfg)
        row = {"episode": ep, "epsilon": agent.epsilon,
               "mean_reward": agent.ep_return / max(agent.ep_decisions, 1),
               "avg_wait_min": m["avg_wait_min"], "headway_cv": m["headway_cv"],
               "greedy_wait_min": "", "greedy_cv": ""}
        if (ep + 1) % args.eval_every == 0:
            g = greedy_eval(agent, cfg, VALID_SEEDS)
            row["greedy_wait_min"], row["greedy_cv"] = g["avg_wait_min"], g["headway_cv"]
            print(f"ep {ep+1:5d}  eps={agent.epsilon:.2f}  "
                  f"train wait={m['avg_wait_min']:.2f}  greedy wait={g['avg_wait_min']:.2f}  "
                  f"greedy CV={g['headway_cv']:.3f}")
        log.append(row)

    agent.save(os.path.join(args.out, "q_table.npz"))
    with open(os.path.join(args.out, "training_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    print(f"saved Q-table and log to {args.out}/")


if __name__ == "__main__":
    main()
