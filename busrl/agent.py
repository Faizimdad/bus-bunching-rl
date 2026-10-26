"""Tabular Q-learning holding agent (one Q-table shared by all buses).

Holding is a semi-Markov decision problem: decisions happen only when a bus is
ready to leave a stop, and the consequence of a hold is observed at that bus's
*next* decision point. The agent therefore keeps, per bus, the last
(state, action, holding cost) and closes the transition when the same bus
reaches its next stop.
"""
from __future__ import annotations

import numpy as np

from .env import Decision

# Bin edges for spacing ratios (1.0 = perfectly even spacing).
GAP_EDGES = np.array([0.3, 0.6, 0.85, 1.15, 1.4, 1.7])
LOAD_EDGES = np.array([0.3, 0.6])  # fraction of capacity


def n_gap_bins() -> int:
    return len(GAP_EDGES) + 1


def n_load_bins() -> int:
    return len(LOAD_EDGES) + 1


def n_states() -> int:
    return n_gap_bins() * n_gap_bins() * n_load_bins()


def encode(dec: Decision) -> int:
    f = int(np.searchsorted(GAP_EDGES, dec.gap_fwd))
    b = int(np.searchsorted(GAP_EDGES, dec.gap_bwd))
    l = int(np.searchsorted(LOAD_EDGES, dec.load / dec.capacity))
    return (f * n_gap_bins() + b) * n_load_bins() + l


def spacing_cost(dec: Decision) -> float:
    """Squared deviation of both gaps from the ideal spacing."""
    gf = min(dec.gap_fwd, 2.5)
    gb = min(dec.gap_bwd, 2.5)
    return 0.5 * ((gf - 1.0) ** 2 + (gb - 1.0) ** 2)


def holding_cost(hold_min: float, load: int, capacity: int) -> float:
    """Holding delays the bus itself (fixed part) and everyone on board."""
    return hold_min * (0.5 + load / capacity)


class QLearningAgent:
    def __init__(self, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.9,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_eps: int = 1200,
                 w_space: float = 1.0, w_hold: float = 0.05, seed: int = 0):
        self.nA = n_actions
        self.Q = np.zeros((n_states(), n_actions))
        self.visits = np.zeros((n_states(), n_actions), dtype=int)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_eps = eps_start, eps_end, eps_decay_eps
        self.w_space, self.w_hold = w_space, w_hold
        self.rng = np.random.default_rng(seed)
        self.episode = 0
        self.training = True
        self._pending: dict[int, tuple[int, int, float]] = {}
        self.ep_return = 0.0
        self.ep_decisions = 0

    # ----------------------------------------------------------- exploration
    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.episode / self.eps_decay_eps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, dec: Decision) -> int:
        s = encode(dec)
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nA))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        # break ties toward the smallest hold (no gratuitous holding)
        return int(best[0])

    # -------------------------------------------------------------- learning
    def reward(self, prev_hold_cost: float, dec_next: Decision) -> float:
        return -self.w_space * spacing_cost(dec_next) - self.w_hold * prev_hold_cost

    def observe(self, bus: int, dec: Decision, action: int, hold_min: float):
        s = encode(dec)
        if bus in self._pending:
            ps, pa, pcost = self._pending[bus]
            r = self.reward(pcost, dec)
            self.ep_return += r
            self.ep_decisions += 1
            if self.training:
                target = r + self.gamma * self.Q[s].max()
                self.Q[ps, pa] += self.alpha * (target - self.Q[ps, pa])
                self.visits[ps, pa] += 1
        self._pending[bus] = (s, action, holding_cost(hold_min, dec.load, dec.capacity))

    def end_episode(self):
        self._pending.clear()
        self.episode += 1

    def start_episode(self):
        self.ep_return = 0.0
        self.ep_decisions = 0

    def save(self, path: str):
        np.savez(path, Q=self.Q, visits=self.visits)

    def load(self, path: str):
        d = np.load(path)
        self.Q, self.visits = d["Q"], d["visits"]
