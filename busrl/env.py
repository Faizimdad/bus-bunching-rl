"""Loop-route bus simulator in which bunching emerges from dwell-time feedback.

Each bus travels a circular route with ``n_stops`` stops. Passengers arrive at
every stop as a Poisson process; dwell time grows with the number of boarding and
alighting passengers. A bus that falls behind finds more waiting passengers,
dwells longer and falls further behind, while its follower catches up. This
positive feedback is what causes buses to bunch.

When a bus has finished serving a stop it reaches a *decision point*; the
controller chooses how long to hold it (0, 2 or 4 min) before it departs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MOVING, DWELL, HOLD = 0, 1, 2


@dataclass
class BusConfig:
    n_stops: int = 12            # stops on the loop
    n_buses: int = 4             # buses in service
    seg_time: float = 2.5        # mean run time between consecutive stops (min)
    seg_cv: float = 0.25         # coefficient of variation of segment run time
    arrival_rate: float = 0.6    # passenger arrivals per stop (pax/min)
    board_time: float = 0.08     # boarding time per passenger (min, ~5 s)
    alight_time: float = 0.04    # alighting time per passenger (min)
    door_time: float = 0.25      # fixed door open/close overhead (min)
    alight_prob: float = 0.15    # each on-board passenger alights w.p. this at a stop
    capacity: int = 70           # bus capacity (pax)
    horizon: float = 240.0       # simulated minutes per episode
    warmup: float = 30.0         # minutes discarded before metrics are recorded
    dt: float = 0.1              # simulation time step (min)
    hold_options: tuple = (0.0, 2.0, 4.0)
    init_jitter: float = 0.15    # initial spacing perturbation (fraction of ideal)


@dataclass
class Decision:
    """Information handed to the controller at a decision point."""
    bus: int
    stop: int
    time: float
    gap_fwd: float   # spatial gap to bus ahead / ideal spacing (1.0 = perfect)
    gap_bwd: float   # spatial gap to bus behind / ideal spacing
    load: int        # passengers on board
    capacity: int


@dataclass
class EpisodeStats:
    headways: list = field(default_factory=list)   # arrival headways at stops (min)
    wait_pax_min: float = 0.0                       # integral of queue length
    arrivals: int = 0                               # passengers who arrived
    boardings: int = 0
    held_pax_min: float = 0.0                       # on-board pax x hold minutes
    holds: list = field(default_factory=list)       # hold chosen at each decision
    trajectory: list = field(default_factory=list)  # (t, bus, cumulative position)


class BusRouteEnv:
    def __init__(self, cfg: BusConfig | None = None, seed: int | None = None,
                 record_trajectory: bool = False):
        self.cfg = cfg or BusConfig()
        self.rng = np.random.default_rng(seed)
        self.record_trajectory = record_trajectory
        self.ideal_spacing = self.cfg.n_stops / self.cfg.n_buses

    # ------------------------------------------------------------------ setup
    def reset(self):
        c = self.cfg
        M = c.n_buses
        base = np.arange(M) * self.ideal_spacing
        jitter = self.rng.uniform(-c.init_jitter, c.init_jitter, M) * self.ideal_spacing
        # buses start exactly at a stop just before it, so they "arrive" immediately
        self.pos = (base + jitter) % c.n_stops          # position on loop in stop units
        self.laps = np.zeros(M)                          # for unwrapped trajectories
        self.phase = np.full(M, MOVING)
        self.speed = np.array([1.0 / self._sample_seg() for _ in range(M)])
        self.timer = np.zeros(M)                         # remaining dwell/hold time
        self.load = self.rng.integers(5, 20, M)
        self.cur_stop = np.full(M, -1)
        self.queue = self.rng.poisson(c.arrival_rate * self.ideal_spacing * c.seg_time / 2,
                                      c.n_stops).astype(int)
        self.last_arrival = np.full(c.n_stops, np.nan)
        self.next_stop = (np.floor(self.pos).astype(int) + 1) % c.n_stops
        self.t = 0.0
        self.stats = EpisodeStats()
        self.pending_hold = np.zeros(M)

    def _sample_seg(self) -> float:
        c = self.cfg
        sigma = np.sqrt(np.log(1 + c.seg_cv ** 2))
        mu = np.log(c.seg_time) - sigma ** 2 / 2
        return float(self.rng.lognormal(mu, sigma))

    # --------------------------------------------------------------- geometry
    def _gaps(self, i: int):
        """Spatial gaps (in units of ideal spacing) to the bus ahead and behind."""
        n = self.cfg.n_stops
        d = (self.pos - self.pos[i]) % n
        d[i] = np.inf
        ahead = d.min()
        dback = (self.pos[i] - self.pos) % n
        dback[i] = np.inf
        behind = dback.min()
        return ahead / self.ideal_spacing, behind / self.ideal_spacing

    def _measuring(self) -> bool:
        return self.t >= self.cfg.warmup

    # -------------------------------------------------------------- boarding
    def _board(self, i: int, k: int) -> int:
        room = self.cfg.capacity - self.load[i]
        n = int(min(self.queue[k], room))
        self.queue[k] -= n
        self.load[i] += n
        if self._measuring():
            self.stats.boardings += n
        return n

    # -------------------------------------------------------------- run loop
    def run(self, controller, learner=None):
        """Simulate one episode.

        ``controller(decision) -> action index`` picks the hold.
        ``learner`` (optional) receives ``observe(bus, decision, action, hold_cost)``
        callbacks so a Q-learning agent can form transitions.
        """
        self.reset()
        c = self.cfg
        n_steps = int(round(c.horizon / c.dt))
        for _ in range(n_steps):
            self.t += c.dt
            arrivals = self.rng.poisson(c.arrival_rate * c.dt, c.n_stops)
            self.queue += arrivals
            if self._measuring():
                self.stats.arrivals += int(arrivals.sum())
                self.stats.wait_pax_min += float(self.queue.sum()) * c.dt

            for i in range(c.n_buses):
                if self.phase[i] == MOVING:
                    self._move(i)
                elif self.phase[i] == DWELL:
                    self.timer[i] -= c.dt
                    if self.timer[i] <= 0:
                        k = self.cur_stop[i]
                        extra = self._board(i, k)        # passengers who came during dwell
                        if extra > 0:
                            self.timer[i] = extra * c.board_time
                        else:
                            self._decide(i, controller, learner)
                elif self.phase[i] == HOLD:
                    self.timer[i] -= c.dt
                    if self.timer[i] <= 0:
                        k = self.cur_stop[i]
                        extra = self._board(i, k)
                        if extra > 0:                    # top up, then leave
                            self.phase[i] = DWELL
                            self.timer[i] = extra * c.board_time
                            self.pending_hold[i] = -1    # marks "already decided"
                        else:
                            self._depart(i)

            if self.record_trajectory and int(round(self.t / c.dt)) % 5 == 0:
                for i in range(c.n_buses):
                    self.stats.trajectory.append((self.t, i, self.laps[i] * c.n_stops + self.pos[i]))
        if learner is not None:
            learner.end_episode()
        return self.stats

    def _move(self, i: int):
        c = self.cfg
        new = self.pos[i] + self.speed[i] * c.dt
        target = self.next_stop[i] if self.next_stop[i] != 0 else c.n_stops
        if new >= target:
            k = self.next_stop[i]
            self.pos[i] = float(k)
            if k == 0:
                self.laps[i] += 1
            self._arrive(i, k)
        else:
            self.pos[i] = new

    def _arrive(self, i: int, k: int):
        c = self.cfg
        if self._measuring() and not np.isnan(self.last_arrival[k]):
            self.stats.headways.append(self.t - self.last_arrival[k])
        self.last_arrival[k] = self.t
        n_alight = int(self.rng.binomial(self.load[i], c.alight_prob))
        self.load[i] -= n_alight
        n_board = self._board(i, k)
        self.cur_stop[i] = k
        self.phase[i] = DWELL
        self.pending_hold[i] = 0
        self.timer[i] = c.door_time + c.board_time * n_board + c.alight_time * n_alight

    def _decide(self, i: int, controller, learner):
        if self.pending_hold[i] == -1:   # boarding top-up after a hold: just leave
            self._depart(i)
            return
        gf, gb = self._gaps(i)
        dec = Decision(i, int(self.cur_stop[i]), self.t, gf, gb, int(self.load[i]),
                       self.cfg.capacity)
        a = int(controller(dec))
        hold = self.cfg.hold_options[a]
        hold_cost_pax = hold * self.load[i]
        if self._measuring():
            self.stats.holds.append(hold)
            self.stats.held_pax_min += hold_cost_pax
        if learner is not None:
            learner.observe(i, dec, a, hold)
        if hold > 0:
            self.phase[i] = HOLD
            self.timer[i] = hold
        else:
            self._depart(i)

    def _depart(self, i: int):
        self.phase[i] = MOVING
        self.speed[i] = 1.0 / self._sample_seg()
        self.next_stop[i] = (self.cur_stop[i] + 1) % self.cfg.n_stops
        self.pos[i] = float(self.cur_stop[i]) + 1e-6
        self.pending_hold[i] = 0


def summarize(stats: EpisodeStats, cfg: BusConfig | None = None) -> dict:
    cfg = cfg or BusConfig()
    measured_hours = (cfg.horizon - cfg.warmup) / 60.0
    h = np.asarray(stats.headways)
    mean_h = h.mean() if len(h) else np.nan
    boardings = max(stats.boardings, 1)
    avg_wait = stats.wait_pax_min / max(stats.arrivals, 1)
    hold_delay = stats.held_pax_min / boardings
    return {
        "avg_wait_min": avg_wait,
        "headway_mean_min": mean_h,
        "headway_cv": h.std() / mean_h if len(h) else np.nan,
        "bunching_pct": 100.0 * float(np.mean(h < 0.25 * mean_h)) if len(h) else np.nan,
        "mean_hold_min": float(np.mean(stats.holds)) if stats.holds else 0.0,
        "hold_delay_per_pax_min": hold_delay,
        "total_cost_per_pax_min": avg_wait + hold_delay,
        "boardings_per_hour": stats.boardings / measured_hours,
    }
