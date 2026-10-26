"""Non-learning controllers used as baselines. All use the same action set
{0, 2, 4} min as the RL agent, so the comparison is like-for-like."""
from __future__ import annotations

import numpy as np

from .env import BusConfig, Decision


def _nearest_option(hold_min: float, options) -> int:
    """Index of the largest available hold that does not exceed ``hold_min``."""
    idx = 0
    for j, h in enumerate(options):
        if h <= hold_min + 1e-9:
            idx = j
    return idx


def no_holding(cfg: BusConfig):
    """Buses leave as soon as boarding is finished."""
    return lambda dec: 0


def forward_headway_rule(cfg: BusConfig, threshold: float = 0.8):
    """Classic threshold control: hold while the gap to the bus ahead is below
    ``threshold`` x ideal spacing, for roughly the time needed to restore it."""
    seg = cfg.seg_time

    def ctrl(dec: Decision) -> int:
        deficit = (threshold - dec.gap_fwd) * (cfg.n_stops / cfg.n_buses) * seg
        return _nearest_option(max(0.0, deficit), cfg.hold_options) if deficit > 0 else 0
    return ctrl


def two_way_rule(cfg: BusConfig):
    """Even-headway rule: hold for half the imbalance between the gap behind and
    the gap ahead (converted to minutes), so the bus moves toward the midpoint."""
    seg = cfg.seg_time
    spacing = cfg.n_stops / cfg.n_buses

    def ctrl(dec: Decision) -> int:
        imbalance = (dec.gap_bwd - dec.gap_fwd) * spacing * seg / 2.0
        return _nearest_option(max(0.0, imbalance), cfg.hold_options)
    return ctrl


BASELINES = {
    "No holding": no_holding,
    "Forward-headway rule": forward_headway_rule,
    "Two-way rule": two_way_rule,
}
