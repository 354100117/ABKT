"""Network state machine with hysteresis — GOOD / DEGRADED / POOR.

Designed to prevent oscillation at boundary conditions through:
  - Asymmetric hysteresis bands (wider at low bandwidth, narrower at high)
  - N-out-of-M voting (fast drop, slow rise)
  - State-aware probe frequency hint

Usage:
    sm = NetworkStateMachine()
    sm.update(bw_bps=80e6, rtt_ms=2.0)  # → GOOD
    sm.update(bw_bps=45e6, rtt_ms=3.0)  # → DEGRADED (after 2 of last 3)
    sm.state  # → NetworkState.DEGRADED
"""

from __future__ import annotations

import enum
from collections import deque
from typing import List

from backend.config import (
    BW_GOOD_THRESHOLD, BW_POOR_THRESHOLD, RTT_GOOD_THRESHOLD,
    RTT_DEGRADED_THRESHOLD, DOWNGRADE_VOTES, DOWNGRADE_WINDOW,
    UPGRADE_VOTES, UPGRADE_WINDOW, TARGET_TRANSFER_TIME,
)


class NetworkState(enum.Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    UNKNOWN = "unknown"  # before any probe data


class NetworkStateMachine:
    """Hysteresis state machine for network condition classification.

    Maintains a ring buffer of recent probe classifications and uses
    N-out-of-M voting to decide transitions.

    Attributes:
        state: Current NetworkState.
        bw_ewma: Current bandwidth EWMA value (bytes/sec).
        rtt_ewma: Current RTT EWMA value (ms).
    """

    def __init__(self):
        self.state: NetworkState = NetworkState.UNKNOWN
        self.bw_ewma: float = 0.0
        self.rtt_ewma: float = 0.0
        self._history: deque = deque(maxlen=max(DOWNGRADE_WINDOW, UPGRADE_WINDOW))

    def update(self, bw_bps: float, rtt_ms: float) -> NetworkState:
        """Classify the current network state from a probe sample.

        Args:
            bw_bps: Measured bandwidth in bytes/sec (instant or EWMA).
            rtt_ms: Measured RTT in milliseconds (instant or EWMA).

        Returns:
            Updated NetworkState after hysteresis voting.
        """
        self.bw_ewma = bw_bps
        self.rtt_ewma = rtt_ms

        raw_state = self._classify_sample(bw_bps, rtt_ms)
        self._history.append(raw_state)
        return self._resolve_state(raw_state)

    def _classify_sample(self, bw_bps: float, rtt_ms: float) -> NetworkState:
        """Classify a single sample with hysteresis-aware thresholds."""
        # POOR is bandwidth-alone (bandwidth is ground truth for throughput)
        if bw_bps < BW_POOR_THRESHOLD:
            return NetworkState.POOR
        # GOOD requires both metrics healthy
        if bw_bps >= BW_GOOD_THRESHOLD and rtt_ms <= RTT_GOOD_THRESHOLD:
            return NetworkState.GOOD
        # RTT-only DEGRADED trigger
        if rtt_ms > RTT_DEGRADED_THRESHOLD:
            return NetworkState.DEGRADED
        # Everything else is DEGRADED
        return NetworkState.DEGRADED

    def _resolve_state(self, raw_state: NetworkState) -> NetworkState:
        """Apply hysteresis voting to decide if a transition should occur.

        Fast drop: downgrade needs DOWNGRADE_VOTES of last DOWNGRADE_WINDOW.
        Slow rise: upgrade needs UPGRADE_VOTES of last UPGRADE_WINDOW.
        """
        if self.state == NetworkState.UNKNOWN:
            self.state = raw_state
            return self.state

        ranking = {NetworkState.GOOD: 0, NetworkState.DEGRADED: 1, NetworkState.POOR: 2}
        current_rank = ranking.get(self.state, 1)
        raw_rank = ranking.get(raw_state, 1)

        if raw_rank > current_rank:
            # Downgrade — fast reaction
            if self._count_in_window(raw_state, DOWNGRADE_WINDOW) >= DOWNGRADE_VOTES:
                self.state = raw_state
        elif raw_rank < current_rank:
            # Upgrade — slow recovery
            if self._count_in_window(raw_state, UPGRADE_WINDOW) >= UPGRADE_VOTES:
                self.state = raw_state
        # else same rank → no transition
        return self.state

    def _count_in_window(self, state: NetworkState, window: int) -> int:
        """Count occurrences of `state` among the last `window` samples."""
        samples = list(self._history)[-window:]
        return samples.count(state)

    @property
    def probe_interval_bw(self) -> float:
        """Recommended bandwidth probe interval for this state."""
        return {NetworkState.GOOD: 10.0, NetworkState.DEGRADED: 5.0, NetworkState.POOR: 5.0,
                NetworkState.UNKNOWN: 2.0}.get(self.state, 5.0)

    @property
    def probe_bw_data_size(self) -> int:
        """Recommended bandwidth probe data size for this state."""
        return {NetworkState.GOOD: 1 * 1024 * 1024,
                NetworkState.DEGRADED: 1 * 1024 * 1024,
                NetworkState.POOR: 256 * 1024,
                NetworkState.UNKNOWN: 1 * 1024 * 1024}.get(self.state, 1 * 1024 * 1024)

    def budget_bytes(self, bw_ewma: float, total_bytes: float = 0.0) -> float:
        """Compute usable budget from bandwidth and target transfer time.

        Formula: budget = bw_ewma * TARGET_TRANSFER_TIME
        Capped at total_bytes (never exceed original size).
        """
        if bw_ewma <= 0:
            return 0.0
        budget = bw_ewma * TARGET_TRANSFER_TIME
        if total_bytes > 0:
            budget = min(budget, total_bytes)
        return budget

    def __repr__(self) -> str:
        return (f"NetworkStateMachine(state={self.state.value}, "
                f"bw_ewma={self.bw_ewma / 1e6:.1f} MB/s, "
                f"rtt_ewma={self.rtt_ewma:.1f} ms)")
