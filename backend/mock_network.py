"""Mock network probe client for offline ABKT pipeline simulation.

Drop-in replacement for NetworkProbeClient that uses synthetic bandwidth/RTT
profiles instead of real TCP probes. Enables single-process testing of the
full ABKT decision pipeline without hardware.

Usage:
    scenario = step_scenario(high_bw=100e6, low_bw=20e6, transition_time=5.0)
    mock = MockNetworkProbeClient(scenario)
    mock.start()

    snapshot = mock.get_snapshot()
    # ... run allocator, quantize ...
    mock.record_transfer(bytes_sent, elapsed, compression_ratio)
    mock.advance_time(1.0)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Tuple

from backend.config import (
    COLD_BW, COLD_START_MARGIN, MIN_SAFETY_MARGIN,
    CONFIDENCE_TRANSFERS, TARGET_TRANSFER_TIME,
    CONFIDENCE_DECAY_SEC, CONFIDENCE_DECAY_HALF_LIFE,
    CONFIDENCE_PROBE_RATIO, BW_PERCENTILE_INDEX,
    BW_WINDOW_SIZE, EWMA_MAX_CHANGE, TRANSFER_BW_ALPHA,
    DIVERGENCE_THRESHOLD,
)
from backend.ewma import EWMA
from backend.state_machine import NetworkState, NetworkStateMachine
from backend.network_probe import NetworkSnapshot


@dataclass
class NetworkProfile:
    """A time-series of (time_offset, bandwidth_bps, rtt_ms) samples."""
    points: List[Tuple[float, float, float]]  # (time_sec, bw_bps, rtt_ms)

    def interpolate(self, t: float) -> Tuple[float, float]:
        """Return (bandwidth_bps, rtt_ms) at time t via linear interpolation."""
        if not self.points:
            return COLD_BW, 0.0
        if t <= self.points[0][0]:
            return self.points[0][1], self.points[0][2]
        if t >= self.points[-1][0]:
            return self.points[-1][1], self.points[-1][2]
        for i in range(len(self.points) - 1):
            t0, bw0, rtt0 = self.points[i]
            t1, bw1, rtt1 = self.points[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / max(t1 - t0, 1e-9)
                bw = bw0 + frac * (bw1 - bw0)
                rtt = rtt0 + frac * (rtt1 - rtt0)
                return bw, rtt
        return self.points[-1][1], self.points[-1][2]


class MockNetworkProbeClient:
    """Mock probe client using synthetic network profiles.

    Drop-in replacement for NetworkProbeClient. Uses virtual time
    instead of real wall-clock time.
    """

    def __init__(self, profile: NetworkProfile):
        self.profile = profile
        self._virtual_time = 0.0
        self._lock = threading.Lock()

        # Same state as real client
        self._bw_window = []
        self._bw_transfer_ewma = EWMA(alpha=TRANSFER_BW_ALPHA)
        self._rtt_ewma = EWMA(alpha=0.5)
        self._state_machine = NetworkStateMachine()
        self._transfer_count = 0
        self._probe_count = 0
        self._last_transfer_time = 0.0
        self._calibrated = False

    def start(self) -> None:
        """No-op for mock (no background thread needed)."""
        pass

    def stop(self) -> None:
        pass

    def advance_time(self, delta: float) -> None:
        """Advance virtual clock by delta seconds."""
        with self._lock:
            self._virtual_time += delta

    def set_time(self, t: float) -> None:
        """Set virtual clock to absolute time t."""
        with self._lock:
            self._virtual_time = t

    @property
    def virtual_time(self) -> float:
        return self._virtual_time

    def probe_now(self, data_size: int = 0) -> None:
        """Simulate a bandwidth probe at current virtual time."""
        bw, rtt = self.profile.interpolate(self._virtual_time)
        with self._lock:
            self._bw_window.append(bw)
            if len(self._bw_window) > BW_WINDOW_SIZE:
                self._bw_window = self._bw_window[-BW_WINDOW_SIZE:]
            self._rtt_ewma.update(rtt)
            self._probe_count += 1
            self._state_machine.update(
                self._get_bw_for_state_machine(),
                self._rtt_ewma.value or rtt,
            )

    def get_effective_bw(self) -> float:
        """Same logic as real client."""
        if self._bw_window:
            sorted_bw = sorted(self._bw_window)
            idx = min(BW_PERCENTILE_INDEX, len(sorted_bw) - 1)
            window_bw = sorted_bw[idx]
        else:
            window_bw = None
        transfer_bw = self._bw_transfer_ewma.value

        if window_bw is None and transfer_bw is None:
            return COLD_BW * COLD_START_MARGIN
        if transfer_bw is None:
            return (window_bw or COLD_BW) * COLD_START_MARGIN

        if window_bw is not None:
            ratio = window_bw / max(transfer_bw, 1.0)
            if ratio > DIVERGENCE_THRESHOLD:
                base = transfer_bw
            else:
                base = min(window_bw, transfer_bw)
        else:
            base = transfer_bw

        transfer_conf = min(self._transfer_count / CONFIDENCE_TRANSFERS, 1.0)
        probe_conf = min(self._probe_count / (CONFIDENCE_TRANSFERS * CONFIDENCE_PROBE_RATIO), 1.0)
        confidence = max(transfer_conf, probe_conf)

        if self._last_transfer_time > 0:
            elapsed = self._virtual_time - self._last_transfer_time
            if elapsed > CONFIDENCE_DECAY_SEC:
                decay = max(0.0, 1.0 - (elapsed - CONFIDENCE_DECAY_SEC) / CONFIDENCE_DECAY_HALF_LIFE)
                confidence *= decay

        margin = COLD_START_MARGIN + confidence * (MIN_SAFETY_MARGIN - COLD_START_MARGIN)
        return base * margin

    def get_snapshot(self, total_bytes: float = 0.0) -> NetworkSnapshot:
        with self._lock:
            bw = self.get_effective_bw()
            rtt = self._rtt_ewma.value if self._rtt_ewma.valid else 0.0
            state = self._state_machine.state
            budget = bw * TARGET_TRANSFER_TIME
        return NetworkSnapshot(
            timestamp=self._virtual_time,
            bandwidth_bps=bw,
            bandwidth_ewma=bw,
            rtt_ms=rtt,
            rtt_ewma=rtt,
            state=state,
            budget_bytes=budget,
        )

    def record_transfer(self, compressed_bytes: int, elapsed_sec: float,
                        compression_ratio: float) -> None:
        if elapsed_sec <= 0.001 or compressed_bytes < 1024:
            return
        measured_bw = compressed_bytes * compression_ratio / elapsed_sec
        with self._lock:
            self._bw_window.append(measured_bw)
            if len(self._bw_window) > BW_WINDOW_SIZE:
                self._bw_window = self._bw_window[-BW_WINDOW_SIZE:]
            self._bw_transfer_ewma.update_clamped(measured_bw, max_change=EWMA_MAX_CHANGE)
            self._transfer_count += 1
            self._last_transfer_time = self._virtual_time
            self._calibrated = True

    def is_calibrated(self) -> bool:
        return self._calibrated

    def _get_bw_for_state_machine(self) -> float:
        if self._bw_window:
            sorted_bw = sorted(self._bw_window)
            return sorted_bw[len(sorted_bw) // 2]
        return COLD_BW

    @property
    def state_machine(self) -> NetworkStateMachine:
        return self._state_machine


# ════════════════════════════════════════════════════════════════════
# Scenario generators
# ════════════════════════════════════════════════════════════════════


def step_scenario(high_bw: float, low_bw: float, transition_time: float,
                  rtt_ms: float = 2.0) -> NetworkProfile:
    """Sudden bandwidth drop at transition_time."""
    return NetworkProfile([
        (0.0, high_bw, rtt_ms),
        (transition_time - 0.001, high_bw, rtt_ms),
        (transition_time, low_bw, rtt_ms),
        (transition_time * 2, low_bw, rtt_ms),
    ])


def ramp_scenario(start_bw: float, end_bw: float, duration: float,
                  rtt_ms: float = 2.0) -> NetworkProfile:
    """Gradual bandwidth change over duration."""
    return NetworkProfile([
        (0.0, start_bw, rtt_ms),
        (duration, end_bw, rtt_ms),
        (duration * 2, end_bw, rtt_ms),
    ])


def oscillate_scenario(high_bw: float, low_bw: float, period: float,
                       num_cycles: int, rtt_ms: float = 2.0) -> NetworkProfile:
    """Periodic bandwidth fluctuation."""
    points = []
    for i in range(num_cycles * 2 + 1):
        t = i * period / 2
        bw = high_bw if i % 2 == 0 else low_bw
        points.append((t, bw, rtt_ms))
    return NetworkProfile(points)


def random_walk_scenario(start_bw: float, step_size: float, num_steps: int,
                         min_bw: float = 5e6, max_bw: float = 200e6,
                         rtt_ms: float = 2.0) -> NetworkProfile:
    """Stochastic bandwidth changes (deterministic seed)."""
    import random
    rng = random.Random(42)
    points = [(0.0, start_bw, rtt_ms)]
    bw = start_bw
    for i in range(1, num_steps + 1):
        delta = rng.uniform(-step_size, step_size)
        bw = max(min_bw, min(max_bw, bw + delta))
        points.append((float(i), bw, rtt_ms))
    return NetworkProfile(points)
