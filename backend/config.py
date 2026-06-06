"""Centralized tuning parameters for the ABKT backend.

All magic numbers extracted from network_probe, state_machine, precision_allocator,
chunked_transfer, token_importance, and prefill_node. Modify here to tune behavior.
"""

from enum import IntEnum


class Precision(IntEnum):
    FP16 = 16
    FP8 = 8
    INT4 = 4
    INT2 = 2


# ════════════════════════════════════════════════════════════════════
# Network probing & bandwidth estimation
# ════════════════════════════════════════════════════════════════════

DEFAULT_PROBE_PORT = 9877
RTT_INTERVAL_SEC = 1.0
BW_ALPHA = 0.3              # bandwidth EWMA smoothing (probe data)
RTT_ALPHA = 0.5             # RTT EWMA smoothing (faster response to congestion)
EWMA_MAX_CHANGE = 0.20      # ±20% cap per EWMA update to prevent oscillation
WARMUP_BYTES = 100 * 1024   # 100 KB TCP warmup payload
MIN_CALIB_BYTES = 1 * 1024 * 1024   # 1 MB — skip calibration for smaller transfers
MIN_CALIB_ELAPSED = 0.100   # 100ms — skip calibration if too fast
COLD_BW = 15e6              # conservative cold-start: 15 MB/s (120 Mbps)

# Sliding window + transfer EWMA
BW_WINDOW_SIZE = 10         # sliding window: last N probe/transfer samples
TRANSFER_BW_ALPHA = 0.5     # transfer EWMA — trust actual transfers more
DIVERGENCE_THRESHOLD = 2.5  # probe/transfer ratio to flag mismatch
COLD_START_MARGIN = 0.7     # 70% of estimate when uncalibrated
MIN_SAFETY_MARGIN = 0.8     # floor: never use more than 80% of estimated bw
CONFIDENCE_TRANSFERS = 5    # full confidence after this many transfers
TARGET_TRANSFER_TIME = 2.0  # target transfer time in seconds

# Socket timeouts (seconds)
PROBE_CONNECT_TIMEOUT = 5.0
PROBE_RTT_TIMEOUT = 10.0

# INT4 quality floor
INT4_HEADROOM = 1.3         # 30% above INT4 for importance-based upgrades
MAX_TRANSFER_TIME = 60.0    # safety cap in seconds


# ════════════════════════════════════════════════════════════════════
# Network state machine (hysteresis)
# ════════════════════════════════════════════════════════════════════

BW_GOOD_THRESHOLD = 40e6        # 40 MB/s → above = GOOD
BW_POOR_THRESHOLD = 15e6        # 15 MB/s → below = POOR (bandwidth-alone)
RTT_GOOD_THRESHOLD = 10.0       # 10 ms → below = GOOD
RTT_DEGRADED_THRESHOLD = 20.0   # 20 ms → above = DEGRADED

# Voting windows
DOWNGRADE_VOTES = 2         # 2 of last 3 to downgrade
DOWNGRADE_WINDOW = 3
UPGRADE_VOTES = 4           # 4 of last 5 to upgrade
UPGRADE_WINDOW = 5


# ════════════════════════════════════════════════════════════════════
# Precision allocation
# ════════════════════════════════════════════════════════════════════

NUM_GROUPS = 4              # token groups per layer for per-group quantization

QUALITY_FIDELITY = {
    Precision.FP16: 1.00,
    Precision.FP8: 0.98,
    Precision.INT4: 0.92,
    Precision.INT2: 0.80,
}

BYTES_PER_ELEMENT = {
    Precision.FP16: 2.0,
    Precision.FP8: 1.0,
    Precision.INT4: 0.5,
    Precision.INT2: 0.25,
}

PREC_NAMES = {16: "FP16", 8: "FP8 ", 4: "INT4", 2: "INT2"}


# ════════════════════════════════════════════════════════════════════
# Chunked transfer
# ════════════════════════════════════════════════════════════════════

MIN_CHUNK_SIZE = 16
MAX_CHUNK_SIZE = 256
TIMING_CHECK_INTERVAL = 2   # check throughput every N chunks
SLOW_THRESHOLD = 0.7        # actual_bw < expected * threshold → downgrade


# ════════════════════════════════════════════════════════════════════
# Token importance (3D scoring weights)
# ════════════════════════════════════════════════════════════════════

IMPORTANCE_ALPHA = 0.6      # attention importance weight
IMPORTANCE_BETA = 0.25      # layer sensitivity weight
IMPORTANCE_GAMMA = 0.15     # position decay weight
POSITION_DECAY_RATE = 0.01  # exponential decay rate for position scoring
