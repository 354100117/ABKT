"""Centralized tuning parameters for the ABKT backend.

All magic numbers extracted from network_probe, state_machine, precision_allocator,
chunked_transfer, token_importance, and prefill_node. Modify here to tune behavior.

Optionally override via YAML/JSON config file: load_config("path/to/config.yaml").
"""

import json
from enum import IntEnum
from pathlib import Path
from typing import Optional


class Precision(IntEnum):
    FP16 = 16
    INT8 = 8       # symmetric INT8 (absmax quantization)
    INT4 = 4
    INT2 = 2

# Backward-compatible alias
Precision.FP8 = Precision.INT8


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

# Confidence time decay (B4)
CONFIDENCE_DECAY_SEC = 120.0    # start decaying after 2 min of no transfers
CONFIDENCE_DECAY_HALF_LIFE = 300.0  # half confidence every 5 min

# Probe-based confidence ramp (B3)
CONFIDENCE_PROBE_RATIO = 3     # probes contribute at 1/3 the rate of transfers

# Bandwidth recovery (B5)
BW_PERCENTILE_INDEX = 0        # use 10th percentile instead of min (0 = index 0)

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
    Precision.INT8: 0.999,  # calibrated: PPL delta ~0.06% on Qwen2.5-3B
    Precision.INT4: 0.99,   # calibrated: PPL delta ~14% on Qwen2.5-3B
    Precision.INT2: 0.85,   # calibrated: PPL delta ~75% on Qwen2.5-3B
}

BYTES_PER_ELEMENT = {
    Precision.FP16: 2.0,
    Precision.INT8: 1.0,
    Precision.INT4: 0.5,
    Precision.INT2: 0.25,
}

PREC_NAMES = {16: "FP16", 8: "INT8", 4: "INT4", 2: "INT2"}


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


# ════════════════════════════════════════════════════════════════════
# Config file loading
# ════════════════════════════════════════════════════════════════════

# All overridable keys (maps YAML key → module-level variable name)
_CONFIG_KEYS = {
    # Network probing
    "rtt_interval_sec", "bw_alpha", "rtt_alpha", "ewma_max_change",
    "warmup_bytes", "min_calib_bytes", "min_calib_elapsed", "cold_bw",
    "bw_window_size", "transfer_bw_alpha", "divergence_threshold",
    "cold_start_margin", "min_safety_margin", "confidence_transfers",
    "target_transfer_time", "confidence_decay_sec", "confidence_decay_half_life",
    "confidence_probe_ratio", "bw_percentile_index",
    "probe_connect_timeout", "probe_rtt_timeout",
    "int4_headroom", "max_transfer_time",
    # State machine
    "bw_good_threshold", "bw_poor_threshold",
    "rtt_good_threshold", "rtt_degraded_threshold",
    "downgrade_votes", "downgrade_window",
    "upgrade_votes", "upgrade_window",
    # Precision allocation
    "num_groups",
    # Chunked transfer
    "min_chunk_size", "max_chunk_size", "timing_check_interval", "slow_threshold",
    # Token importance
    "importance_alpha", "importance_beta", "importance_gamma", "position_decay_rate",
}


def load_config(path: str) -> None:
    """Load config overrides from a YAML or JSON file.

    Only keys listed in _CONFIG_KEYS are applied. Unknown keys are ignored
    with a warning. Values are written to this module's global namespace.

    Args:
        path: Path to YAML (.yaml/.yml) or JSON (.json) config file.
    """
    import sys
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError("PyYAML required for YAML config: pip install pyyaml")
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix} (use .yaml or .json)")

    module = sys.modules[__name__]
    applied = 0
    for key, value in data.items():
        key_lower = key.lower()
        if key_lower in _CONFIG_KEYS:
            attr_name = key_lower.upper()
            if hasattr(module, attr_name):
                old = getattr(module, attr_name)
                # Cast to same type as original
                try:
                    if isinstance(old, float):
                        value = float(value)
                    elif isinstance(old, int):
                        value = int(value)
                except (TypeError, ValueError):
                    pass
                setattr(module, attr_name, value)
                applied += 1
            else:
                print(f"[config] Warning: unknown key '{key}' (ignored)")
        else:
            print(f"[config] Warning: not overridable: '{key}' (ignored)")

    print(f"[config] Loaded {applied} overrides from {path}")
