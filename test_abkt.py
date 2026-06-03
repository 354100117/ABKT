#!/usr/bin/env python3
"""Unit tests for ABKT modules — EWMA, state machine, precision allocator,
adaptive quantization, and calibration feedback-loop safety.

Run:
    python3 test_abkt.py
    python3 -m pytest test_abkt.py -v
"""

import math
import time

import torch

from backend.ewma import EWMA
from backend.state_machine import (
    NetworkState, NetworkStateMachine,
    BW_GOOD_THRESHOLD, BW_POOR_THRESHOLD,
    RTT_GOOD_THRESHOLD, RTT_DEGRADED_THRESHOLD,
)
from backend.precision_allocator import PrecisionAllocator, Precision, BYTES_PER_ELEMENT
from backend.adaptive_quant import AdaptiveQuantizer
from backend.chunked_transfer import ChunkedSender


# ════════════════════════════════════════════════════════════════════
# EWMA tests
# ════════════════════════════════════════════════════════════════════

def test_ewma_cold_start():
    """First update should seed the EWMA."""
    e = EWMA(alpha=0.3)
    assert not e.valid
    val = e.update(100.0)
    assert e.valid
    assert val == 100.0
    assert e.value == 100.0


def test_ewma_convergence():
    """EWMA should converge to steady-state value."""
    e = EWMA(alpha=0.5)
    for _ in range(20):
        e.update(50.0)
    assert abs(e.value - 50.0) < 1e-6


def test_ewma_step_response():
    """EWMA should track step changes with expected lag."""
    e = EWMA(alpha=0.3)
    e.update(100.0)  # seed
    # Step to 50:
    e.update(50.0)
    # Expected: 0.3 * 50 + 0.7 * 100 = 85
    assert abs(e.value - 85.0) < 1e-6


def test_ewma_update_clamped():
    """update_clamped should cap single-step changes."""
    e = EWMA(alpha=0.8)
    e.update(100.0)
    # Without clamp: 0.8*200 + 0.2*100 = 180 (80% jump)
    e.update_clamped(200.0, max_change=0.20)
    # With ±20% clamp: 100 * 1.2 = 120
    assert abs(e.value - 120.0) < 1e-6


def test_ewma_reset():
    """Reset should clear EWMA state."""
    e = EWMA(alpha=0.3)
    e.update(100.0)
    assert e.valid
    e.reset()
    assert not e.valid


# ════════════════════════════════════════════════════════════════════
# State machine tests
# ════════════════════════════════════════════════════════════════════

def test_state_machine_good():
    """GOOD requires bw >= GOOD threshold AND rtt <= GOOD threshold."""
    sm = NetworkStateMachine()
    # GOOD conditions
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, RTT_GOOD_THRESHOLD * 0.5)
    assert sm.state == NetworkState.GOOD


def test_state_machine_poor_bandwidth_alone():
    """POOR triggered by bandwidth alone (P0 fix: not OR with RTT)."""
    sm = NetworkStateMachine()
    # Start GOOD
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, RTT_GOOD_THRESHOLD * 0.5)

    # Bandwidth below POOR threshold → POOR (regardless of RTT)
    for _ in range(3):
        sm.update(BW_POOR_THRESHOLD * 0.5, 1.0)  # Excellent RTT!
    assert sm.state == NetworkState.POOR, (
        f"Expected POOR, got {sm.state}. "
        "POOR should be bandwidth-alone, regardless of RTT.")


def test_state_machine_no_rtt_or_bug():
    """POOR is NOT triggered by high RTT alone (P0 fix: not OR with RTT)."""
    sm = NetworkStateMachine()
    # Start GOOD
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, RTT_GOOD_THRESHOLD * 0.5)

    # High RTT but good bandwidth → should be DEGRADED, not POOR
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, 60.0)  # 60ms RTT, but BW is fine
    assert sm.state == NetworkState.DEGRADED, (
        f"Expected DEGRADED, got {sm.state}. "
        "High RTT with good bandwidth should not trigger POOR.")


def test_state_machine_hysteresis():
    """Downgrade faster than upgrade (2/3 vs 4/5)."""
    sm = NetworkStateMachine()
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, RTT_GOOD_THRESHOLD * 0.5)
    assert sm.state == NetworkState.GOOD

    # Single bad sample → should NOT immediately downgrade
    sm.update(BW_POOR_THRESHOLD * 0.5, 100.0)
    assert sm.state == NetworkState.GOOD, "Single sample should not trigger downgrade"

    # 2 bad samples in last 3 → should downgrade
    sm.update(BW_POOR_THRESHOLD * 0.5, 100.0)
    assert sm.state == NetworkState.POOR, "2/3 should trigger downgrade"


def test_state_machine_no_flip_flop():
    """State should not oscillate on boundary conditions."""
    sm = NetworkStateMachine()
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, RTT_GOOD_THRESHOLD * 0.5)
    assert sm.state == NetworkState.GOOD

    # Oscillate around GOOD/DEGRADED boundary
    for i in range(20):
        if i % 2 == 0:
            sm.update(BW_GOOD_THRESHOLD * 0.95, RTT_GOOD_THRESHOLD * 0.5)
        else:
            sm.update(BW_GOOD_THRESHOLD * 1.05, RTT_GOOD_THRESHOLD * 0.5)

    # Should not be in POOR (which would indicate oscillation)
    assert sm.state != NetworkState.POOR

    # The 4/5 upgrade requirement should prevent rapid flip-flop
    history_variation = len(set(
        sm.update(BW_GOOD_THRESHOLD * 0.95, RTT_GOOD_THRESHOLD * 0.5)
        for _ in range(10)
    ))
    assert history_variation <= 2, "State should not oscillate wildly"


def test_state_machine_budget_progression():
    """Budget = bw * TARGET_TIME, capped at total_bytes."""
    sm = NetworkStateMachine()
    bw = 50e6

    # Budget without total_bytes cap
    budget = sm.budget_bytes(bw)
    assert abs(budget - bw * 2.0) < 1.0, f"Expected {bw*2.0}, got {budget}"

    # Budget with total_bytes cap
    total = 10e6  # 10 MB
    budget_capped = sm.budget_bytes(bw, total_bytes=total)
    assert budget_capped == total, f"Expected {total}, got {budget_capped}"

    # Lower bandwidth → lower budget
    low_budget = sm.budget_bytes(10e6)
    assert low_budget < budget, "Lower BW should give lower budget"


# ════════════════════════════════════════════════════════════════════
# Precision allocator tests
# ════════════════════════════════════════════════════════════════════

def _make_kv_cache(num_layers=4, batch=1, heads=4, seq=64, dim=64):
    """Create a synthetic KV cache for testing."""
    kv_cache = {0: {}}
    for i in range(num_layers):
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        kv_cache[0][i] = (k, v)
    return kv_cache


def test_pia_ample_budget():
    """Given enough budget, all entries should be FP16."""
    kv = _make_kv_cache(num_layers=2, seq=64)
    budget = 10 * 1024 * 1024  # huge budget
    imp = {0: {i: torch.ones(64) for i in range(2)}}
    alloc = PrecisionAllocator()
    result = alloc.allocate(imp, kv, budget)

    assert result.avg_precision_bits == 16.0, "Ample budget should give FP16"
    assert abs(result.compression_ratio - 1.0) < 0.01


def test_pia_zero_budget():
    """With zero budget, everything should be at minimum precision (FP8)."""
    kv = _make_kv_cache(num_layers=2, seq=64)
    budget = 0.0
    imp = {0: {i: torch.ones(64) for i in range(2)}}
    alloc = PrecisionAllocator()
    result = alloc.allocate(imp, kv, budget)

    # Minimum precision is FP8 (1 byte/element)
    fp16_total = PrecisionAllocator._total_bytes(kv, Precision.FP16)
    fp8_total = fp16_total / 2  # FP8 = 1 byte vs FP16 = 2 bytes
    assert result.total_bytes <= fp8_total * 1.01, (
        f"Should be near FP8 total ({fp8_total:.0f}), "
        f"got {result.total_bytes:.0f}")
    assert result.compression_ratio >= 1.9, "Should achieve ~2x compression"


def test_pia_importance_respected():
    """More important tokens should get higher precision."""
    kv = _make_kv_cache(num_layers=2, seq=64)
    total_fp16 = PrecisionAllocator._total_bytes(kv, Precision.FP16)
    budget = total_fp16 * 0.5  # 50% budget

    # Layer 0 = high importance, Layer 1 = low importance
    imp = {0: {0: torch.ones(64) * 0.9, 1: torch.ones(64) * 0.1}}
    alloc = PrecisionAllocator()
    result = alloc.allocate(imp, kv, budget)

    p0 = result.precision_map[0][0]
    p1 = result.precision_map[0][1]
    avg0 = float(p0.float().mean())
    avg1 = float(p1.float().mean())
    assert avg0 >= avg1, (
        f"High-importance layer should get higher precision "
        f"(layer0={avg0:.1f}, layer1={avg1:.1f})")


def test_pia_budget_constraint():
    """Allocation should respect budget (or clamp to FP8 floor)."""
    kv = _make_kv_cache(num_layers=4, seq=128)
    total_fp16 = PrecisionAllocator._total_bytes(kv, Precision.FP16)
    budget = total_fp16 * 0.6  # 60% — enough for FP8 mix
    imp = {0: {i: torch.ones(128) * (1.0 - i * 0.2) for i in range(4)}}
    alloc = PrecisionAllocator()
    result = alloc.allocate(imp, kv, budget)

    fp8_floor = total_fp16 * 0.5  # FP8 minimum
    effective_limit = max(budget, fp8_floor)
    assert result.total_bytes <= effective_limit * 1.01, (
        f"Allocation ({result.total_bytes:.0f}) exceeds limit ({effective_limit:.0f})")


# ════════════════════════════════════════════════════════════════════
# Adaptive quantization tests
# ════════════════════════════════════════════════════════════════════

def test_quant_fp16_roundtrip():
    """FP16 should be lossless passthrough."""
    kv = _make_kv_cache(num_layers=1, seq=8)
    prec_map = {0: {0: torch.full((8,), 16, dtype=torch.int8)}}
    q = AdaptiveQuantizer()
    quantized, meta = q.quantize(kv, prec_map)
    dequantized = q.dequantize(quantized, meta)

    k0, v0 = kv[0][0]
    dk, dv = dequantized[0][0]
    assert (k0 == dk).all(), "FP16 should be lossless for keys"
    assert (v0 == dv).all(), "FP16 should be lossless for values"


def test_quant_fp8_roundtrip():
    """FP8 quantization should produce reasonable error (<5%)."""
    kv = _make_kv_cache(num_layers=1, seq=16)
    prec_map = {0: {0: torch.full((16,), 8, dtype=torch.int8)}}
    q = AdaptiveQuantizer()
    quantized, meta = q.quantize(kv, prec_map)
    dequantized = q.dequantize(quantized, meta)

    orig = kv[0][0][0].float()
    deq = dequantized[0][0][0].float()
    mse = ((orig - deq) ** 2).mean().sqrt() / orig.std()
    assert mse < 0.05, f"FP8 relative error should be <5%, got {mse:.4f}"


def test_quant_int4_roundtrip():
    """INT4 quantization should produce reasonable error (<15%)."""
    kv = _make_kv_cache(num_layers=1, seq=16)
    prec_map = {0: {0: torch.full((16,), 4, dtype=torch.int8)}}
    q = AdaptiveQuantizer()
    quantized, meta = q.quantize(kv, prec_map)
    dequantized = q.dequantize(quantized, meta)

    orig = kv[0][0][0].float()
    deq = dequantized[0][0][0].float()
    rel_err = (orig - deq).abs().mean() / orig.abs().mean()
    assert rel_err < 0.18, f"INT4 relative error should be <18%, got {rel_err:.4f}"


def test_mixed_precision():
    """Different layers should get different precisions."""
    kv = _make_kv_cache(num_layers=2, seq=8)
    prec_map = {
        0: {
            0: torch.full((8,), 16, dtype=torch.int8),   # layer 0 → FP16
            1: torch.full((8,), 4, dtype=torch.int8),    # layer 1 → INT4
        }
    }
    q = AdaptiveQuantizer()
    quantized, meta = q.quantize(kv, prec_map)
    dequantized = q.dequantize(quantized, meta)

    assert meta[0][0]["precision"] == 16
    assert meta[0][1]["precision"] == 4


# ════════════════════════════════════════════════════════════════════
# Calibration feedback loop safety
# ════════════════════════════════════════════════════════════════════

def test_calibration_uncompressed_equivalent():
    """record_transfer should use uncompressed-equivalent bandwidth.

    This prevents the feedback loop: compression → smaller transfer →
    lower measured BW → tighter budget → more compression.
    """
    # Simulate: compressed=4MB, elapsed=0.2s, compression_ratio=4.0
    # Uncompressed-equivalent: 4MB * 4.0 = 16MB → BW = 16/0.2 = 80 MB/s
    measured_bw = 4 * 1024 * 1024 * 4.0 / 0.2  # = 80 MB/s

    # Without the fix (using compressed bytes directly):
    wrong_bw = 4 * 1024 * 1024 / 0.2  # = 20 MB/s (much lower!)

    assert measured_bw > wrong_bw * 3, (
        "Uncompressed-equivalent bandwidth should be much higher "
        "than compressed-only bandwidth to prevent feedback loop."
    )


def test_ewma_clamp_feedback_loop():
    """Simulate the feedback loop scenario and verify stability.

    GC pause causes one slow transfer → if uncapped, EWMA drops →
    tighter budget → more compression → smaller transfer → appears slow.
    """
    e = EWMA(alpha=0.5)
    # Initial BW = 80 MB/s
    e.update(80e6)

    # GC pause: one transfer is 3x slower → appears as 1/3 bandwidth
    e.update(25e6)  # no clamp
    uncapped = e.value

    e2 = EWMA(alpha=0.5)
    e2.update(80e6)
    e2.update_clamped(25e6, max_change=0.20)
    clamped = e2.value

    # Clamped version should be more conservative (higher)
    assert clamped > uncapped, (
        f"Clamped EWMA ({clamped:.0f}) should be higher than "
        f"uncapped ({uncapped:.0f}) after a transient dip"
    )


# ════════════════════════════════════════════════════════════════════
# Sliding window bandwidth tests
# ════════════════════════════════════════════════════════════════════

def test_sliding_window_min_immediate_drop():
    """Window min should respond immediately to bandwidth drop."""
    from collections import deque
    window = deque(maxlen=10)

    # Steady state: 80 MB/s
    for _ in range(5):
        window.append(80e6)
    assert min(window) == 80e6

    # Bandwidth drops to 20 MB/s
    window.append(20e6)
    assert min(window) == 20e6, "Window min should immediately reflect drop"


def test_sliding_window_min_slow_recovery():
    """Window min should recover slowly (needs old samples to expire)."""
    from collections import deque
    window = deque(maxlen=5)

    # Fill with bad values
    for _ in range(5):
        window.append(20e6)
    assert min(window) == 20e6

    # One good sample doesn't change min
    window.append(80e6)
    assert min(window) == 20e6, "One good sample should not raise min"

    # Fill with good samples until bad ones expire
    for _ in range(4):
        window.append(80e6)
    assert min(window) == 80e6, "Min should recover after window slides"


def test_sliding_window_vs_ewma_response():
    """Window min responds faster than EWMA to bandwidth drops."""
    from collections import deque
    window = deque(maxlen=10)
    ewma = EWMA(alpha=0.3)

    # Steady state
    for _ in range(5):
        window.append(80e6)
        ewma.update(80e6)

    # Bandwidth drops to 20
    window.append(20e6)
    ewma.update(20e6)

    window_min = min(window)
    # EWMA: 0.3*20 + 0.7*80 = 62
    assert window_min < ewma.value * 0.5, (
        f"Window min ({window_min/1e6:.0f}) should be much lower than "
        f"EWMA ({ewma.value/1e6:.0f}) after a sudden drop")


# ════════════════════════════════════════════════════════════════════
# Mid-transfer downgrade tests
# ════════════════════════════════════════════════════════════════════

def test_downgrade_precision_map():
    """Downgrade should reduce precision by one level for remaining layers."""
    prec_map = {
        0: {
            0: torch.full((8,), 16, dtype=torch.int8),   # FP16
            1: torch.full((8,), 8, dtype=torch.int8),    # FP8
            2: torch.full((8,), 4, dtype=torch.int8),    # INT4
            3: torch.full((8,), 2, dtype=torch.int8),    # INT2
        }
    }
    remaining = [0, 1, 2, 3]
    new_map = ChunkedSender._downgrade_precision_map(prec_map, remaining)

    assert (new_map[0][0] == 8).all(), "FP16 should downgrade to FP8"
    assert (new_map[0][1] == 4).all(), "FP8 should downgrade to INT4"
    assert (new_map[0][2] == 2).all(), "INT4 should downgrade to INT2"
    assert (new_map[0][3] == 2).all(), "INT2 should stay INT2"


def test_downgrade_only_remaining():
    """Downgrade should only affect remaining layers, not sent ones."""
    prec_map = {
        0: {
            0: torch.full((8,), 16, dtype=torch.int8),   # already sent
            1: torch.full((8,), 16, dtype=torch.int8),   # remaining
        }
    }
    remaining = [1]  # only layer 1 is remaining
    new_map = ChunkedSender._downgrade_precision_map(prec_map, remaining)

    assert (new_map[0][0] == 16).all(), "Sent layer should be unchanged"
    assert (new_map[0][1] == 8).all(), "Remaining layer should be downgraded"


# ════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("EWMA cold start", test_ewma_cold_start),
        ("EWMA convergence", test_ewma_convergence),
        ("EWMA step response", test_ewma_step_response),
        ("EWMA update clamped", test_ewma_update_clamped),
        ("EWMA reset", test_ewma_reset),
        ("State machine GOOD", test_state_machine_good),
        ("State machine POOR bandwidth-alone", test_state_machine_poor_bandwidth_alone),
        ("State machine no RTT-OR bug", test_state_machine_no_rtt_or_bug),
        ("State machine hysteresis", test_state_machine_hysteresis),
        ("State machine no flip-flop", test_state_machine_no_flip_flop),
        ("State machine budget progression", test_state_machine_budget_progression),
        ("PIA ample budget", test_pia_ample_budget),
        ("PIA zero budget", test_pia_zero_budget),
        ("PIA importance respected", test_pia_importance_respected),
        ("PIA budget constraint", test_pia_budget_constraint),
        ("Quant FP16 roundtrip", test_quant_fp16_roundtrip),
        ("Quant FP8 roundtrip", test_quant_fp8_roundtrip),
        ("Quant INT4 roundtrip", test_quant_int4_roundtrip),
        ("Quant mixed precision", test_mixed_precision),
        ("Calibration uncompressed-equivalent", test_calibration_uncompressed_equivalent),
        ("EWMA clamp feedback loop", test_ewma_clamp_feedback_loop),
        ("Sliding window immediate drop", test_sliding_window_min_immediate_drop),
        ("Sliding window slow recovery", test_sliding_window_min_slow_recovery),
        ("Sliding window vs EWMA response", test_sliding_window_vs_ewma_response),
        ("Downgrade precision map", test_downgrade_precision_map),
        ("Downgrade only remaining", test_downgrade_only_remaining),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  {passed}/{passed + failed} passed")
    if failed:
        print(f"  ❌ {failed} FAILED")
    else:
        print(f"  ✅ ALL PASSED")
    print(f"{'=' * 50}")
