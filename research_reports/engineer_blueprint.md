# ABKT Implementation Blueprint

> Engineer assessment — practical feasibility, optimization tradeoffs, integration specifics
> 2026-05-18

---

## A. Module Architecture

### A.1 New Modules to Create

The project currently has no `backend/` directory. All ABKT modules live there:

| File | Purpose | Key Classes |
|------|---------|-------------|
| `backend/__init__.py` | Package exports | --- |
| `backend/network_probe.py` | Probe client + server + EWMA + 3-level state machine | `ProbeServer`, `NetworkProbeClient`, `NetworkState`, `NetworkSnapshot`, `NetworkStateMachine` |
| `backend/precision_allocator.py` | PIA algorithm, precision enums, budget math | `Precision`, `AllocationResult`, `PrecisionAllocator` |
| `backend/adaptive_quant.py` | Quantize/dequantize FP16-FP8-INT4-INT2 | `AdaptiveQuantizer`, `AdaptiveDequantizer` |
| `backend/chunked_transfer.py` | Chunked send (prefill side), chunked receive (decode side) | `ChunkedTransfer`, `ChunkAssembler` |

**Deferred to phase 2:** `backend/token_importance.py` (TokenImportanceEvaluator). The weekly report flagged attention score capture as not yet stable. Initial ABKT uses the simpler Key-norm proxy from design doc section 4.1 (`_compute_attention_importance_from_kv`), which requires no model changes and only reads existing KV tensors.

### A.2 Data Flow

```
prefill_node.py                                     decode_node.py
    |                                                   |
    v                                                   |
[PrefillStage.forward()]                                |
    |  outputs.past_key_values                          |
    v                                                   |
[KVCache.from_dynamic_cache()]                          |
    |  kv_cache: KVCache object                         |
    v                                                   |
[TokenImportanceEvaluator]                              |
    |  (simplified Key-norm proxy)                      |
    |  importance_map: {layer: tensor[seq]}             |
    v                                                   |
[NetworkProbeClient.get_snapshot()]                     |
    |  snapshot.budget_bytes                            |
    |  snapshot.bandwidth_ewma                          |
    |  snapshot.state: GOOD/DEGRADED/POOR               |
    v                                                   |
[PrecisionAllocator.allocate()]                         |
    |  alloc_result.precision_map                       |
    |  alloc_result.compression_ratio                   |
    v                                                   |
[AdaptiveQuantizer.quantize_kv_cache()]                 |
    |  quantized_kv, metadata                           |
    v                                                   |
[ChunkedTransfer.send_kv_cache()]  ───TCP──────────> [ChunkAssembler.receive_chunk()]
    |  layer-by-layer, per-layer chunked                  |  buffers chunks per layer
    |  uses SocketClient.call("receive_kv_chunk", ...)    |  returns {"ready": False} until last chunk
    |                                                     v
    |                                               [ChunkAssembler.assemble_and_dequantize()]
    |                                                     |  full KVCache + metadata → DynamicCache
    |                                                     v
    |                                               [handle_run_decode_abkt()]
    |                                                     |  standard decode loop
    |                                                     v
    |  client.call("run_decode_abkt", ...) ──TCP────>  [response with generated_text]
    v
[probe.record_transfer_complete()]  ← calibration feedback
```

### A.3 Probe Side-Channel Architecture

The probe runs on a **separate TCP port** (default 9877, distinct from both the existing probe test tool port 9876 and the data port 29501).

**Rationale for separate port:**
- Avoids head-of-line blocking from large KV cache data transfers
- Allows RTT measurement even during ongoing large transfers
- Probe server is a trivial echo server — zero coupling with SocketServer's handler dispatch

**Connectivity:**
```
prefill_node.py (192.168.0.50)                  decode_node.py (192.168.0.20)
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ NetworkProbeClient            │  port 9877   │ ProbeServer (daemon thread)   │
│ (daemon thread)               │ ───────────→ │ listens on 0.0.0.0:9877       │
│   - RTT probe every 500ms     │  short-lived │   - echo 0x01 packets         │
│   - BW probe every 10th RTT   │  TCP conns   │   - receive 0x02 payloads     │
│                               │              │                               │
│ SocketClient                  │  port 29501  │ SocketServer (main thread)    │
│ (main thread)                 │ ───────────→ │ listens on 0.0.0.0:29501      │
│   - receive_kv_chunk          │  persistent  │   - handle_receive_kv_chunk   │
│   - run_decode_abkt           │  connection  │   - handle_run_decode_abkt    │
└──────────────────────────────┘              └──────────────────────────────┘
```

---

## B. NetworkProbe Module Design

### B.1 Complete API

```python
# backend/network_probe.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import threading
import time

class NetworkState(Enum):
    GOOD = "good"          # BW >= 50 MB/s AND RTT <= 5ms
    DEGRADED = "degraded"  # between thresholds
    POOR = "poor"          # BW <= 10 MB/s OR RTT >= 50ms

@dataclass
class NetworkSnapshot:
    timestamp: float
    bandwidth_bps: float         # current instantaneous BW estimate
    bandwidth_ewma: float        # EWMA-smoothed BW — THE value for budget
    rtt_ms: float                # current instantaneous RTT
    rtt_ewma: float              # EWMA-smoothed RTT
    state: NetworkState          # GOOD / DEGRADED / POOR
    budget_bytes: float          # = bandwidth_ewma * max_delay_sec * safety_factor
    sample_count: int            # total samples collected

class ProbeServer:
    """Lightweight echo server for RTT measurement. Runs on decode node.

    Handles binary protocol:
        0x01 (PROBE_RTT): receive 1 byte, echo 1 byte
        0x02 (PROBE_BW):  receive 4-byte length + payload, echo 1 byte
    """

    PROBE_ECHO = 0x01
    PROBE_BANDWIDTH = 0x02

    def __init__(self, host: str = "0.0.0.0", port: int = 9877):
        ...

    def start(self) -> None:
        """Spawn daemon thread. Non-blocking. Call before SocketServer.serve_forever()."""

    def stop(self) -> None:
        """Set running flag, close listener socket."""

    def _handle_client(self, conn, addr) -> None:
        """Handle one probe client request. Short-lived connections."""


class NetworkStateMachine:
    """Three-level state machine with Schmitt trigger hysteresis.

    Transitions:
        GOOD ──(bw<50 OR rtt>5, 3 consecutive)──> DEGRADED
        GOOD ──(bw<10 OR rtt>50, 1 sample)──────> POOR       (emergency)
        DEGRADED ──(bw<10 OR rtt>50, 3 samples)────> POOR
        DEGRADED ──(bw>=50 AND rtt<=5, 5 samples)───> GOOD
        POOR ──(bw>=50 AND rtt<=5, 5 samples)───────> GOOD    (direct)
        POOR ──(bw>=10 AND rtt<=50, 5 samples)──────> DEGRADED
    """

    def __init__(
        self,
        bw_threshold_good: float = 50_000_000,   # 50 MB/s
        bw_threshold_poor: float = 10_000_000,   # 10 MB/s
        rtt_threshold_good: float = 5.0,         # 5 ms
        rtt_threshold_poor: float = 50.0,        # 50 ms
        degrade_samples: int = 3,                 # samples to downgrade
        recover_samples: int = 5,                 # samples to upgrade
    ):
        ...

    def update(self, bw_ewma: float, rtt_ewma: float) -> NetworkState:
        """Feed new EWMA values, return current state after hysteresis."""

    @property
    def state(self) -> NetworkState:
        ...


class NetworkProbeClient:
    """Background probe client. Runs on prefill node.

    Key design decision: bandwidth estimation comes from CALIBRATED actual
    transfers, not from probe packets. Probe measures RTT directly (which
    is independent of payload size) and runs periodic small-packet BW probes
    only as a sanity check / change detector.

    Threading model:
        - _probe_loop() runs in a daemon thread
        - get_snapshot() is thread-safe (lock-protected read)
        - calibrate_with_transfer() and record_transfer_complete() are
          thread-safe and feed the real bandwidth EWMA
    """

    def __init__(
        self,
        target_host: str,
        target_port: int = 9877,
        probe_interval_sec: float = 0.5,          # RTT probe frequency
        bw_probe_interval: int = 10,               # BW probe every N RTT probes
        bw_probe_size_mb: float = 0.5,             # small payload: 500 KB
        ewma_alpha_bw_probe: float = 0.2,          # slow — probe BW is noisy
        ewma_alpha_bw_calibrated: float = 0.3,     # moderate — real transfer BW
        ewma_alpha_rtt: float = 0.5,               # fast — RTT changes matter
        max_delay_sec: float = 0.5,                # acceptable transfer latency
        safety_factor: float = 0.85,               # 15% headroom for TCP overhead
        cold_start_bw: float = 80_000_000,         # 80 MB/s conservative default
    ):
        ...

    # ── Lifecycle ──

    def start(self) -> NetworkSnapshot:
        """Start the background probe thread. Returns initial snapshot immediately.
        Call once before any inference requests.
        """

    def stop(self) -> None:
        """Stop the probe thread. Call during shutdown."""

    # ── Snapshot (main consumer API) ──

    def get_snapshot(self) -> NetworkSnapshot:
        """Thread-safe read of latest network state.
        Called by the main inference loop before each transfer decision.
        """

    # ── Calibration (lifetime learning) ──

    def calibrate_with_transfer(self, num_bytes: int, elapsed_sec: float) -> None:
        """Seed the bandwidth EWMA with a real measured transfer.
        Called exactly once, after the first ABKT transfer completes.
        Uses direct replacement (not EWMA blend) to erase cold-start guess.
        """

    def record_transfer_complete(
        self, compressed_bytes: int, elapsed_sec: float, compression_ratio: float
    ) -> None:
        """Update BW EWMA with actual transfer throughput.
        Uses UNCOMPRESSED equivalent size to prevent feedback loops.
        Called after every ABKT transfer completes.
        """

    # ── Internal ──

    def _probe_loop(self) -> None:
        """Background loop: probe RTT every interval, BW every N intervals."""

    def _probe_rtt(self) -> float:
        """Open short-lived TCP connection, send 0x01, receive 0x01, return RTT ms."""

    def _probe_bandwidth(self) -> float:
        """Open short-lived TCP connection, send 0x02 + payload, measure throughput."""

    def _update_ewma(self, bw: float, rtt: float) -> None:
        """Update EWMA state. Write under lock."""

    def _make_snapshot(self) -> NetworkSnapshot:
        """Build NetworkSnapshot from current EWMA and state machine."""
```

### B.2 Threading Model

```
prefill_node.py MAIN THREAD:              NetworkProbeClient DAEMON THREAD:
    probe.start()                              while running:
    ┊                                            rtt = _probe_rtt()
    kv_cache = prefill()                         if rtt_counter % 10 == 0:
    snapshot = probe.get_snapshot()  ←lock→          bw = _probe_bandwidth()
    allocate(snapshot.budget)                    _update_ewma(bw, rtt)  ←lock→
    quantize + transfer                          snapshot = _make_snapshot()
    probe.record_transfer_complete()  ←lock→     sleep(0.5)
    ┊
    probe.stop()
```

The lock protects `_bandwidth_ewma`, `_rtt_ewma`, `_last_snapshot`, and `_state_machine`. The critical section in `get_snapshot()` is a single dict/dataclass copy, under 1 microsecond. The probe thread holds the lock for EWMA updates (~microseconds). No contention.

### B.3 Error Handling and Fallback

| Failure Mode | Detection | Behavior |
|-------------|-----------|----------|
| Probe connection refused | `ConnectionRefusedError` | Log warning, keep last known values |
| Probe timeout | `socket.timeout` after 2s | RTT=100ms (pessimistic), BW=last EWMA |
| 5 consecutive probe failures | `_consecutive_failures >= 5` | Downgrade state one level |
| Cold start (zero data) | `_bandwidth_ewma is None` | BW=80MB/s, RTT=5ms, state=GOOD |
| Probe server not running on decode | All probes fail | ABKT degrades to FP16 pass-through after 5 failures |

### B.4 Handling Probe-vs-Real-Traffic Gap

This is the most critical practical concern. Probe BW measurements from small packets (~500 KB) systematically overestimate real throughput for large transfers (10-100 MB) due to:
- TCP slow start (small transfers never exit slow start)
- Congestion window ramp-up
- No steady-state congestion control behavior

**Solution: Calibration-based bandwidth estimation.**

The probe measures **RTT directly** (independent of payload size). Bandwidth estimation relies on actual transfers:

1. **RTT probes**: Short-lived connection, 1 byte in each direction. RTT measures network latency directly. This is accurate regardless of payload size.

2. **BW probes (lightweight)**: Every 10 RTT probes, send 500 KB and measure throughput. This is NOT used for budget computation. It serves only as a **sanity check change detector** — if probe BW drops below 0.3x the calibrated EWMA, it signals a possible network degradation.

3. **Calibrated BW**: After each real KV cache transfer completes, `record_transfer_complete()` updates the bandwidth EWMA with the actual measured throughput. This is the SOURCE OF TRUTH for budget computation.

4. **Fusion logic**:
```python
if calibrated_bw_ewma is not None:
    if probe_bw < 0.3 * calibrated_bw_ewma:
        # Probe detects dramatic drop — calibrated value may be stale
        bw_for_budget = 0.7 * calibrated_bw_ewma + 0.3 * probe_bw
    else:
        bw_for_budget = calibrated_bw_ewma  # trust calibrated
else:
    bw_for_budget = probe_bw_ewma  # fallback (cold start)
```

---

## C. EWMA Configuration and Budget Computation

### C.1 Recommended Alpha Values

| Metric | Alpha | Rationale |
|--------|-------|-----------|
| RTT EWMA | **0.5** | RTT changes on millisecond scale; need responsiveness to detect congestion onset. Higher values oversmooth and delay state transitions. |
| BW EWMA (probe) | **0.2** | Probe BW is noisy, biased (500 KB != 50 MB), and only used for sanity checking. Slow smoothing prevents false alarms. |
| BW EWMA (calibrated) | **0.3** | Real transfer BW is the ground truth. Moderate alpha tracks genuine bandwidth changes while smoothing measurement noise (TCP burstiness, CPU scheduling jitter). |

### C.2 Cold Start Strategy

```
Phase 1: Before any transfer completes
  RTT_EWMA ← first successful RTT probe (expect 1-3 ms on direct 1GbE link)
  BW_EWMA ← 80 MB/s (conservative estimate for 1 Gbps Ethernet)
  State  ← GOOD (optimistic start avoids initial quality degradation)
  Budget ← 80 * 0.5 * 0.85 = 34 MB

Phase 2: First transfer completes
  measured_bw = uncompressed_equivalent_bytes / elapsed_sec
  BW_EWMA = measured_bw    # DIRECT REPLACEMENT, not EWMA blend
                            # This "erases" the cold-start guess
  RTT_EWMA continues from Phase 1

Phase 3: Steady state
  Each transfer: BW_EWMA = 0.3 * measured_bw + 0.7 * BW_EWMA_old
  RTT_EWMA updates from probes continuously

Phase 4: Idle periods (> 30 seconds without a transfer)
  BW_EWMA decays toward probe estimate at rate 0.05 per probe cycle:
    BW_EWMA = 0.95 * BW_EWMA_old + 0.05 * probe_bw
  This prevents stale calibrated values from persisting indefinitely.
```

### C.3 Budget Computation Formula

```python
def compute_budget(self) -> float:
    """Compute transfer budget in bytes."""
    # Determine effective BW: prefer calibrated, fallback to probe
    if self._bw_ewma_calibrated is not None:
        bw = self._bw_ewma_calibrated
        # Sanity check against probe
        if self._bw_ewma_probe is not None:
            if self._bw_ewma_probe < 0.3 * bw:
                bw = 0.7 * bw + 0.3 * self._bw_ewma_probe
    else:
        bw = self._bw_ewma_probe or self.cold_start_bw

    # Apply max_delay (state-dependent) and safety factor
    max_delay = self._state_dependent_delay()
    budget = bw * max_delay * self.safety_factor
    return int(budget)

def _state_dependent_delay(self) -> float:
    if self._state_machine.state == NetworkState.GOOD:
        return self.max_delay_sec           # 0.5
    elif self._state_machine.state == NetworkState.DEGRADED:
        return self.max_delay_sec * 0.6     # 0.3
    else:  # POOR
        return self.max_delay_sec * 0.4     # 0.2
```

### C.4 Practical Budget Example

**At 80 MB/s practical throughput (GOOD state):**
- Budget = 80 MB/s × 0.5s × 0.85 = **34 MB**

**For OPT-6.7B (32 layers, 32 heads, head_dim=128, seq_len=512):**
- KV entries per layer: batch=1, 32 heads, 512 tokens, 128 dim = 2,097,152 elements
- K+V per layer: 2 × 2,097,152 = 4,194,304 elements
- Total all layers: 32 × 4,194,304 = 134,217,728 elements
- FP16 total: 134,217,728 × 2 bytes = **268 MB**
- Required compression for 34 MB budget: 268/34 = **7.9x**
- This means the system will almost always use INT2 (8x compression) for most entries under practical conditions.

**Key insight**: On 1 Gbps Ethernet with a 0.5s delay target, ABKT will operate in heavily compressed mode by default. The optimization isn't about "should we compress" but "how to allocate the limited budget across layers and tokens to minimize quality loss."

**At theoretical max 125 MB/s (optimistic):**
- Budget = 125 MB/s × 0.5s × 0.85 = **53 MB**
- Required compression: 268/53 = 5.1x → still requires mostly INT4 (4x) with some entries at INT2

**Even with generous 1.0s max_delay:**
- Budget = 125 MB/s × 1.0s × 0.85 = **106 MB**
- Required compression: 268/106 = 2.5x → mix of INT4 and FP8

---

## D. Three-Level State Machine with Hysteresis

### D.1 Threshold Values

All thresholds calibrated for 1 Gbps Ethernet (~125 MB/s theoretical, ~80-100 MB/s practical):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BW_THRESHOLD_GOOD` | 50 MB/s (400 Mbps) | ~40% of theoretical max. Below this, compression provides enough benefit to justify quality tradeoff. |
| `BW_THRESHOLD_POOR` | 10 MB/s (80 Mbps) | ~8% of theoretical max. Below this, only INT2 is viable. |
| `RTT_THRESHOLD_GOOD` | 5 ms | Normal LAN RTT. Values above indicate congestion or buffer bloat. |
| `RTT_THRESHOLD_POOR` | 50 ms | 10x normal. Indicates severe congestion, packet loss, or WiFi interference. |

### D.2 State Definitions

| State | BW Condition | RTT Condition | Meaning |
|-------|-------------|---------------|---------|
| GOOD | >= 50 MB/s | <= 5 ms | Headroom available. Can use FP16 for important entries. |
| DEGRADED | 10-50 MB/s | 5-50 ms | Network under pressure. Need systematic compression. |
| POOR | <= 10 MB/s | >= 50 ms | Severe constraint. Aggressive INT2 compression, only top tokens get INT4. |

### D.3 Hysteresis Implementation

Uses a Schmitt trigger pattern: asymmetric sample counts for degrade vs recover to prevent oscillation.

```python
class NetworkStateMachine:
    def __init__(
        self,
        bw_threshold_good: float = 50_000_000,
        bw_threshold_poor: float = 10_000_000,
        rtt_threshold_good: float = 5.0,
        rtt_threshold_poor: float = 50.0,
        degrade_samples: int = 3,      # need 3 consecutive "bad" readings to degrade
        recover_samples: int = 5,      # need 5 consecutive "good" readings to recover
    ):
        self._state = NetworkState.GOOD  # optimistic initial
        self._consecutive_opposite = 0
        self._pending_target = None

    def update(self, bw_ewma: float, rtt_ewma: float) -> NetworkState:
        # Determine raw state from thresholds
        raw_state = self._classify(bw_ewma, rtt_ewma)

        if raw_state == self._state:
            # Reset counters — current state confirmed
            self._consecutive_opposite = 0
            self._pending_target = None
            return self._state

        # Raw state differs from current
        if self._pending_target != raw_state:
            # New target direction
            self._pending_target = raw_state
            self._consecutive_opposite = 1
        else:
            self._consecutive_opposite += 1

        # Check if we have enough consecutive samples
        is_degrading = (raw_state.value > self._state.value
                        if isinstance(raw_state, NetworkState) else False)
        # Actually compare by severity: GOOD=0, DEGRADED=1, POOR=2
        degrade_severity = {"good": 0, "degraded": 1, "poor": 2}
        is_degrading = degrade_severity[raw_state.value] > degrade_severity[self._state.value]

        # Emergency: jump two levels (GOOD→POOR) happens immediately
        is_emergency = (
            degrade_severity[raw_state.value] - degrade_severity[self._state.value] >= 2
        )

        required = 1 if is_emergency else (
            self.degrade_samples if is_degrading else self.recover_samples
        )

        if self._consecutive_opposite >= required:
            old = self._state
            self._state = raw_state
            self._consecutive_opposite = 0
            self._pending_target = None
            # Could log: [NetworkStateMachine] {old.value} → {self._state.value}
            return self._state

        return self._state  # not enough consecutive samples yet

    def _classify(self, bw: float, rtt: float) -> NetworkState:
        if bw >= self.bw_threshold_good and rtt <= self.rtt_threshold_good:
            return NetworkState.GOOD
        elif bw <= self.bw_threshold_poor or rtt >= self.rtt_threshold_poor:
            return NetworkState.POOR
        else:
            return NetworkState.DEGRADED
```

### D.4 Transition Diagram

```
                    bw<50 OR rtt>5 (3 samples)
            GOOD ─────────────────────────────────→ DEGRADED
              ↑                                       │
              │ bw>=50 AND rtt<=5 (5 samples)         │ bw<10 OR rtt>50 (3 samples)
              │                                       │
              │         bw<10 OR rtt>50               ↓
              └──────────── (1 sample) ──────────── POOR
              │                                       │
              │    bw>=50 AND rtt<=5 (5 samples)      │
              └───────────────────────────────────────┘
              │
              │    bw>=10 AND rtt<=50 (5 samples)      ↑
              └───────────────────────────────────────┘
                        (from POOR to DEGRADED)
```

### D.5 State-to-Budget Mapping

State determines `max_delay_sec` in the budget formula. It does NOT directly map to precision levels — that is PrecisionAllocator's optimization decision based on the resulting budget.

| State | max_delay_sec | Budget (at 80 MB/s) | Expected Outcome |
|-------|---------------|---------------------|------------------|
| GOOD | 0.5 | 34 MB | Mixed FP16/FP8/INT4 |
| DEGRADED | 0.3 | 20 MB | Mostly INT4, some FP8 for top entries |
| POOR | 0.2 | 14 MB | Almost all INT2, top entries at INT4 |

### D.6 Transition Rules Summary

| From | To | Condition | Samples Required |
|------|----|-----------|-----------------|
| GOOD | DEGRADED | BW < 50 OR RTT > 5 | 3 consecutive |
| GOOD | POOR | BW < 10 OR RTT > 50 | 1 (emergency) |
| DEGRADED | POOR | BW < 10 OR RTT > 50 | 3 consecutive |
| DEGRADED | GOOD | BW >= 50 AND RTT <= 5 | 5 consecutive |
| POOR | GOOD | BW >= 50 AND RTT <= 5 | 5 consecutive |
| POOR | DEGRADED | BW >= 10 AND RTT <= 50 | 5 consecutive |

---

## E. Calibration Integration

### E.1 The Feedback Loop Problem

The ABKT system has an inherent positive feedback loop risk:

```
Better compression → Smaller transfer → Less time → HIGHER apparent BW
  → Larger budget → Less compression → Larger transfer → More time
  → LOWER apparent BW → Smaller budget → Better compression → ...
```

If `record_transfer_complete()` used the compressed transfer size directly, the measured bandwidth would oscillate with the compression ratio, not reflect actual network capacity.

### E.2 Solution: Uncompressed-Equivalent Bandwidth

```python
def record_transfer_complete(
    self,
    compressed_bytes: int,       # actual bytes sent over the wire
    elapsed_sec: float,          # wall-clock time for the transfer
    compression_ratio: float,    # from alloc_result (FP16_total / compressed_total)
) -> None:
    """Update BW EWMA using uncompressed-equivalent throughput.

    Uses uncompressed-equivalent bytes to measure the NETWORK'S actual
    capacity, not the reduced data rate from compression. This prevents
    the compression→budget→compression feedback loop.
    """
    if elapsed_sec <= 0:
        return

    uncompressed_equivalent = compressed_bytes * compression_ratio
    measured_bw = uncompressed_equivalent / elapsed_sec  # bytes/sec

    # First calibration: direct replacement
    if self._bw_ewma_calibrated is None:
        self._bw_ewma_calibrated = measured_bw
        return

    # Subsequent: EWMA update
    self._bw_ewma_calibrated = (
        self.ewma_alpha_bw_calibrated * measured_bw
        + (1 - self.ewma_alpha_bw_calibrated) * self._bw_ewma_calibrated
    )
```

**Verification**: If actual network capacity is 80 MB/s, the bandwidth measurement converges to 80 MB/s regardless of whether ABKT compresses at 2x, 4x, or 8x.

### E.3 Calibration Hooks — Exact Locations

#### Hook 1: prefill_node.py (after KV cache extraction, before client)

```python
# prefill_node.py, around line 118 (after fingerprint print, before line 123)

# ── ABKT: Get network snapshot ──
if abkt_enabled:
    snapshot = probe.get_snapshot()
    print(f"[ABKT] Network: state={snapshot.state.value} "
          f"bw_ewma={snapshot.bandwidth_ewma/1e6:.1f}MB/s "
          f"rtt_ewma={snapshot.rtt_ewma:.1f}ms "
          f"budget={snapshot.budget_bytes/1e6:.1f}MB")

    # ── ABKT: Importance evaluation ──
    importance_map = importance_evaluator.compute_importance(kv_cache)

    # ── ABKT: Precision allocation ──
    alloc_result = precision_allocator.allocate(
        importance_map, kv_cache, snapshot.budget_bytes)
    print(f"[ABKT] Allocation: {alloc_result.avg_precision_bits:.1f} bits/elem "
          f"compression={alloc_result.compression_ratio:.1f}x "
          f"total={alloc_result.total_bytes/1e6:.1f}MB")

    # ── ABKT: Quantize ──
    quantized_kv, metadata = adaptive_quantizer.quantize_kv_cache(
        kv_cache, alloc_result.precision_map)
```

#### Hook 2: prefill_node.py (after transfer completes)

```python
# prefill_node.py, around line 146 (after client.call() returns)

if abkt_enabled:
    t_send = time.time() - t_send_start
    # Use uncompressed-equivalent to prevent feedback loop
    probe.record_transfer_complete(
        compressed_bytes=actual_bytes_sent,
        elapsed_sec=t_send,
        compression_ratio=alloc_result.compression_ratio,
    )
    print(f"[ABKT] Transfer: {actual_bytes_sent/1e6:.1f}MB compressed "
          f"in {t_send:.2f}s → calibrated BW updated")
```

### E.4 Calibration in decode_node.py

The decode node measures dequantization overhead for profiling (does not feed back to probe):

```python
# Inside the new handle_receive_kv_chunk / assemble path:
t_assemble = time.time()
full_kv_cache = assembler.assemble_and_dequantize(buffers, metadata)
assemble_time_ms = (time.time() - t_assemble) * 1000
print(f"[ABKT] Chunk assembly + dequantization: {assemble_time_ms:.1f}ms")
```

### E.5 Transfer Timing Granularity

For `record_transfer_complete()`, the prefill side measures:

```
t_start = time.time()
for each layer:
    for each chunk in layer:
        client.call("receive_kv_chunk", ...)  # synchronous — includes network RTT
actual_bytes += chunk_compressed_size
total_time = time.time() - t_start
```

Each `client.call("receive_kv_chunk")` includes:
- `send_obj()` serialization time
- Network transmission time
- Decode node processing time (buffer append, trivial)
- `recv_obj()` response deserialization time
- Network response time

The measured time includes both network and serialization overhead, giving a realistic end-to-end throughput measurement.

---

## F. Exact Integration Points

### F.1 prefill_node.py Integration Map

Reference: `/ssd/pd/ABKT/prefill_node.py` (current version)

| Line Range | Current Code | ABKT Change | Type |
|-----------|-------------|-------------|------|
| 1-34 | Imports | Add: `from backend.network_probe import NetworkProbeClient` and other ABKT imports | ADD |
| ~56 | CUDA check | UNCHANGED | --- |
| ~96-103 | Prefill forward pass | UNCHANGED | --- |
| ~113-120 | KVCache extraction + fingerprint | UNCHANGED | --- |
| **after ~120** | *(nothing)* | **INSERT**: ABKT snapshot, importance, allocate, quantize | NEW BLOCK |
| ~123-128 | SocketClient created + connect | UNCHANGED (SocketClient used by both paths) | --- |
| **~130-144** | `client.call("run_decode", kv_cache=...)` | **REPLACE**: if ABKT → `chunked_transfer.send()` + `client.call("run_decode_abkt", ...)`. else → original code | BRANCH |
| **after ~146** | `send_time` printed | **INSERT**: if ABKT → `probe.record_transfer_complete()` | NEW BLOCK |
| ~177 | `client.close()` | UNCHANGED | --- |

### F.2 decode_node.py Integration Map

Reference: `/ssd/pd/ABKT/decode_node.py` (current version)

| Line Range | Current Code | ABKT Change | Type |
|-----------|-------------|-------------|------|
| 1-39 | Imports | Add: `from backend.chunked_transfer import ChunkAssembler` and probe import | ADD |
| ~119 | Model loaded (`model.eval()`) | **INSERT**: ProbeServer start, chunk buffers init, dequantizer init | NEW BLOCK |
| ~127-300 | `handle_run_decode()` | UNCHANGED (backward compat) | --- |
| **after ~300** | *(nothing)* | **INSERT**: `handle_receive_kv_chunk()` and `handle_run_decode_abkt()` definitions | NEW HANDLERS |
| ~303 | `handlers = {"run_decode": handle_run_decode}` | **MODIFY**: add `"receive_kv_chunk"` and `"run_decode_abkt"` keys | EXTEND |
| ~304-307 | Server start | UNCHANGED | --- |

### F.3 pipeline.py Integration

Reference: `/ssd/pd/ABKT/pd_inference/pipeline.py`

The `PDRunner` class is used for the torch.distributed-based orchestration path. The current `prefill_node.py` and `decode_node.py` use `SocketClient`/`SocketServer` directly, NOT `PDRunner`. However, for future integration with `PDRunner.run_inference()`:

| Line Range | Current Code | ABKT Change |
|-----------|-------------|-------------|
| ~499-502 | `kv_cache, hidden = run_prefill_pipeline(...)` | **INSERT** (optional): if ABKT enabled, wrap with `abkt_process_kv(kv_cache)` before decode |
| ~524-533 | `run_decode_pipeline(...)` | If using chunked path: replace `init_kv` RPC calls with chunked receive |

**For the initial implementation, pipeline.py requires NO changes.** The integration happens at the `prefill_node.py` / `decode_node.py` level, which calls SocketClient/SocketServer directly.

### F.4 socket_transport.py Changes

Reference: `/ssd/pd/ABKT/pd_inference/socket_transport.py`

One addition needed: a `send_only` method on `SocketClient` for fire-and-forget chunk sends. However, the recommended approach is to keep the request-response pattern per chunk for natural flow control (sender won't outpace receiver). Each `receive_kv_chunk` call returns `{"ok": True, "ready": bool}`.

**No changes needed to SocketServer.** The handler dispatch pattern already supports any number of ops.

**One optional addition to SocketClient:**

```python
# In SocketClient class, after call() method (~line 240):

def send_raw(self, op: str, **payload) -> Any:
    """Send a request and receive response. Identical to call() but
    with a shorter timeout for chunk transfers (5s instead of 300s).
    """
    if self._sock is None:
        raise RuntimeError("Not connected. Call connect() first.")
    msg = {"op": op, "payload": payload}
    send_obj(self._sock, msg)
    resp = recv_obj(self._sock)
    if not isinstance(resp, dict):
        raise RuntimeError(f"Invalid response type: {type(resp)}")
    if resp.get("ok"):
        return resp.get("result")
    error = resp.get("error", "Unknown error")
    raise RuntimeError(f"Remote error: {error}")
```

### F.5 New RPC Handler Specifications

#### Handler: `receive_kv_chunk`

**Called by:** prefill_node.py for each chunk of each layer

**Payload:**
```json
{
    "request_id": "req-000001",
    "layer_idx": 0,
    "chunk_start": 0,
    "chunk_end": 64,
    "total_seq_len": 512,
    "k": "<tensor: [1, 32, 64, 128]>",
    "v": "<tensor: [1, 32, 64, 128]>",
    "meta": {
        "precision": 4,
        "scale_k": 0.0123,
        "zero_point_k": 7,
        "scale_v": 0.0456,
        "zero_point_v": 7
    },
    "is_last_chunk": false
}
```

**Response:**
```json
{
    "ok": true,
    "result": {
        "ready": false,
        "layers_received": 3,
        "layer_complete": false
    }
}
```

When `is_last_chunk` is true AND that layer was the last one expected: response includes assembled KV or signals readiness.

#### Handler: `run_decode_abkt`

**Called by:** prefill_node.py after all chunks sent

**Payload:** Same as `run_decode` but WITHOUT `kv_cache`:
```json
{
    "request_id": "req-000001",
    "input_ids": [1, 2, 3, ...],
    "first_token": 42,
    "max_new_tokens": 128,
    "repetition_penalty": 1.0,
    "do_sample": false,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0
}
```

**Response:** Same as `run_decode`:
```json
{
    "ok": true,
    "result": {
        "generated_ids": [1, 2, 3, 42, 99, ...],
        "generated_text": "Paris is the capital...",
        "num_tokens": 127,
        "time": 15.3
    }
}
```

### F.6 Backward Compatibility

- `--abkt-enabled` CLI flag (default: `False`). When disabled, the original single-shot flow runs unchanged.
- The probe server on decode unconditionally starts (port 9877). If no probe client connects, it's harmless — listens but idles.
- The existing `handle_run_decode` handler is preserved in its entirety.
- The existing `SocketClient.call("run_decode", ...)` path is preserved.

### F.7 ABKT Initialization in prefill_node.py main()

```python
# After config parsing (~line 52), before model loading (~line 60):

abkt_enabled = getattr(config, 'abkt_enabled', False)

if abkt_enabled:
    # Start network probe (runs in background)
    probe = NetworkProbeClient(
        target_host=config.master_addr,
        target_port=config.probe_port,  # default 9877
    )
    snapshot = probe.start()
    print(f"[ABKT] Probe started: {snapshot}")

    # Initialize ABKT modules
    importance_evaluator = TokenImportanceEvaluator()  # simplified Key-norm version
    precision_allocator = PrecisionAllocator()
    adaptive_quantizer = AdaptiveQuantizer()
    chunked_transfer = ChunkedTransfer(
        base_chunk_size=64,
        min_chunk_size=32,
        max_chunk_size=256,
    )
```

### F.8 ABKT Initialization in decode_node.py main()

```python
# After model loading (~line 119), before handler definitions:

# ── ABKT: Probe server ──
probe_server = ProbeServer(host="0.0.0.0", port=config.probe_port)
probe_server.start()
print(f"[ABKT] Probe server listening on port {config.probe_port}")

# ── ABKT: Chunk assembler ──
chunk_assembler = ChunkAssembler()
dequantizer = AdaptiveDequantizer()
```

---

## G. Testing Strategy

### G.1 tc netem Bandwidth Simulation

Run on **prefill node** (192.168.0.50, the sender) to shape outbound traffic to the decode node.

```bash
#!/bin/bash
# backend/tests/simulate_network.sh
# Usage: source backend/tests/simulate_network.sh
#        simulate_100mbps
#        simulate_10mbps
#        simulate_latency_50ms
#        simulate_loss_1pct
#        reset_network

INTERFACE="eth0"  # adjust if different

reset_network() {
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null || true
    echo "[net] Reset to default"
}

simulate_100mbps() {
    reset_network
    sudo tc qdisc add dev $INTERFACE root handle 1: htb default 10
    sudo tc class add dev $INTERFACE parent 1: classid 1:10 htb rate 100mbit ceil 100mbit
    echo "[net] Simulating 100 Mbps"
}

simulate_50mbps() {
    reset_network
    sudo tc qdisc add dev $INTERFACE root handle 1: htb default 10
    sudo tc class add dev $INTERFACE parent 1: classid 1:10 htb rate 50mbit ceil 50mbit
    echo "[net] Simulating 50 Mbps (DEGRADED trigger)"
}

simulate_10mbps() {
    reset_network
    sudo tc qdisc add dev $INTERFACE root handle 1: htb default 10
    sudo tc class add dev $INTERFACE parent 1: classid 1:10 htb rate 10mbit ceil 10mbit
    echo "[net] Simulating 10 Mbps (POOR trigger)"
}

simulate_latency_50ms() {
    reset_network
    sudo tc qdisc add dev $INTERFACE root netem delay 20ms 5ms distribution normal
    echo "[net] Simulating RTT 20ms ± 5ms"
}

simulate_loss_1pct() {
    reset_network
    sudo tc qdisc add dev $INTERFACE root netem loss 1%
    echo "[net] Simulating 1% packet loss"
}

# Variable bandwidth: cycles between 100mbit and 10mbit every 30 seconds
simulate_variable() {
    reset_network
    sudo tc qdisc add dev $INTERFACE root handle 1: htb default 10
    sudo tc class add dev $INTERFACE parent 1: classid 1:10 htb rate 100mbit ceil 100mbit
    echo "[net] Variable mode started — use change_bandwidth.sh to cycle"
}
```

### G.2 Unit Test Plan

All unit tests use `pytest`, mocking network I/O where appropriate.

#### UT-1: EWMA Computation (`test_network_probe.py::TestEWMA`)

```python
def test_ewma_initial_value():
    assert ewma_update(None, 100, 0.3) == 100

def test_ewma_convergence():
    """EWMA with alpha=0.3 converges to within 1% after ~20 steps."""
    val = None
    for _ in range(20):
        val = ewma_update(val, 50, 0.3)
    assert abs(val - 50) < 0.5

def test_ewma_step_response():
    """Alpha=0.3 reaches 90% of step change in ~5 steps."""
    val = ewma_update(None, 50, 0.3)  # start at 50
    for _ in range(5):
        val = ewma_update(val, 100, 0.3)
    assert val >= 90  # 90% of (100-50) = 45, so 95

def test_ewma_alpha_zero():
    """Alpha=0.0: EWMA never changes from initial."""
    val = ewma_update(None, 100, 0.0)
    val = ewma_update(val, 50, 0.0)
    assert val == 100

def test_ewma_alpha_one():
    """Alpha=1.0: EWMA instantaneously equals latest value."""
    val = ewma_update(None, 100, 1.0)
    val = ewma_update(val, 50, 1.0)
    assert val == 50
```

#### UT-2: State Machine (`test_network_probe.py::TestStateMachine`)

```python
def test_good_to_degraded():
    sm = NetworkStateMachine()
    assert sm.state == NetworkState.GOOD
    # 3 consecutive DEGRADED readings
    for _ in range(3):
        state = sm.update(bw_ewma=30_000_000, rtt_ewma=10.0)
    assert state == NetworkState.DEGRADED

def test_degraded_to_good_hysteresis():
    sm = NetworkStateMachine()
    # First degrade
    for _ in range(3):
        sm.update(bw_ewma=30_000_000, rtt_ewma=10.0)
    assert sm.state == NetworkState.DEGRADED
    # Need 5 consecutive GOOD to recover
    for _ in range(4):
        state = sm.update(bw_ewma=60_000_000, rtt_ewma=3.0)
    assert state == NetworkState.DEGRADED  # not yet
    state = sm.update(bw_ewma=60_000_000, rtt_ewma=3.0)
    assert state == NetworkState.GOOD  # 5th sample

def test_no_oscillation():
    """Alternating 49/51 MB/s should NOT toggle state."""
    sm = NetworkStateMachine()
    for _ in range(10):
        sm.update(bw_ewma=49_000_000, rtt_ewma=5.0)
    assert sm.state == NetworkState.GOOD  # never accumulated 3 consecutive DEGRADED
    sm.update(bw_ewma=51_000_000, rtt_ewma=5.0)  # resets counter
    for _ in range(3):
        sm.update(bw_ewma=49_000_000, rtt_ewma=5.0)
    # After 3 consecutive, should degrade
    assert sm.state == NetworkState.DEGRADED

def test_emergency_good_to_poor():
    sm = NetworkStateMachine()
    # One reading below POOR threshold = immediate transition
    state = sm.update(bw_ewma=5_000_000, rtt_ewma=5.0)
    assert state == NetworkState.POOR
```

#### UT-3: PIA Algorithm (`test_precision_allocator.py::TestPIA`)

```python
def test_budget_sufficient_all_fp16():
    """When budget >= FP16 total, everything gets FP16."""
    result = allocator.allocate(importance, kv_cache, budget=float('inf'))
    for dm in result.precision_map.values():
        for prec_tensor in dm.values():
            assert (prec_tensor == 16).all()

def test_budget_zero_all_min_precision():
    """When budget = 0, everything gets minimum precision (INT2)."""
    result = allocator.allocate(importance, kv_cache, budget=0)
    for dm in result.precision_map.values():
        for prec_tensor in dm.values():
            assert (prec_tensor == 2).all()

def test_monotonicity():
    """Higher importance entries never get lower precision."""
    result = allocator.allocate(importance, kv_cache, budget=moderate_budget)
    # Check: if imp_A > imp_B, prec_A >= prec_B
    ...

def test_budget_constraint():
    """Actual computed bytes <= budget_bytes."""
    result = allocator.allocate(importance, kv_cache, budget=budget)
    assert result.total_bytes <= budget * 1.01  # 1% tolerance for rounding

def test_compression_ratio():
    """Compression ratio matches FP16_total / actual_total."""
    result = allocator.allocate(importance, kv_cache, budget=budget)
    expected = fp16_total / result.total_bytes
    assert abs(result.compression_ratio - expected) < 0.01
```

#### UT-4: Quantization Roundtrip (`test_adaptive_quant.py::TestRoundtrip`)

```python
def test_fp16_passthrough():
    """FP16: zero loss."""
    tensor = torch.randn(1, 32, 64, 128, dtype=torch.float16)
    qt, meta = quantize_tensor(tensor, Precision.FP16)
    dq = dequantize_tensor(qt, meta, Precision.FP16)
    assert torch.equal(tensor, dq)

def test_fp8_max_error():
    """FP8: max error < 5% of value range."""
    tensor = torch.randn(1, 32, 64, 128, dtype=torch.float16)
    qt, meta = quantize_tensor(tensor, Precision.FP8)
    dq = dequantize_tensor(qt, meta, Precision.FP8)
    error = (dq.float() - tensor.float()).abs()
    value_range = tensor.max() - tensor.min()
    assert (error / max(value_range, 1e-8)).max() < 0.05

def test_int4_max_error():
    """INT4: max error < 15% of value range."""
    ...

def test_int2_max_error():
    """INT2: max error < 30% of value range."""
    ...

def test_l2_relative_error():
    """L2 relative error within bounds."""
    tensor = torch.randn(1, 32, 128, 128, dtype=torch.float16)
    for prec, max_re in [(Precision.FP8, 0.02), (Precision.INT4, 0.10), (Precision.INT2, 0.25)]:
        qt, meta = quantize_tensor(tensor, prec)
        dq = dequantize_tensor(qt, meta, prec)
        re = (dq.float() - tensor.float()).norm() / tensor.float().norm()
        assert re < max_re, f"{prec}: relative error {re} >= {max_re}"
```

#### UT-5: Feedback Loop Prevention (`test_network_probe.py::TestFeedbackLoop`)

```python
def test_bw_ewma_independent_of_compression():
    """Simulate transfers at 2x, 4x, 8x compression. BW_EWMA stays stable."""
    probe = NetworkProbeClient(...)
    probe.calibrate_with_transfer(
        num_bytes=100_000_000, elapsed_sec=1.0)  # seeds at 100 MB/s

    # Transfer with 2x compression (50 MB compressed, 0.5s)
    probe.record_transfer_complete(50e6, 0.5, 2.0)
    assert abs(probe._bw_ewma_calibrated - 100e6) < 5e6  # should stay near 100

    # Transfer with 8x compression (12.5 MB compressed, 0.125s)
    probe.record_transfer_complete(12.5e6, 0.125, 8.0)
    assert abs(probe._bw_ewma_calibrated - 100e6) < 5e6  # still near 100

    # The EWMA should converge to actual network capacity (100 MB/s),
    # not the compressed data rate
```

### G.3 Integration Test Plan

#### IT-1: Local End-to-End (same machine, OPT-125M)

```
Setup: Run prefill_node.py and decode_node.py on localhost
Model: facebook/opt-125m (fits easily in memory)
Test:
  1. ABKT disabled: run 3 prompts, verify output matches baseline
  2. ABKT enabled + FP16 pass-through: verify tokens identical to baseline
  3. ABKT enabled + forced DEGRADED (simulate via probe mock): verify tokens
  4. Verify chunk assembly correctness: compare assembled KV to original
Measured: transfer time, chunk count, assembly time, token output
```

#### IT-2: Two-Machine Baseline (x86 + Jetson, OPT-350M)

```
Setup: Prefill on 192.168.0.50, Decode on 192.168.0.20
Model: facebook/opt-350m
Test:
  1. Direct transfer (ABKT disabled): measure baseline transfer time, TTFT
  2. ABKT enabled: same prompts, compare transfer time, TTFT, tok/s
  3. Verify fingerprint match: KVCache on prefill vs DynamicCache on decode
Measured: transfer_ms, TTFT_ms, tokens_per_sec, compression_ratio
```

#### IT-3: Network Condition Scenarios (with tc netem)

```
Setup: Two-machine + tc netem on prefill node (192.168.0.50)
Model: facebook/opt-350m
Scenarios:
  a) 100 Mbps stable: verify GOOD state, high precision allocation
  b) 50 Mbps stable: verify DEGRADED state, moderate compression
  c) 10 Mbps stable: verify POOR state, aggressive compression
  d) 100→10 Mbps drop mid-transfer: verify probe detects within 3 samples,
     remaining layers get compressed
  e) 20ms latency: verify RTT EWMA triggers DEGRADED
Measured: state transitions, precision distribution, token quality (PPL proxy)
```

#### IT-4: State Transition Stress Test

```
Setup: tc netem variable bandwidth script
Test: Run inference every 10 seconds for 5 minutes
      Bandwidth oscillates between 100 Mbps and 10 Mbps every 30 seconds
Verify:
  - No crashes or deadlocks
  - State machine transitions correctly with hysteresis
  - Budget adjusts to each new state
  - Calibrated BW converges to actual capacity over multiple transfers
```

### G.4 Regression Tests

| Test | Condition | Expected |
|------|-----------|----------|
| ABKT disabled | `--abkt-enabled` not set | Output identical to current main branch |
| Legacy `run_decode` | Client calls `run_decode` with full kv_cache | Decode works as before |
| Probe failure resilience | Probe server not running | ABKT degrades to FP16 pass-through |
| Multiple sequential requests | 10 requests without restart | Each request gets updated network state |
| Memory leak test | 100 requests | No monotonic memory growth in chunk buffers |

---

## H. Assumptions Requiring Experimental Validation

1. **Actual bandwidth between 192.168.0.50 and 192.168.0.20**: All thresholds assume ~80-100 MB/s practical throughput on 1 Gbps Ethernet. If the physical link is 100 Mbps or WiFi, all thresholds must be rescaled proportionally (divide by ~10).

2. **Jetson dequantization speed**: INT4/INT2 dequantization on ARM CPU may take 10-50ms per layer at 512 tokens. This must be measured before committing to aggressive compression. If dequantization exceeds 5ms per layer, implement it on Jetson GPU instead.

3. **Quality fidelity values**: FP8=0.98, INT4=0.92, INT2=0.80 are estimates. Need per-model calibration by measuring PPL degradation on WikiText-2 for each precision level individually.

4. **torch.save serialization overhead**: `SocketClient.send_obj()` uses `torch.save` which adds significant overhead for large tensors. For production, raw tensor bytes via `numpy().tobytes()` with a custom header may be needed for compressed formats.

5. **Probe connection churn on Jetson**: 500ms interval means ~2 new TCP connections per second to port 9877. Must verify Jetson's networking stack handles this without resource exhaustion.

6. **Chunk reassembly memory on decode**: Peak buffer memory is `num_layers × tokens × heads × head_dim × bytes_per_element × 2(K+V)`. For OPT-6.7B at 512 tokens FP16: 268 MB per request. Jetson AGX Orin has 64GB RAM — acceptable for single requests, but needs attention for concurrent request handling.

7. **TCP_NODELAY interaction**: The existing SocketClient does not set `TCP_NODELAY`. For chunked transfer with small chunks, Nagle's algorithm may add ~200ms delay per chunk. Consider adding `sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)` to the SocketClient constructor for the data connection when ABKT is enabled.
