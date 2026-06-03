# Network Probe Integration Analysis for Prefill->Decode Pipeline

**Date:** 2026-05-18
**Context:** ABKT prefill (x86, 192.168.0.50) -> decode (Jetson Orin, 192.168.0.20) over 1 GbE

## Existing Code Integration Points

The key touchpoints between the network module and the transfer pipeline are:

| Phase | Code Location | What happens |
|-------|--------------|--------------|
| Before prefill | (none yet) | Probe starts, state initialized |
| After prefill forward pass | `prefill_node.py:113-115` | KV cache extracted |
| Before transfer | `PrecisionAllocator.allocate()` takes `budget_bytes` | Budget from `NetworkProbeClient.get_snapshot()` |
| During transfer | `ChunkedSender.send_all()` per-chunk sends | No mid-transfer monitoring currently |
| After transfer | (none yet) | `record_transfer()` calibration call |

The critical gap is that `prefill_node.py` currently uses `SocketClient.call("run_decode", kv_cache=...)` directly, bypassing the entire ABKT network-aware pipeline (no probing, no budget, no precision allocation, no chunked transfer).

---

## 1. Probe Lifecycle: Long-Running Daemon vs Per-Request

### Recommendation: Long-running daemon with lazy-start on first request

**Start the probe server on decode node and probe client on prefill node at process startup, not per-request.**

**Why:**

1. **EWMA needs history to be meaningful.** The BW EWMA with alpha=0.3 reaches ~84% of steady-state after 5 samples. At 10s intervals for bandwidth probes (GOOD state), that's 50 seconds. If you restart probing per-request, the EWMA is seeded by a single cold-start value and never reaches a reliable estimate for short prompts.

2. **RTT probes (1s interval) provide fast congestion detection.** RTT spikes are the earliest signal of network contention. A daemon that probes RTT every second catches congestion before the first byte of KV cache is sent.

3. **Persistent TCP connection amortizes handshake overhead.** The RTT probe connection in `_get_rtt_conn()` is kept open between probes. Creating a fresh connection per request would add ~1-3ms of three-way-handshake noise to every RTT measurement.

**Integration point:** In `prefill_node.py` main(), after CUDA check and before model loading:

```python
# Start network probe daemon (best-effort, won't block if decode node isn't up yet)
probe = NetworkProbeClient(target_host=config.master_addr, target_port=9877)
probe.start()
```

The probe server (`ProbeServer`) should be started on the decode node at process startup via `decode_node.py`, before the model is loaded. This gives a few seconds of probing history by the time the first request arrives.

**Edge case: decode node not ready yet.** The probe client's background thread will fail silently (all exceptions caught in `_probe_loop`). When the decode node comes online, the next probe cycle will succeed and start populating the EWMA. This is acceptable -- the first request will use the cold-start default.

### Do NOT run warmup_connection() blindly for every request

`warmup_connection()` sends 100KB of dummy data. This is useful before the first-ever KV cache transfer to open the TCP congestion window, but doing it per-request adds 100KB of unnecessary traffic. Call it once after probe startup when `is_calibrated()` first becomes true, or skip it entirely if the probe's own bandwidth probes (1MB payloads) have already opened the window.

---

## 2. Cold Start Problem

### Analysis of strategies

| Strategy | Latency overhead | Quality impact | Implementation effort |
|----------|-----------------|----------------|----------------------|
| A: Conservative default | 0ms | INT2 for everything when actual BW is good | Already implemented (`COLD_BW = 15e6`) |
| B: Quick bandwidth probe | ~50-100ms | Near-optimal after probe | Need to add a single-shot fast probe |
| C: Prefill time proxy | 0ms | Unreliable (compute-bound, not network-bound) | Not worth pursuing |

### Recommendation: Strategy B (pipelined quick probe) with A as fallback

**Pipeline the quick probe alongside model loading:**

1. Start `ProbeServer` on decode node before model loading (it's a thread, non-blocking)
2. Start `NetworkProbeClient` on prefill node before model loading
3. While the model loads (~5-15 seconds on x86), the probe thread runs RTT probes at 1s intervals and a first bandwidth probe within ~1s
4. By the time model loading completes, the EWMA has 5-15 RTT samples and at least 1 bandwidth sample (1MB burst at t=1s)

**This gives us a calibrated estimate before the first request arrives, with zero additional latency.**

**Concrete integration:**

```python
# In prefill_node.py main(), ORDER MATTERS:
# 1. Start probe client (immediately begins background probing)
probe = NetworkProbeClient(target_host=config.master_addr, target_port=9877)
probe.start()

# 2. Load model (probe runs concurrently during this ~10-15s window)
prefill_stage = PrefillStage(...)
prefill_stage.load()

# 3. By now, probe has 10+ RTT samples and 1+ BW sample
#    Force one calibrated BW probe if not yet calibrated
if not probe.is_calibrated():
    probe._probe_bandwidth(data_size=2 * 1024 * 1024)  # 2MB, synchronous, ~20ms on 1GbE
    # If still not calibrated (decode node unreachable), fall through to COLD_BW

# 4. Now use the snapshot for the first request
snapshot = probe.get_snapshot()
```

**Cold start value tuning:** The current `COLD_BW = 15e6` is reasonable for the "decode node unreachable" case. If the decode node is reachable but probing failed, lower `COLD_BW` slightly to 12e6 to be more conservative. We'd rather waste latency (by compressing too much) than waste quality (by sending too little data in a tight budget).

**Why not Strategy C (prefill time proxy):** Prefill forward pass time is dominated by compute (matrix multiplies on GPU), not the network. A 32-layer model on a 128-token prompt might take 200ms on GPU vs 500ms on CPU. Network bandwidth cannot be inferred from this -- it's a completely different resource. The only correlation is "if prefill is slow because the machine is overloaded, network might also be slow," but this is too weak to use as a bandwidth estimate.

---

## 3. Calibration Integration

### Where to call `record_transfer()`

**Right after the send completes, with the actual compressed byte count and elapsed time.**

The exact insertion point in the integrated pipeline:

```python
# Pseudocode for the integrated prefill_node flow:
snapshot = probe.get_snapshot()                          # 1. Get budget
importance = evaluator.compute(kv_cache, ...)            # 2. Score tokens
alloc_result = allocator.allocate(importance, kv_cache,  # 3. Allocate precision
                                   snapshot.budget_bytes)
quantized, metadata = quantizer.quantize(kv_cache,       # 4. Quantize
                                         alloc_result.precision_map)

# 5. Send (with timing)
t0 = time.time()
sender.send_all(quantized, metadata, importance,
                snapshot.bandwidth_bps, request_id)
elapsed = time.time() - t0

# 6. Calibrate AFTER send completes
total_compressed = _sum_bytes(quantized)                 # actual compressed bytes sent
probe.record_transfer(
    compressed_bytes=total_compressed,
    elapsed_sec=elapsed,
    compression_ratio=alloc_result.compression_ratio,    # FP16 / compressed size
)
```

### The compression_ratio parameter

The ratio is available from `AllocationResult.compression_ratio`, computed as `total_fp16_size / actual_compressed_size`. This is exact -- no estimation needed. The flow:

1. `PrecisionAllocator.allocate()` returns `AllocationResult` with `.compression_ratio`
2. `AdaptiveQuantizer.quantize()` produces the actual compressed tensors
3. After `ChunkedSender.send_all()` completes, we have real elapsed time and real compressed bytes
4. `record_transfer(compressed_bytes, elapsed, compression_ratio)` computes uncompressed-equivalent BW

The order is: allocate -> quantize -> send -> record. The compression ratio from step 1 matches what was actually sent in step 2, so there's no estimation gap.

### Why calibrate on real transfers instead of bandwidth probes

The probe's `_probe_bandwidth()` sends 1MB of uniform zero bytes. The actual KV cache transfer is different in every way: larger (2-512 MB), the data is non-uniform (affects TCP's behavior less than one might think, but still), and it's chunked across multiple send calls. Calibration from real transfers is always more accurate for predicting the next real transfer.

**The 1MB bandwidth probes should be treated as a fast, cheap signal for state transitions, not the primary bandwidth estimate.** The EWMA design already prioritizes calibration (higher weight via `CALIBRATION_ALPHA=0.5` when called through `calibrate_with_transfer`, vs `update_clamped` for probe samples).

---

## 4. State-Driven Adaptation

### The oscillation concern: does `update_clamped(±20%)` suffice?

Let's trace the potential feedback loop:

1. Network is POOR -> budget shrinks (0.2s * 0.7 margin) -> more INT4/INT2 compression
2. 4x smaller payload -> transfer completes faster -> calibration sees high BW
3. High BW -> state upgrades to GOOD -> budget expands -> less compression
4. Large payload -> transfer is slow -> calibration sees low BW -> downgrade again

The ±20% clamp prevents step (3) from being too aggressive -- even if calibration measures 80 MB/s when EWMA was 10 MB/s, the EWMA only rises to 12 MB/s (10 * 1.2). Then the next calibration might push it to 14.4, then 17.3, etc. This is a slow ramp-up, not an instant swing.

**However**, the state machine adds a second feedback mechanism: state transitions change `max_delay` and `safety_margin`, which are step-function changes to the budget. Going from POOR (0.2s * 0.7) to DEGRADED (0.3s * 0.8) is a 1.7x budget increase even before the BW EWMA changes.

**Is this a problem in practice?** Probably not severe, because:
- The hysteresis voting (4/5 to upgrade) means the EWMA must show 4 out of 5 samples above the threshold before upgrading
- At alpha=0.3 and ±20% clamp, it takes ~5-6 calibration events to go from 10 MB/s to 40 MB/s
- With 10s between bandwidth probes, that's 50-60 seconds for a full recovery. This is intentionally slow.

**But there is one gap:** during rapid state transitions caused by RTT changes (not BW). RTT can spike and recover in seconds. The state machine maps RTT > 20ms to DEGRADED regardless of bandwidth. If RTT briefly spikes to 25ms for 2 samples (2 seconds), the state drops to DEGRADED, the budget drops, and the next request gets more aggressive compression. But the RTT recovers by the time the request starts. This is a false-positive degradation.

**Recommendation: Add an RTT EWMA vs instant-RTT distinction to the state machine.**

```python
# In state_machine.py _classify_sample():
# Use EWMA RTT (not instant) for DEGRADED trigger
# This prevents single-sample RTT spikes from triggering state changes
```

Currently, the state machine is called with `_rtt_ewma.value` (the EWMA, not the instant value). Looking at `_probe_rtt()`:

```python
self._state_machine.update(
    self._bw_ewma.value or COLD_BW,
    self._rtt_ewma.value or rtt,   # <-- uses EWMA, good
)
```

This is already using the EWMA RTT for state decisions. The EWMA at alpha=0.5 smooths single-sample spikes: a spike from 5ms to 30ms for one sample pushes the EWMA to 0.5*30 + 0.5*5 = 17.5ms, still below DEGRADED threshold (20ms). So this is adequately handled.

### Should probe interval decrease in POOR state?

**Current code already does this.** `probe_interval_bw` returns 5.0s for POOR/DEGRADED vs 10.0s for GOOD. This is correct: faster probing in poor conditions to detect recovery sooner.

**But there's a subtlety:** the POOR state also reduces `probe_bw_data_size` to 256KB (from 1MB). The intent is to avoid competing with real traffic during congestion. This is the right instinct, but 256KB is too small for an accurate bandwidth measurement on 1 GbE -- the transfer completes in ~2ms, which is within noise range of TCP scheduling. A 1MB probe is only 8ms on a 1 GbE link in GOOD state, and even in POOR state (if BW is ~6 MB/s), it's ~170ms. Neither is disruptive compared to a 100-500ms KV cache transfer.

**Recommendation: Keep 1MB as the minimum probe size, regardless of state.** The 256KB reduction is premature optimization and hurts measurement accuracy more than it helps reduce contention. In POOR state, the KV cache transfers will be small anyway (due to high compression), so a 1MB probe might actually be LARGER than the real traffic -- which is fine, it just means the probe data dominates the measurement.

---

## 5. Emergency Downgrade During Transfer

### The problem

The current `ChunkedSender.send_all()` sorts layers by importance (highest first) and sends them sequentially. If bandwidth collapses mid-transfer:
- Early layers (high importance) were already sent at the precision budgeted pre-transfer
- Late layers (low importance) haven't been sent yet

**This is actually the desired behavior** since important layers are sent first. But there's no mechanism to downgrade the remaining layers if bandwidth collapses.

### Design: Per-layer timing check with emergency re-quantization

Add a callback to `ChunkedSender` that checks elapsed time vs expected time after each layer:

```python
class ChunkedSender:
    def __init__(self, send_fn, quantizer, on_layer_complete=None):
        ...
        self.on_layer_complete = on_layer_complete  # Callable(layer_idx, elapsed, bytes_sent)

    def send_all(self, quantized_kv, ...):
        t_start = time.time()
        total_bytes = 0
        for dnode in sorted(quantized_kv.keys()):
            for lidx in layer_order:
                kv = layer_cache.get(lidx)
                if kv is None:
                    continue
                # ... send chunks for this layer ...
                layer_bytes = self._layer_size(kv)
                total_bytes += layer_bytes
                elapsed = time.time() - t_start
                expected_elapsed = total_bytes / bandwidth_bps  # from pre-transfer snapshot

                if elapsed > expected_elapsed * 2.0:
                    # Bandwidth is half of expected -> emergency
                    if self.on_layer_complete:
                        self.on_layer_complete(lidx, elapsed, total_bytes, emergency=True)
```

The emergency callback would:
1. Re-query `probe.get_snapshot()` for the latest EWMA (RTT probes continue during transfer)
2. If state is now POOR/DEGRADED and was previously GOOD, re-run `PrecisionAllocator.allocate()` for the remaining layers with the new, smaller budget
3. Re-quantize the remaining layers and continue sending

**Critical design constraint:** The decode-side `ChunkAssembler` must know the precision of each layer to dequantize. Currently, precision metadata is per-layer in `metadata`. If a layer is re-quantized mid-transfer, the metadata must be updated.

**Simpler alternative (recommended for Phase 1): Don't re-quantize mid-transfer. Instead, detect the degradation and cancel + restart the transfer with updated budget.**

The cost of cancellation is: the decode node discards partial chunks (via `ChunkAssembler.cancel()`), the prefill node re-allocates precision with the new (tighter) budget, re-quantizes, and re-sends. The cost is roughly 1x the transfer time of already-sent layers. This is simpler than mid-stream re-quantization and handles the edge case correctly.

**For Phase 2:** If cancellation overhead is too high (e.g., 200MB already sent before degradation), implement the mid-transfer re-quantization approach described above.

### Detection mechanism

While `ChunkedSender` is sending, the probe thread continues running in the background (it's a daemon thread). The RTT probes fire every 1 second. If RTT spikes mid-transfer (detectable on next `get_snapshot()` call), bandwidth has likely collapsed.

**The check should happen between layers, not between chunks.** Chunks are small (16-256 tokens), checking every chunk would add overhead. A layer is typically 1/32 of the total transfer, so checking 32 times per transfer is reasonable.

### Concrete integration:

```python
# In ChunkedSender.send_all(), after each layer:
if lidx % 4 == 0:  # Check every 4th layer (~8 checks per 32-layer transfer)
    current_snapshot = probe.get_snapshot()
    if current_snapshot.state == NetworkState.POOR and original_state == NetworkState.GOOD:
        logger.warning(f"Emergency: state degraded GOOD->POOR mid-transfer at layer {lidx}")
        # Option A: Cancel and restart
        return {"status": "emergency_restart", "layer_completed": lidx}
        # Option B: Re-budget remaining layers
        remaining_layers = layers[lidx+1:]
        new_budget = current_snapshot.budget_bytes - bytes_already_sent
        # ... re-allocate, re-quantize, continue ...
```

---

## 6. Concurrent Probing vs Transfer

### The contention question

Bandwidth probes send 1MB of data on a new TCP connection. The KV cache transfer sends data on a different connection (the `SocketClient` connection on port 29501, or potentially the same probe port 9877). Two TCP connections competing on the same 1 GbE link will roughly share bandwidth equally during the probe burst (~8ms for 1MB at 1 Gbps).

**Is this a problem?** For a 100ms KV cache transfer, a concurrent 8ms probe steals ~8% of the bandwidth during that window. This is measurable but not catastrophic. For a 500ms transfer, it's 1.6%.

### Recommendation: Pause bandwidth probes during KV cache transfer, but continue RTT probes

**Why pause bandwidth probes:**
1. Calibration from the real transfer is more accurate than a 1MB probe anyway
2. Avoids any measurement interference
3. The probe data competes with the KV cache on the shared 1 GbE link

**Why continue RTT probes:**
1. RTT probes are 64 bytes round-trip -- negligible bandwidth impact
2. RTT is the earliest signal of congestion (spikes before bandwidth drops)
3. Enables mid-transfer emergency detection (Section 5)

**Implementation:** Add a `pause_bw_probes` / `resume_bw_probes` method to `NetworkProbeClient`:

```python
class NetworkProbeClient:
    def __init__(self, ...):
        ...
        self._bw_probes_paused = False

    def pause_bw_probes(self):
        """Pause bandwidth probes during KV cache transfer."""
        self._bw_probes_paused = True

    def resume_bw_probes(self):
        self._bw_probes_paused = False

    def _probe_loop(self):
        ...
        if do_bw and not self._bw_probes_paused:
            self._probe_bandwidth(data_size=bw_data_size)
        else:
            self._probe_rtt()  # RTT always runs
```

This adds ~6 lines to `network_probe.py`. The prefill node calls `probe.pause_bw_probes()` before `sender.send_all()` and `probe.resume_bw_probes()` after.

### Alternative: Use the data connection for RTT measurement

Instead of the separate RTT connection, the prefill node could measure RTT by timing the TCP ACK for the last chunk sent. This is "free" (no extra connection) and more representative of the actual data path. But it requires application-level ACKs from the decode node, which the current `ChunkedSender`/`SocketClient` protocol doesn't support.

**Not recommended for Phase 1.** The persistent RTT connection is lightweight enough.

---

## Summary: Recommended Integration Pipeline

Here's the end-to-end flow for a request, with concrete code placement:

```
1. PROCESS STARTUP (prefill_node.py and decode_node.py)
   ├── decode_node: ProbeServer(port=9877).start()     # daemon thread
   └── prefill_node: probe = NetworkProbeClient(...).start()  # daemon thread
                      probe runs RTT every 1s, BW every 5-10s during model load

2. PER-REQUEST (prefill_node.py)
   ├── Run prefill forward pass (existing)
   ├── snapshot = probe.get_snapshot()                  # network_probe.py:198
   ├── imp = evaluator.compute(kv_cache, ...)            # token_importance.py:31
   ├── alloc = allocator.allocate(imp, kv_cache,         # precision_allocator.py:83
   │                              snapshot.budget_bytes)
   ├── quantized, meta = quantizer.quantize(kv_cache,    # adaptive_quant.py:24
   │                                       alloc.precision_map)
   ├── probe.pause_bw_probes()                           # network_probe.py (new)
   ├── sender.send_all(quantized, meta, imp,             # chunked_transfer.py:54
   │                    snapshot.bandwidth_bps, req_id)
   │   └── [every 4 layers: check probe.get_snapshot()   # emergency detection
   │        for state degradation]
   ├── probe.resume_bw_probes()                          # network_probe.py (new)
   └── probe.record_transfer(                            # network_probe.py:231
           compressed_bytes=total_sent,
           elapsed_sec=elapsed,
           compression_ratio=alloc.compression_ratio)
```

## Specific Code Changes Required

### `backend/network_probe.py` (3 additions)

1. `pause_bw_probes()` / `resume_bw_probes()` — 6 lines
2. Expose `is_calibrated()` — already exists at line 267
3. Force-first-bw-probe: expose a synchronous `probe_bandwidth_once(data_size=2MB)` for the pipelined cold-start flow

### `backend/chunked_transfer.py` (1 addition)

4. Mid-transfer state check: add optional `probe_client` parameter to `send_all()`. After every Nth layer, call `probe.get_snapshot()` and compare state to pre-transfer state. Return early with `{"status": "emergency_degraded", ...}` if state worsens.

### `prefill_node.py` (major integration)

5. Initialize `NetworkProbeClient` at startup
6. Replace `SocketClient.call("run_decode", kv_cache=...)` with the full ABKT pipeline: allocate -> quantize -> chunked_send -> calibrate
7. Handle emergency degradation response (cancel + retry with tighter budget)

### `decode_node.py` (1 addition)

8. Start `ProbeServer` at startup

### `backend/state_machine.py` (1 tune)

9. Consider reducing `probe_bw_data_size` for POOR back to 1MB (currently 256KB) for measurement accuracy -- or keep 256KB and accept the noise tradeoff. The 256KB value is fine for detecting state transitions but poor for EWMA accuracy. Since calibration from real transfers is the primary BW source, 256KB probes are adequate for state detection.

---

## Open Questions

1. **What happens if decode node restarts?** The probe client detects connection failure, resets the RTT connection, and continues probing. The EWMA doesn't reset (by design), so bandwidth estimate persists across decode restarts. This is acceptable since bandwidth is a property of the link, not the endpoint.

2. **Multiple concurrent requests?** The current prefill_node.py is single-request. If concurrent requests are added later, the probe daemon serves all of them. The `record_transfer()` calibration must be serialized (the `_lock` already ensures this). Concurrent transfers would produce interleaved calibration events, which is fine for EWMA.

3. **Should we measure the decode node's outbound bandwidth too?** The decode node sends small responses (token IDs, ~4 bytes per token). Not worth probing. The 1 GbE link is symmetric in capacity, so the prefill->decode measurement applies in both directions.

4. **Probe port vs data port:** The probe server uses port 9877. Data transfer uses port 29501 (SocketClient). These are separate connections. On a 1 GbE link, two TCP connections share the same bottleneck queue at the switch/NIC. The probe measurements are representative of the data path because the bottleneck is the link, not the port.
