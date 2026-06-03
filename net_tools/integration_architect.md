# ABKT Integration Architecture

Precision allocation, adaptive quantization, and chunked transfer for the prefill-to-decode KV cache pipeline.

---

## 1. High-Level Data Flow

```
prefill forward pass
  -> KVCache.from_dynamic_cache(past)          # extract KV (unchanged)
  -> _convert_to_abkt(kv_cache)                # KVCache -> {0: {layer: (k,v)}}
  -> TokenImportanceEvaluator.compute()         # per-token importance [0..1]
  -> probe.get_snapshot()                       # budget_bytes from network state
  -> PrecisionAllocator.allocate()              # precision_map per layer
  -> AdaptiveQuantizer.quantize()               # quantized_kv + metadata
  -> SocketClient.send_raw(chunks...)           # stream chunks to decode
  -> SocketClient.recv_obj()                    # wait for decode result
```

```
decode side (new handler)
  -> ChunkReceiver loop: recv_raw() chunks
  -> ChunkAssembler.add_chunk() per chunk
  -> on_complete: dequantize -> DynamicCache -> decode loop
```

---

## 2. Insertion Points in prefill_node.py

### 2.1 Module lifecycle (persistent, cross-request)

Network probing must start EARLY, before model loading, because:
- Model loading takes 5-15 seconds, providing probing warmup time
- Cold-start bandwidth is the single biggest risk to quality

**File: `/ssd/pd/ABKT/prefill_node.py`**

| Line | What changes | Why |
|------|-------------|-----|
| After line 16 | Add ABKT imports | New dependencies |
| After line 50 (config parse) | `probe = NetworkProbeClient(config.master_addr); probe.start()` | Start probing BEFORE model load |
| After line 76 (model loaded) | `probe.warmup_connection()` | Open TCP congestion window on probe connection |
| After line 113 (KVCache.from_dynamic_cache) | Insert ABKT pipeline: importance -> snapshot -> allocate -> quantize | Core pipeline |
| Line 126 (client.connect) | Keep this, client is used for the init RPC AND chunk streaming | SocketClient gets a `send_raw` method |
| Lines 133-144 (client.call) | Replace with chunked send + final result wait | See section 2.3 |
| After line 175 (finally: client.close) | `probe.stop()` | Clean shutdown |

### 2.2 New imports (insert after line 16)

```python
# Add after line 16 in prefill_node.py:
from backend.network_probe import NetworkProbeClient, ProbeServer
from backend.token_importance import TokenImportanceEvaluator
from backend.precision_allocator import PrecisionAllocator
from backend.adaptive_quant import AdaptiveQuantizer
from backend.chunked_transfer import ChunkedSender
```

### 2.3 Core pipeline insertion (replaces lines 113-144)

**Current code (lines 113-144):**
```python
    past = outputs.past_key_values
    kv_cache = KVCache.from_dynamic_cache(past)
    # ... print diagnostics ...
    client = SocketClient(config.master_addr, config.master_port, timeout=600.0)
    client.connect()
    result = client.call("run_decode", kv_cache=kv_cache.to_transport_dict(), ...)
```

**Replacement:**

```python
    # ── Extract KV cache (unchanged) ──
    past = outputs.past_key_values
    kv_cache = KVCache.from_dynamic_cache(past)
    # ... print diagnostics (unchanged) ...

    # ══════════════════════════════════════════════════════════
    # ABKT pipeline: importance -> allocate -> quantize -> send
    # ══════════════════════════════════════════════════════════

    # Step 1: Convert KVCache to ABKT internal format
    # KVCache layers: [{layer_idx, k, v}] -> {0: {layer_idx: (k, v)}}
    abkt_kv = _kvcache_to_abkt(kv_cache)

    # Step 2: Evaluate token importance
    evaluator = TokenImportanceEvaluator()
    importance_map = evaluator.compute(abkt_kv, num_layers, kv_cache.seq_len)

    # Step 3: Get network snapshot (bandwidth, RTT, budget)
    snapshot = probe.get_snapshot()
    print(f"[prefill] Network snapshot: {snapshot}")
    budget = snapshot.budget_bytes

    # Step 4: Allocate precision under budget
    allocator = PrecisionAllocator()
    allocation = allocator.allocate(importance_map, abkt_kv, budget)
    print(f"[prefill] Allocation: avg_bits={allocation.avg_precision_bits:.1f} "
          f"compression={allocation.compression_ratio:.1f}x "
          f"bytes={allocation.total_bytes/1e6:.1f}MB / budget={budget/1e6:.1f}MB")

    # Step 5: Quantize
    quantizer = AdaptiveQuantizer()
    quantized_kv, q_metadata = quantizer.quantize(abkt_kv, allocation.precision_map)

    # ══════════════════════════════════════════════════════════
    # Transfer
    # ══════════════════════════════════════════════════════════

    print(f"\n[prefill] Connecting to decode node at "
          f"{config.master_addr}:{config.master_port}...")

    client = SocketClient(config.master_addr, config.master_port, timeout=600.0)
    try:
        client.connect()

        # Generate request ID and send chunk init
        request_id = f"req_{int(time.time()*1000)}"
        client.send_raw({
            "op": "run_decode_abkt",
            "payload": {
                "request_id": request_id,
                "input_ids": input_ids_t[0].tolist(),
                "first_token": first_token,
                "max_new_tokens": config.max_new_tokens,
                "repetition_penalty": config.repetition_penalty,
                "do_sample": config.do_sample,
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "num_layers": num_layers,
                "seq_len": kv_cache.seq_len,
                "q_metadata": q_metadata,         # scales, zero points
                "importance_map": importance_map,  # for layer ordering
            }
        })

        # Stream chunks
        sender = ChunkedSender(
            send_fn=lambda payload: client.send_raw({
                "op": "kv_chunk",
                "payload": payload
            }),
            quantizer=quantizer,
        )
        t_send = time.time()
        sender.send_all(quantized_kv, q_metadata, importance_map,
                        snapshot.bandwidth_bps, request_id)

        # Signal end-of-chunks and wait for decode result
        client.send_raw({"op": "decode_start", "payload": {"request_id": request_id}})
        result = client.recv_obj()  # blocking wait for final result

        send_time = time.time() - t_send
        print(f"[prefill] Received result in {send_time:.2f}s")

        # ── Record transfer for calibration ──
        compressed_bytes = allocation.total_bytes
        probe.record_transfer(
            compressed_bytes, send_time, allocation.compression_ratio
        )

        # ... display result (unchanged from lines 149-167) ...
```

### 2.4 Helper: KVCache to ABKT format conversion

Add this function to prefill_node.py (or in a shared utility, see section 6):

```python
def _kvcache_to_abkt(kv_cache: KVCache) -> dict:
    """Convert KVCache transport format to ABKT internal format.

    Input:  KVCache.layers = [KVLayerCache(layer_idx=i, k=..., v=...), ...]
    Output: {0: {layer_idx: (k_tensor, v_tensor)}}
    """
    result = {0: {}}
    for layer in kv_cache.layers:
        result[0][layer.layer_idx] = (layer.k, layer.v)
    return result
```

---

## 3. Probing: When It Starts and Persists

### 3.1 ProbeServer on decode node

**File: `/ssd/pd/ABKT/decode_node.py`**

Insert after line 118 (`model.eval()`):

```python
    # ── Start probe server (background daemon on port 9877) ──
    probe_server = ProbeServer(host="0.0.0.0", port=9877)
    probe_server.start()
    print(f"[decode] Probe server started on port 9877")
```

This runs as a daemon thread alongside the SocketServer. No new process needed.

### 3.2 NetworkProbeClient on prefill node — persistent

**Decision: Persistent, NOT per-request.**

The probe client runs for the entire lifetime of `prefill_node.py main()`. It starts before model loading and stops after the final `client.close()`.

Rationale:
- Probing needs 5+ seconds of history for stable EWMA estimates
- Model loading takes 5-15 seconds, providing free warmup time
- Per-request probing would repeat cold-start on every request
- Probe overhead is minimal: 1 RTT probe/sec (64 bytes) + 1 BW probe/10 sec (1 MB)
- The persistent connection to ProbeServer is maintained across requests

### 3.3 Startup timeline

```
t=0.0s   probe.start()                    # RTT probing begins immediately
t=0.0s   _get_num_layers()                # fast config read
t=0.5s   model loading begins             # 5-15s of GPU work
         [probing continues in background]
t=10s    model loaded                     # EWMA now has 10 RTT samples
t=10.5s  probe.warmup_connection()        # TCP CWND warmup
t=11s    prefill forward pass             # 0.1-2s
t=13s    get_snapshot()                   # bandwidth estimate available
t=13s    quantize + send                  # uses calibrated budget
```

---

## 4. Cold Start Problem

### 4.1 Scenario

On the first request, `_bw_ewma.valid` is `False`. `get_snapshot()` returns `COLD_BW = 15e6` (15 MB/s, ~120 Mbps).

### 4.2 Handling strategy

**Conservative first transfer:**
- State = UNKNOWN, so `budget_safety_margin = 0.8`, `max_delay_sec = 0.3`
- budget = 15e6 * 0.3 * 0.8 = 3.6 MB
- For a 32-layer model with 1024 tokens (512 MB FP16), that means ~142x compression needed
- PIA will allocate minimum precision (INT2) across all tokens
- This is acceptable for the first request — quality recovers on subsequent requests

**Alternative (recommended for v1): two-phase first request:**
1. First transfer: no quantization (FP16), used ONLY for calibration
2. `probe.record_transfer(actual_bytes, elapsed, compression_ratio=1.0)` seeds the BW EWMA
3. Second request: uses calibrated budget

But this wastes the first user request. Better approach:

**Use BW probes as pre-calibration:**
- The `_probe_bandwidth()` method runs periodic BW probes (1 MB data transfer)
- By the time the first prefill completes, at least one BW probe should have run
- BW probe interval during UNKNOWN state = 5 seconds (from state_machine.py line 143)
- If model loading takes >5 seconds, `_bw_ewma.valid` will be True before the first snapshot

**Code change in network_probe.py, line 277:** Reduce BW probe interval during UNKNOWN from 5s to 2s for faster cold start:

```python
# In state_machine.py, line 143, change:
NetworkState.UNKNOWN: 5.0
# To:
NetworkState.UNKNOWN: 2.0  # faster cold-start calibration
```

**Guarantee at least one BW probe before first snapshot:**

In `prefill_node.py`, after `probe.start()` and after model loading, add:

```python
    # Ensure at least one BW probe has completed
    if not probe.is_calibrated():
        print("[prefill] Running initial BW probe for calibration...")
        # Force a synchronous BW probe
        probe._probe_bandwidth(data_size=1 * 1024 * 1024)
        # With 2s RTT interval active, we may already have one, but this guarantees it
```

Actually, `_probe_bandwidth` is private. Better to add a public method:

**Add to `NetworkProbeClient` in `backend/network_probe.py`:**
```python
    def probe_now(self) -> None:
        """Force an immediate bandwidth probe (used for cold-start calibration)."""
        self._probe_bandwidth(data_size=1 * 1024 * 1024)
```

### 4.3 Budget formula reminder

```
budget_bytes = bandwidth_bps * max_delay_sec * safety_margin

State       max_delay  safety_margin  Example at 100 MB/s
GOOD        0.5        1.0            50 MB
DEGRADED    0.3        0.8            24 MB
POOR        0.2        0.7            14 MB
UNKNOWN     0.3        0.8            24 MB  (but BW=COLD_BW=15 => 3.6 MB)
```

---

## 5. Decode-Side Changes

### 5.1 New RPC handler: `run_decode_abkt`

**File: `/ssd/pd/ABKT/decode_node.py`**

Add new imports after line 30:
```python
from backend.chunked_transfer import ChunkAssembler
from backend.adaptive_quant import AdaptiveQuantizer
```

Add ProbeServer start (section 3.1 above).

Add a new handler registration. The current code registers `{"run_decode": handle_run_decode}` at line 303. Change to:

```python
    handlers = {
        "run_decode": handle_run_decode,       # legacy path (unchanged)
        "run_decode_abkt": handle_run_decode_abkt,  # new ABKT path
    }
```

### 5.2 New handler implementation

Add `handle_run_decode_abkt` as a top-level function in decode_node.py alongside `handle_run_decode`:

```python
    def handle_run_decode_abkt(
        request_id,
        input_ids,
        first_token,
        max_new_tokens,
        repetition_penalty,
        do_sample,
        temperature,
        top_k,
        top_p,
        num_layers,
        seq_len,
        q_metadata,
        importance_map,
    ):
        """ABKT handler: receives quantized KV as chunks, assembles, dequantizes, decodes."""
        req_id = _request_counter[0]
        _request_counter[0] += 1
        print(f"\n[decode] === ABKT Request #{req_id} (id={request_id}) ===")

        # ── Phase 1: Receive chunks and assemble ──
        quantizer = AdaptiveQuantizer()
        assembled_kv = {}   # will be set by on_complete callback
        assembly_done = threading.Event()

        def on_assembly_complete(req_id_str: str, dequantized_kv: dict):
            print(f"[decode] Assembly complete for {req_id_str}")
            assembled_kv["data"] = dequantized_kv
            assembly_done.set()

        chunk_assembler = ChunkAssembler(quantizer=quantizer, on_complete=on_assembly_complete)
        print(f"[decode] Waiting for KV chunks (seq_len={seq_len}, layers={num_layers})...")

        # The server's _handle_client loop will call this function for each incoming
        # "kv_chunk" message. We need to receive them from the same connection.
        # Strategy: the caller (SocketServer._handle_client) reads messages in a loop.
        # For "run_decode_abkt", we return a special marker, and the server enters
        # a chunk-receive loop.

        # This handler is called by SocketServer._handle_client which is inside a
        # message-receive loop. The handler returns a marker object, and the server
        # switches to chunk-receive mode for this connection.

        return {
            "_abkt_phase": "awaiting_chunks",
            "_assembler": chunk_assembler,
            "_assembly_done": assembly_done,
            "_assembled_kv": assembled_kv,
            "_decode_params": {
                "input_ids": input_ids,
                "first_token": first_token,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
            "_request_id": request_id,
            "_req_num": req_id,
        }
```

### 5.3 SocketServer modification for chunk streaming

**File: `/ssd/pd/ABKT/pd_inference/socket_transport.py`**

The `_handle_client` method (line 127) needs to support a "streaming mode" where, after the initial RPC call, the server continues reading raw messages (not request-response) until a terminal message arrives.

**Modified `_handle_client` (replaces lines 127-161):**

```python
    def _handle_client(self) -> None:
        """Process requests from the connected client."""
        while self._running:
            try:
                msg = recv_obj(self._client)
            except (ConnectionError, struct.error):
                break
            except Exception as e:
                print(f"[socket_server] Receive error: {e}")
                break

            if not isinstance(msg, dict):
                self._send_error("invalid_message")
                continue

            op = msg.get("op")
            payload = msg.get("payload") or {}

            if op == "shutdown":
                self._send_ok("bye")
                self._running = False
                break

            handler = self.handlers.get(op)
            if handler is None:
                self._send_error(f"unknown_op:{op}")
                continue

            try:
                result = handler(**payload)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_error(str(e))
                continue

            # ── ABKT streaming mode ──
            if isinstance(result, dict) and result.get("_abkt_phase") == "awaiting_chunks":
                self._handle_abkt_stream(result)
                break  # stream complete, connection closes

            self._send_ok(result)

    def _handle_abkt_stream(self, ctx: dict) -> None:
        """Receive KV chunks in streaming mode, then run decode when complete.

        ctx is the return value from handle_run_decode_abkt containing
        the chunk assembler, event, and decode parameters.
        """
        assembler = ctx["_assembler"]
        assembly_done = ctx["_assembly_done"]
        assembled_kv = ctx["_assembled_kv"]
        decode_params = ctx["_decode_params"]
        request_id = ctx["_request_id"]
        req_num = ctx["_req_num"]

        try:
            while self._running:
                try:
                    msg = recv_obj(self._client)
                except (ConnectionError, struct.error):
                    break
                except Exception as e:
                    print(f"[socket_server] Stream receive error: {e}")
                    break

                if not isinstance(msg, dict):
                    continue

                op = msg.get("op")
                payload = msg.get("payload") or {}

                if op == "kv_chunk":
                    # Feed chunk to assembler
                    chunk_req_id = payload.get("request_id", "")
                    if chunk_req_id == request_id:
                        assembler.add_chunk(
                            request_id=chunk_req_id,
                            layer_idx=payload["layer_idx"],
                            chunk_start=payload["chunk_start"],
                            chunk_end=payload["chunk_end"],
                            total_seq_len=payload["total_seq_len"],
                            k=payload["k"],
                            v=payload["v"],
                            meta=payload.get("meta", {}),
                            is_last_chunk=payload.get("is_last_chunk", False),
                        )

                elif op == "decode_start":
                    # All chunks sent. Wait for assembly completion.
                    if assembly_done.wait(timeout=60.0):
                        dequantized = assembled_kv.get("data")
                        if dequantized is None:
                            self._send_error("Assembly failed: no data")
                            return

                        # Convert ABKT format back to KVCache + run decode
                        result = _run_abkt_decode(
                            dequantized_kv=dequantized,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            decode_params=decode_params,
                            req_num=req_num,
                            pick_token_fn=_pick_token,
                        )
                        self._send_ok(result)
                    else:
                        self._send_error("Assembly timeout")
                    return

                else:
                    print(f"[socket_server] Unknown stream op: {op}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_error(str(e))
        finally:
            assembler.cancel(request_id)
```

### 5.4 Decode execution helper

Add to `decode_node.py`, above `handle_run_decode_abkt`:

```python
    def _run_abkt_decode(dequantized_kv, model, tokenizer, device, decode_params,
                         req_num, pick_token_fn):
        """Execute decode loop from assembled+dequantized KV cache.

        dequantized_kv: {decode_node: {layer: (k, v)}} in FP16
        """
        # Reconstruct DynamicCache
        dcache = DynamicCache()
        layer_kv = dequantized_kv.get(0, {})
        for lidx in sorted(layer_kv.keys()):
            kv = layer_kv[lidx]
            if kv is None:
                continue
            k, v = kv
            dcache.update(k.to(device), v.to(device), lidx)

        # The rest is identical to handle_run_decode from line 177 onward:
        input_ids = decode_params["input_ids"]
        first_token = decode_params["first_token"]
        max_new_tokens = decode_params["max_new_tokens"]
        do_sample = decode_params["do_sample"]
        temperature = decode_params["temperature"]
        top_k = decode_params["top_k"]
        top_p = decode_params["top_p"]
        repetition_penalty = decode_params.get("repetition_penalty", 1.0)

        if first_token is not None:
            generated = list(input_ids) + [first_token]
            tokens_to_generate = max_new_tokens - 1
        else:
            generated = list(input_ids)
            tokens_to_generate = max_new_tokens

        t_start = time.time()

        # First-step sanity check (same as handle_run_decode lines 193-237)
        verify_did_run = False
        if first_token is not None and tokens_to_generate >= 0:
            past_len = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
            verify_attn_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            verify_token = torch.tensor([[generated[-1]]], device=device)
            with torch.no_grad():
                verify_out = model(
                    input_ids=verify_token,
                    attention_mask=verify_attn_mask,
                    past_key_values=dcache,
                    use_cache=True,
                )
            verify_logits = verify_out.logits[:, -1, :].float()
            verify_token_id = pick_token_fn(
                verify_logits[0],
                do_sample=do_sample, temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
            generated.append(verify_token_id)
            tokens_to_generate -= 1
            dcache = verify_out.past_key_values
            verify_did_run = True

        # Main decode loop (same as handle_run_decode lines 239-286)
        for step in range(max(tokens_to_generate, 0)):
            current_token = torch.tensor([[generated[-1]]], device=device)
            step_past_len = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
            step_attn_mask = torch.ones((1, step_past_len + 1), dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(
                    input_ids=current_token,
                    attention_mask=step_attn_mask,
                    past_key_values=dcache,
                    use_cache=True,
                )
            dcache = outputs.past_key_values
            step_logits = outputs.logits[:, -1, :].clone().float()

            if repetition_penalty != 1.0 and step > 0:
                for tid in generated[len(input_ids):]:
                    if tid < step_logits.shape[-1]:
                        if step_logits[0, tid] > 0:
                            step_logits[0, tid] /= repetition_penalty
                        else:
                            step_logits[0, tid] *= repetition_penalty

            next_token = pick_token_fn(
                step_logits[0],
                do_sample=do_sample, temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
            generated.append(next_token)

            if step % 10 == 0 or step == tokens_to_generate - 1:
                elapsed = time.time() - t_start
                tps = (step + 1) / elapsed if elapsed > 0 else 0

        total_time = time.time() - t_start
        gen_tokens = len(generated) - len(input_ids)
        generated_text = decode_tokens(tokenizer, generated)
        print(f"[decode] Request #{req_num} complete: "
              f"{gen_tokens} new tokens in {total_time:.2f}s "
              f"({gen_tokens / total_time:.1f} tok/s)")

        return {
            "generated_ids": generated,
            "generated_text": generated_text,
            "num_tokens": gen_tokens,
            "time": total_time,
        }
```

---

## 6. SocketClient: New `send_raw` method

**File: `/ssd/pd/ABKT/pd_inference/socket_transport.py`**

Add to `SocketClient` class (after `call` method, line 239):

```python
    def send_raw(self, obj: Any) -> None:
        """Send an object without waiting for a response.

        For streaming mode where the caller sends multiple messages
        before receiving a single final response.
        """
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() first.")
        msg = obj  # obj should already be {"op": ..., "payload": ...}
        send_obj(self._sock, msg)

    def recv_obj(self) -> Any:
        """Receive a single object from the socket.

        For use after streaming sends, to receive the final response.
        This is the SAME as the existing recv_obj function, exposed as a method.
        """
        from pd_inference.socket_transport import recv_obj as _recv
        return _recv(self._sock)
```

**Note:** `recv_obj` already exists as a module-level function. We just need to expose it on the client.

---

## 7. RPC Protocol Changes Summary

### 7.1 New operations

| Operation | Direction | Description |
|-----------|-----------|-------------|
| `run_decode_abkt` | prefill -> decode | Init chunked transfer with metadata |
| `kv_chunk` | prefill -> decode | One chunk of quantized KV (layer slice) |
| `decode_start` | prefill -> decode | Signal end of chunks; trigger assembly + decode |
| `run_decode` | prefill -> decode | **Preserved** as legacy fallback path |

### 7.2 Protocol flow

```
PREFILL                                DECODE
  |                                       |
  |-- send_raw({op: "run_decode_abkt"}) ->|  handler returns _abkt_phase marker
  |                                       |  server enters stream mode
  |-- send_raw({op: "kv_chunk"}) -------->|  ChunkAssembler.add_chunk()
  |-- send_raw({op: "kv_chunk"}) -------->|  ChunkAssembler.add_chunk()
  |       ... (N chunks) ...              |
  |-- send_raw({op: "decode_start"}) ---->|  wait assembly_done
  |                                       |  dequantize + decode
  |<------- recv_obj({ok: True}) ---------|  send result
  |                                       |
```

### 7.3 Message formats

**run_decode_abkt payload:**
```json
{
    "request_id": "req_1716076800123",
    "input_ids": [...],
    "first_token": 42,
    "max_new_tokens": 128,
    "repetition_penalty": 1.0,
    "do_sample": false,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "num_layers": 32,
    "seq_len": 1024,
    "q_metadata": {0: {0: {"precision": 8, "scale_k": 0.01, ...}, ...}},
    "importance_map": {0: {0: tensor, ...}}
}
```

**kv_chunk payload:**
```json
{
    "request_id": "req_1716076800123",
    "layer_idx": 0,
    "chunk_start": 0,
    "chunk_end": 64,
    "total_seq_len": 1024,
    "k": tensor([1, 4, 64, 64]),
    "v": tensor([1, 4, 64, 64]),
    "meta": {"precision": 8, "scale_k": 0.01, "zero_k": 0, ...},
    "is_last_chunk": false
}
```

---

## 8. Shared Utility: Format Conversion

A consistent conversion between KVCache transport format and ABKT internal format is needed in both nodes. Recommend adding to `pd_inference/kv_cache.py`:

```python
# Add to KVCache class in /ssd/pd/ABKT/pd_inference/kv_cache.py:

def to_abkt_dict(self) -> dict:
    """Convert to ABKT internal format: {decode_node: {layer_idx: (k, v)}}."""
    result = {0: {}}
    for layer in self.layers:
        result[0][layer.layer_idx] = (layer.k, layer.v)
    return result

@classmethod
def from_abkt_dict(cls, d: dict) -> "KVCache":
    """Convert from ABKT internal format back to KVCache."""
    layers = []
    layer_cache = d.get(0, {})
    for layer_idx in sorted(layer_cache.keys()):
        kv = layer_cache[layer_idx]
        if kv is None:
            continue
        k, v = kv
        layers.append(KVLayerCache(layer_idx=layer_idx, k=k, v=v))
    return cls(layers=layers)
```

---

## 9. Error Handling and Edge Cases

### 9.1 Budget exceeds FP16 size

If `budget >= total_fp16_size`, PrecisionAllocator returns uniform FP16 with `compression_ratio = 1.0`. The Quantizer passes through without modification. In this case, chunked transfer still happens (no behavior change needed), but each chunk contains full-precision tensors.

**Optimization:** When `compression_ratio == 1.0`, skip chunking entirely and fall back to legacy `run_decode`:

```python
    if allocation.compression_ratio == 1.0:
        # Budget ample: use legacy single-message path
        result = client.call("run_decode",
            kv_cache=kv_cache.to_transport_dict(),
            input_ids=input_ids_t[0].tolist(),
            first_token=first_token,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )
    else:
        # ABKT path with chunked transfer
        ...
```

### 9.2 Chunk assembly timeout

If not all chunks arrive, `assembly_done.wait(timeout=60.0)` times out. The server sends an error response and cancels the assembler buffer for that request_id.

### 9.3 Probe server connection failure

`NetworkProbeClient` handles connection failures gracefully (returns `None` from `_get_rtt_conn()`). The prefill pipeline proceeds with cold-start defaults if probing is unavailable.

### 9.4 Empty KV cache (edge case)

If `kv_cache.seq_len == 0`, skip the ABKT pipeline entirely and send an empty kv_cache via legacy path.

---

## 10. Files Changed (Summary)

| File | Changes |
|------|---------|
| `prefill_node.py` | Add imports (L17-21), probe.start() (L51), insertion after KVCache extraction (L113-144 replaced), probe.stop() (L175+) |
| `decode_node.py` | Add imports (L31-32), ProbeServer start (after L118), new handler `handle_run_decode_abkt`, helper `_run_abkt_decode`, handler registration (L303) |
| `socket_transport.py` | Modify `_handle_client` to dispatch ABKT stream mode, add `_handle_abkt_stream()`, add `SocketClient.send_raw()` and `SocketClient.recv_obj()` |
| `kv_cache.py` | Add `KVCache.to_abkt_dict()` and `KVCache.from_abkt_dict()` |
| `state_machine.py` | Change UNKNOWN probe interval from 5.0 to 2.0 (line 143) |
| `network_probe.py` | Add `NetworkProbeClient.probe_now()` public method |

### Files NOT changed

| File | Reason |
|------|--------|
| `backend/precision_allocator.py` | Works as-is with correct input format |
| `backend/adaptive_quant.py` | Works as-is |
| `backend/token_importance.py` | Works as-is |
| `backend/chunked_transfer.py` | Works as-is (send_fn hook handles integration) |
| `backend/ewma.py` | No changes needed |
| `pd_inference/config.py` | No changes needed |
| `pd_inference/model.py` | No changes needed |

---

## 11. Testing Strategy

### 11.1 Integration test points

1. **Cold start:** First request with no probing history. Budget = COLD_BW-based. Verify quantized transfer completes and produces non-degenerate output.

2. **Calibrated:** Second request after first transfer. Budget reflects measured BW. Verify compression ratio is proportional to budget.

3. **Ample budget:** Simulate high bandwidth (good network). Verify FP16 passthrough path is taken.

4. **Constrained budget:** Simulate low bandwidth. Verify aggressive quantization + chunked transfer completes.

5. **Legacy fallback:** Verify `run_decode` still works when ABKT is disabled (run `prefill_node.py` without `--abkt` flag).

### 11.2 Proposed CLI flag

```python
# In prefill_node.py argument parser:
parser.add_argument("--abkt", action="store_true", default=True,
                    help="Enable ABKT adaptive transfer (default: on)")
parser.add_argument("--no-abkt", action="store_false", dest="abkt",
                    help="Disable ABKT, use legacy full-KV transfer")
```

---

## 12. Open Questions

1. **Should chunked transfer use a separate TCP connection?** Current design uses the SAME connection as the RPC channel, switching to stream mode. This is simpler but means only one request can use the connection at a time (acceptable since SocketServer already serializes connections). A separate connection would add complexity but allow pipelining.

2. **Should importance_map be sent in the init message or derived on decode side?** Current design sends it. Could be re-derived on decode side from the dequantized KV, saving network bytes at the cost of compute.

3. **Prefill node currently runs single-request (not a server).** When prefill_node becomes a persistent server, NetworkProbeClient should be initialized ONCE at server startup, not per-request. This design anticipates that.
