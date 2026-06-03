# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ABKT (Adaptive Bitrate KV Cache Transfer) is a PD-separated (Prefill-Decode) LLM inference system that splits transformer model execution across two networked machines. The **prefill node** (RTX 5060 Ti, 192.168.0.50) processes the prompt and extracts KV cache; the **decode node** (Jetson Orin, 192.168.0.20) runs autoregressive token generation. KV cache is transferred over TCP with adaptive mixed-precision quantization (FP16/FP8/INT4/INT2) based on real-time network conditions.

## Commands

```bash
# Run full PD inference (orchestrates both nodes via SSH)
./run_all.sh /path/to/model "Your prompt" 64

# Run individual nodes
./run_prefill.sh /path/to/model "prompt" [max_tokens] [layer_split]
./run_decode.sh /path/to/model [port] [layer_split]

# Run tests
python3 test_abkt.py
python3 -m pytest test_abkt.py -v

# Run with sampling
./run_all.sh /path/to/model "Hello" 64 --sample --temperature 0.8 --top-k 50 --top-p 0.9

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Two-Package Structure

- **`pd_inference/`** — Core PD inference infrastructure: config (`PDConfig`), KV cache serialization (`KVCache`/`KVLayerCache`), TCP socket transport (`SocketServer`/`SocketClient`), model stages (`PrefillStage`/`DecodeStage`), and utility functions.
- **`backend/`** — ABKT adaptive transfer system: network probing, bandwidth estimation, precision allocation, quantization, and chunked streaming.

### Data Flow (ABKT Path)

1. **NetworkProbeClient** (prefill) continuously measures RTT and bandwidth to the decode node's **ProbeServer** (port 9877)
2. Prefill runs forward pass, extracts KV cache via `KVCache.from_dynamic_cache()`
3. **TokenImportanceEvaluator** scores each token's importance using Key L2-norm
4. **PrecisionAllocator** (PIA algorithm) greedily assigns precision levels (FP16→FP8→INT4→INT2) per layer under a bandwidth-derived byte budget
5. **AdaptiveQuantizer** quantizes the KV cache according to the allocation
6. **ChunkedSender** streams quantized chunks to decode node in importance order
7. **ChunkAssembler** (decode) receives chunks, reassembles, dequantizes, and triggers decode
8. Decode node runs autoregressive generation using `model.forward()` with `DynamicCache`

### Key Design Decisions

- **Dual-EWMA bandwidth estimation**: probe-based EWMA + transfer-based EWMA (ground truth). `get_effective_bw()` takes the minimum with confidence-scaled safety margin.
- **Feedback loop prevention**: `record_transfer()` uses uncompressed-equivalent bandwidth (`compressed_bytes * compression_ratio / elapsed`). EWMA updates are clamped to ±20% per step.
- **Hysteresis state machine**: NetworkState (GOOD/DEGRADED/POOR) with N-out-of-M voting (2/3 to downgrade, 4/5 to upgrade). POOR is bandwidth-alone, not OR'd with RTT.
- **Per-layer minimum precision**: bottom 1/3 layers → FP8 floor, middle → INT4, top → INT2. When budget is infeasible, lowest-importance layers are dropped (max 50%).
- **Socket protocol**: 4-byte length prefix + `torch.save`/`torch.load` serialization. Supports both request-response (`call()`) and streaming (`send_raw()`/ABKT chunked mode).

### Model Support

`ModelArch` abstracts OPT (learned positional embeddings, `model.model.decoder`) and Qwen2.5/Llama (RoPE, `model.model`). Partial model loading from safetensors is supported via `try_load_partial_model()` for layer-split scenarios.

### ABKT vs Legacy Transfer

- **Legacy path** (`run_decode` op): when budget is ample (compression_ratio=1.0), KV cache is sent as a single `torch.save` payload
- **ABKT path** (`run_decode_abkt` + `kv_chunk` + `decode_start` ops): chunked streaming with quantization, used when compression is needed

## Key Files

| File | Purpose |
|------|---------|
| `prefill_node.py` | Prefill node entry point — full ABKT pipeline |
| `decode_node.py` | Decode node entry point — RPC server + autoregressive loop |
| `pd_inference/kv_cache.py` | KVCache serialization (DynamicCache ↔ transport dict ↔ ABKT dict) |
| `pd_inference/socket_transport.py` | TCP socket server/client with ABKT streaming support |
| `pd_inference/model.py` | PrefillStage/DecodeStage with partial model loading |
| `backend/network_probe.py` | ProbeServer + NetworkProbeClient (EWMA, calibration, dual-BW) |
| `backend/precision_allocator.py` | PIA algorithm — greedy importance-weighted precision allocation |
| `backend/adaptive_quant.py` | FP16/FP8/INT4/INT2 quantize/dequantize |
| `backend/chunked_transfer.py` | ChunkedSender (prefill) + ChunkAssembler (decode) |
| `backend/state_machine.py` | NetworkState hysteresis machine (GOOD/DEGRADED/POOR) |
| `backend/token_importance.py` | Key L2-norm importance scoring |
| `test_abkt.py` | Unit tests for EWMA, state machine, precision allocator, quantization |
