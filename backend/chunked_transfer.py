"""Chunked KV Cache transfer with layer-level streaming.

Design:
  - Layer-by-layer streaming: prefetch, quantize, and send each layer
  - Chunk size adapts to bandwidth
  - Decode side receives chunks, assembles, dequantizes, and initializes KV

Key insight from research: chunk-level timing is used for in-transfer
anomaly detection (emergency downgrade), NOT for calibration (which runs
on dedicated probes and full-transfer timing).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from backend.adaptive_quant import AdaptiveQuantizer
from backend.precision_allocator import AllocationResult, Precision, BYTES_PER_ELEMENT

logger = logging.getLogger(__name__)

MIN_CHUNK_SIZE = 16
MAX_CHUNK_SIZE = 256
TIMING_CHECK_INTERVAL = 2   # check throughput every N chunks
SLOW_THRESHOLD = 0.7        # actual_bw < expected * threshold → downgrade


# ════════════════════════════════════════════════════════════════════
# ChunkedSender — prefill side
# ════════════════════════════════════════════════════════════════════


class ChunkedSender:
    """Streams KV cache layers to decode node, with adaptive chunk sizing
    and mid-transfer bandwidth adaptation.

    When bandwidth drops during transfer, remaining unstarted layers are
    re-quantized at a lower precision to keep transfer time bounded.

    Usage:
        sender = ChunkedSender(
            send_fn=rpc.send, quantizer=quantizer,
            original_kv=abkt_kv, precision_map=allocation.precision_map,
        )
        sender.send_all(quantized_kv, metadata, importance_map, bandwidth_bps, request_id)
    """

    def __init__(self, send_fn: Callable, quantizer: AdaptiveQuantizer,
                 original_kv: Optional[Dict] = None,
                 precision_map: Optional[Dict] = None):
        self.send_fn = send_fn
        self.quantizer = quantizer
        self.original_kv = original_kv    # unquantized data for re-quantization
        self.precision_map = precision_map  # current precision allocation

    def send_all(
        self,
        quantized_kv: Dict,
        metadata: Dict,
        importance_map: Dict,
        bandwidth_bps: float,
        request_id: str,
        on_progress: "Callable[[int, int], None] | None" = None,
    ) -> None:
        """Send all layers in importance order, with mid-transfer adaptation."""
        chunk_size = self._adaptive_chunk_size(bandwidth_bps)
        layer_order = self._sort_layers(quantized_kv, importance_map)
        total_chunks = 0
        total_bytes = 0
        t_start = time.time()

        # Pre-calculate total expected bytes for progress tracking
        total_expected = 0
        for dnode in quantized_kv:
            for lidx in layer_order:
                kv = quantized_kv[dnode].get(lidx)
                if kv is not None:
                    k, v = kv
                    total_expected += k.nbytes + v.nbytes

        # Track which layers have been sent vs are still pending
        sent_layers = set()
        meta_sent = set()   # layers whose metadata has been sent
        active_quantized = quantized_kv
        active_metadata = metadata

        for dnode in sorted(active_quantized.keys()):
            node_meta = active_metadata.get(dnode, {})
            layer_cache = active_quantized[dnode]

            for lidx in layer_order:
                kv = layer_cache.get(lidx)
                if kv is None:
                    continue
                k, v = kv
                seq_len = k.shape[2]
                meta = node_meta.get(lidx, {})

                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    k_chunk = k[:, :, start:end, :].contiguous()
                    v_chunk = v[:, :, start:end, :].contiguous()
                    chunk_bytes = k_chunk.nbytes + v_chunk.nbytes
                    total_bytes += chunk_bytes
                    total_chunks += 1
                    if total_chunks % 10 == 0 or end >= seq_len:
                        print(f"[sender] chunk #{total_chunks} layer={lidx} "
                              f"[{start}:{end}] {chunk_bytes/1024:.1f}KB "
                              f"last={end >= seq_len}")
                    # Send metadata only with the first chunk of each layer
                    # to avoid redundant serialization (~6 KB × 84 chunks/layer)
                    include_meta = lidx not in meta_sent
                    msg = {
                        "request_id": request_id,
                        "layer_idx": lidx,
                        "chunk_start": start,
                        "chunk_end": end,
                        "total_seq_len": seq_len,
                        "k": k_chunk,
                        "v": v_chunk,
                        "meta": meta if include_meta else {},
                        "is_last_chunk": (end >= seq_len),
                    }
                    self.send_fn(msg)
                    if include_meta:
                        meta_sent.add(lidx)
                    if on_progress is not None:
                        on_progress(total_bytes, total_expected)

                    # ── Mid-transfer bandwidth check ──
                    if (total_chunks % TIMING_CHECK_INTERVAL == 0
                            and self.original_kv is not None
                            and self.precision_map is not None):
                        remaining = [l for l in layer_order if l not in sent_layers]
                        new_q, new_m = self._check_and_downgrade(
                            total_bytes, t_start, bandwidth_bps,
                            remaining, active_quantized, active_metadata,
                        )
                        if new_q is not None:
                            active_quantized = new_q
                            active_metadata = new_m

                sent_layers.add(lidx)

        elapsed = time.time() - t_start
        print(f"[sender] All chunks sent: {total_chunks} chunks, "
              f"{total_bytes/1e6:.1f} MB in {elapsed:.2f}s "
              f"({total_bytes/elapsed/1e6:.1f} MB/s)")

    def _check_and_downgrade(
        self,
        bytes_sent: int,
        t_start: float,
        expected_bw: float,
        remaining_layers: List[int],
        quantized_kv: Dict,
        metadata: Dict,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Check actual throughput and downgrade remaining layers if slow.

        Returns (new_quantized, new_metadata) if downgrade happened, else (None, None).
        """
        elapsed = time.time() - t_start
        if elapsed < 0.1:
            return None, None

        actual_bw = bytes_sent / elapsed
        if actual_bw >= expected_bw * SLOW_THRESHOLD:
            return None, None

        # Bandwidth dropped — downgrade remaining layers
        print(f"[sender] BW drop detected: actual={actual_bw/1e6:.1f} MB/s "
              f"< expected*{SLOW_THRESHOLD}={expected_bw*SLOW_THRESHOLD/1e6:.1f} MB/s")
        print(f"[sender] Downgrading {len(remaining_layers)} remaining layers")

        new_prec_map = self._downgrade_precision_map(
            self.precision_map, remaining_layers
        )
        new_q, new_m = self.quantizer.quantize(self.original_kv, new_prec_map)

        # Update precision_map so subsequent checks use the new allocation
        self.precision_map = new_prec_map
        return new_q, new_m

    @staticmethod
    def _downgrade_precision_map(
        precision_map: Dict, remaining_layers: List[int]
    ) -> Dict:
        """Downgrade precision for remaining layers by one level."""
        DOWNGRADE = {
            Precision.FP16: Precision.FP8,
            Precision.FP8: Precision.INT4,
            Precision.INT4: Precision.INT2,
            Precision.INT2: Precision.INT2,
        }
        new_map = {}
        for dnode, layer_map in precision_map.items():
            new_map[dnode] = {}
            for lidx, prec_tensor in layer_map.items():
                if lidx in remaining_layers:
                    avg_prec = int(round(float(prec_tensor.float().mean())))
                    cur = Precision(avg_prec) if avg_prec in {16, 8, 4, 2} else Precision.FP16
                    downgraded = DOWNGRADE[cur]
                    new_map[dnode][lidx] = torch.full(
                        prec_tensor.shape, downgraded.value, dtype=torch.int8
                    )
                    if downgraded != cur:
                        print(f"[sender]   layer {lidx}: {cur.name} → {downgraded.name}")
                else:
                    new_map[dnode][lidx] = prec_tensor
        return new_map

    @staticmethod
    def _adaptive_chunk_size(bandwidth_bps: float) -> int:
        bw_norm = min(bandwidth_bps / 100e6, 1.0)
        return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE,
                   int(MIN_CHUNK_SIZE + (MAX_CHUNK_SIZE - MIN_CHUNK_SIZE) * bw_norm)))

    @staticmethod
    def _sort_layers(kv_cache: Dict, importance_map: Dict) -> List[int]:
        layer_imp: Dict[int, float] = {}
        for dnode, layer_cache in kv_cache.items():
            imp = importance_map.get(dnode, {})
            for lidx in layer_cache:
                if lidx not in layer_imp:
                    t = imp.get(lidx)
                    layer_imp[lidx] = float(t.mean().item()) if t is not None else 0.5
        return sorted(layer_imp, key=lambda l: layer_imp[l], reverse=True)


# ════════════════════════════════════════════════════════════════════
# ChunkAssembler — decode side
# ════════════════════════════════════════════════════════════════════


class ChunkAssembler:
    """Receives and assembles chunked KV cache layers.

    Thread-safe. On receiving the last chunk for a request,
    assembles all chunks, dequantizes, and notifies the callback.

    Usage:
        assembler = ChunkAssembler(dequantize_fn, on_complete)
        assembler.add_chunk(request_id, layer_idx, ...)
    """

    def __init__(self, quantizer: AdaptiveQuantizer,
                 on_complete: Callable[[str, Dict], None],
                 num_layers: int = 0):
        self.quantizer = quantizer
        self.on_complete = on_complete
        self.num_layers = num_layers
        self._lock = threading.Lock()
        self._buffers: Dict[str, Dict[int, dict]] = {}  # req_id → layer → chunks

    def add_chunk(self, request_id: str, layer_idx: int,
                  chunk_start: int, chunk_end: int, total_seq_len: int,
                  k: torch.Tensor, v: torch.Tensor, meta: dict,
                  is_last_chunk: bool) -> Optional[bool]:
        """Store a chunk. Returns True when full KV is ready and callback fired."""
        with self._lock:
            buf = self._buffers.setdefault(request_id, {})
            info = buf.setdefault(layer_idx, {
                "chunks": [], "total_seq_len": total_seq_len, "meta": meta
            })
            info["chunks"].append((chunk_start, chunk_end, k, v))
            n_chunks = len(info["chunks"])

            print(f"[assembler] chunk layer={layer_idx} [{chunk_start}:{chunk_end}] "
                  f"chunks_for_layer={n_chunks} is_last={is_last_chunk} "
                  f"layers_in_buf={len(buf)}/{self.num_layers}")

            if not is_last_chunk:
                return None

            # Wait for all expected layers before checking completeness
            if self.num_layers > 0 and len(buf) < self.num_layers:
                print(f"[assembler] Waiting for layers: {len(buf)}/{self.num_layers}")
                return None

            # Check all layers complete
            if not all(self._is_complete(info) for info in buf.values()):
                incomplete = [l for l, info in buf.items() if not self._is_complete(info)]
                print(f"[assembler] Layers incomplete: {incomplete}")
                return None

            # All chunks received — assemble and dequantize
            print(f"[assembler] All {len(buf)} layers complete, assembling...")
            t0 = time.time()
            assembled = self._assemble(request_id, buf)
            print(f"[assembler] Assembled in {time.time() - t0:.3f}s")
            # Save meta before deleting buffer (we're already under self._lock)
            meta_for_deq = {0: {l: buf[l]["meta"] for l in assembled}}
            del self._buffers[request_id]

        try:
            q = self.quantizer
            t1 = time.time()
            print(f"[assembler] Dequantizing {len(assembled)} layers...")
            dequantized = q.dequantize({0: assembled}, meta_for_deq)
            print(f"[assembler] Dequantized in {time.time() - t1:.3f}s")
            t2 = time.time()
            self.on_complete(request_id, dequantized)
            print(f"[assembler] Callback done in {time.time() - t2:.3f}s")
        except Exception as e:
            print(f"[assembler] ERROR during assembly/dequant: {e}")
            import traceback
            traceback.print_exc()
        return True

    @staticmethod
    def _is_complete(info: dict) -> bool:
        """Check if all chunks for a layer have been received."""
        total = info["total_seq_len"]
        covered = sorted(info["chunks"], key=lambda x: x[0])
        pos = 0
        for start, end, _, _ in covered:
            if start > pos:
                return False
            pos = max(pos, end)
        return pos >= total

    def _assemble(self, request_id: str, buf: dict) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Concatenate chunked tensors into full (k, v) per layer."""
        result = {}
        for lidx, info in buf.items():
            chunks = sorted(info["chunks"], key=lambda x: x[0])
            k_parts, v_parts = [], []
            for _, _, k_chunk, v_chunk in chunks:
                k_parts.append(k_chunk)
                v_parts.append(v_chunk)
            print(f"[assembler]   layer {lidx}: cat {len(k_parts)} chunks "
                  f"dtype={k_parts[0].dtype} device={k_parts[0].device}")
            result[lidx] = (torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2))
        print(f"[assembler] Concatenation done for {len(result)} layers")
        return result

    def cancel(self, request_id: str) -> None:
        """Drop pending chunks for a cancelled request."""
        with self._lock:
            self._buffers.pop(request_id, None)
