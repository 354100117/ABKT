"""KV cache transport module for EdgePD.

Provides serializable KV cache containers for transfer between
prefill and decode nodes over TCP.

Design:
    KVLayerCache — per-layer (k, v) tensor pair, serializable
    KVCache      — ordered list of KVLayerCache, converts to/from
                   transformers DynamicCache, supports network transport

Usage (prefill side):
    kv_cache = KVCache.from_dynamic_cache(outputs.past_key_values)
    data = kv_cache.to_transport_dict()
    client.call("run_decode", kv_cache=data, ...)

Usage (decode side):
    kv_cache = KVCache.from_transport_dict(data)
    dcache = kv_cache.to_dynamic_cache(device)
    outputs = model(input_ids, past_key_values=dcache, use_cache=True)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


@dataclass
class KVLayerCache:
    """Per-layer KV cache for transport between nodes.

    Attributes:
        layer_idx: Global layer index in the model.
        k: Key tensor on CPU, shape [batch, num_heads, seq_len, head_dim].
        v: Value tensor on CPU, shape [batch, num_heads, seq_len, head_dim].
    """

    layer_idx: int
    k: torch.Tensor
    v: torch.Tensor

    def to_device(self, device: torch.device) -> "KVLayerCache":
        return KVLayerCache(
            layer_idx=self.layer_idx,
            k=self.k.to(device),
            v=self.v.to(device),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "k": self.k,
            "v": self.v,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KVLayerCache":
        return cls(
            layer_idx=int(d["layer_idx"]),
            k=d["k"],
            v=d["v"],
        )


@dataclass
class KVCache:
    """Ordered multi-layer KV cache for node-to-node transport.

    Each layer's (k, v) pair is stored in a KVLayerCache. The order
    matches the model's layer order (0..num_layers-1).

    Serialization produces a plain dict of tensors compatible with
    torch.save / torch.load (used by SocketClient/SocketServer).
    """

    layers: List[KVLayerCache] = field(default_factory=list)

    # ── Properties ──

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        """Cached sequence length from the first valid layer."""
        for layer in self.layers:
            k = layer.k
            if k is not None and hasattr(k, "shape") and k.ndim >= 3:
                return int(k.shape[2])
        return 0

    # ── Serialization ──

    def to_transport_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for torch.save network transport.

        All tensors are kept on CPU for serialization.
        """
        return {"layers": [ly.to_dict() for ly in self.layers]}

    def to_abkt_dict(self) -> Dict[int, Dict[int, tuple]]:
        """Convert to ABKT internal format: {decode_node: {layer_idx: (k, v)}}."""
        result: Dict[int, Dict[int, tuple]] = {0: {}}
        for layer in self.layers:
            result[0][layer.layer_idx] = (layer.k, layer.v)
        return result

    @classmethod
    def from_abkt_dict(cls, d: Dict[int, Dict[int, tuple]]) -> "KVCache":
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

    @classmethod
    def from_transport_dict(cls, d: Dict[str, Any]) -> "KVCache":
        """Deserialize from transport dict (e.g. after torch.load)."""
        return cls(layers=[KVLayerCache.from_dict(ly) for ly in d["layers"]])

    # ── DynamicCache conversion ──

    @classmethod
    def from_dynamic_cache(cls, cache: Any) -> "KVCache":
        """Extract KV cache from a transformers DynamicCache (v5.0.0).

        Uses ``cache.layers[i].keys`` / ``.values`` which are the
        full (past + current) key/value tensors per layer.
        """
        if not hasattr(cache, "layers"):
            raise TypeError(
                f"Expected DynamicCache with 'layers' attribute, got {type(cache)}"
            )
        layers: List[KVLayerCache] = []
        for idx, dyn_layer in enumerate(cache.layers):
            k = dyn_layer.keys.detach().cpu()
            v = dyn_layer.values.detach().cpu()
            layers.append(KVLayerCache(layer_idx=idx, k=k, v=v))
        return cls(layers=layers)

    def to_dynamic_cache(self, device: torch.device) -> Any:
        """Build a transformers DynamicCache pre-populated with stored KV.

        Uses ``DynamicCache.update()`` which lazily initialises empty
        sequence-length-0 tensors, then concatenates our stored K/V.
        The result is a properly shaped DynamicCache ready for
        ``model.forward(…, past_key_values=cache, use_cache=True)``.
        """
        if DynamicCache is None:
            raise ImportError(
                "transformers.cache_utils.DynamicCache is required. "
                "Install transformers >= 5.0.0."
            )
        dcache = DynamicCache()
        for layer in self.layers:
            k = layer.k.to(device)
            v = layer.v.to(device)
            # update() lazily inits empty K/V and cat(empty, stored) → stored
            dcache.update(k, v, layer.layer_idx)
        return dcache

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.num_layers}, seq_len={self.seq_len})"
        )


# ── KV Cache diagnostics ──


def kv_cache_fingerprint(kv_cache: KVCache, sample_layers=(0, -1)) -> str:
    """Return a compact fingerprint of a KVCache for cross-node comparison.

    Computes per-layer shape, dtype, device, and a hash of the raw bytes
    of the first and last layers so that prefill and decode can verify
    the transport was lossless.

    Args:
        kv_cache: The KVCache to fingerprint.
        sample_layers: Layer indices (or negative indices) to sample.

    Returns:
        A multi-line string with layer-by-layer diagnostics.
    """
    lines = [f"KVCache fingerprint: {kv_cache}"]
    indices = []
    for idx in sample_layers:
        if idx < 0:
            idx = kv_cache.num_layers + idx
        if 0 <= idx < kv_cache.num_layers:
            indices.append(idx)

    for idx in indices:
        ly = kv_cache.layers[idx]
        k_hash = _tensor_hash(ly.k)
        v_hash = _tensor_hash(ly.v)
        k_stats = _tensor_stats(ly.k)
        v_stats = _tensor_stats(ly.v)
        lines.append(
            f"  layer[{idx}] k={tuple(ly.k.shape)} {ly.k.dtype} "
            f"mean={k_stats[0]:.6f} std={k_stats[1]:.6f} "
            f"min={k_stats[2]:.6f} max={k_stats[3]:.6f} "
            f"hash={k_hash}"
        )
        lines.append(
            f"           v={tuple(ly.v.shape)} {ly.v.dtype} "
            f"mean={v_stats[0]:.6f} std={v_stats[1]:.6f} "
            f"min={v_stats[2]:.6f} max={v_stats[3]:.6f} "
            f"hash={v_hash}"
        )
    return "\n".join(lines)


def dynamic_cache_fingerprint(dcache, sample_layers=(0, -1)) -> str:
    """Return a compact fingerprint of a DynamicCache (decode-side).

    Compares against KVCache fingerprints to verify the reconstruction.
    """
    lines = ["DynamicCache fingerprint:"]
    if not hasattr(dcache, "layers"):
        lines.append("  (not a DynamicCache)")
        return "\n".join(lines)

    seen = getattr(dcache, "_seen_tokens", None)
    num_layers = len(dcache.layers)
    lines.append(f"  _seen_tokens={seen}  num_layers={num_layers}")
    seq = dcache.get_seq_length(0) if num_layers > 0 else 0
    lines.append(f"  get_seq_length(0)={seq}")

    indices = []
    for idx in sample_layers:
        if idx < 0:
            idx = num_layers + idx
        if 0 <= idx < num_layers:
            indices.append(idx)

    for idx in indices:
        layer = dcache.layers[idx]
        k = layer.keys
        v = layer.values
        if k is None or k.numel() == 0:
            lines.append(f"  layer[{idx}] EMPTY")
            continue
        k_hash = _tensor_hash(k)
        v_hash = _tensor_hash(v)
        k_stats = _tensor_stats(k)
        v_stats = _tensor_stats(v)
        lines.append(
            f"  layer[{idx}] k={tuple(k.shape)} {k.dtype} "
            f"mean={k_stats[0]:.6f} std={k_stats[1]:.6f} "
            f"min={k_stats[2]:.6f} max={k_stats[3]:.6f} "
            f"hash={k_hash}"
        )
        lines.append(
            f"           v={tuple(v.shape)} {v.dtype} "
            f"mean={v_stats[0]:.6f} std={v_stats[1]:.6f} "
            f"min={v_stats[2]:.6f} max={v_stats[3]:.6f} "
            f"hash={v_hash}"
        )
    return "\n".join(lines)


def _tensor_hash(t: torch.Tensor) -> str:
    """Return a short hash of a tensor's raw bytes for integrity checks."""
    if t is None or t.numel() == 0:
        return "empty"
    raw = t.detach().cpu().numpy().tobytes()
    return hashlib.md5(raw).hexdigest()[:8]


def _tensor_stats(t: torch.Tensor):
    """Return (mean, std, min, max) of a float tensor."""
    if t is None or t.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    f = t.detach().float()
    return (
        float(f.mean()),
        float(f.std()),
        float(f.min()),
        float(f.max()),
    )
