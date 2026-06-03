"""Common utilities for PD inference.

Includes:
    - Tokenizer loading and encoding
    - KV cache manipulation
    - Tensor debugging helpers
    - Model partial loading from safetensors
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.modeling_utils import init_empty_weights
except ImportError:
    try:
        from transformers.utils.generic import init_empty_weights
    except ImportError:
        init_empty_weights = None

try:
    from safetensors import safe_open as _safe_open
except ImportError:
    _safe_open = None


# ── Tokenizer ──


def load_tokenizer(model_name: str) -> Optional[AutoTokenizer]:
    """Load tokenizer from model path.

    Returns None if loading fails (uses fallback ASCII encoding).
    """
    if not model_name:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"[tokenizer] Loaded from {model_name}")
        return tokenizer
    except Exception as e:
        print(f"[tokenizer] Failed to load from '{model_name}': {e}")
        return None


def encode_prompt(prompt: str, tokenizer: Optional[AutoTokenizer]) -> List[int]:
    """Encode a prompt string to token IDs.

    Falls back to ASCII encoding if no tokenizer is available.
    """
    if tokenizer is None:
        ids = [min(ord(c), 255) for c in prompt][:128]
        return ids if ids else [0]
    ids = tokenizer.encode(prompt)
    if not ids:
        fallback = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
        ids = [fallback]
    return ids


def decode_tokens(tokenizer: Optional[AutoTokenizer], ids: List[int]) -> str:
    """Decode token IDs to text."""
    if tokenizer is None:
        return str(ids)
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return str(ids)


# ── Device ──


def get_device() -> torch.device:
    """Get the available device (CUDA preferred)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def module_device(module: Optional[torch.nn.Module]) -> str:
    """Get the device string of a module's first parameter."""
    if module is None:
        return "None"
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "no-params"
    except Exception:
        return "unknown"


def module_dtype(module: Optional[torch.nn.Module], fallback=torch.float16) -> torch.dtype:
    """Get the dtype of a module's first parameter."""
    if module is None:
        return fallback
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback
    except Exception:
        return fallback


# ── Tensor debugging ──


def tensor_summary(tensor, name="tensor", max_vals=4) -> str:
    """Compact string summary of a tensor's shape, dtype, and statistics."""
    if tensor is None:
        return f"{name}=None"
    if not isinstance(tensor, torch.Tensor):
        return f"{name}=type={type(tensor).__name__}"
    if tensor.numel() == 0:
        return f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} empty"
    flat = tensor.detach().reshape(-1)
    sample = flat[:max_vals].float().cpu().tolist()
    mean = float(flat.float().mean())
    std = float(flat.float().std(unbiased=False)) if flat.numel() > 1 else 0.0
    return (
        f"{name}=shape={tuple(tensor.shape)} dtype={tensor.dtype} {tensor.device} "
        f"mean={mean:.6f} std={std:.6f} sample={sample}"
    )


def kv_shape_str(kv) -> str:
    """Compact string of KV cache shape."""
    if kv is None:
        return "None"
    kv_t = _kv_to_tuple(kv)
    if kv_t is not None:
        return f"k={tuple(kv_t[0].shape)} v={tuple(kv_t[1].shape)}"
    return type(kv).__name__


# ── KV cache helpers ──


def _kv_to_tuple(present):
    """Convert KV cache (tuple, DynamicLayer, or DynamicCache) to (k, v) tuple."""
    if present is None:
        return None
    if hasattr(present, 'keys') and hasattr(present, 'values'):
        # DynamicLayer (transformers >= 5.0.0)
        k = present.keys
        v = present.values
        return k, v
    if hasattr(present, 'key_cache') and hasattr(present, 'value_cache'):
        # DynamicCache (transformers >= 4.46)
        k = present.key_cache[-1] if present.key_cache else None
        v = present.value_cache[-1] if present.value_cache else None
        if k is None or v is None:
            return None
        return k, v
    if isinstance(present, (tuple, list)) and len(present) >= 2:
        return present[0], present[1]
    return None


def detach_kv_to_cpu(present):
    """Detach KV cache pair and move to CPU."""
    if present is None:
        return None
    kv = _kv_to_tuple(present)
    if kv is None:
        return None
    k, v = kv
    return k.detach().cpu(), v.detach().cpu()


def detach_kv_to_device(present, device):
    """Detach KV cache pair and move to device."""
    if present is None:
        return None
    kv = _kv_to_tuple(present)
    if kv is None:
        return None
    k, v = kv
    return k.detach().to(device), v.detach().to(device)


def merge_kv_parts(parts):
    """Merge multiple KV cache parts (from batch splitting) into one."""
    if not parts:
        return None
    first = parts[0]
    if isinstance(first, dict):
        merged = {}
        for part in parts:
            if part is None:
                continue
            for node_idx, node_kv in part.items():
                dst = merged.setdefault(node_idx, {})
                for layer_idx, kv in node_kv.items():
                    if kv is None:
                        continue
                    k, v = kv
                    if layer_idx in dst and dst[layer_idx] is not None:
                        prev_k, prev_v = dst[layer_idx]
                        dst[layer_idx] = (torch.cat([prev_k, k], dim=0), torch.cat([prev_v, v], dim=0))
                    else:
                        dst[layer_idx] = (k, v)
        return merged
    if isinstance(first, list):
        merged = [None for _ in range(len(first))]
        for part in parts:
            if part is None:
                continue
            for idx, kv in enumerate(part):
                if kv is None:
                    continue
                k, v = kv
                if merged[idx] is not None:
                    prev_k, prev_v = merged[idx]
                    merged[idx] = (torch.cat([prev_k, k], dim=0), torch.cat([prev_v, v], dim=0))
                else:
                    merged[idx] = (k, v)
        return merged
    return first


def split_kv_cache_by_length(kv_cache, lengths: List[int]):
    """Split a batched KV cache by sequence lengths for per-request handling.

    Args:
        kv_cache: Dict[node_idx][layer_idx] = (k, v) with batch dim > 1
        lengths: List of sequence lengths per request

    Returns:
        List of per-request KV caches (same structure, batch dim = 1)
    """
    if kv_cache is None:
        return [None for _ in range(len(lengths))]

    if not isinstance(kv_cache, dict):
        return [kv_cache for _ in range(len(lengths))]

    per_req = []
    for i, seq_len in enumerate(lengths):
        req_kv = {}
        for decode_node_idx, layer_cache in kv_cache.items():
            node_kv = {}
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    node_kv[layer_idx] = None
                    continue
                if not isinstance(kv, (list, tuple)):
                    node_kv[layer_idx] = None
                    continue
                k, v = kv
                node_kv[layer_idx] = (
                    k[i:i + 1, :, :seq_len, :].contiguous(),
                    v[i:i + 1, :, :seq_len, :].contiguous(),
                )
            req_kv[decode_node_idx] = node_kv
        per_req.append(req_kv)
    return per_req


def infer_past_len(kv_cache) -> int:
    """Infer the past sequence length from a KV cache.

    Works with dict or list format KV caches.
    """
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, dict):
        for _, layer_cache in kv_cache.items():
            if isinstance(layer_cache, dict):
                for _, kv in layer_cache.items():
                    if kv is None:
                        continue
                    if not isinstance(kv, (list, tuple)):
                        continue
                    k, _ = kv
                    return int(k.shape[2])
    elif isinstance(kv_cache, list):
        for kv in kv_cache:
            if kv is None:
                continue
            if not isinstance(kv, (list, tuple)):
                continue
            k, _ = kv
            return int(k.shape[2])
    return 0


# ── Attention mask ──


def prepare_layer_attention_mask(attention_mask, hidden_states, layer_past):
    """Prepare 4D causal attention mask for a decoder layer.

    Adds causal masking on top of the padding mask (if provided).
    Returns a tensor of shape (batch, 1, tgt_len, src_len).
    """
    if attention_mask is None:
        return None
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, device=hidden_states.device)
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    if attention_mask.dim() == 4:
        return attention_mask.to(hidden_states.dtype)
    if attention_mask.dim() != 2:
        return attention_mask

    bsz, tgt_len = hidden_states.shape[:2]
    past_len = 0
    if layer_past is not None:
        kv = _kv_to_tuple(layer_past)
        if kv is not None:
            k, _ = kv
            if torch.is_tensor(k):
                past_len = int(k.shape[2])
    src_len = past_len + tgt_len

    causal = torch.full(
        (tgt_len, src_len),
        torch.finfo(hidden_states.dtype).min,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    causal = torch.triu(causal, diagonal=1 + past_len)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgt_len, src_len)

    if attention_mask.shape[1] == src_len:
        pad_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
        pad_mask = (1.0 - pad_mask) * torch.finfo(hidden_states.dtype).min
        causal = causal + pad_mask

    return causal


def compute_position_embeds(embed_positions, *, bs, seq_len, past_len, device,
                            input_ids=None, pad_token_id=None):
    """Compute positional embeddings for OPT models."""
    if embed_positions is None:
        return None

    position_ids = torch.arange(
        int(past_len),
        int(past_len) + seq_len,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0).expand(bs, seq_len)

    try:
        pos_embeds = embed_positions(position_ids)
        return pos_embeds
    except Exception:
        return embed_positions(position_ids)


# ── Batch padding ──


def pad_batch(ids_batch: List[List[int]], pad_id: int):
    """Pad a batch of token ID sequences to equal length.

    Returns:
        (padded_ids, attention_mask)
    """
    max_len = max(len(x) for x in ids_batch) if ids_batch else 0
    padded = []
    masks = []
    for ids in ids_batch:
        pad_len = max_len - len(ids)
        padded.append(ids + [pad_id] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)
    return padded, masks


# ── SafeTensors partial loading ──


def _find_safetensors_index(model_dir: str):
    """Find the safetensors index file in a model directory."""
    try:
        files = os.listdir(model_dir)
    except Exception:
        return None
    if "model.safetensors.index.json" in files:
        return os.path.join(model_dir, "model.safetensors.index.json")
    candidates = sorted([f for f in files if f.endswith(".safetensors.index.json")])
    if candidates:
        return os.path.join(model_dir, candidates[0])
    return None


def _find_single_safetensors(model_dir: str):
    """Find a single safetensors file in a model directory."""
    try:
        files = os.listdir(model_dir)
    except Exception:
        return None
    if "model.safetensors" in files:
        return os.path.join(model_dir, "model.safetensors")
    candidates = [f for f in files if f.endswith(".safetensors")]
    if len(candidates) == 1:
        return os.path.join(model_dir, candidates[0])
    return None


def _load_safetensors_subset(model_dir: str, prefixes):
    """Load only the tensors matching given prefixes from safetensors files.

    This avoids loading the entire model into memory.
    """
    if _safe_open is None:
        return None
    index_path = _find_safetensors_index(model_dir)
    state_dict = {}
    if index_path:
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            return None
        weight_map = index.get("weight_map", {})
        file_to_keys = defaultdict(list)
        for key, filename in weight_map.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    file_to_keys[filename].append(key)
                    break
        if not file_to_keys:
            return None
        for filename, keys in file_to_keys.items():
            path = os.path.join(model_dir, filename)
            with _safe_open(path, framework="pt", device="cpu") as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key)
        return state_dict
    single_path = _find_single_safetensors(model_dir)
    if not single_path:
        return None
    with _safe_open(single_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    state_dict[key] = f.get_tensor(key)
                    break
    return state_dict if state_dict else None


def _build_prefixes_for_range(model, layer_range, load_embed, load_lm_head):
    """Build key prefixes for loading a subset of layers.

    Supports OPT (model.decoder.*) and Qwen2.5/Llama (model.*) architectures.
    """
    keys = list(model.state_dict().keys())
    has_model_decoder = any(k.startswith("model.decoder.") for k in keys)
    has_model_layers = any(k.startswith("model.layers.") for k in keys)

    if has_model_decoder:
        decoder_prefix = "model.decoder."
    elif has_model_layers:
        decoder_prefix = "model."
    else:
        decoder_prefix = "decoder."

    has_model_lm_head = any(k.startswith("model.lm_head.") for k in keys)
    lm_head_prefix = "model.lm_head." if has_model_lm_head else "lm_head."
    start, end = layer_range
    prefixes = [f"{decoder_prefix}layers.{i}." for i in range(start, end)]

    if load_embed:
        prefixes.append(f"{decoder_prefix}embed_tokens.")
        # embed_positions only exists in OPT-style models
        if has_model_decoder:
            prefixes.append(f"{decoder_prefix}embed_positions.")
            project_in_key = f"{decoder_prefix}project_in."
            if any(k.startswith(project_in_key) for k in keys):
                prefixes.append(project_in_key)

    if load_lm_head:
        # OPT uses final_layer_norm, Qwen2.5/Llama use norm
        if has_model_decoder:
            prefixes.append(f"{decoder_prefix}final_layer_norm.")
            project_out_key = f"{decoder_prefix}project_out."
            if any(k.startswith(project_out_key) for k in keys):
                prefixes.append(project_out_key)
        else:
            prefixes.append(f"{decoder_prefix}norm.")
        prefixes.append(lm_head_prefix)
    return prefixes


def try_load_partial_model(model_name: str, layer_range, load_embed: bool, load_lm_head: bool):
    """Try to load only a subset of model layers from safetensors files.

    Returns the model with only needed layers materialized, or None on failure.

    Args:
        model_name: Path to model directory
        layer_range: (start, end) tuple of layer indices to load
        load_embed: Whether to load embedding layers (first node)
        load_lm_head: Whether to load the LM head (last node)
    """
    if _safe_open is None or init_empty_weights is None:
        return None
    if not model_name or not os.path.isdir(model_name):
        return None

    # Build a full model skeleton with init_empty_weights
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Determine which prefixes to load
    prefixes = _build_prefixes_for_range(model, layer_range, load_embed, load_lm_head)
    state_dict = _load_safetensors_subset(model_name, prefixes)
    if not state_dict:
        return None

    # Remap state_dict keys if needed (e.g., decoder. → model.decoder.)
    keys = list(model.state_dict().keys())
    has_model_decoder = any(k.startswith("model.decoder.") for k in keys)
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if has_model_decoder and key.startswith("decoder."):
            new_key = "model." + key
        elif not has_model_decoder and key.startswith("model.decoder."):
            new_key = key[len("model."):]
        if has_model_lm_head := any(k.startswith("model.lm_head.") for k in keys):
            if new_key.startswith("lm_head."):
                new_key = "model." + new_key
        elif new_key.startswith("model.lm_head."):
            new_key = new_key[len("model."):]
        remapped[new_key] = value

    # Load into the empty model
    try:
        incompatible = model.load_state_dict(remapped, strict=False, assign=True)
    except TypeError:
        incompatible = model.load_state_dict(remapped, strict=False)

    missing = getattr(incompatible, "missing_keys", []) or []
    critical_missing = []
    for key in missing:
        for prefix in prefixes:
            if key.startswith(prefix):
                critical_missing.append(key)
                break
    if critical_missing:
        print(f"[partial_load] Missing {len(critical_missing)} critical keys, falling back")
        return None

    if load_lm_head and hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass

    return model


def load_full_model(model_name: str, device, dtype=torch.float16):
    """Load the full model from HuggingFace.

    This is the simple fallback when partial loading is not available.
    """
    print(f"[model] Loading full model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    print(f"[model] Loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
          f"~{mem_gb:.1f} GB on {device}")
    return model


def build_layer_ranges(num_layers: int, num_stages: int) -> List[Tuple[int, int]]:
    """Distribute layers evenly across stages.

    Args:
        num_layers: Total number of transformer layers
        num_stages: Number of stages to split across

    Returns:
        List of (start, end) tuples for each stage
    """
    if num_stages <= 0:
        return []
    base = num_layers // num_stages
    rem = num_layers % num_stages
    counts = [base + (1 if i < rem else 0) for i in range(num_stages)]
    ranges = []
    start = 0
    for c in counts:
        end = start + c
        ranges.append((start, end))
        start = end
    return ranges
