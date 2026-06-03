"""Pipeline orchestration for PD-separated inference.

Handles the full inference flow:
    1. Prefill: runs prompt through prefill stages → KV cache + first token
    2. KV cache transfer: sends KV cache from prefill to decode nodes
    3. Decode: runs autoregressive generation on decode nodes
    4. Result collection: gathers generated tokens and decodes to text

Supports both local (same process) and remote (RPC) stage execution.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from pd_inference.config import PDConfig
from pd_inference.model import DecodeStage, PrefillStage
from pd_inference.rpc import RPCClient
from pd_inference.utils import (
    decode_tokens,
    detach_kv_to_device,
    infer_past_len,
    kv_shape_str,
    load_tokenizer,
    split_kv_cache_by_length,
    tensor_summary,
)

_DEBUG = os.environ.get("EDGEPD_DEBUG", "0") == "1"


def _dprint(msg: str):
    if _DEBUG:
        print(msg)


# ── Repetition penalty ──


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_token_ids: List[int],
    penalty: float = 1.0,
) -> torch.Tensor:
    """Apply repetition penalty to logits."""
    if penalty == 1.0 or not generated_token_ids:
        return logits
    if logits.dim() == 2:
        logits = logits[0]
    for token_id in generated_token_ids:
        if token_id < logits.shape[-1]:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


# ════════════════════════════════════════════════════════════════════
# Prefill
# ════════════════════════════════════════════════════════════════════


def run_prefill_pipeline(
    prefill_stages: List[PrefillStage],
    input_ids,
    attention_mask,
    rpc: Optional[RPCClient] = None,
) -> Tuple[Any, Any]:
    """Run the prefill pipeline across all prefill stages.

    Each stage processes the prompt through its assigned layers.
    Hidden states flow from stage to stage. The last stage also
    produces the KV cache for all layers.

    Args:
        prefill_stages: List of PrefillStage objects (or StageHandle wrappers)
        input_ids: Padded input IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        rpc: RPC client for remote stage execution

    Returns:
        (merged_kv_cache, last_hidden_states)
        merged_kv_cache: dict of all layer KV caches
        last_hidden_states: hidden states from the last stage
    """
    hidden = None
    attn = attention_mask
    merged_kv_cache = {}

    for stage_idx, stage in enumerate(prefill_stages):
        t0 = time.time()

        # Call prefill forward (local or remote)
        hidden, stage_kv, attn = _call_prefill(stage, input_ids, hidden, attn, rpc)
        elapsed = time.time() - t0

        _dprint(f"[pipeline] Prefill stage {stage_idx} | "
                f"{tensor_summary(hidden, 'h')} "
                f"kv_type={type(stage_kv).__name__} "
                f"time={elapsed:.3f}s")

        # Merge KV cache
        if isinstance(stage_kv, dict):
            for decode_node_idx, layer_cache in stage_kv.items():
                if decode_node_idx is None:
                    continue
                if decode_node_idx not in merged_kv_cache:
                    merged_kv_cache[decode_node_idx] = {}
                for layer_idx, kv in layer_cache.items():
                    if layer_idx is None:
                        continue
                    merged_kv_cache[decode_node_idx][layer_idx] = kv
        elif stage_kv is not None:
            merged_kv_cache.setdefault(0, {})
            for layer_idx, kv in enumerate(stage_kv):
                if kv is not None:
                    merged_kv_cache[0][layer_idx] = kv

    return merged_kv_cache if merged_kv_cache else None, hidden


def _call_prefill(stage, input_ids, hidden_states, attention_mask, rpc):
    """Call prefill forward on a stage, either locally or via RPC."""
    if hasattr(stage, 'is_local') and not stage.is_local():
        # Remote call
        return rpc.call(
            stage.rank,
            "prefill_forward",
            {
                "input_ids": input_ids,
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
            },
        )
    # Local call
    lock = getattr(stage, "_lock", None)
    if lock is not None:
        with lock:
            return stage.forward(input_ids, hidden_states, attention_mask)
    return stage.forward(input_ids, hidden_states, attention_mask)


# ════════════════════════════════════════════════════════════════════
# Decode
# ════════════════════════════════════════════════════════════════════


def run_decode_pipeline(
    decode_stages: List[DecodeStage],
    input_ids: List[int],
    kv_cache: Any,
    max_new_tokens: int,
    request_id: str,
    rpc: Optional[RPCClient] = None,
    prefill_token_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
) -> List[int]:
    """Run autoregressive decode pipeline.

    Args:
        decode_stages: List of DecodeStage objects
        input_ids: Single request's input IDs
        kv_cache: KV cache from prefill (dict format)
        max_new_tokens: Maximum tokens to generate
        request_id: Request identifier
        rpc: RPC client for remote stage execution
        prefill_token_ids: Token IDs from prefill last node (used as first token)
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled)

    Returns:
        List of all generated token IDs (including input IDs)
    """
    if not decode_stages:
        return input_ids

    _dprint(f"[decode] START req={request_id} "
            f"input_len={len(input_ids)} max_new={max_new_tokens}")

    # Initialize KV cache on all decode stages
    for stage_idx, stage in enumerate(decode_stages):
        # Extract this stage's portion of the KV cache
        stage_kv = _normalize_kv_for_stage(stage, kv_cache)
        _dprint(f"[decode] init_kv stage={stage_idx} "
                f"kv={kv_shape_str(stage_kv[0]) if stage_kv and stage_kv[0] else 'mixed'}")
        _call_init_kv(stage, request_id, stage_kv, rpc)

    # Determine past length from KV cache
    past_len = infer_past_len(kv_cache)

    # Generate tokens autoregressively
    generated = list(input_ids)
    generated_token_ids: List[int] = []

    for step in range(max_new_tokens):
        # Determine input token for this step
        if step == 0 and prefill_token_ids is not None:
            token_batch = [prefill_token_ids]
        else:
            token_batch = [[generated[-1]]]

        # Run through decode stages
        hidden = None
        for stage_idx, stage in enumerate(decode_stages):
            if step == 0 and stage_idx == 0:
                step_hidden = None  # first stage embeds from token IDs
            else:
                step_hidden = hidden

            if step_hidden is not None and isinstance(step_hidden, torch.Tensor) and step_hidden.dim() == 1:
                step_hidden = step_hidden.unsqueeze(0).unsqueeze(0)

            hidden = _call_decode_step(stage, request_id, token_batch, step_hidden, past_len, rpc)

        # Compute logits on the last decode stage
        last_stage = decode_stages[-1]
        logits = _call_compute_logits(last_stage, hidden, rpc)

        if isinstance(logits, torch.Tensor):
            logits_tensor = logits
        else:
            logits_tensor = torch.tensor(logits)

        if logits_tensor.dim() == 2:
            logits_tensor = logits_tensor.unsqueeze(1)
        if logits_tensor.shape[1] == 0:
            raise RuntimeError("decode logits seq length is 0")

        # Select next token (greedy with repetition penalty)
        step_logits = logits_tensor[:, -1, :].clone()
        _apply_repetition_penalty(step_logits, generated_token_ids, repetition_penalty)
        next_token = torch.argmax(step_logits, dim=-1).item()

        generated.append(next_token)
        generated_token_ids.append(next_token)
        past_len += 1

        _dprint(f"[decode] step={step} token={next_token} past_len={past_len}")

    # Clear KV cache
    for stage in decode_stages:
        _call_clear_kv(stage, request_id, rpc)

    _dprint(f"[decode] DONE req={request_id} generated={len(generated)} tokens")
    return generated


def _normalize_kv_for_stage(stage, kv_cache):
    """Extract this stage's portion from a merged KV cache.

    The merged KV cache format from prefill is:
        {decode_stage_id: {layer_idx: (k, v), ...}, ...}

    This function extracts the list of (k, v) tuples for this stage,
    ordered by layer index, suitable for DecodeStage.init_kv().
    """
    if kv_cache is None:
        return None

    # Determine number of layers this stage handles
    num_stage_layers = len(stage.layers) if hasattr(stage, 'layers') else 0

    if isinstance(kv_cache, dict):
        # Look up this stage's portion
        stage_id = getattr(stage, 'stage_id', 0)
        node_cache = kv_cache.get(stage_id)

        # Fallback: try first available key
        if node_cache is None and kv_cache:
            node_cache = kv_cache[next(iter(kv_cache.keys()))]

        if node_cache is None:
            return [None] * num_stage_layers

        # Convert dict to list: [layer0_kv, layer1_kv, ...]
        if isinstance(node_cache, dict):
            max_layer = max(node_cache.keys()) + 1 if node_cache else 0
            num_layers = max(max_layer, num_stage_layers)
            result = [None] * num_layers
            for layer_idx, kv in node_cache.items():
                if layer_idx < num_layers:
                    result[layer_idx] = kv
            return result
        return node_cache

    # Non-dict format: use as-is
    return kv_cache


def _call_init_kv(stage, request_id, kv_cache, rpc):
    """Initialize KV cache on a decode stage (local or remote)."""
    if hasattr(stage, 'is_local') and not stage.is_local():
        return rpc.call(stage.rank, "init_kv", {
            "request_id": request_id,
            "kv_cache": kv_cache,
        })
    return stage.init_kv(request_id, kv_cache)


def _call_decode_step(stage, request_id, input_ids, hidden_states, past_len, rpc):
    """Run a decode step (local or remote)."""
    if hasattr(stage, 'is_local') and not stage.is_local():
        return rpc.call(stage.rank, "decode_step", {
            "request_id": request_id,
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "past_len": past_len,
        })
    return stage.decode_step(request_id, input_ids, hidden_states, past_len)


def _call_compute_logits(stage, hidden_states, rpc):
    """Compute logits (local or remote)."""
    if hasattr(stage, 'is_local') and not stage.is_local():
        return rpc.call(stage.rank, "compute_logits", {
            "hidden_states": hidden_states,
        })
    return stage.compute_logits(hidden_states)


def _call_clear_kv(stage, request_id, rpc):
    """Clear KV cache (local or remote)."""
    if hasattr(stage, 'is_local') and not stage.is_local():
        try:
            return rpc.call(stage.rank, "clear_kv", {"request_id": request_id})
        except Exception:
            return None
    return stage.clear_kv(request_id)


# ════════════════════════════════════════════════════════════════════
# StageHandle (for remote execution)
# ════════════════════════════════════════════════════════════════════


class StageHandle:
    """Wrapper for a stage that may be local or remote.

    Usage:
        handle = StageHandle(rank=1, stage_id=0, stage=None)  # remote
        handle = StageHandle(rank=0, stage_id=0, stage=stage_obj)  # local
    """

    def __init__(self, rank: int, stage_id: int, stage=None):
        self.rank = rank
        self.stage_id = stage_id
        self.stage = stage

    def is_local(self) -> bool:
        return self.stage is not None

    @property
    def layers(self):
        if self.stage is not None:
            return self.stage.layers
        return []

    def __repr__(self):
        return f"StageHandle(rank={self.rank}, id={self.stage_id}, local={self.is_local()})"


# ════════════════════════════════════════════════════════════════════
# Full pipeline (orchestrator)
# ════════════════════════════════════════════════════════════════════


class PDRunner:
    """Orchestrates the full PD-separated inference pipeline.

    Manages prefill stages, decode stages, and the communication
    between them across distributed nodes.
    """

    def __init__(self, config: PDConfig, rpc: Optional[RPCClient] = None):
        self.config = config
        self.rpc = rpc
        self.tokenizer = load_tokenizer(config.model_name)

        # Build prefill stages
        self.prefill_stages = self._build_prefill_stages(config)
        # Build decode stages
        self.decode_stages = self._build_decode_stages(config)

    def _build_prefill_stages(self, config: PDConfig) -> List:
        """Build prefill stage(s) for the configured node.

        On rank 0 (prefill node), creates a local PrefillStage.
        On rank 1 (decode node), creates a StageHandle pointing to rank 0.
        """
        stages = []
        num_layers = self._get_num_layers()
        layer_range = config.prefill_layer_range

        if layer_range == (0, 0):
            # Both nodes load all layers — use full range
            layer_range = (0, num_layers)

        load_embed = (layer_range[0] == 0)
        load_lm_head = True  # last prefill stage computes logits

        if config.is_prefill:
            # Local stage
            stage = PrefillStage(
                stage_id=0,
                layer_range=layer_range,
                model_name=config.model_name,
                load_embed=load_embed,
                load_lm_head=load_lm_head,
                num_layers_total=num_layers,
            )
            stages.append(stage)
        else:
            # Remote handle
            stages.append(StageHandle(rank=0, stage_id=0, stage=None))

        return stages

    def _build_decode_stages(self, config: PDConfig) -> List:
        """Build decode stage(s) for the configured node."""
        stages = []
        num_layers = self._get_num_layers()
        layer_range = config.decode_layer_range

        if layer_range == (0, 0):
            layer_range = (0, num_layers)

        load_embed = (layer_range[0] == 0)
        load_lm_head = True  # last decode stage computes logits

        if config.is_decode:
            stage = DecodeStage(
                stage_id=0,
                layer_range=layer_range,
                model_name=config.model_name,
                load_embed=load_embed,
                load_lm_head=load_lm_head,
                num_layers_total=num_layers,
            )
            stages.append(stage)
        else:
            stages.append(StageHandle(rank=1, stage_id=0, stage=None))

        return stages

    def _get_num_layers(self) -> int:
        """Get the total number of transformer layers."""
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(self.config.model_name)
        return getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 32))

    def load_all(self):
        """Load model components for all local stages."""
        for stage in self.prefill_stages:
            if not getattr(stage, 'is_local', lambda: True)():
                continue
            stage.load()
        for stage in self.decode_stages:
            if not getattr(stage, 'is_local', lambda: True)():
                continue
            stage.load()
        print("[PDRunner] All local stages loaded.")

    def run_inference(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Run the full PD-separated inference pipeline on a single prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Max tokens to generate (default: from config)
            repetition_penalty: Repetition penalty (default: from config)

        Returns:
            Generated text (input + new tokens)
        """
        from pd_inference.utils import encode_prompt, pad_batch

        max_new = max_new_tokens or self.config.max_new_tokens
        rep_penalty = repetition_penalty or self.config.repetition_penalty

        # ── Encode prompt ──
        input_ids = encode_prompt(prompt, self.tokenizer)
        print(f"[inference] Prompt: '{prompt}' ({len(input_ids)} tokens)")
        print(f"[inference] Generating {max_new} tokens...")

        # ── Run prefill ──
        t0 = time.time()
        padded, mask = pad_batch([input_ids], pad_id=(
            self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
        ))

        kv_cache, hidden = run_prefill_pipeline(
            self.prefill_stages, padded, mask, self.rpc
        )
        prefill_time = time.time() - t0
        print(f"[inference] Prefill done: {prefill_time:.3f}s")

        # ── Get first token from prefill ──
        t1 = time.time()
        last_stage = self.prefill_stages[-1]
        if getattr(last_stage, 'is_local', lambda: True)():
            first_tokens = last_stage.get_next_token(hidden, [len(input_ids)])
        elif self.rpc is not None:
            first_tokens = self.rpc.call(
                last_stage.rank,
                "prefill_last_token_ids",
                {"hidden_states": hidden, "lengths": [len(input_ids)]},
            )
        else:
            first_tokens = []

        if isinstance(first_tokens, torch.Tensor):
            first_tokens = first_tokens.detach().cpu().tolist()
        print(f"[inference] First token: {first_tokens}")

        # ── Run decode ──
        generated = run_decode_pipeline(
            self.decode_stages,
            input_ids,
            kv_cache,
            max_new,
            request_id="req-000001",
            rpc=self.rpc,
            prefill_token_ids=first_tokens,
            repetition_penalty=rep_penalty,
        )
        decode_time = time.time() - t1
        total_time = time.time() - t0

        # ── Decode result ──
        gen_tokens = len(generated) - len(input_ids)
        generated_text = decode_tokens(self.tokenizer, generated)
        print(f"[inference] Done: {gen_tokens} tokens in {total_time:.2f}s "
              f"({gen_tokens / total_time:.1f} tok/s)")

        return generated_text

    def get_worker_handlers(self) -> Dict[str, callable]:
        """Get RPC handler functions for worker nodes.

        Called on decode node to register handlers with RPCServer.
        """
        # Find local decode stage
        local_decode = None
        for stage in self.decode_stages:
            if getattr(stage, 'is_local', lambda: False)():
                local_decode = stage.stage if hasattr(stage, 'stage') else stage
                break

        if local_decode is None:
            return {}

        def _decode_step(request_id, input_ids, hidden_states, past_len):
            return local_decode.decode_step(request_id, input_ids, hidden_states, past_len)

        def _init_kv(request_id, kv_cache):
            return local_decode.init_kv(request_id, kv_cache)

        def _clear_kv(request_id):
            return local_decode.clear_kv(request_id)

        def _compute_logits(hidden_states):
            return local_decode.compute_logits(hidden_states)

        return {
            "decode_step": _decode_step,
            "init_kv": _init_kv,
            "clear_kv": _clear_kv,
            "compute_logits": _compute_logits,
        }

    def get_prefill_handlers(self) -> Dict[str, callable]:
        """Get RPC handler functions for prefill worker nodes.

        Called on prefill node to register handlers with RPCServer
        (e.g., when prefill node also serves RPC for logits computation).
        """
        local_prefill = None
        for stage in self.prefill_stages:
            if not getattr(stage, 'is_local', lambda: False)():
                continue
            local_prefill = stage.stage if hasattr(stage, 'stage') else stage
            break

        if local_prefill is None:
            return {}

        def _prefill_forward(input_ids, hidden_states, attention_mask):
            with local_prefill._lock:
                return local_prefill.forward(input_ids, hidden_states, attention_mask)

        def _prefill_last_token_ids(hidden_states, lengths):
            return local_prefill.get_next_token(hidden_states, lengths)

        return {
            "prefill_forward": _prefill_forward,
            "prefill_last_token_ids": _prefill_last_token_ids,
        }
