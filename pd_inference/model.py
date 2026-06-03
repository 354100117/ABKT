"""Model stages for PD-separated inference.

Core classes:
    - PrefillStage: Runs the prompt through its assigned layers
    - DecodeStage: Runs autoregressive decoding using KV cache

Each stage loads only its required model components for memory efficiency.
"""

from __future__ import annotations

import os
import threading
import traceback
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from pd_inference.utils import (
    compute_position_embeds,
    detach_kv_to_cpu,
    detach_kv_to_device,
    get_device,
    kv_shape_str,
    module_device,
    module_dtype,
    prepare_layer_attention_mask,
    tensor_summary,
    try_load_partial_model,
)

# ── Environment flags ──

_DEBUG = os.environ.get("EDGEPD_DEBUG", "0") == "1"


def _dprint(msg: str):
    if _DEBUG:
        print(msg)


# ── Trace hidden states (first, last, and sample layers) ──


def _should_trace_layer(local_idx: int, num_layers: int) -> bool:
    """Check if we should trace hidden states for this layer."""
    if num_layers <= 0:
        return False
    return local_idx in {0, 1, num_layers - 1}


# ════════════════════════════════════════════════════════════════════
# Model architecture abstraction
# ════════════════════════════════════════════════════════════════════


class ModelArch:
    """Unified access to model components across architectures.

    Handles structural differences between OPT (learned positional embeddings,
    model.model.decoder) and Qwen2.5/Llama-style models (RoPE, model.model).
    """

    def __init__(self, model):
        config = model.config
        model_type = getattr(config, "model_type", "")

        if model_type in ("qwen2",):
            self._backbone = model.model
            self._norm = self._backbone.norm
            self._embed_tokens = self._backbone.embed_tokens
            self._embed_positions = None
            self._project_in = None
            self._project_out = None
            self._use_rope = True
        else:  # OPT and similar architectures
            self._backbone = model.model.decoder
            self._norm = getattr(self._backbone, "final_layer_norm", None)
            self._embed_tokens = self._backbone.embed_tokens
            self._embed_positions = getattr(self._backbone, "embed_positions", None)
            self._project_in = getattr(self._backbone, "project_in", None)
            self._project_out = getattr(self._backbone, "project_out", None)
            self._use_rope = False

    @property
    def layers(self):
        return self._backbone.layers

    @property
    def norm(self):
        return self._norm

    @property
    def embed_tokens(self):
        return self._embed_tokens

    @property
    def embed_positions(self):
        return self._embed_positions

    @property
    def project_in(self):
        return self._project_in

    @property
    def project_out(self):
        return self._project_out

    @property
    def use_rope(self):
        return self._use_rope


# ════════════════════════════════════════════════════════════════════
# PrefillStage
# ════════════════════════════════════════════════════════════════════


class PrefillStage:
    """Prefill stage: processes prompt through assigned layers.

    For the first prefill node, this includes token embedding and
    positional embedding. For the last prefill node, this includes
    final layer norm and LM head for logit computation.

    The prefill runs a single forward pass (not autoregressive)
    and extracts the KV cache for use by decode stages.
    """

    def __init__(
        self,
        stage_id: int,
        layer_range: Tuple[int, int],
        model_name: str,
        load_embed: bool = False,
        load_lm_head: bool = False,
        num_layers_total: Optional[int] = None,
    ):
        self.stage_id = stage_id
        self.layer_range = layer_range          # (start, end) of layers this stage handles
        self.model_name = model_name
        self.load_embed = load_embed             # first node: load embed_tokens + embed_positions
        self.load_lm_head = load_lm_head         # last node: load final_layer_norm + lm_head
        self.device = get_device()

        # Model components (to be loaded)
        self.model = None                        # Full model (for simple forward)
        self._arch: Optional[ModelArch] = None   # Architecture abstraction
        self.layers: Optional[List[nn.Module]] = None
        self.lm_head: Optional[nn.Module] = None
        self.num_layers: int = 0
        self.pad_token_id: Optional[int] = None

        # Config
        self._config = AutoConfig.from_pretrained(model_name) if model_name else None
        if self._config:
            self.num_layers = getattr(self._config, "num_hidden_layers",
                                     getattr(self._config, "num_layers", 0))
            self.pad_token_id = getattr(self._config, "pad_token_id", None)
        if num_layers_total is not None:
            self.num_layers = num_layers_total

        # Thread safety
        self._lock = threading.Lock()

    def load(self):
        """Load model components for this stage."""
        if self.layers is not None:
            return   # already loaded

        start, end = self.layer_range
        _dprint(f"[PrefillStage {self.stage_id}] Loading layers {start}-{end}, "
                f"embed={self.load_embed}, lm_head={self.load_lm_head}")

        # ── Try partial loading first ──
        if start != 0 or end != 0:
            # Non-zero layer range means partial — use layer range directly
            partial_model = try_load_partial_model(
                self.model_name, self.layer_range, self.load_embed, self.load_lm_head
            )
            if partial_model is not None:
                self._load_from_partial(partial_model)
                print(f"[PrefillStage {self.stage_id}] Loaded partial model "
                      f"layers={start}-{end} embed={self.load_embed} lm_head={self.load_lm_head}")
                return

        # ── Fall back to full model ──
        print(f"[PrefillStage {self.stage_id}] Loading full model from {self.model_name}")
        self._load_full_model()

    def _load_full_model(self):
        """Load the full model and extract needed components."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.pad_token_id = self.model.config.pad_token_id
        self._arch = ModelArch(self.model)

        start, end = self.layer_range if self.layer_range != (0, 0) else (0, self.num_layers)

        # Extract LM head components (last node)
        if self.load_lm_head or end >= self.num_layers:
            self.lm_head = self.model.lm_head.to(self.device)

        # Extract layers
        self.layers = [self._arch.layers[i].to(self.device) for i in range(start, end)]
        print(f"[PrefillStage {self.stage_id}] Loaded {len(self.layers)} layers [{start}-{end})")

    def _load_from_partial(self, model):
        """Load components from a partially loaded model."""
        self.model = model.to(self.device)
        self.model.eval()
        self.pad_token_id = self.model.config.pad_token_id
        self._arch = ModelArch(self.model)

        start, end = self.layer_range

        if self.load_lm_head:
            self.lm_head = self.model.lm_head.to(self.device)

        self.layers = [self._arch.layers[i].to(self.device) for i in range(start, end)]

    def forward(
        self,
        input_ids,
        hidden_states=None,
        attention_mask=None,
    ):
        """Run prefill forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len] or None if hidden_states provided
            hidden_states: Previous hidden states (from upstream stage) or None
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            (hidden_states, kv_cache, attention_mask)
            hidden_states: Tensor on CPU
            kv_cache: Dict[node_idx][layer_idx] = (k, v) — layer-mapped KV cache
            attention_mask: Updated attention mask on CPU (or None)
        """
        if self.layers is None:
            self.load()

        start, end = self.layer_range
        _dprint(f"[Prefill {self.stage_id}] forward start | "
                f"hidden={'given' if hidden_states is not None else 'None'} "
                f"layer_range=[{start},{end})")

        # ── Compute embeddings if this is the first stage ──
        position_ids = None
        if hidden_states is None:
            if self.model is not None:
                # Full model path: run all layers at once
                return self._forward_full_model(input_ids, attention_mask)
            else:
                # Partial model: compute embeddings, then run assigned layers
                input_ids_t = torch.tensor(input_ids, device=self.device)
                bs, seq_len = input_ids_t.shape
                arch = self._arch
                hidden_states = arch.embed_tokens(input_ids_t)
                if arch.use_rope:
                    position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
                else:
                    if arch.embed_positions is not None:
                        pos_embeds = compute_position_embeds(
                            arch.embed_positions,
                            bs=bs, seq_len=seq_len, past_len=0,
                            device=self.device, input_ids=input_ids_t,
                            pad_token_id=self.pad_token_id,
                        )
                        hidden_states = hidden_states + pos_embeds
                    if arch.project_in is not None:
                        hidden_states = arch.project_in(hidden_states)

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.to(self.device)
        else:
            hidden_states = torch.tensor(hidden_states, device=self.device)
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.device)
            else:
                attention_mask = torch.tensor(attention_mask, device=self.device)

        # ── Run layers ──
        local_kv_cache = []
        use_rope = self._arch.use_rope if self._arch else False
        for local_idx, layer in enumerate(self.layers):
            global_idx = start + local_idx
            layer_past = None
            layer_attn = prepare_layer_attention_mask(attention_mask, hidden_states, layer_past)

            layer_kwargs = dict(
                attention_mask=layer_attn,
                use_cache=True,
                past_key_values=layer_past,
            )
            if use_rope and position_ids is not None:
                layer_kwargs["position_ids"] = position_ids
            out = layer(hidden_states, **layer_kwargs)

            if torch.is_tensor(out):
                hidden_states = out
            elif isinstance(out, (tuple, list)):
                hidden_states = out[0]
            else:
                hidden_states = out.last_hidden_state

            # Extract present KV
            present = None
            if isinstance(out, (tuple, list)):
                if len(out) >= 3 and out[2] is not None:
                    present = out[2]
                elif len(out) >= 2 and out[1] is not None:
                    present = out[1]
            present_attr = getattr(out, "past_key_values", None)
            if present_attr is not None:
                present = present_attr
            elif hasattr(out, "present") and out.present is not None:
                present = out.present

            if present is not None:
                try:
                    kv_cpu = detach_kv_to_cpu(present)
                    local_kv_cache.append(kv_cpu)
                except Exception:
                    local_kv_cache.append(None)
            else:
                local_kv_cache.append(None)

            _dprint(f"[Prefill {self.stage_id}] layer[{local_idx}] global={global_idx} "
                    f"{tensor_summary(hidden_states, 'h')} "
                    f"kv={kv_shape_str(local_kv_cache[-1])}")

        # ── Build KV cache dict ──
        # For PD-split: map from local layer index to global decode node index
        # Currently supports 1 prefill + 1 decode: all layers go to decode node 0
        kv_cache = {0: {i: kv for i, kv in enumerate(local_kv_cache)}}

        attn_out = attention_mask.detach().cpu() if attention_mask is not None else None
        _dprint(f"[Prefill {self.stage_id}] done | hidden={tensor_summary(hidden_states)} "
                f"kv_layers={sum(1 for x in local_kv_cache if x is not None)}"
                f"/{len(local_kv_cache)}")

        return hidden_states.detach().cpu(), kv_cache, attn_out

    def _forward_full_model(self, input_ids, attention_mask):
        """Forward through entire model, extract this stage's KV cache."""
        input_ids_t = torch.tensor(input_ids, device=self.device)
        attn_mask = None
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                attn_mask = attention_mask.to(self.device)
            else:
                attn_mask = torch.tensor(attention_mask, device=self.device)
            if attn_mask.dtype in (torch.int64, torch.int32, torch.float32):
                target_dtype = module_dtype(self.lm_head if self.load_lm_head else self.layers[0]
                                           if self.layers else None)
                if self.model is not None:
                    target_dtype = getattr(self.model, "dtype", target_dtype)
                attn_mask = attn_mask.to(target_dtype)

        with torch.no_grad():
            outputs = self.model(
                input_ids_t,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
            )

        # Extract hidden states at the end of this stage's layer range
        start, end = self.layer_range
        hidden_states = outputs.hidden_states[end] if hasattr(outputs, 'hidden_states') and outputs.hidden_states else None
        if hidden_states is None:
            # Extract from last_hidden_state
            hidden_states = outputs.last_hidden_state

        # Extract KV cache for this stage's layers
        # Handle DynamicCache (transformers 5.0.0) and legacy tuple formats
        past = outputs.past_key_values
        local_kv_cache = []
        if past is not None:
            if hasattr(past, 'layers'):
                # DynamicCache v5.0.0+: past.layers is List[DynamicLayer]
                for i in range(start, min(end, len(past.layers))):
                    try:
                        layer = past.layers[i]
                        k = layer.keys
                        v = layer.values
                        local_kv_cache.append((k.detach().cpu(), v.detach().cpu()))
                    except Exception:
                        local_kv_cache.append(None)
            else:
                # Legacy DynamicCache or tuple format
                if hasattr(past, 'to_legacy_cache'):
                    past = past.to_legacy_cache()
                for i in range(start, min(end, len(past))):
                    try:
                        layer_kv = past[i]
                        if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                            k, v = layer_kv
                            local_kv_cache.append((k.detach().cpu(), v.detach().cpu()))
                        else:
                            local_kv_cache.append(None)
                    except Exception:
                        local_kv_cache.append(None)

        kv_cache = {0: {i: kv for i, kv in enumerate(local_kv_cache)}}
        attn_out = attn_mask.detach().cpu() if attn_mask is not None else None
        _dprint(f"[Prefill {self.stage_id}] full model done | "
                f"{tensor_summary(hidden_states, 'h')} "
                f"kv_layers={sum(1 for x in local_kv_cache if x is not None)}/{len(local_kv_cache)}")

        if hasattr(hidden_states, 'detach'):
            hidden_states = hidden_states.detach().cpu()
        return hidden_states, kv_cache, attn_out

    def compute_logits(self, hidden_states):
        """Compute logits from hidden states for the last prefill node.

        Args:
            hidden_states: Hidden states from the last layer

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        if self.lm_head is None:
            raise RuntimeError(
                f"PrefillStage {self.stage_id} has no lm_head "
                f"(load_lm_head={self.load_lm_head})"
            )

        if isinstance(hidden_states, torch.Tensor):
            h = hidden_states.to(self.device).float()
        else:
            h = torch.tensor(hidden_states, device=self.device).float()

        if self._arch is not None:
            if self._arch.norm is not None:
                h = self._arch.norm.to(device=h.device, dtype=torch.float32)(h)
            if self._arch.project_out is not None:
                h = self._arch.project_out.to(device=h.device, dtype=torch.float32)(h)

        bias = self.lm_head.bias.float() if self.lm_head.bias is not None else None
        logits = torch.nn.functional.linear(
            h, self.lm_head.weight.float(), bias
        )
        return logits.detach().cpu()

    def get_next_token(self, hidden_states, lengths: List[int]) -> List[int]:
        """Get the next token IDs from hidden states.

        Uses the last token of each sequence for autoregressive generation.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_size]
            lengths: Sequence lengths for each item in the batch

        Returns:
            List of next token IDs
        """
        logits = self.compute_logits(hidden_states)
        next_tokens = []
        for i, length in enumerate(lengths):
            token_logits = logits[i, length - 1, :]
            next_token = int(torch.argmax(token_logits, dim=-1).item())
            next_tokens.append(next_token)
        return next_tokens


# ════════════════════════════════════════════════════════════════════
# DecodeStage
# ════════════════════════════════════════════════════════════════════


class DecodeStage:
    """Decode stage: runs autoregressive decoding using KV cache.

    For the first decode node, this includes token embedding and
    positional embedding. For the last decode node, this includes
    final layer norm and LM head for logit computation.

    Each request maintains its own KV cache for the duration of
    autoregressive generation.
    """

    def __init__(
        self,
        stage_id: int,
        layer_range: Tuple[int, int],
        model_name: str,
        load_embed: bool = False,
        load_lm_head: bool = False,
        num_layers_total: Optional[int] = None,
    ):
        self.stage_id = stage_id
        self.layer_range = layer_range          # (start, end) of layers this stage handles
        self.model_name = model_name
        self.load_embed = load_embed             # first node: embed
        self.load_lm_head = load_lm_head         # last node: final_layer_norm + lm_head
        self.device = get_device()

        # Model components
        self._arch: Optional[ModelArch] = None   # Architecture abstraction
        self.layers: Optional[List[nn.Module]] = None
        self.lm_head: Optional[nn.Module] = None
        self.num_layers: int = 0
        self.pad_token_id: Optional[int] = None

        # Config
        self._config = AutoConfig.from_pretrained(model_name) if model_name else None
        if self._config:
            self.num_layers = getattr(self._config, "num_hidden_layers",
                                     getattr(self._config, "num_layers", 0))
            self.pad_token_id = getattr(self._config, "pad_token_id", None)
        if num_layers_total is not None:
            self.num_layers = num_layers_total

        # Per-request KV cache: {request_id: [(k, v), ...]}
        self._kv_cache: Dict[str, List] = {}
        self._cache_lock = threading.Lock()
        self._lock = threading.Lock()

    def load(self):
        """Load model components for this stage."""
        if self.layers is not None:
            return

        start, end = self.layer_range
        _dprint(f"[DecodeStage {self.stage_id}] Loading layers {start}-{end}, "
                f"embed={self.load_embed}, lm_head={self.load_lm_head}")

        # ── Try partial loading first ──
        if start != 0 or end != 0:
            partial_model = try_load_partial_model(
                self.model_name, self.layer_range, self.load_embed, self.load_lm_head
            )
            if partial_model is not None:
                self._load_from_partial(partial_model)
                print(f"[DecodeStage {self.stage_id}] Loaded partial model "
                      f"layers={start}-{end} embed={self.load_embed} lm_head={self.load_lm_head}")
                return

        # ── Fall back to full model ──
        print(f"[DecodeStage {self.stage_id}] Loading full model from {self.model_name}")
        self._load_full_model()

    def _load_full_model(self):
        """Load the full model and extract needed components."""
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model_ref = self.model_ref.to(self.device)
        self.model_ref.eval()
        self.pad_token_id = self.model_ref.config.pad_token_id
        self._arch = ModelArch(self.model_ref)

        start, end = self.layer_range if self.layer_range != (0, 0) else (0, self.num_layers)

        if self.load_lm_head or end >= self.num_layers:
            self.lm_head = self.model_ref.lm_head.to(self.device)

        self.layers = [self._arch.layers[i].to(self.device) for i in range(start, end)]
        print(f"[DecodeStage {self.stage_id}] Loaded {len(self.layers)} layers [{start}-{end})")

    def _load_from_partial(self, model):
        """Load components from a partially loaded model."""
        model = model.to(self.device)
        model.eval()
        self.pad_token_id = model.config.pad_token_id
        self._arch = ModelArch(model)

        start, end = self.layer_range

        if self.load_lm_head:
            self.lm_head = model.lm_head.to(self.device)

        self.layers = [self._arch.layers[i].to(self.device) for i in range(start, end)]

    def init_kv(self, request_id: str, kv_cache):
        """Initialize KV cache for a request from prefill output.

        Args:
            request_id: Unique request identifier
            kv_cache: List of (k, v) tuples from prefill, one per layer
        """
        if self.layers is None:
            self.load()

        local_len = len(self.layers) if self.layers else 0
        cache = [None for _ in range(local_len)]
        if kv_cache is not None:
            for idx in range(min(local_len, len(kv_cache))):
                kv = kv_cache[idx]
                if kv is None:
                    continue
                k, v = kv
                cache[idx] = (k.to(self.device), v.to(self.device))

        non_none = sum(1 for c in cache if c is not None)
        _dprint(f"[Decode {self.stage_id}] init_kv req={request_id} "
                f"non_none={non_none}/{len(cache)}")

        with self._cache_lock:
            self._kv_cache[request_id] = cache

    def clear_kv(self, request_id: str):
        """Clear KV cache for a completed request."""
        with self._cache_lock:
            self._kv_cache.pop(request_id, None)

    def decode_step(self, request_id: str, input_ids, hidden_states=None, past_len=0):
        """Run a single autoregressive decode step.

        Args:
            request_id: Request identifier
            input_ids: Input token IDs for this step (batch x 1)
            hidden_states: Previous hidden states (from upstream stage) or None for first stage
            past_len: Past sequence length (excluding current tokens)

        Returns:
            Hidden states tensor on CPU
        """
        if self.layers is None:
            self.load()

        # Get KV cache for this request
        with self._cache_lock:
            cache = self._kv_cache.get(request_id)
        if cache is None:
            cache = [None for _ in range(len(self.layers))]
            with self._cache_lock:
                self._kv_cache[request_id] = cache

        # Compute embeddings if this is the first stage
        position_ids = None
        if hidden_states is None:
            input_ids = torch.tensor(input_ids, device=self.device)
            bs, seq = input_ids.shape
            arch = self._arch
            hidden_states = arch.embed_tokens(input_ids)
            if arch.use_rope:
                position_ids = torch.arange(
                    int(past_len), int(past_len) + seq,
                    device=self.device, dtype=torch.long,
                ).unsqueeze(0)
            else:
                pos_embeds = compute_position_embeds(
                    arch.embed_positions,
                    bs=bs, seq_len=seq, past_len=int(past_len),
                    device=self.device, input_ids=input_ids,
                    pad_token_id=self.pad_token_id,
                )
                if pos_embeds is not None:
                    hidden_states = hidden_states + pos_embeds
                if arch.project_in is not None:
                    hidden_states = arch.project_in(hidden_states)

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.to(self.device)
        else:
            hidden_states = torch.tensor(hidden_states, device=self.device)

        start, end = self.layer_range

        # Run through assigned layers with KV cache
        use_rope = self._arch.use_rope if self._arch else False
        for local_idx, layer in enumerate(self.layers):
            layer_past = cache[local_idx]
            if layer_past is not None:
                layer_past = detach_kv_to_device(layer_past, self.device)

            # Note: In transformers 5.0.0+ with SDPA, causal masking is
            # handled internally by the attention module. Passing a 4D mask
            # here conflicts with the internal SDPA mask expansion.
            layer_kwargs = dict(
                attention_mask=None,
                use_cache=True,
                past_key_values=layer_past,
            )
            if use_rope and position_ids is not None:
                layer_kwargs["position_ids"] = position_ids
            out = layer(hidden_states, **layer_kwargs)

            if torch.is_tensor(out):
                hidden_states = out
            elif isinstance(out, (tuple, list)):
                hidden_states = out[0]
            else:
                hidden_states = out.last_hidden_state

            # Extract updated KV cache
            present = None
            if isinstance(out, (tuple, list)):
                if len(out) >= 3 and out[2] is not None:
                    present = out[2]
                elif len(out) >= 2 and out[1] is not None:
                    present = out[1]
            present_attr = getattr(out, "past_key_values", None)
            if present_attr is not None:
                present = present_attr
            elif hasattr(out, "present") and out.present is not None:
                present = out.present

            if present is not None:
                present_dev = detach_kv_to_device(present, self.device)
                cache[local_idx] = present_dev

            if _should_trace_layer(local_idx, len(self.layers)):
                _dprint(f"[Decode {self.stage_id}] layer[{local_idx}] "
                        f"{tensor_summary(hidden_states, 'h')} "
                        f"kv={kv_shape_str(cache[local_idx])}")

        # Update cache
        with self._cache_lock:
            self._kv_cache[request_id] = cache

        _dprint(f"[Decode {self.stage_id}] step done | "
                f"{tensor_summary(hidden_states, 'h')}")
        return hidden_states.detach().cpu()

    def compute_logits(self, hidden_states) -> torch.Tensor:
        """Compute logits from final hidden states (last decode node only).

        Args:
            hidden_states: Hidden states from the last decode layer

        Returns:
            Logits tensor
        """
        if self.lm_head is None:
            raise RuntimeError(
                f"DecodeStage {self.stage_id} has no lm_head "
                f"(load_lm_head={self.load_lm_head})"
            )

        if isinstance(hidden_states, torch.Tensor):
            h = hidden_states.to(self.device).float()
        else:
            h = torch.tensor(hidden_states, device=self.device).float()

        if self._arch is not None:
            if self._arch.norm is not None:
                h = self._arch.norm.to(device=h.device, dtype=torch.float32)(h)
            if self._arch.project_out is not None:
                h = self._arch.project_out.to(device=h.device, dtype=torch.float32)(h)

        bias = self.lm_head.bias.float() if self.lm_head.bias is not None else None
        logits = torch.nn.functional.linear(h, self.lm_head.weight.float(), bias)
        return logits.detach().cpu()

    def run_full_decode(
        self,
        request_id: str,
        kv_cache,
        input_ids: List[int],
        max_new_tokens: int,
        repetition_penalty: float = 1.0,
    ) -> List[int]:
        """Run the full autoregressive decode loop locally.

        Receives KV cache from prefill and runs all decode steps on this node.

        Args:
            request_id: Unique request identifier
            kv_cache: KV cache from prefill in format {decode_node_id: {layer_idx: (k, v)}}
            input_ids: Prompt token IDs
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty (1.0 = disabled)

        Returns:
            Complete list of token IDs (prompt + generated)
        """
        import time

        if self.layers is None:
            self.load()

        # Normalize KV cache: {0: {0: (k,v), ...}} -> [(k,v), ...]
        normalized_kv = _normalize_kv_cache(kv_cache, len(self.layers))
        self.init_kv(request_id, normalized_kv)

        # Determine past length from KV cache
        past_len = _infer_kv_past_len(normalized_kv)

        # Autoregressive decode loop
        generated = list(input_ids)
        generated_token_ids: List[int] = []

        print(f"[decode] Starting autoregressive loop: past_len={past_len}, "
              f"max_new={max_new_tokens}")
        t0 = time.time()

        for step in range(max_new_tokens):
            # Input is the last generated token
            current_token = generated[-1]

            # Run single decode step
            hidden = self.decode_step(
                request_id, [[current_token]], None, past_len
            )

            # Compute logits
            logits_tensor = self.compute_logits(hidden)
            if logits_tensor.dim() == 2:
                logits_tensor = logits_tensor.unsqueeze(1)

            # Select next token (greedy with repetition penalty)
            step_logits = logits_tensor[:, -1, :].clone().float()
            if repetition_penalty != 1.0 and generated_token_ids:
                for token_id in generated_token_ids:
                    if token_id < step_logits.shape[-1]:
                        if step_logits[0, token_id] > 0:
                            step_logits[0, token_id] /= repetition_penalty
                        else:
                            step_logits[0, token_id] *= repetition_penalty

            next_token = int(torch.argmax(step_logits, dim=-1).item())
            generated.append(next_token)
            generated_token_ids.append(next_token)
            past_len += 1

            if step % 10 == 0 or step == max_new_tokens - 1:
                elapsed = time.time() - t0
                tps = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"[decode] step={step+1}/{max_new_tokens} "
                      f"token={next_token} past_len={past_len} "
                      f"tps={tps:.1f}")

        elapsed_total = time.time() - t0
        print(f"[decode] Done: {len(generated_token_ids)} tokens in "
              f"{elapsed_total:.2f}s ({len(generated_token_ids)/elapsed_total:.1f} tok/s)")

        # Clean up
        self.clear_kv(request_id)
        return generated


def _normalize_kv_cache(kv_cache, num_layers: int) -> List:
    """Normalize KV cache from prefill format to list format for DecodeStage.

    Input:  {decode_node_id: {layer_idx: (k, v)}}
    Output: [(k, v), ...] ordered by layer index
    """
    if kv_cache is None:
        return [None] * num_layers

    if isinstance(kv_cache, list):
        return kv_cache

    if isinstance(kv_cache, dict):
        # Find the layer cache (first value in the outer dict)
        layer_cache = None
        for _, v in kv_cache.items():
            if isinstance(v, dict):
                layer_cache = v
                break
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                # Direct (k, v) pair
                return list(kv_cache.values())

        if layer_cache is None:
            return [None] * num_layers

        result = [None] * max(num_layers, max(layer_cache.keys()) + 1 if layer_cache else 0)
        for layer_idx, kv_pair in layer_cache.items():
            if layer_idx < len(result) and kv_pair is not None:
                result[layer_idx] = kv_pair
        return result[:num_layers]

    return [None] * num_layers


def _infer_kv_past_len(kv_cache) -> int:
    """Infer past sequence length from normalized KV cache list."""
    if kv_cache is None:
        return 0
    for kv in kv_cache:
        if kv is None:
            continue
        if isinstance(kv, (list, tuple)) and len(kv) >= 2:
            k, _ = kv
            if hasattr(k, 'shape') and len(k.shape) >= 3:
                return int(k.shape[2])
    return 0
