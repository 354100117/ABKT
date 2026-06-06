"""Token importance evaluation — three-dimensional scoring.

Scores each KV cache entry along three dimensions:
1. Attention importance (alpha=0.6): mean attention weight received by each token
2. Layer sensitivity (beta=0.25): coefficient of variation of Key tensor per layer
3. Position decay (gamma=0.15): exponential decay favoring recent tokens

When attention_weights is None (backward-compatible), falls back to Key L2-norm
proxy for dimension 1, and uniform weights for dimensions 2-3.

Reference: ABKT_创新方案.md section 4.1
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class TokenImportanceEvaluator:
    """Evaluates token importance using three-dimensional scoring.

    Usage:
        evaluator = TokenImportanceEvaluator()
        importance = evaluator.compute(kv_cache, num_layers, seq_len)
        # With attention weights (full 3D scoring):
        importance = evaluator.compute(kv_cache, num_layers, seq_len,
                                       attention_weights=attn_weights)
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.25, gamma: float = 0.15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        num_layers: int,
        seq_len: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """Compute importance scores for all entries.

        Args:
            kv_cache: {decode_node: {layer_idx: (K, V)}}.
            num_layers: Total number of layers.
            seq_len: Sequence length.
            attention_weights: Optional [batch, heads, seq, seq] attention
                weights from the prefill forward pass. When provided, enables
                full 3D scoring; when None, uses Key L2-norm proxy.

        Returns:
            {decode_node: {layer_idx: importance_tensor[seq_len]}} in [0, 1].
        """
        # Dimension 1: attention importance
        attn_importance = self._compute_attention_importance(
            attention_weights, kv_cache, seq_len
        )

        # Dimension 2: layer sensitivity (per-decode-node, per-layer)
        layer_sensitivity = self._compute_layer_sensitivity(kv_cache)

        # Dimension 3: position decay (shared across all entries)
        pos_decay = self._compute_position_decay(seq_len)

        # Fuse and normalize per (dnode, layer)
        result = {}
        for dnode, layer_cache in kv_cache.items():
            result[dnode] = {}
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                # Attention dimension: [seq]
                attn_score = attn_importance.get((dnode, lidx), attn_importance.get("_default"))
                # Layer sensitivity: scalar
                lscore = layer_sensitivity.get((dnode, lidx), 0.5)
                # Fuse: [seq]
                fused = (self.alpha * attn_score
                         + self.beta * lscore
                         + self.gamma * pos_decay)
                # Normalize to [0, 1]
                fmin, fmax = fused.min(), fused.max()
                if fmax > fmin:
                    fused = (fused - fmin) / (fmax - fmin)
                else:
                    fused = torch.ones_like(fused)
                result[dnode][lidx] = fused
        return result

    def _compute_attention_importance(
        self,
        attention_weights: Optional[torch.Tensor],
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        seq_len: int,
    ) -> Dict:
        """Dimension 1: per-token importance from attention or Key L2-norm proxy.

        Returns dict keyed by (dnode, lidx) with [seq] tensors, plus a
        '_default' key for the proxy fallback.
        """
        if attention_weights is not None:
            # attention_weights: [batch, heads, seq, seq]
            # Mean over batch and heads → [seq, seq], then sum columns → [seq]
            # (how much total attention each token receives)
            attn = attention_weights.float().mean(dim=(0, 1))  # [seq, seq]
            score = attn.sum(dim=0)  # [seq]
            if score.max() > score.min():
                score = (score - score.min()) / (score.max() - score.min())
            else:
                score = torch.ones(seq_len, device=score.device)
            return {"_default": score}

        # Fallback: Key L2-norm proxy (backward-compatible)
        proxy = {}
        for dnode, layer_cache in kv_cache.items():
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, _ = kv
                proxy[(dnode, lidx)] = self._score_from_key(k)
        # Use mean across layers as default
        if proxy:
            stacked = torch.stack(list(proxy.values()))
            proxy["_default"] = stacked.mean(dim=0)
        return proxy

    @staticmethod
    def _compute_layer_sensitivity(
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Dict[Tuple[int, int], float]:
        """Dimension 2: layer sensitivity via coefficient of variation (CV).

        Higher CV → more variation in Key magnitudes → more sensitive to
        quantization error → higher importance weight.
        Returns {(dnode, lidx): float} in [0, 1].
        """
        cvs = {}
        for dnode, layer_cache in kv_cache.items():
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, _ = kv
                # CV = std / mean of absolute Key values
                flat = k.float().abs().reshape(-1)
                mean_val = flat.mean()
                if mean_val > 0:
                    cv = (flat.std() / mean_val).item()
                else:
                    cv = 0.0
                cvs[(dnode, lidx)] = cv

        if cvs:
            cv_min = min(cvs.values())
            cv_max = max(cvs.values())
            if cv_max > cv_min:
                cvs = {k: (v - cv_min) / (cv_max - cv_min) for k, v in cvs.items()}
            else:
                cvs = {k: 0.5 for k in cvs}
        return cvs

    @staticmethod
    def _compute_position_decay(seq_len: int, rate: float = 0.01) -> torch.Tensor:
        """Dimension 3: exponential position decay.

        Recent tokens (higher indices) get higher scores.
        Returns [seq_len] tensor in (0, 1].
        """
        positions = torch.arange(seq_len, dtype=torch.float32)
        decay = torch.exp(-rate * (seq_len - 1 - positions))
        # Normalize to [0, 1]
        return decay / decay.max()

    @staticmethod
    def _score_from_key(k: torch.Tensor) -> torch.Tensor:
        """Compute normalized token importance from Key tensor (L2-norm proxy)."""
        # k shape: [batch, heads, seq, head_dim]
        norm = torch.linalg.vector_norm(k.float(), dim=(0, 1, 3))  # [seq]
        if norm.max() > norm.min():
            scores = (norm - norm.min()) / (norm.max() - norm.min())
        else:
            scores = torch.ones_like(norm)
        return scores  # [seq]
