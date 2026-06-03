"""Token importance evaluation — simplified Key-norm proxy.

Phase 1 uses Key L2-norm as a lightweight proxy for attention importance.
Phase 2 will upgrade to the full three-dimensional scoring (attention score +
layer sensitivity + position decay) once attention score capture is stable.

Reference: ABKT_创新方案.md section 4.1
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class TokenImportanceEvaluator:
    """Evaluates token importance using Key L2-norm proxy.

    Usage:
        evaluator = TokenImportanceEvaluator()
        importance = evaluator.compute(kv_cache, num_layers, seq_len)
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.25, gamma: float = 0.15):
        # Weights for phase-2 fusion (currently unused)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        num_layers: int,
        seq_len: int,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """Compute importance scores for all entries.

        Uses Key L2-norm as a proxy for attention importance.
        Returns {decode_node: {layer_idx: importance_tensor[seq_len]}} in [0, 1].
        """
        result = {}
        for dnode, layer_cache in kv_cache.items():
            result[dnode] = {}
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                score = self._score_from_key(k)
                result[dnode][lidx] = score
        return result

    @staticmethod
    def _score_from_key(k: torch.Tensor) -> torch.Tensor:
        """Compute normalized token importance from Key tensor.

        Higher L2-norm → higher attention contribution → more important.
        """
        # k shape: [batch, heads, seq, head_dim]
        norm = torch.linalg.vector_norm(k.float(), dim=(0, 1, 3))  # [seq]
        if norm.max() > norm.min():
            scores = (norm - norm.min()) / (norm.max() - norm.min())
        else:
            scores = torch.ones_like(norm)
        return scores  # [seq]
