#!/usr/bin/env python3
"""PPL evaluation for ABKT KV cache quantization strategies.

Computes perplexity on WikiText-2 test set using sliding window evaluation.
Each strategy applies a different precision allocation to the KV cache,
then measures PPL on continuation tokens using the quantized cache.

This simulates the full ABKT pipeline offline on a single GPU:
  1. Prefill → extract KV cache (FP16)
  2. Apply precision allocation strategy
  3. Quantize → dequantize (simulates transfer roundtrip)
  4. Decode continuation with reconstructed KV cache
  5. Measure cross-entropy loss → PPL

Usage:
    # Quick smoke test (5 windows)
    python tools/eval_ppl.py --model /ssd/models/opt-2.7b-safetensors --max-windows 5

    # Compare all strategies
    python tools/eval_ppl.py --model /ssd/models/opt-2.7b-safetensors --output results.json

    # Run specific strategies
    python tools/eval_ppl.py --model /ssd/models/opt-2.7b-safetensors --strategy abkt uniform_fp8

    # Ablation experiment
    python tools/eval_ppl.py --model /ssd/models/opt-2.7b-safetensors \\
        --strategy abkt_full abkt_no_attn abkt_no_layer abkt_no_position abkt_no_fidelity
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from backend.precision_allocator import PrecisionAllocator, Precision
from backend.adaptive_quant import AdaptiveQuantizer
from backend.token_importance import TokenImportanceEvaluator
from pd_inference.kv_cache import KVCache

# ── Strategy definitions ──

ALL_STRATEGIES = [
    "fp16", "abkt", "uniform_fp8", "uniform_int4", "uniform_int2", "random",
]

ABLATION_STRATEGIES = [
    "abkt_full", "abkt_no_attn", "abkt_no_layer", "abkt_no_position", "abkt_no_fidelity",
]

ABLATION_CONFIGS = {
    "abkt_full":        {"alpha": 0.6,  "beta": 0.25, "gamma": 0.15, "fidelity": True},
    "abkt_no_attn":     {"alpha": 0.0,  "beta": 0.5,  "gamma": 0.33, "fidelity": True},
    "abkt_no_layer":    {"alpha": 0.75, "beta": 0.0,  "gamma": 0.25, "fidelity": True},
    "abkt_no_position": {"alpha": 0.7,  "beta": 0.3,  "gamma": 0.0,  "fidelity": True},
    "abkt_no_fidelity": {"alpha": 0.6,  "beta": 0.25, "gamma": 0.15, "fidelity": False},
}


def evaluate_strategy(
    model,
    windows: List[List[int]],
    strategy: str,
    context_len: int = 2048,
    stride: int = 512,
    budget_ratio: float = 0.5,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Evaluate a single strategy and return PPL + stats."""
    quantizer = AdaptiveQuantizer()
    allocator = PrecisionAllocator()

    # Determine evaluator params for ablation
    if strategy in ABLATION_CONFIGS:
        cfg = ABLATION_CONFIGS[strategy]
        evaluator = TokenImportanceEvaluator(
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"]
        )
        use_fidelity = cfg["fidelity"]
        display_name = strategy
    elif strategy == "abkt":
        evaluator = TokenImportanceEvaluator()
        use_fidelity = True
        display_name = f"abkt(budget={budget_ratio})"
    else:
        evaluator = None
        use_fidelity = True
        display_name = strategy

    total_loss = 0.0
    total_tokens = 0
    total_bytes_sum = 0.0
    total_fp16_sum = 0.0
    t_start = time.time()

    for win_idx, window_ids in enumerate(windows):
        # a) Extract KV cache via forward pass on context portion
        context_input = torch.tensor([window_ids[:stride]], device=device)
        with torch.no_grad():
            context_out = model(context_input, use_cache=True)
        orig_kv = context_out.past_key_values

        # b) Convert to ABKT format
        kv_cache_obj = KVCache.from_dynamic_cache(orig_kv)
        abkt_kv = kv_cache_obj.to_abkt_dict()
        num_layers = kv_cache_obj.num_layers
        seq_len = kv_cache_obj.seq_len
        del orig_kv, context_out

        # c) Apply strategy to get precision_map
        if strategy == "fp16":
            result = allocator.allocate_uniform(abkt_kv, Precision.FP16)
        elif strategy == "uniform_fp8":
            result = allocator.allocate_uniform(abkt_kv, Precision.FP8)
        elif strategy == "uniform_int4":
            result = allocator.allocate_uniform(abkt_kv, Precision.INT4)
        elif strategy == "uniform_int2":
            result = allocator.allocate_uniform(abkt_kv, Precision.INT2)
        elif strategy == "random":
            result = allocator.allocate_random(abkt_kv, seed=seed)
        elif strategy in ("abkt",) or strategy in ABLATION_CONFIGS:
            importance = evaluator.compute(abkt_kv, num_layers, seq_len)
            total_fp16 = PrecisionAllocator._total_bytes(abkt_kv, Precision.FP16)
            budget = total_fp16 * budget_ratio
            result = allocator.allocate(
                importance, abkt_kv, budget,
                num_layers_total=num_layers,
                use_quality_fidelity=use_fidelity,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # d) Quantize → dequantize (simulates transfer roundtrip)
        if strategy == "fp16":
            # No quantization — use original KV directly
            reconstructed_kv = abkt_kv
        else:
            quantized_kv, metadata = quantizer.quantize(abkt_kv, result.precision_map)
            reconstructed_kv = quantizer.dequantize(quantized_kv, metadata)

        # e) Reconstruct DynamicCache from dequantized KV
        recon_obj = KVCache.from_abkt_dict(reconstructed_kv)
        dyn_cache = recon_obj.to_dynamic_cache(device)

        # f) Compute PPL on continuation tokens with reconstructed cache
        cont_ids = window_ids[stride:context_len]
        cont_input = torch.tensor([cont_ids], device=device)
        position_ids = torch.arange(stride, stride + len(cont_ids), device=device).unsqueeze(0)

        with torch.no_grad():
            cont_out = model(
                cont_input,
                past_key_values=dyn_cache,
                position_ids=position_ids,
            )

        # g) Compute loss: logits[i] predicts target[i+1]
        # cont_out.logits shape: [1, seq, vocab]
        logits = cont_out.logits[:, :-1, :].contiguous().float()
        targets = cont_input[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
        total_bytes_sum += result.total_bytes
        total_fp16_sum += PrecisionAllocator._total_bytes(abkt_kv, Precision.FP16)

        # Cleanup
        del dyn_cache, recon_obj, cont_out, logits, targets
        torch.cuda.empty_cache()

        # Progress
        if (win_idx + 1) % 10 == 0 or win_idx == 0:
            running_ppl = math.exp(total_loss / max(total_tokens, 1))
            elapsed = time.time() - t_start
            print(f"  [{display_name}] window {win_idx + 1}/{len(windows)}: "
                  f"PPL={running_ppl:.4f}  ({elapsed:.1f}s)")

    ppl = math.exp(total_loss / max(total_tokens, 1))
    avg_bytes = total_bytes_sum / len(windows)
    avg_fp16 = total_fp16_sum / len(windows)
    elapsed = time.time() - t_start

    return {
        "strategy": strategy,
        "ppl": ppl,
        "avg_bits": result.avg_precision_bits,
        "compression_ratio": avg_fp16 / max(avg_bytes, 1),
        "total_bytes": avg_bytes,
        "fp16_bytes": avg_fp16,
        "num_windows": len(windows),
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
    }


def print_comparison_table(results: List[dict], model_name: str,
                           dataset_name: str, context_len: int, stride: int):
    """Print formatted comparison table."""
    print(f"\n{'=' * 75}")
    print(f" ABKT Evaluation Results")
    print(f" Model: {model_name}  |  Dataset: {dataset_name}"
          f"  |  Context: {context_len}  |  Stride: {stride}")
    print(f"{'=' * 75}")
    header = f"{'Strategy':<22} {'PPL':>10} {'Compress':>10} {'AvgBits':>8} {'Size(MB)':>10} {'Time':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['strategy']:<22} {r['ppl']:>10.4f} "
              f"{r['compression_ratio']:>9.2f}x "
              f"{r['avg_bits']:>8.1f} "
              f"{r['total_bytes'] / 1e6:>10.2f} "
              f"{r['elapsed_sec']:>7.1f}s")
    print(f"{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(description="ABKT PPL Evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1",
                        help="Dataset config")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--strategy", nargs="+", default=None,
                        help=f"Strategies to evaluate. Options: {ALL_STRATEGIES + ABLATION_STRATEGIES}")
    parser.add_argument("--budget-ratio", type=float, default=0.5,
                        help="Budget as fraction of FP16 size (for abkt)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit windows for fast testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.strategy is None:
        args.strategy = ALL_STRATEGIES

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).to(args.device).eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers, device={args.device}")

    print(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    ds = load_dataset(args.dataset, args.dataset_config, split="test")
    all_text = "\n\n".join(ex["text"] for ex in ds if ex["text"].strip())
    all_ids = tokenizer.encode(all_text)
    print(f"Dataset: {len(all_ids)} tokens")

    # Build sliding windows
    windows = []
    for i in range(0, len(all_ids) - args.context_len + 1, args.stride):
        windows.append(all_ids[i:i + args.context_len])
    if args.max_windows:
        windows = windows[:args.max_windows]
    print(f"Windows: {len(windows)} (context={args.context_len}, stride={args.stride})")

    # Run each strategy
    results = []
    for strategy in args.strategy:
        print(f"\nEvaluating: {strategy}")
        r = evaluate_strategy(
            model, windows, strategy,
            context_len=args.context_len,
            stride=args.stride,
            budget_ratio=args.budget_ratio,
            seed=args.seed,
            device=args.device,
        )
        results.append(r)
        print(f"  DONE: PPL={r['ppl']:.4f}, avg_bits={r['avg_bits']:.1f}, "
              f"compression={r['compression_ratio']:.2f}x, {r['elapsed_sec']:.1f}s")

    # Print table
    model_name = os.path.basename(args.model.rstrip("/"))
    print_comparison_table(results, model_name, args.dataset_config,
                           args.context_len, args.stride)

    # Save JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
