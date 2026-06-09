#!/usr/bin/env python3
"""QUALITY_FIDELITY calibration and per-layer sensitivity measurement.

Empirically measures quality fidelity values for each precision level by
quantizing KV cache and measuring PPL degradation. Also measures per-layer
sensitivity by quantizing one layer at a time.

Usage:
    # Calibrate quality fidelity values
    python tools/calibrate_fidelity.py --model /ssd/models/opt-2.7b-safetensors

    # Per-layer sensitivity (slower)
    python tools/calibrate_fidelity.py --model /ssd/models/opt-2.7b-safetensors --per-layer

    # Quick test
    python tools/calibrate_fidelity.py --model /ssd/models/opt-2.7b-safetensors --max-windows 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from backend.precision_allocator import PrecisionAllocator, Precision
from backend.adaptive_quant import AdaptiveQuantizer
from pd_inference.kv_cache import KVCache


def compute_ppl(model, windows, abkt_kv_template, precision_map,
                context_len=2048, stride=512, device="cuda"):
    """Compute PPL with a given precision map on the given windows."""
    quantizer = AdaptiveQuantizer()
    total_loss = 0.0
    total_tokens = 0

    for window_ids in windows:
        # Quantize → dequantize
        quantized_kv, metadata = quantizer.quantize(abkt_kv_template, precision_map)
        reconstructed_kv = quantizer.dequantize(quantized_kv, metadata)

        # Reconstruct DynamicCache
        recon_obj = KVCache.from_abkt_dict(reconstructed_kv)
        dyn_cache = recon_obj.to_dynamic_cache(device)

        # Compute PPL on continuation
        cont_ids = window_ids[stride:context_len]
        cont_input = torch.tensor([cont_ids], device=device)
        position_ids = torch.arange(stride, stride + len(cont_ids), device=device).unsqueeze(0)

        with torch.no_grad():
            cont_out = model(
                cont_input,
                past_key_values=dyn_cache,
                position_ids=position_ids,
            )

        logits = cont_out.logits[:, :-1, :].contiguous().float()
        targets = cont_input[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

        del dyn_cache, recon_obj, cont_out, logits, targets
        torch.cuda.empty_cache()

    return math.exp(total_loss / max(total_tokens, 1))


def calibrate_fidelity(model, windows, abkt_kv, num_layers, seq_len,
                        context_len=2048, stride=512, device="cuda"):
    """Measure quality fidelity for each precision level."""
    allocator = PrecisionAllocator()

    # FP16 baseline
    print("  Measuring FP16 baseline PPL...")
    fp16_map = allocator._uniform_map(abkt_kv, Precision.FP16)
    ppl_fp16 = compute_ppl(model, windows, abkt_kv, fp16_map,
                            context_len, stride, device)
    print(f"  FP16 PPL: {ppl_fp16:.4f}")

    results = {"FP16": {"ppl": ppl_fp16, "fidelity": 1.0}}

    for prec_name, precision in [("INT8", Precision.INT8),
                                   ("INT4", Precision.INT4),
                                   ("INT2", Precision.INT2)]:
        print(f"  Measuring {prec_name} PPL...")
        prec_map = allocator._uniform_map(abkt_kv, precision)
        ppl = compute_ppl(model, windows, abkt_kv, prec_map,
                          context_len, stride, device)
        fidelity = 1.0 - (ppl - ppl_fp16) / ppl_fp16
        results[prec_name] = {"ppl": ppl, "fidelity": fidelity}
        print(f"  {prec_name} PPL: {ppl:.4f}, fidelity: {fidelity:.4f}")

    return results


def measure_per_layer_sensitivity(model, windows, abkt_kv, num_layers, seq_len,
                                   context_len=2048, stride=512, device="cuda"):
    """Measure PPL impact of quantizing each layer individually to INT4."""
    allocator = PrecisionAllocator()

    # FP16 baseline
    print("  Measuring FP16 baseline PPL...")
    fp16_map = allocator._uniform_map(abkt_kv, Precision.FP16)
    ppl_fp16 = compute_ppl(model, windows, abkt_kv, fp16_map,
                            context_len, stride, device)
    print(f"  FP16 PPL: {ppl_fp16:.4f}")

    sensitivity = {}
    for layer_idx in range(num_layers):
        # Start with all FP16, set one layer to INT4
        prec_map = {}
        for dnode, layer_cache in abkt_kv.items():
            prec_map[dnode] = {}
            for lidx in layer_cache:
                if lidx == layer_idx:
                    prec_map[dnode][lidx] = torch.full(
                        (4,), Precision.INT4.value, dtype=torch.int8)
                else:
                    prec_map[dnode][lidx] = torch.full(
                        (4,), Precision.FP16.value, dtype=torch.int8)

        ppl = compute_ppl(model, windows, abkt_kv, prec_map,
                          context_len, stride, device)
        delta = ppl - ppl_fp16
        sensitivity[layer_idx] = delta
        print(f"  Layer {layer_idx:2d}: PPL={ppl:.4f} (delta={delta:+.4f})")

    return {"fp16_ppl": ppl_fp16, "per_layer_delta": sensitivity}


def main():
    parser = argparse.ArgumentParser(description="ABKT Quality Fidelity Calibration")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--max-windows", type=int, default=10,
                        help="Number of windows (more = better calibration)")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--per-layer", action="store_true", default=False,
                        help="Also measure per-layer sensitivity (slow)")
    parser.add_argument("--output", default="calibration_results.json",
                        help="Output JSON")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).to(args.device).eval()
    num_layers = model.config.num_hidden_layers
    print(f"Model: {num_layers} layers")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = "\n\n".join(ex["text"] for ex in ds if ex["text"].strip())
    all_ids = tokenizer.encode(all_text)

    windows = []
    for i in range(0, len(all_ids) - args.context_len + 1, args.stride):
        windows.append(all_ids[i:i + args.context_len])
    if args.max_windows:
        windows = windows[:args.max_windows]
    print(f"Windows: {len(windows)}")

    # Extract KV cache from first window
    context_input = torch.tensor([windows[0][:args.stride]], device=args.device)
    with torch.no_grad():
        context_out = model(context_input, use_cache=True)
    orig_kv = context_out.past_key_values
    kv_cache_obj = KVCache.from_dynamic_cache(orig_kv)
    abkt_kv = kv_cache_obj.to_abkt_dict()
    seq_len = kv_cache_obj.seq_len
    del orig_kv, context_out
    torch.cuda.empty_cache()

    output = {}

    # Quality fidelity calibration
    print("\n=== Quality Fidelity Calibration ===")
    fidelity = calibrate_fidelity(
        model, windows, abkt_kv, num_layers, seq_len,
        args.context_len, args.stride, args.device)
    output["quality_fidelity"] = fidelity

    # Print recommended values
    print(f"\n=== Recommended QUALITY_FIDELITY ===")
    print(f"  FP16: 1.00")
    for name in ["INT8", "INT4", "INT2"]:
        f = fidelity[name]["fidelity"]
        print(f"  {name}: {f:.4f}")

    # Per-layer sensitivity
    if args.per_layer:
        print("\n=== Per-Layer Sensitivity ===")
        sensitivity = measure_per_layer_sensitivity(
            model, windows, abkt_kv, num_layers, seq_len,
            args.context_len, args.stride, args.device)
        output["per_layer_sensitivity"] = sensitivity

        # Find natural boundaries
        deltas = [sensitivity["per_layer_delta"][i] for i in range(num_layers)]
        sorted_layers = sorted(range(num_layers), key=lambda i: deltas[i])
        print(f"\n  Most sensitive layers (top 5): {sorted_layers[:5]}")
        print(f"  Least sensitive layers (top 5): {sorted_layers[-5:]}")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
