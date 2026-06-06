#!/usr/bin/env python3
"""EdgePD — Decode Node

Runs on the decode machine (Jetson Orin @ 192.168.0.20).

Starts a TCP socket server, receives KV cache from the prefill node,
and runs the full autoregressive decode loop using the standard
HuggingFace ``model.forward()`` API.

Architecture (transformers 5.0.0):
    1. Load the full model (e.g. OPTForCausalLM, Qwen2ForCausalLM).
    2. Receive serialised KVCache from prefill.
    3. Build a transformers DynamicCache and pre-populate it with
       received KV tensors.
    4. Run autoregressive decode: for each step, call
       ``model.forward(input_ids, past_key_values=cache, use_cache=True)``.
       The model handles cache_position, attention masks, and position
       embeddings internally.

Usage:
    python decode_node.py --model-name /path/to/model [--port 29501]
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from pd_inference.config import PDConfig
from pd_inference.kv_cache import KVCache, dynamic_cache_fingerprint
from pd_inference.socket_transport import SocketServer, send_obj, recv_obj
from pd_inference.utils import decode_tokens, load_tokenizer, get_device
from backend.config import DEFAULT_PROBE_PORT
from backend.network_probe import ProbeServer
from backend.chunked_transfer import ChunkAssembler
from backend.adaptive_quant import AdaptiveQuantizer


# ── Sampling utilities ──

def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Sample one token from logits with temperature, top-k, top-p filtering.

    Args:
        logits: 1D float tensor of shape [vocab_size].
        temperature: Scale logits before softmax (>1 = more random, <1 = sharper).
            Values <= 0 fall back to greedy argmax.
        top_k: Keep only the k highest-probability tokens (0 = disabled).
        top_p: Nucleus sampling — keep smallest set whose cumulative prob >= p
            (1.0 = disabled).

    Returns:
        Sampled token id (int).
    """
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())

    scaled = logits / temperature

    # top-k: zero out everything except the k largest logits
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        threshold = scaled.topk(k, dim=-1).values.min()
        scaled = torch.where(scaled < threshold, torch.tensor(float('-inf'), device=scaled.device), scaled)

    # top-p (nucleus): sort descending, keep cumulative prob up to p
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Remove tokens after cumulative exceeds top_p
        cutoff = (cumulative > top_p).nonzero(as_tuple=True)
        if len(cutoff[0]) > 0:
            first_exceed = cutoff[0][0].item()
            sorted_probs[first_exceed + 1:] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

    # Normalize if any probs were zeroed
    if probs.sum() > 0:
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


def main():
    config = _parse_decode_config()
    print("=" * 60)
    print(" EdgePD Decode Node")
    print(f" Model: {config.model_name}")
    print(f" Port: {config.master_port}")
    print(f" Layer split: {config.layer_split or 'none (all layers)'}")
    print("=" * 60)

    device = get_device()
    if torch.cuda.is_available():
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[decode] CUDA: {torch.cuda.get_device_name(0)} "
              f"({mem_gb:.1f} GB)")
    else:
        print("[decode] WARNING: CUDA not available, using CPU")

    # ── Load model ──
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print(f"[decode] Model loaded in {time.time() - t0:.1f}s")

    # ── Start probe server (daemon, background thread) ──
    probe_server = ProbeServer(host="0.0.0.0", port=DEFAULT_PROBE_PORT)
    probe_server.start()
    print(f"[decode] Probe server started on port {DEFAULT_PROBE_PORT}")

    # ── Load tokenizer ──
    tokenizer = load_tokenizer(config.model_name)

    # ── RPC handler: full autoregressive decode using model.forward() ──
    _request_counter = [0]

    def _pick_token(logits_1d, *, do_sample, temperature, top_k, top_p):
        if do_sample:
            return sample_token(logits_1d, temperature=temperature,
                                top_k=top_k, top_p=top_p)
        else:
            return int(logits_1d.argmax(dim=-1).item())

    def handle_run_decode(
        kv_cache,
        input_ids,
        first_token=None,
        max_new_tokens=128,
        repetition_penalty=1.0,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        _client_socket=None,
    ):
        req_id = _request_counter[0]
        _request_counter[0] += 1

        print(f"\n[decode] === Request #{req_id} ===")

        # Step 1: Deserialise and build DynamicCache
        if isinstance(kv_cache, dict):
            kv = KVCache.from_transport_dict(kv_cache)
        else:
            # Backward compat: already a KVCache-like structure
            kv = _legacy_normalise(kv_cache, model)

        print(f"[decode] Received KVCache: {kv}")

        dcache = kv.to_dynamic_cache(device)

        # ── Diagnostic: verify DynamicCache reconstruction ──
        print(dynamic_cache_fingerprint(dcache))

        # Two-step protocol: if input_ids is None, send ack and wait for params
        if input_ids is None and _client_socket is not None:
            print(f"[decode] Two-step mode: sending KV ack...")
            send_obj(_client_socket, {"ok": True, "result": {"ack": "kv_received"}})
            print(f"[decode] Waiting for decode params...")
            params = recv_obj(_client_socket)
            if not isinstance(params, dict):
                return {"generated_text": "[ERROR] Invalid params", "generated_ids": [], "num_tokens": 0, "time": 0}
            input_ids = params.get("input_ids", [])
            first_token = params.get("first_token")
            max_new_tokens = params.get("max_new_tokens", 128)
            repetition_penalty = params.get("repetition_penalty", 1.0)
            do_sample = params.get("do_sample", False)
            temperature = params.get("temperature", 1.0)
            top_k = params.get("top_k", 0)
            top_p = params.get("top_p", 1.0)
            print(f"[decode] Params received: {max_new_tokens} tokens, "
                  f"{'sample' if do_sample else 'greedy'}")

        # Step 2: Autoregressive decode loop using model.forward()
        sample_cfg = ""
        if do_sample:
            sample_cfg = (f" sample(t={temperature:.1f}"
                          + (f", top_k={top_k}" if top_k > 0 else "")
                          + (f", top_p={top_p:.1f}" if top_p < 1.0 else "")
                          + ")")
        else:
            sample_cfg = " greedy"
        print(f"[decode] Decode strategy:{sample_cfg}")

        # The first token after prefill comes from the prefill node's logits
        # to avoid duplicating the last prompt token in the KV cache.
        if first_token is not None:
            generated = list(input_ids) + [first_token]
            tokens_to_generate = max_new_tokens - 1
            first_token_text = tokenizer.decode([first_token]) if tokenizer else str(first_token)
            print(f"[decode] Starting from prefill first token: {first_token} ({repr(first_token_text)})")
        else:
            generated = list(input_ids)
            tokens_to_generate = max_new_tokens
        t_start = time.time()

        # ── First-step sanity check ──
        # Run one forward pass on the decode side to verify the KV cache
        # produces valid output (non-degenerate logits).  This also
        # serves as the first real decode step, so we fold its results
        # into generated[] to avoid duplicating tokens in the main loop.
        verify_did_run = False
        if first_token is not None and tokens_to_generate >= 0:
            # The attention_mask must cover the FULL causal window
            # (past_len + current_len), not just the current token.
            past_len = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
            verify_attn_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            verify_token = torch.tensor([[generated[-1]]], device=device)
            with torch.no_grad():
                verify_out = model(
                    input_ids=verify_token,
                    attention_mask=verify_attn_mask,
                    past_key_values=dcache,
                    use_cache=True,
                )
            verify_logits = verify_out.logits[:, -1, :].float()
            top5_vals, top5_ids = torch.topk(verify_logits, k=min(5, verify_logits.shape[-1]))
            top5_probs = torch.softmax(verify_logits, dim=-1)[0, top5_ids[0]]
            entropy = float(-(top5_probs * top5_probs.log()).sum())

            print(f"[decode] First decode step sanity check: "
                  f"logits_min={verify_logits.min().item():.4f} "
                  f"logits_max={verify_logits.max().item():.4f} "
                  f"logits_mean={verify_logits.mean().item():.4f} "
                  f"entropy={entropy:.4f} "
                  f"top5_ids={top5_ids[0].tolist()}")

            if torch.isnan(verify_logits).any() or torch.isinf(verify_logits).any():
                print(f"[decode] CRITICAL: logits contain NaN or Inf! KV cache may be corrupted.")
            elif entropy < 1e-6:
                print(f"[decode] WARNING: very low entropy ({entropy:.6f}) — "
                      f"logits may be collapsed, indicating KV cache issue.")

            # This verify step already consumed one decode iteration:
            # dcache now contains prompt_kv + first_token_kv.
            # Append the predicted token so the main loop continues from there.
            verify_token_id = _pick_token(
                verify_logits[0],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated.append(verify_token_id)
            tokens_to_generate -= 1
            dcache = verify_out.past_key_values
            verify_did_run = True

        for step in range(max(tokens_to_generate, 0)):
            current_token = torch.tensor([[generated[-1]]], device=device)

            # Build attention mask for the current decode step.
            # The mask covers the full causal window [batch, past_len + current_len].
            # SDPA handles causal masking internally;
            # we just mark all tokens as valid (no padding).
            step_past_len = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
            step_attn_mask = torch.ones((1, step_past_len + 1), dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(
                    input_ids=current_token,
                    attention_mask=step_attn_mask,
                    past_key_values=dcache,
                    use_cache=True,
                )

            # The model mutates dcache in-place; outputs.past_key_values
            # is the same object (or a reference to it).
            dcache = outputs.past_key_values
            step_logits = outputs.logits[:, -1, :].clone().float()

            # Repetition penalty
            if repetition_penalty != 1.0 and step > 0:
                for tid in generated[len(input_ids):]:
                    if tid < step_logits.shape[-1]:
                        if step_logits[0, tid] > 0:
                            step_logits[0, tid] /= repetition_penalty
                        else:
                            step_logits[0, tid] *= repetition_penalty

            next_token = _pick_token(
                step_logits[0],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated.append(next_token)

            display_step = step + (0 if first_token is None else 1)
            total_display = tokens_to_generate + (0 if first_token is None else 1)
            if step % 10 == 0 or step == tokens_to_generate - 1:
                elapsed = time.time() - t_start
                tps = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"[decode] step={display_step + 1}/{total_display} "
                      f"token={next_token} tps={tps:.1f}")

        total_time = time.time() - t_start
        gen_tokens = len(generated) - len(input_ids)
        generated_text = decode_tokens(tokenizer, generated[len(input_ids):])
        print(f"[decode] Request #{req_id} complete: "
              f"{gen_tokens} new tokens in {total_time:.2f}s "
              f"({gen_tokens / total_time:.1f} tok/s)")

        return {
            "generated_ids": generated,
            "generated_text": generated_text,
            "num_tokens": gen_tokens,
            "time": total_time,
        }

    # ── ABKT streaming handler ──

    def handle_run_decode_abkt(
        request_id, input_ids, first_token=None, max_new_tokens=128,
        repetition_penalty=1.0, do_sample=False, temperature=1.0,
        top_k=0, top_p=1.0, num_layers=32, seq_len=0, q_metadata=None,
        _client_socket=None,
    ):
        """Init ABKT chunked transfer: returns stream context for SocketServer."""
        req_id = _request_counter[0]
        _request_counter[0] += 1
        print(f"\n[decode] === ABKT Request #{req_id} (id={request_id}) ===")

        quantizer = AdaptiveQuantizer()
        assembled_kv: dict = {}
        assembly_done = threading.Event()

        def on_complete(req_id_str: str, dequantized_kv: dict):
            assembled_kv[request_id] = dequantized_kv
            assembly_done.set()

        chunk_assembler = ChunkAssembler(
            quantizer=quantizer, on_complete=on_complete,
            num_layers=num_layers,
        )

        def run_decode_from_abkt(dequantized_kv: dict) -> dict:
            """Reconstruct DynamicCache from dequantized KV and run decode."""
            dcache = DynamicCache()
            layer_kv = dequantized_kv.get(0, {})
            n_layers = len(layer_kv)
            print(f"[decode] ABKT: Reconstructing DynamicCache from {n_layers} layers")

            # Diagnostic: check dequantized data
            first_lidx = min(layer_kv.keys()) if layer_kv else -1
            if first_lidx >= 0:
                fk, fv = layer_kv[first_lidx]
                print(f"[decode] ABKT: layer {first_lidx} k={list(fk.shape)} "
                      f"dtype={fk.dtype} "
                      f"mean={fk.float().mean():.4f} std={fk.float().std():.4f} "
                      f"min={fk.float().min():.4f} max={fk.float().max():.4f}")

            for lidx in sorted(layer_kv.keys()):
                kv = layer_kv[lidx]
                if kv is None:
                    continue
                k, v = kv
                dcache.update(k.to(device), v.to(device), lidx)

            # DynamicCache fingerprint
            seq = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
            print(f"[decode] ABKT: DynamicCache layers={len(dcache.layers)} "
                  f"seq_len={seq}")
            print(dynamic_cache_fingerprint(dcache))

            generated = list(input_ids)
            if first_token is not None:
                generated.append(first_token)
                tokens_left = max_new_tokens - 1
            else:
                tokens_left = max_new_tokens

            t_start = time.time()

            # First-step sanity check
            if first_token is not None and tokens_left >= 0:
                past_len = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
                attn = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
                tok = torch.tensor([[generated[-1]]], device=device)
                with torch.no_grad():
                    out = model(tok, attention_mask=attn,
                                past_key_values=dcache, use_cache=True)
                verify_logits = out.logits[:, -1, :].float()
                top5_vals, top5_ids = torch.topk(verify_logits, k=min(5, verify_logits.shape[-1]))
                top5_probs = torch.softmax(verify_logits, dim=-1)[0, top5_ids[0]]
                entropy = float(-(top5_probs * top5_probs.log()).sum())
                top5_text = [tokenizer.decode([tid]) if tokenizer else str(tid)
                             for tid in top5_ids[0].tolist()]
                print(f"[decode] ABKT sanity: logits_min={verify_logits.min().item():.4f} "
                      f"logits_max={verify_logits.max().item():.4f} "
                      f"mean={verify_logits.mean().item():.4f} "
                      f"entropy={entropy:.4f} "
                      f"top5={list(zip(top5_ids[0].tolist(), top5_text))}")
                if torch.isnan(verify_logits).any() or torch.isinf(verify_logits).any():
                    print(f"[decode] ABKT CRITICAL: NaN/Inf in logits!")
                elif entropy < 1e-6:
                    print(f"[decode] ABKT WARNING: near-zero entropy — logits collapsed!")

                tid = _pick_token(verify_logits[0],
                                  do_sample=do_sample, temperature=temperature,
                                  top_k=top_k, top_p=top_p)
                generated.append(tid)
                tokens_left -= 1
                dcache = out.past_key_values

            for step in range(max(tokens_left, 0)):
                tok = torch.tensor([[generated[-1]]], device=device)
                past_len = dcache.get_seq_length(0) if len(dcache.layers) > 0 else 0
                attn = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(tok, attention_mask=attn,
                                past_key_values=dcache, use_cache=True)
                dcache = out.past_key_values
                logits = out.logits[:, -1, :].clone().float()
                if repetition_penalty != 1.0:
                    for tid_rep in generated[len(input_ids):]:
                        if tid_rep < logits.shape[-1]:
                            logits[0, tid_rep] /= repetition_penalty if logits[0, tid_rep] > 0 else logits[0, tid_rep] * repetition_penalty
                next_id = _pick_token(logits[0], do_sample=do_sample,
                                      temperature=temperature, top_k=top_k, top_p=top_p)
                generated.append(next_id)

            total_time = time.time() - t_start
            gen_tokens = len(generated) - len(input_ids)
            text = decode_tokens(tokenizer, generated[len(input_ids):])
            print(f"[decode] ABKT Request #{req_id} complete: "
                  f"{gen_tokens} tokens in {total_time:.2f}s "
                  f"({gen_tokens / total_time:.1f} tok/s)")
            return {
                "generated_ids": generated, "generated_text": text,
                "num_tokens": gen_tokens, "time": total_time,
            }

        return {
            "_abkt_stream": True,
            "_assembler": chunk_assembler,
            "_assembly_done": assembly_done,
            "_assembled_kv": assembled_kv,
            "_request_id": request_id,
            "_decode_fn": run_decode_from_abkt,
        }

    # ── Start server ──
    handlers = {
        "run_decode": handle_run_decode,
        "run_decode_abkt": handle_run_decode_abkt,
    }
    server = SocketServer("0.0.0.0", config.master_port, handlers)
    print(f"[decode] RPC server ready, waiting for connections "
          f"on port {config.master_port}...")
    server.serve_forever()
    print("[decode] Done.")


def _legacy_normalise(kv_raw, model):
    """Normalise legacy dict-format KV cache from older prefill code."""
    layer_list = []
    if isinstance(kv_raw, dict):
        for node_id, layer_dict in kv_raw.items():
            if isinstance(layer_dict, dict):
                for idx, kv_pair in sorted(layer_dict.items()):
                    if kv_pair is not None and len(kv_pair) >= 2:
                        k, v = kv_pair
                        layer_list.append(
                            dict(layer_idx=idx, k=k.cpu(), v=v.cpu())
                        )
    return KVCache.from_transport_dict({"layers": layer_list})


def _parse_decode_config() -> PDConfig:
    """Parse minimal config for decode node."""
    import argparse

    parser = argparse.ArgumentParser(description="EdgePD Decode Node")
    parser.add_argument("--model-name", required=True,
                        help="Model path or HuggingFace name")
    parser.add_argument("--port", type=int, default=29501,
                        help="TCP port to listen on (default: 29501)")
    parser.add_argument("--layer-split", default=None,
                        help="Layer split boundary (e.g. '12')")

    args = parser.parse_args()

    layer_split = None
    if args.layer_split and args.layer_split.lower() not in ("", "none", "all"):
        try:
            boundary = int(args.layer_split)
            layer_split = (boundary, boundary)
        except ValueError:
            print(f"[config] Invalid --layer-split '{args.layer_split}', ignoring")

    return PDConfig(
        master_port=args.port,
        model_name=args.model_name,
        layer_split=layer_split,
    )


if __name__ == "__main__":
    main()
