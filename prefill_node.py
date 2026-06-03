#!/usr/bin/env python3
"""EdgePD — Prefill Node

Runs on the prefill machine (RTX 5060 Ti @ 192.168.0.50).

Processes the prompt through the model, extracts KV cache via the
KVCache transport container, and sends it to the decode node.

Usage:
    # 单次推理
    python prefill_node.py --model-name /path/to/model --prompt "Hello" \\
        --decode-host 192.168.0.20 --decode-port 29501

    # 交互式多轮对话
    python prefill_node.py --model-name /path/to/model --interactive \\
        --decode-host 192.168.0.20

    # 带系统提示的多轮对话
    python prefill_node.py --model-name /path/to/model --interactive \\
        --system-prompt "你是一个有帮助的AI助手" --max-new-tokens 512
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from pd_inference.config import PDConfig
from pd_inference.kv_cache import KVCache, kv_cache_fingerprint
from pd_inference.model import PrefillStage
from pd_inference.socket_transport import SocketClient
from pd_inference.utils import (
    encode_prompt,
    load_tokenizer,
    pad_batch,
)
from backend.network_probe import NetworkProbeClient
from backend.token_importance import TokenImportanceEvaluator
from backend.precision_allocator import PrecisionAllocator, Precision
from backend.adaptive_quant import AdaptiveQuantizer
from backend.chunked_transfer import ChunkedSender


# ════════════════════════════════════════════════════════════════════
# 核心: 单轮推理
# ════════════════════════════════════════════════════════════════════


def run_prefill_turn(
    input_ids: list,
    prefill_stage: PrefillStage,
    tokenizer,
    config: PDConfig,
    probe: NetworkProbeClient,
    num_layers: int,
) -> dict:
    """运行单轮 prefill + KV 传输 + decode。

    Args:
        input_ids: 完整对话历史的 token IDs (包含所有之前的轮次)
        prefill_stage: 已加载的模型
        tokenizer: tokenizer
        config: 配置
        probe: 网络探测客户端
        num_layers: 模型层数

    Returns:
        dict with keys: generated_text, generated_ids, num_tokens, time
    """
    # ── Run prefill forward pass ──
    padded, mask = pad_batch([input_ids], pad_id=(
        tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None
        else 0
    ))
    input_ids_t = torch.tensor(padded, device=prefill_stage.device)
    attn_mask = None
    if mask is not None:
        attn_mask = torch.tensor(mask, device=prefill_stage.device)

    print(f"[prefill] Running prefill ({len(input_ids)} tokens)...")
    t_prefill = time.time()
    with torch.no_grad():
        outputs = prefill_stage.model(
            input_ids_t,
            attention_mask=attn_mask,
            use_cache=True,
        )

    prefill_time = time.time() - t_prefill
    print(f"[prefill] Prefill complete in {prefill_time:.3f}s")

    # ── Get first token from prefill logits ──
    first_logits = outputs.logits[:, -1, :].float()
    first_token = int(first_logits.argmax(dim=-1).item())
    first_text = tokenizer.decode([first_token]) if tokenizer else str(first_token)
    print(f"[prefill] First predicted token: {first_token} ({repr(first_text)})")

    # ── Extract KV cache via KVCache transport container ──
    past = outputs.past_key_values
    kv_cache = KVCache.from_dynamic_cache(past)
    print(f"[prefill] Extracted KV cache: {kv_cache}")
    for ly in kv_cache.layers[:1]:
        print(f"[prefill]   layer {ly.layer_idx}: k={list(ly.k.shape)}, "
              f"v={list(ly.v.shape)}")
    print(kv_cache_fingerprint(kv_cache))

    # ════════════════════════════════════════════════════════════════
    # ABKT: Adaptive Bitrate KV Cache Transfer
    # ════════════════════════════════════════════════════════════════

    if not probe.is_calibrated():
        print("[prefill] Cold start: forcing initial BW probe...")
        probe.probe_now(data_size=4 * 1024 * 1024)

    # Step 1: Convert to ABKT format
    abkt_kv = kv_cache.to_abkt_dict()

    # Step 2: Evaluate token importance
    evaluator = TokenImportanceEvaluator()
    importance_map = evaluator.compute(abkt_kv, num_layers, kv_cache.seq_len)

    # Step 3: Get network snapshot and compute dynamic budget
    total_fp16 = PrecisionAllocator._total_bytes(abkt_kv, Precision.FP16)
    snapshot = probe.get_snapshot(total_bytes=total_fp16)
    budget = snapshot.budget_bytes

    print(f"[prefill] ABKT: state={snapshot.state.value} "
          f"bw={snapshot.bandwidth_ewma/1e6:.1f} MB/s "
          f"budget={budget/1e6:.1f} MB "
          f"fp16_total={total_fp16/1e6:.1f} MB")

    # Step 4: Allocate precision under budget
    allocator = PrecisionAllocator()
    allocation = allocator.allocate(importance_map, abkt_kv, budget,
                                    num_layers_total=num_layers)

    # Handle infeasible budget
    original_num_layers = num_layers
    if not allocation.feasible and allocation.dropped_layers:
        MIN_LAYERS_FRACTION = 0.5
        max_droppable = max(1, int(num_layers * (1 - MIN_LAYERS_FRACTION)))
        dropped = allocation.dropped_layers[:max_droppable]
        print(f"[prefill] ABKT: Budget infeasible — dropping {len(dropped)}/{len(allocation.dropped_layers)} "
              f"suggested layers (max_droppable={max_droppable}): {dropped}")
        for dnode in list(abkt_kv.keys()):
            for lidx in dropped:
                abkt_kv[dnode].pop(lidx, None)
                importance_map.get(dnode, {}).pop(lidx, None)
        num_layers -= len(dropped)
        allocation = allocator.allocate(importance_map, abkt_kv, budget,
                                        num_layers_total=original_num_layers)
        print(f"[prefill] ABKT: After dropping — avg_bits={allocation.avg_precision_bits:.1f} "
              f"compression={allocation.compression_ratio:.1f}x "
              f"bytes={allocation.total_bytes/1e6:.1f} MB")

    print(f"[prefill] ABKT: avg_bits={allocation.avg_precision_bits:.1f} "
          f"compression={allocation.compression_ratio:.1f}x "
          f"bytes={allocation.total_bytes/1e6:.1f} MB")

    # Step 5: Quantize
    quantizer = AdaptiveQuantizer()
    quantized_kv, q_metadata = quantizer.quantize(abkt_kv, allocation.precision_map)

    # ════════════════════════════════════════════════════════════════
    # Transfer to decode node
    # ════════════════════════════════════════════════════════════════

    client = SocketClient(config.master_addr, config.master_port, timeout=600.0)
    t_send = time.time()
    result = None
    try:
        client.connect()

        if allocation.compression_ratio == 1.0:
            print("[prefill] Budget ample, using legacy FP16 transfer...")
            result = client.call(
                "run_decode",
                kv_cache=kv_cache.to_transport_dict(),
                input_ids=input_ids_t[0].tolist(),
                first_token=first_token,
                max_new_tokens=config.max_new_tokens,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )
            send_time = time.time() - t_send
        else:
            request_id = f"req_{int(time.time() * 1000)}"
            print(f"[prefill] ABKT: Starting chunked transfer (id={request_id})")

            t_init = time.time()
            client.send_raw({
                "op": "run_decode_abkt",
                "payload": {
                    "request_id": request_id,
                    "input_ids": input_ids_t[0].tolist(),
                    "first_token": first_token,
                    "max_new_tokens": config.max_new_tokens,
                    "repetition_penalty": config.repetition_penalty,
                    "do_sample": config.do_sample,
                    "temperature": config.temperature,
                    "top_k": config.top_k,
                    "top_p": config.top_p,
                    "num_layers": num_layers,
                    "seq_len": kv_cache.seq_len,
                    "q_metadata": q_metadata,
                },
            })
            print(f"[prefill] ABKT: Init sent in {time.time() - t_init:.3f}s")

            probe.pause_bw_probes()
            t_chunks = time.time()
            sender = ChunkedSender(
                send_fn=lambda payload: client.send_raw({
                    "op": "kv_chunk",
                    "payload": payload,
                }),
                quantizer=quantizer,
                original_kv=abkt_kv,
                precision_map=allocation.precision_map,
            )
            sender.send_all(quantized_kv, q_metadata, importance_map,
                            snapshot.bandwidth_bps, request_id)
            print(f"[prefill] ABKT: All chunks sent in {time.time() - t_chunks:.3f}s")
            probe.resume_bw_probes()

            print(f"[prefill] ABKT: Sending decode_start...")
            client.send_raw({"op": "decode_start", "payload": {"request_id": request_id}})
            print(f"[prefill] ABKT: Waiting for decode result...")
            result = client.recv_obj()
            send_time = time.time() - t_send

        # ── Calibration ──
        probe.record_transfer(
            compressed_bytes=allocation.total_bytes,
            elapsed_sec=send_time,
            compression_ratio=allocation.compression_ratio,
        )

        print(f"[prefill] Transfer complete in {send_time:.2f}s")

    except Exception as e:
        print(f"[ERROR] Prefill failed: {e}")
        import traceback
        traceback.print_exc()
        return {"generated_text": f"[ERROR] {e}", "generated_ids": [], "num_tokens": 0, "time": 0}
    finally:
        client.close()

    # ── Parse result ──
    if isinstance(result, dict):
        return result
    return {"generated_text": str(result), "generated_ids": [], "num_tokens": 0, "time": 0}


# ════════════════════════════════════════════════════════════════════
# 单次推理模式 (原始行为)
# ════════════════════════════════════════════════════════════════════


def run_single_shot(config, prefill_stage, tokenizer, probe, num_layers):
    """单次推理: 处理一个 prompt 并退出。"""
    prompt = config.prompt or "The capital of France is"
    input_ids = encode_prompt(prompt, tokenizer)
    print(f"[prefill] Prompt: '{prompt}' ({len(input_ids)} tokens)")

    result = run_prefill_turn(
        input_ids=input_ids,
        prefill_stage=prefill_stage,
        tokenizer=tokenizer,
        config=config,
        probe=probe,
        num_layers=num_layers,
    )

    # ── Display result ──
    generated_text = result.get("generated_text", str(result))
    num_tokens = result.get("num_tokens", 0)
    total_time = result.get("time", 0)

    print(f"\n{'=' * 60}")
    print(f" [RESULT] Generated text:")
    print(f"{'=' * 60}")
    print(generated_text)
    print(f"{'=' * 60}")
    if num_tokens:
        print(f" Generated {num_tokens} tokens in {total_time:.1f}s "
              f"({num_tokens / total_time:.1f} tok/s)")

    return result


# ════════════════════════════════════════════════════════════════════
# 交互式多轮对话模式
# ════════════════════════════════════════════════════════════════════


def run_interactive(config, prefill_stage, tokenizer, probe, num_layers):
    """交互式多轮对话: 持续接受用户输入，维护对话历史。"""
    print(f"\n{'=' * 60}")
    print(f" EdgePD 交互式多轮对话")
    print(f" 输入消息后按回车发送")
    print(f" 输入 /quit 退出, /clear 清除历史, /info 查看状态")
    print(f"{'=' * 60}\n")

    # 构建初始 token IDs (系统提示 + 可选的初始 prompt)
    conversation_ids = []

    system_prompt = config.system_prompt
    if system_prompt:
        # 用 tokenizer 的 chat template 如果可用
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "system", "content": system_prompt}]
                conversation_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                print(f"[prefill] System prompt loaded ({len(conversation_ids)} tokens)")
            except Exception:
                # Fallback: 直接 tokenize
                conversation_ids = encode_prompt(system_prompt, tokenizer)
                print(f"[prefill] System prompt loaded ({len(conversation_ids)} tokens, raw)")
        else:
            conversation_ids = encode_prompt(system_prompt, tokenizer)

    # 如果有初始 prompt，作为第一轮用户输入
    initial_prompt = config.prompt
    if initial_prompt:
        print(f"[prefill] Initial prompt: '{initial_prompt}'")

    turn_count = 0
    total_gen_tokens = 0
    total_prefill_tokens = 0

    while True:
        try:
            # 获取用户输入
            if turn_count == 0 and initial_prompt:
                user_input = initial_prompt
                print(f"User: {user_input}")
            else:
                user_input = input("User: ").strip()

            if not user_input:
                continue

            # 命令处理
            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd in ("/quit", "/exit", "/q"):
                    print("[prefill] 退出对话")
                    break
                elif cmd == "/clear":
                    conversation_ids = []
                    if system_prompt:
                        if hasattr(tokenizer, 'apply_chat_template'):
                            try:
                                messages = [{"role": "system", "content": system_prompt}]
                                conversation_ids = tokenizer.apply_chat_template(
                                    messages, tokenize=True, add_generation_prompt=False
                                )
                            except Exception:
                                conversation_ids = encode_prompt(system_prompt, tokenizer)
                        else:
                            conversation_ids = encode_prompt(system_prompt, tokenizer)
                    turn_count = 0
                    total_gen_tokens = 0
                    total_prefill_tokens = 0
                    print("[prefill] 对话历史已清除")
                    continue
                elif cmd == "/info":
                    print(f"[info] 轮次: {turn_count}, "
                          f"对话 tokens: {len(conversation_ids)}, "
                          f"累计生成: {total_gen_tokens}, "
                          f"模型: {config.model_name}")
                    continue
                else:
                    print(f"[prefill] 未知命令: {user_input}")
                    continue

            # 构建本轮的完整输入
            if hasattr(tokenizer, 'apply_chat_template') and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                # 使用 chat template
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                # 重建对话历史 (简化: 只用原始文本)
                messages.append({"role": "user", "content": user_input})
                try:
                    turn_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                    # 第一轮: 直接使用
                    # 后续轮: 需要累积 (但 chat template 会包含完整格式)
                    if turn_count == 0:
                        full_ids = turn_ids
                    else:
                        # 追加本轮用户输入 (去掉之前的 system prompt 部分)
                        # 简单方案: 每轮重新构建
                        full_ids = turn_ids  # TODO: 更好的多轮累积
                except Exception:
                    new_ids = encode_prompt(user_input, tokenizer)
                    full_ids = conversation_ids + new_ids
            else:
                # 无 chat template: 简单拼接
                new_ids = encode_prompt(user_input, tokenizer)
                full_ids = conversation_ids + new_ids

            total_prefill_tokens += len(full_ids)
            turn_count += 1

            # 运行 prefill + decode
            result = run_prefill_turn(
                input_ids=full_ids,
                prefill_stage=prefill_stage,
                tokenizer=tokenizer,
                config=config,
                probe=probe,
                num_layers=num_layers,
            )

            # 显示结果
            generated_text = result.get("generated_text", "")
            num_tokens = result.get("num_tokens", 0)
            total_time = result.get("time", 0)
            generated_ids = result.get("generated_ids", [])

            # 从 generated_text 中提取助手回复 (去掉原始 prompt 部分)
            if tokenizer and generated_ids:
                # generated_ids 包含 input_ids + new tokens
                new_token_ids = generated_ids[len(full_ids):]
                assistant_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            else:
                assistant_text = generated_text

            print(f"\nAssistant: {assistant_text}")
            if num_tokens and total_time > 0:
                print(f"  [{num_tokens} tokens, {total_time:.1f}s, "
                      f"{num_tokens/total_time:.1f} tok/s]")
            print()

            # 更新对话历史
            if generated_ids:
                conversation_ids = generated_ids
            else:
                # Fallback: 用 encode_prompt 重新编码
                conversation_ids = full_ids
                if tokenizer and generated_text:
                    # 追加生成的文本
                    gen_ids = encode_prompt(assistant_text, tokenizer)
                    conversation_ids = conversation_ids + gen_ids

            total_gen_tokens += num_tokens

        except KeyboardInterrupt:
            print("\n[prefill] 中断 (Ctrl+C 再次按退出)")
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                print("\n[prefill] 退出")
                break
        except EOFError:
            print("\n[prefill] 退出")
            break

    # 汇总
    print(f"\n{'=' * 60}")
    print(f" 会话结束: {turn_count} 轮, 累计生成 {total_gen_tokens} tokens")
    print(f"{'=' * 60}")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main():
    config = _parse_prefill_config()
    mode = "interactive" if config.interactive else "single"

    print("=" * 60)
    print(" EdgePD Prefill Node")
    print(f" Model: {config.model_name}")
    print(f" Decode node: {config.master_addr}:{config.master_port}")
    print(f" Mode: {mode}")
    if config.system_prompt:
        print(f" System prompt: '{config.system_prompt[:50]}...'")
    if not config.interactive:
        print(f" Prompt: '{config.prompt}'")
    print(f" Max tokens: {config.max_new_tokens}")
    sample_info = "sample" if config.do_sample else "greedy"
    if config.do_sample:
        sample_info += (f" (t={config.temperature:.1f}"
                        + (f", top_k={config.top_k}" if config.top_k > 0 else "")
                        + (f", top_p={config.top_p:.1f}" if config.top_p < 1.0 else "")
                        + ")")
    print(f" Decode: {sample_info}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] Prefill node requires CUDA GPU")
        sys.exit(1)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[prefill] CUDA: {torch.cuda.get_device_name(0)} "
          f"({mem_gb:.1f} GB)")

    # ── Start network probe ──
    probe = NetworkProbeClient(
        target_host=config.master_addr, target_port=9877,
    )
    probe.start()
    print(f"[prefill] Network probe started (host={config.master_addr}:9877)")

    # ── Load model ──
    num_layers = _get_num_layers(config.model_name)
    layer_range = config.prefill_layer_range
    if layer_range == (0, 0):
        layer_range = (0, num_layers)

    print(f"[prefill] Loading model layers {layer_range[0]}-{layer_range[1] - 1}...")
    t0 = time.time()
    prefill_stage = PrefillStage(
        stage_id=0,
        layer_range=layer_range,
        model_name=config.model_name,
        load_embed=(layer_range[0] == 0),
        load_lm_head=False,
        num_layers_total=num_layers,
    )
    prefill_stage.load()
    print(f"[prefill] Model loaded in {time.time() - t0:.1f}s")

    # ── Load tokenizer ──
    tokenizer = load_tokenizer(config.model_name)

    try:
        if config.interactive:
            run_interactive(config, prefill_stage, tokenizer, probe, num_layers)
        else:
            run_single_shot(config, prefill_stage, tokenizer, probe, num_layers)
    finally:
        probe.stop()

    print("[prefill] Done.")


def _parse_prefill_config() -> PDConfig:
    """Parse config for prefill node."""
    import argparse

    parser = argparse.ArgumentParser(description="EdgePD Prefill Node")
    parser.add_argument("--model-name", required=True,
                        help="Model path or HuggingFace name")
    parser.add_argument("--decode-host", default="192.168.0.20",
                        help="Decode node IP (default: 192.168.0.20)")
    parser.add_argument("--decode-port", type=int, default=29501,
                        help="Decode node port (default: 29501)")
    parser.add_argument("--prompt", default=None,
                        help="Input prompt (single-shot mode)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = disabled)")
    parser.add_argument("--do-sample", action="store_true", default=False,
                        help="Use sampling instead of greedy argmax")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature (higher=more random)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (0=disabled)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling threshold (1.0=disabled)")
    parser.add_argument("--layer-split", default=None,
                        help="Layer split boundary (e.g. '12')")

    # 多轮对话参数
    parser.add_argument("--interactive", "-i", action="store_true", default=False,
                        help="Interactive multi-turn chat mode")
    parser.add_argument("--system-prompt", default=None,
                        help="System prompt for interactive mode")

    args = parser.parse_args()

    layer_split = None
    if args.layer_split and args.layer_split.lower() not in ("", "none", "all"):
        try:
            boundary = int(args.layer_split)
            layer_split = (boundary, boundary)
        except ValueError:
            print(f"[config] Invalid --layer-split '{args.layer_split}', ignoring")

    # 单次模式需要 prompt
    if not args.interactive and args.prompt is None:
        args.prompt = "The capital of France is"

    return PDConfig(
        master_addr=args.decode_host,
        master_port=args.decode_port,
        model_name=args.model_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        layer_split=layer_split,
        interactive=args.interactive,
        system_prompt=args.system_prompt,
    )


def _get_num_layers(model_name: str) -> int:
    """Get the total number of transformer layers from model config."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        return getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 32))
    except Exception:
        return 32


if __name__ == "__main__":
    main()
