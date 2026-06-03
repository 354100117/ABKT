"""Configuration for PD-separated inference.

Usage:
    from pd_inference.config import PDConfig, parse_args

    config = parse_args()
    # config.master_addr  — prefill node IP
    # config.master_port  — TCP port for distributed init
    # config.model_name   — model path
    # config.layer_split  — (prefill_end, decode_start) or None
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PDConfig:
    """Configuration for PD-separated inference."""

    # ── Distributed ──
    master_addr: str = "192.168.0.50"       # Prefill node IP (rank 0)
    master_port: int = 29500                # TCP port
    rank: int = 0                           # 0=prefill, 1=decode
    world_size: int = 2                     # total nodes

    # ── Model ──
    model_name: str = ""                    # model path or HuggingFace name
    dtype: str = "float16"                  # float16 / float32
    max_context_tokens: int = 2048          # max total context length

    # ── Layer split ──
    # If set, each node loads only its assigned layers.
    # (prefill_layer_end, decode_layer_start)
    # e.g. (12, 12) means prefill has layers 0-11, decode has layers 12-23
    # None means both nodes load all layers (simple pipeline parallelism).
    layer_split: Optional[Tuple[int, int]] = None

    # ── Inference ──
    max_new_tokens: int = 128               # default tokens to generate
    repetition_penalty: float = 1.0         # 1.0 = disabled
    do_sample: bool = False                 # whether to use sampling (vs greedy argmax)
    temperature: float = 1.0                # softmax temperature (higher = more random)
    top_k: int = 0                          # top-k filtering (0 = disabled)
    top_p: float = 1.0                      # nucleus sampling (1.0 = disabled)

    # ── Prompt (prefill node only) ──
    prompt: str = ""                        # input prompt for prefill node

    # ── Interactive mode (prefill node only) ──
    interactive: bool = False               # multi-turn chat mode
    system_prompt: str = ""                 # system prompt for interactive mode

    def __post_init__(self):
        assert self.world_size == 2, "PDConfig currently requires exactly 2 nodes"

    @property
    def is_prefill(self) -> bool:
        return self.rank == 0

    @property
    def is_decode(self) -> bool:
        return self.rank == 1

    @property
    def prefill_layer_range(self) -> Tuple[int, int]:
        """Layer range for prefill node."""
        if self.layer_split is not None:
            return (0, self.layer_split[0])
        return (0, 0)  # 0,0 means "all layers"

    @property
    def decode_layer_range(self) -> Tuple[int, int]:
        """Layer range for decode node."""
        if self.layer_split is not None:
            return (self.layer_split[1], 0)  # second element = num_layers filled later
        return (0, 0)  # all layers


def parse_args(argv: Optional[List[str]] = None) -> PDConfig:
    """Parse command-line arguments into PDConfig.

    Compatible with torch.distributed env vars: MASTER_ADDR, MASTER_PORT,
    RANK, WORLD_SIZE are used as defaults.
    """
    parser = argparse.ArgumentParser(
        description="EdgePD: PD-separated LLM Inference"
    )

    # Distributed — defaults read from standard env vars
    parser.add_argument("--master-addr",
                        default=os.environ.get("MASTER_ADDR", "192.168.0.50"),
                        help="Prefill node IP (rank 0) [env: MASTER_ADDR]")
    parser.add_argument("--master-port", type=int,
                        default=int(os.environ.get("MASTER_PORT", "29500")),
                        help="TCP port for distributed init [env: MASTER_PORT]")
    parser.add_argument("--rank", type=int,
                        default=int(os.environ["RANK"]) if "RANK" in os.environ else None,
                        required="RANK" not in os.environ,
                        help="Node rank (0=prefill, 1=decode) [env: RANK]")
    parser.add_argument("--world-size", type=int,
                        default=int(os.environ.get("WORLD_SIZE", "2")),
                        help="Total number of nodes [env: WORLD_SIZE]")

    # Model
    parser.add_argument("--model-name", required=True,
                        help="Model path or HuggingFace name")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "float32"],
                        help="Model dtype")
    parser.add_argument("--max-context-tokens", type=int, default=2048,
                        help="Max total context length")

    # Layer split
    parser.add_argument("--layer-split", default=None,
                        help="Layer split boundary, e.g. '12' means prefill=0-11, decode=12-23. "
                             "Omit or 'all' for both loading all layers.")

    # Inference
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true", default=False,
                        help="Use sampling instead of greedy argmax")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature (higher=more random)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Top-k sampling (0=disabled)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling threshold (1.0=disabled)")

    # Prompt
    parser.add_argument("--prompt", default="",
                        help="Input prompt (prefill node only)")

    args = parser.parse_args(argv)

    # Parse layer split
    layer_split = None
    if args.layer_split and args.layer_split.lower() not in ("", "none", "all"):
        try:
            boundary = int(args.layer_split)
            layer_split = (boundary, boundary)
        except ValueError:
            print(f"[config] Invalid --layer-split '{args.layer_split}', ignoring")
    else:
        layer_split = None  # both nodes load all layers

    return PDConfig(
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        world_size=args.world_size,
        model_name=args.model_name,
        dtype=args.dtype,
        max_context_tokens=args.max_context_tokens,
        layer_split=layer_split,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        prompt=args.prompt,
    )
