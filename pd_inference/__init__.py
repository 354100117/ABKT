"""EdgePD Inference: Distributed PD-separated LLM inference.

Package structure:
    config.py    — Configuration dataclass + CLI parser
    rpc.py       — RPC communication (based on torch.distributed)
    utils.py     — Common utilities (tokenizer, KV cache, model loading)
    model.py     — PrefillStage and DecodeStage (core inference stages)
    pipeline.py  — Pipeline orchestration (prefill + decode execution)
"""
