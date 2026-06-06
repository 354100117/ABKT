"""EdgePD Inference: Distributed PD-separated LLM inference.

Package structure:
    config.py          — Configuration dataclass + CLI parser
    kv_cache.py        — KVCache serialization (DynamicCache ↔ transport dict)
    socket_transport.py — TCP socket server/client with ABKT streaming
    model.py           — PrefillStage and DecodeStage (core inference stages)
    utils.py           — Common utilities (tokenizer, encoding, device)
"""
