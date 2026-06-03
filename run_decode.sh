#!/bin/bash
# ── EdgePD: Start Decode Node ──
# Usage: ./run_decode.sh /path/to/model [port] [layer_split]
#
# Starts the decode node which listens for KV cache from the prefill node.

MODEL_PATH="${1:-/ssd/models/opt-2.7b-safetensors}"
PORT="${2:-29501}"
LAYER_SPLIT="${3:-}"

ARGS="--model-name $MODEL_PATH --port $PORT"
[ -n "$LAYER_SPLIT" ] && ARGS="$ARGS --layer-split $LAYER_SPLIT"

echo "Starting Decode Node on port $PORT..."
echo "Model: $MODEL_PATH"
python3 decode_node.py $ARGS
