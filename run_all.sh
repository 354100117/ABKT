#!/bin/bash
# ── EdgePD: Run full PD-separated inference ──
# Usage: ./run_all.sh [model_path] ["prompt"] [max_tokens] [--sample] [--temperature T] [--top-k K] [--top-p P] [layer_split]
#
# Orchestrates both prefill and decode nodes:
# 1. Syncs code to prefill node (192.168.0.50)
# 2. Starts decode node on this machine (192.168.0.20)
# 3. Starts prefill node on 192.168.0.50 via SSH
# 4. Displays results
#
# Examples:
#   ./run_all.sh /path/to/model "Hello" 64 --sample --temperature 0.8
#   ./run_all.sh /path/to/model "Hello" 64 12              # layer split at layer 12

set -e

MODEL_PATH="${1:-/ssd/models/opt-2.7b-safetensors}"
PROMPT="${2:-The capital of France is}"
MAX_TOKENS="${3:-64}"

# ── Parse remaining arguments (flags + optional layer_split) ──
DO_SAMPLE=""
TEMPERATURE="1.0"
TOP_K="0"
TOP_P="1.0"
LAYER_SPLIT=""
shift 3 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample) DO_SAMPLE="--do-sample" ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --top-k) TOP_K="$2"; shift ;;
        --top-p) TOP_P="$2"; shift ;;
        --*) echo "[orchestrator] WARNING: unknown flag '$1', ignoring" ;;
        *)  LAYER_SPLIT="$1" ;;  # positional: treat as layer_split number
    esac
    shift
done

PREFILL_HOST="192.168.0.50"
DECODE_HOST="192.168.0.20"
DECODE_PORT="29501"
PROJECT_DIR="/ssd/pd/ABKT"

echo "============================================================"
echo " EdgePD: PD-Separated Inference"
echo "============================================================"
echo " Model:     $MODEL_PATH"
echo " Prompt:    \"$PROMPT\""
echo " Max tokens: $MAX_TOKENS"
echo " Layer split: ${LAYER_SPLIT:-none (all layers)}"
[ -n "$DO_SAMPLE" ] && echo " Decode:    sample (t=$TEMPERATURE, top_k=$TOP_K, top_p=$TOP_P)" || echo " Decode:    greedy"
echo " Prefill:   $PREFILL_HOST (RTX 5060 Ti)"
echo " Decode:    $DECODE_HOST (Jetson Orin)"
echo "============================================================"
echo ""

# ── Step 1: Sync code to prefill node ──
echo "[orchestrator] Syncing code to prefill node ($PREFILL_HOST)..."
ssh $PREFILL_HOST "mkdir -p $PROJECT_DIR/pd_inference" 2>/dev/null || true
scp -q $PROJECT_DIR/prefill_node.py $PROJECT_DIR/run_prefill.sh \
       $PROJECT_DIR/config.yaml $PROJECT_DIR/requirements.txt \
       $PREFILL_HOST:$PROJECT_DIR/
scp -q $PROJECT_DIR/pd_inference/*.py $PREFILL_HOST:$PROJECT_DIR/pd_inference/
echo "[orchestrator] Code synced OK"

# ── Step 2: Start decode node (local, background) ──
echo ""
echo "[orchestrator] Starting decode node on $DECODE_HOST:$DECODE_PORT..."
DECODE_LOG="/tmp/decode_node_$$.log"
cd $PROJECT_DIR

DECODE_ARGS="--model-name $MODEL_PATH --port $DECODE_PORT"
[ -n "$LAYER_SPLIT" ] && DECODE_ARGS="$DECODE_ARGS --layer-split $LAYER_SPLIT"

python3 decode_node.py $DECODE_ARGS > "$DECODE_LOG" 2>&1 &
DECODE_PID=$!
echo "[orchestrator] Decode node PID: $DECODE_PID (log: $DECODE_LOG)"

# ── Step 3: Wait for decode node to be ready ──
echo "[orchestrator] Waiting for decode node to start..."
for i in {1..120}; do
    # Check if process is still alive
    if ! kill -0 $DECODE_PID 2>/dev/null; then
        echo "[orchestrator] ERROR: Decode node died! Check $DECODE_LOG"
        cat "$DECODE_LOG"
        exit 1
    fi
    # Check if port is listening
    if nc -z 127.0.0.1 $DECODE_PORT 2>/dev/null; then
        break
    fi
    sleep 2
    if [ $((i % 10)) -eq 0 ]; then
        echo "[orchestrator] Still waiting for decode node (${i}s)..."
    fi
done

if ! nc -z 127.0.0.1 $DECODE_PORT 2>/dev/null; then
    echo "[orchestrator] WARNING: Port $DECODE_PORT not responding, but proceeding anyway..."
fi
echo "[orchestrator] Decode node is ready"

# ── Step 4: Start prefill node on remote machine ──
echo ""
echo "[orchestrator] Starting prefill node on $PREFILL_HOST..."
PREFILL_LOG="/tmp/prefill_node_$$.log"

PREFILL_ARGS="--model-name $MODEL_PATH --prompt \"$PROMPT\" --max-new-tokens $MAX_TOKENS"
PREFILL_ARGS="$PREFILL_ARGS --decode-host $DECODE_HOST --decode-port $DECODE_PORT"
[ -n "$LAYER_SPLIT" ] && PREFILL_ARGS="$PREFILL_ARGS --layer-split $LAYER_SPLIT"
[ -n "$DO_SAMPLE" ] && PREFILL_ARGS="$PREFILL_ARGS $DO_SAMPLE"
PREFILL_ARGS="$PREFILL_ARGS --temperature $TEMPERATURE --top-k $TOP_K --top-p $TOP_P"

ssh $PREFILL_HOST "cd $PROJECT_DIR && python3 prefill_node.py $PREFILL_ARGS" 2>&1 | tee "$PREFILL_LOG"
PREFILL_EXIT=${PIPESTATUS[0]}

echo ""

# ── Step 5: Wait for decode to finish and show its output ──
echo "[orchestrator] Decode node output:"
sleep 2
cat "$DECODE_LOG" 2>/dev/null || echo "(decode log empty)"

# ── Step 6: Cleanup ──
echo ""
echo "[orchestrator] Shutting down..."
if kill -0 $DECODE_PID 2>/dev/null; then
    kill $DECODE_PID 2>/dev/null || true
    wait $DECODE_PID 2>/dev/null || true
fi
echo "[orchestrator] Done. Exit code: $PREFILL_EXIT"
