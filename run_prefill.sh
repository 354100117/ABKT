#!/bin/bash
# ── EdgePD: Start Prefill Node ──
#
# 单次推理:
#   ./run_prefill.sh /path/to/model "Your prompt" [max_tokens]
#
# 交互式多轮对话:
#   ./run_prefill.sh /path/to/model --interactive
#   ./run_prefill.sh /path/to/model -i --system-prompt "你是一个有帮助的助手"
#   ./run_prefill.sh /path/to/model -i --max-new-tokens 512 --do-sample
#
# 透传所有参数到 prefill_node.py:
#   ./run_prefill.sh /path/to/model --interactive --temperature 0.8 --top-k 50

MODEL_PATH="${1:?用法: $0 /path/to/model [prompt 或 --interactive] [max_tokens]}"
shift

# 如果第二个参数不是以 -- 开头，当作 prompt (向后兼容)
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]] && [[ ! "$1" =~ ^-i$ ]]; then
    PROMPT="$1"
    shift
    MAX_TOKENS="${1:-128}"
    shift
    exec python3 prefill_node.py \
        --model-name "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --max-new-tokens "${MAX_TOKENS}" \
        "$@"
else
    # 交互式模式或其他参数，全部透传
    exec python3 prefill_node.py \
        --model-name "$MODEL_PATH" \
        "$@"
fi
