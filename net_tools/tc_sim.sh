#!/bin/bash
# ABKT 网络模拟快捷入口 — 转发到 tc_fluct.py
#
# 用法:
#   sudo ./tc_sim.sh                    # 交互式选择场景
#   sudo ./tc_sim.sh mid_drop           # 直接运行指定场景
#   sudo ./tc_sim.sh gradual --log-file results.csv
#   sudo ./tc_sim.sh reset              # 清除所有 tc 规则
#   sudo ./tc_sim.sh list               # 列出所有场景
#
# 场景:
#   mid_drop      ABKT 核心: 正常→骤降→恢复 (验证中途降级)
#   gradual       渐进退化再恢复 (验证状态机转换)
#   sudden        瞬间骤降 (验证滑动窗口即时响应)
#   jitter        周期性抖动 (验证状态机滞回防抖)
#   loss_cause    丢包导致带宽下降 (验证非带宽因素)
#   realistic     多阶段真实剖面
#   spike         短暂带宽尖峰 (验证不会过度乐观)
#   all_test      综合测试: 依次运行所有场景
#   reset         清除所有 tc 规则

if [ "$(id -u)" -ne 0 ]; then
    echo "错误: 需要 root 权限，请使用: sudo $0 $*"
    exit 1
fi

DIR="$(dirname "$0")"
DEV="${TC_DEV:-}"

case "${1:-}" in
    reset)
        DEV_ARG="${DEV:+--dev $DEV}"
        exec python3 "$DIR/tc_fluct.py" --reset $DEV_ARG
        ;;
    list)
        exec python3 "$DIR/tc_fluct.py" --list
        ;;
    "")
        # 无参数: 交互式选择
        DEV_ARG="${DEV:+--dev $DEV}"
        exec python3 "$DIR/tc_fluct.py" $DEV_ARG
        ;;
    *)
        # 第一个参数作为 preset，其余透传
        PRESET="$1"
        shift
        DEV_ARG="${DEV:+--dev $DEV}"
        exec python3 "$DIR/tc_fluct.py" --preset "$PRESET" $DEV_ARG "$@"
        ;;
esac
