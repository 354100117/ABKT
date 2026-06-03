#!/usr/bin/env python3
"""ABKT 网络波动模拟 — 基于 tc 的真实场景测试工具。

针对 ABKT (Adaptive Bitrate KV Cache Transfer) 系统设计，模拟 PD 分离推理中
prefill→decode 节点之间的各种网络条件。

实际测试环境:
  - Prefill: RTX 5060 Ti @ 192.168.0.50
  - Decode:  Jetson Orin @ 192.168.0.20 (eno1)
  - 链路: 1 GbE 有线局域网
  - 实测吞吐: ~1.27 MB/s (10 Mbps) — 受 Jetson CPU/内存限制

用法:
  sudo ./tc_fluct.py                              # 交互式选择场景
  sudo ./tc_fluct.py --preset mid_drop            # 中途带宽骤降
  sudo ./tc_fluct.py --preset gradual             # 渐进退化
  sudo ./tc_fluct.py --preset jitter              # 周期性抖动
  sudo ./tc_fluct.py --preset realistic           # 多阶段真实剖面
  sudo ./tc_fluct.py --preset sudden              # 瞬间骤降
  sudo ./tc_fluct.py --preset loss_cause          # 丢包导致的带宽下降
  sudo ./tc_fluct.py --preset spike               # 短暂带宽尖峰
  sudo ./tc_fluct.py --preset all_test            # 综合测试序列
  sudo ./tc_fluct.py --reset                      # 清除所有 tc 规则

  sudo ./tc_fluct.py --preset mid_drop --dev eno1 --log-file results.csv

场景说明:
  mid_drop      ABKT 核心测试: 正常→骤降→恢复，验证中途降级机制
  gradual       渐进退化再恢复，验证状态机 GOOD→DEGRADED→POOR 转换
  sudden        瞬间骤降，验证滑动窗口最小值的即时响应
  jitter        周期性高低交替，验证状态机滞回防抖
  loss_cause    丢包导致有效带宽下降，验证非带宽因素的鲁棒性
  realistic     多阶段真实剖面 (正常→拥堵→恢复→稳定)
  spike         短暂带宽尖峰，验证不会因瞬时高带宽过度乐观
  all_test      综合测试: 依次运行所有场景 (每场景 60s)
"""

import argparse
import csv
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple


# ════════════════════════════════════════════════════════════════════
# 场景定义 — 针对 ABKT 测试设计
# ════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """网络场景配置。"""
    name: str
    description: str
    duration: float          # 总时长 (秒)
    phases: list             # [(持续秒, bandwidth_mbps, delay_ms, loss_pct), ...]
    auto_correlate: bool = True  # 带宽低时自动增加延迟


SCENARIOS = {
    # ── ABKT 核心测试场景 ──

    "mid_drop": Scenario(
        name="mid_drop",
        description="ABKT 核心: 正常→骤降→恢复 (验证中途降级)",
        duration=90,
        phases=[
            (20, 80, 2, 0),    # 正常: 80 Mbps, 2ms RTT
            (5, 5, 5, 0),      # 骤降: 5 Mbps — 应触发中途降级
            (20, 5, 8, 0),     # 持续低带宽
            (5, 20, 4, 0),     # 部分恢复
            (20, 60, 2, 0),    # 恢复正常
            (20, 80, 2, 0),    # 完全恢复
        ],
    ),

    "gradual": Scenario(
        name="gradual",
        description="渐进退化再恢复 (验证状态机转换)",
        duration=120,
        phases=[
            (20, 80, 2, 0),    # GOOD 状态
            (20, 40, 3, 0),    # 进入 DEGRADED
            (20, 10, 5, 0),    # 进入 POOR
            (20, 5, 8, 0),     # 深度 POOR
            (20, 30, 4, 0),    # 恢复到 DEGRADED
            (20, 70, 2, 0),    # 恢复到 GOOD
        ],
    ),

    "sudden": Scenario(
        name="sudden",
        description="瞬间骤降 (验证滑动窗口即时响应)",
        duration=60,
        phases=[
            (25, 100, 2, 0),   # 高带宽稳态
            (1, 2, 10, 0),     # 瞬间跌到 2 Mbps
            (9, 2, 10, 0),     # 持续 10s
            (1, 80, 2, 0),     # 瞬间恢复
            (24, 80, 2, 0),    # 稳态恢复
        ],
    ),

    "jitter": Scenario(
        name="jitter",
        description="周期性抖动 (验证状态机滞回防抖)",
        duration=80,
        phases=[
            # 每 10s 交替: 高→低→高→低 ...
            (10, 80, 2, 0),
            (10, 5, 5, 0),
            (10, 80, 2, 0),
            (10, 5, 5, 0),
            (10, 80, 2, 0),
            (10, 5, 5, 0),
            (10, 80, 2, 0),
            (10, 5, 5, 0),
        ],
    ),

    "loss_cause": Scenario(
        name="loss_cause",
        description="丢包导致有效带宽下降 (验证非带宽因素)",
        duration=80,
        phases=[
            (20, 80, 2, 0),     # 正常: 无丢包
            (20, 80, 2, 5),     # 带宽不变但 5% 丢包 → 有效吞吐下降
            (20, 80, 2, 15),    # 15% 丢包 → 严重退化
            (20, 80, 2, 0),     # 恢复: 无丢包
        ],
    ),

    "realistic": Scenario(
        name="realistic",
        description="多阶段真实剖面 (模拟一天中的网络变化)",
        duration=180,
        phases=[
            (30, 80, 2, 0),     # 早晨: 网络空闲
            (20, 50, 3, 0),     # 开始有负载
            (15, 20, 5, 2),     # 高峰: 带宽下降+丢包
            (10, 5, 10, 5),     # 严重拥堵
            (15, 15, 6, 2),     # 缓慢恢复
            (30, 40, 3, 0),     # 负载减轻
            (30, 70, 2, 0),     # 恢复正常
            (30, 80, 2, 0),     # 完全恢复
        ],
    ),

    "spike": Scenario(
        name="spike",
        description="短暂带宽尖峰 (验证不会过度乐观)",
        duration=60,
        phases=[
            (20, 30, 3, 0),    # 中等带宽
            (3, 200, 1, 0),    # 短暂尖峰 (可能来自其他流量结束)
            (7, 30, 3, 0),     # 回到中等
            (3, 150, 1, 0),    # 另一个尖峰
            (27, 30, 3, 0),    # 稳态
        ],
    ),

    "all_test": Scenario(
        name="all_test",
        description="综合测试: 依次运行所有场景 (每场景 60s)",
        duration=0,  # 特殊: 由外部循环控制
        phases=[],   # 特殊: 由外部循环控制
    ),
}


# ════════════════════════════════════════════════════════════════════
# tc 操作
# ════════════════════════════════════════════════════════════════════


def _detect_interface() -> str:
    """自动检测主要网络接口。"""
    try:
        result = subprocess.run(
            ["ip", "route", "get", "192.168.0.1"],
            capture_output=True, text=True, timeout=2,
        )
        for part in result.stdout.split():
            if part in ("eno1", "eth0", "enp0s25", "enp3s0"):
                return part
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ip", "-br", "link", "show"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "UP" and parts[0] != "lo":
                return parts[0]
    except Exception:
        pass

    return "eno1"


def _tc(dev: str, action: str, args: str) -> int:
    """执行 tc 命令，返回返回码。"""
    cmd = f"tc {action} dev {dev} {args}"
    return os.system(cmd)


def _tc_reset(dev: str) -> None:
    """清除所有 tc 规则，恢复正常网络。"""
    _tc(dev, "qdisc del", "root 2>/dev/null || true")
    print(f"[tc] 已清除 {dev} 上的所有规则")


def _tc_apply(dev: str, bw_mbps: float, delay_ms: float = 0,
              loss_pct: float = 0) -> None:
    """应用带宽限制 + 延迟 + 丢包。

    tc 层级:
      root → htb (带宽) → netem (延迟/丢包)
    """
    # 确保清除旧规则
    _tc(dev, "qdisc del", "root 2>/dev/null || true")

    # HTB 带宽限制
    rate_kbit = max(1, int(bw_mbps * 1000))
    _tc(dev, "qdisc add", "root handle 1: htb default 10")
    _tc(dev, "class add",
        f"parent 1: classid 1:10 htb rate {rate_kbit}kbit burst {max(16, rate_kbit // 100)}kbit")

    # netem 延迟 + 丢包
    netem_parts = []
    if delay_ms > 0:
        jitter = max(1, delay_ms * 0.2)  # 20% 抖动
        netem_parts.append(f"delay {delay_ms:.0f}ms {jitter:.0f}ms")
    if loss_pct > 0:
        netem_parts.append(f"loss {loss_pct:.1f}%")

    if netem_parts:
        _tc(dev, "qdisc add",
            f"parent 1:10 handle 10: netem {' '.join(netem_parts)}")


def _tc_update_bw(dev: str, bw_mbps: float) -> None:
    """仅更新带宽 (保留 netem 规则)。"""
    rate_kbit = max(1, int(bw_mbps * 1000))
    _tc(dev, "class replace",
        f"parent 1: classid 1:10 htb rate {rate_kbit}kbit burst {max(16, rate_kbit // 100)}kbit")


def _tc_update_netem(dev: str, delay_ms: float = 0, loss_pct: float = 0) -> None:
    """更新 netem 规则 (延迟+丢包)。"""
    _tc(dev, "qdisc del", "parent 1:10 handle 10: 2>/dev/null || true")
    netem_parts = []
    if delay_ms > 0:
        jitter = max(1, delay_ms * 0.2)
        netem_parts.append(f"delay {delay_ms:.0f}ms {jitter:.0f}ms")
    if loss_pct > 0:
        netem_parts.append(f"loss {loss_pct:.1f}%")
    if netem_parts:
        _tc(dev, "qdisc add",
            f"parent 1:10 handle 10: netem {' '.join(netem_parts)}")


# ════════════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════════════


def _bar(value: float, max_val: float, width: int = 20) -> str:
    """生成带宽条形图。"""
    n = max(0, min(width, int(value / max(max_val, 1) * width)))
    return "█" * n + "░" * (width - n)


def _bw_tag(bw_mbps: float) -> str:
    """带宽等级标签 (对应 ABKT 状态机)。"""
    if bw_mbps >= 50:
        return "GOOD   "
    elif bw_mbps >= 20:
        return "DEGRADED"
    elif bw_mbps >= 5:
        return "POOR   "
    else:
        return "CRITICAL"


def _format_phase_desc(bw: float, delay: float, loss: float) -> str:
    """格式化当前阶段描述。"""
    parts = [f"{bw:.0f}Mbps"]
    if delay > 0:
        parts.append(f"delay={delay:.0f}ms")
    if loss > 0:
        parts.append(f"loss={loss:.0f}%")
    return " ".join(parts)


# ════════════════════════════════════════════════════════════════════
# CSV 日志
# ════════════════════════════════════════════════════════════════════


def _open_csv(path: str):
    """打开 CSV 日志文件。"""
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow([
        "timestamp_sec", "scenario", "phase",
        "target_bw_mbps", "delay_ms", "loss_pct",
        "abkt_state", "comment",
    ])
    fh.flush()
    return fh, writer


def _log_csv(writer, fh, t: float, scenario: str, phase: int,
             bw: float, delay: float, loss: float,
             abkt_state: str = "", comment: str = ""):
    """写入一条 CSV 记录。"""
    writer.writerow([
        f"{t:.2f}", scenario, phase,
        f"{bw:.2f}", f"{delay:.1f}", f"{loss:.1f}",
        abkt_state, comment,
    ])
    fh.flush()


# ════════════════════════════════════════════════════════════════════
# 带宽↔延迟关联
# ════════════════════════════════════════════════════════════════════


def _correlated_delay(bw_mbps: float, base_delay: float = 2.0) -> float:
    """带宽低 → 延迟高 (模拟拥塞效应)。"""
    if bw_mbps >= 50:
        return base_delay
    elif bw_mbps >= 20:
        return base_delay + 2
    elif bw_mbps >= 10:
        return base_delay + 5
    elif bw_mbps >= 5:
        return base_delay + 10
    else:
        return base_delay + 20


# ════════════════════════════════════════════════════════════════════
# 场景执行
# ════════════════════════════════════════════════════════════════════


def _estimate_abkt_state(bw_mbps: float) -> str:
    """估算 ABKT 状态机在给定带宽下的状态。

    基于 state_machine.py 的阈值:
      BW_GOOD_THRESHOLD = 40e6 B/s ≈ 40 MB/s ≈ 320 Mbps
      BW_POOR_THRESHOLD = 15e6 B/s ≈ 15 MB/s ≈ 120 Mbps

    注意: 这里用 tc 限制的是 Mbps (网络层)，而 ABKT 测量的是 MB/s (应用层)。
    应用层吞吐通常为网络层的 80-90% (TCP 开销)。
    """
    # tc Mbps → 应用层 MB/s (约 0.115 倍)
    app_bw_mbps = bw_mbps * 0.85  # TCP 效率
    app_bw_bytes = app_bw_mbps * 1e6 / 8  # → bytes/sec

    if app_bw_bytes >= 40e6:
        return "GOOD"
    elif app_bw_bytes >= 15e6:
        return "DEGRADED"
    else:
        return "POOR"


def run_scenario(scenario: Scenario, dev: str, log_writer=None, log_fh=None,
                 start_time: float = 0) -> None:
    """执行单个网络场景。"""
    print(f"\n{'=' * 65}")
    print(f"  场景: {scenario.name}")
    print(f"  描述: {scenario.description}")
    print(f"  时长: {scenario.duration:.0f}s")
    print(f"  接口: {dev}")
    print(f"{'=' * 65}")

    max_bw = max(bw for _, bw, _, _ in scenario.phases)
    t_offset = time.time()
    phase_start = 0

    for phase_idx, (duration, bw, delay, loss) in enumerate(scenario.phases):
        # 应用 tc 规则
        if phase_idx == 0:
            _tc_apply(dev, bw, delay, loss)
        else:
            _tc_update_bw(dev, bw)
            _tc_update_netem(dev, delay, loss)

        abkt_state = _estimate_abkt_state(bw)
        desc = _format_phase_desc(bw, delay, loss)
        print(f"\n  阶段 {phase_idx + 1}/{len(scenario.phases)}: "
              f"{desc}  →  ABKT≈{abkt_state}")

        # 等待阶段持续时间，每秒更新显示
        phase_end = phase_start + duration
        while True:
            t = time.time() - t_offset
            if t >= phase_end:
                break

            elapsed_in_phase = t - phase_start
            progress = elapsed_in_phase / duration if duration > 0 else 1

            # 进度条
            bar_width = 30
            filled = int(progress * bar_width)
            bar = "━" * filled + "╌" * (bar_width - filled)

            # 带宽可视化
            bw_bar = _bar(bw, max_bw, 16)
            tag = _bw_tag(bw)

            # 延迟/丢包信息
            extras = []
            if delay > 0:
                extras.append(f"d={delay:.0f}ms")
            if loss > 0:
                extras.append(f"l={loss:.0f}%")
            extra_str = "  ".join(extras)

            line = (f"\r  [{t:5.0f}s] {bar} "
                    f"{bw_bar} {bw:6.1f}Mbps [{tag}]")
            if extra_str:
                line += f"  {extra_str}"
            print(line, end="", flush=True)

            # CSV 日志
            if log_writer:
                abs_t = t + start_time
                _log_csv(log_writer, log_fh, abs_t, scenario.name,
                         phase_idx + 1, bw, delay, loss, abkt_state)

            time.sleep(1)

        phase_start = phase_end

    # 清除 tc 规则
    _tc_reset(dev)
    print(f"\n  场景 {scenario.name} 完成")


# ════════════════════════════════════════════════════════════════════
# 交互式场景选择
# ════════════════════════════════════════════════════════════════════


def _interactive_select() -> str:
    """交互式选择场景。"""
    print("\n" + "=" * 65)
    print("  ABKT 网络波动模拟 — 场景选择")
    print("=" * 65)
    print()
    print("  可用场景:")
    print()

    names = list(SCENARIOS.keys())
    for i, name in enumerate(names):
        sc = SCENARIOS[name]
        print(f"    [{i + 1}] {name:15s}  {sc.description}")

    print()
    print("    [0] 退出")
    print()

    while True:
        try:
            choice = input("  选择场景编号: ").strip()
            if choice == "0":
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
            print(f"  无效选择，请输入 0-{len(names)}")
        except ValueError:
            print("  请输入数字")
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ABKT 网络波动模拟 — 基于 tc 的真实场景测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
场景说明:
  mid_drop      ABKT 核心: 正常→骤降→恢复 (验证中途降级)
  gradual       渐进退化再恢复 (验证状态机转换)
  sudden        瞬间骤降 (验证滑动窗口即时响应)
  jitter        周期性抖动 (验证状态机滞回防抖)
  loss_cause    丢包导致带宽下降 (验证非带宽因素)
  realistic     多阶段真实剖面
  spike         短暂带宽尖峰 (验证不会过度乐观)
  all_test      综合测试: 依次运行所有场景

示例:
  sudo ./tc_fluct.py --preset mid_drop
  sudo ./tc_fluct.py --preset gradual --log-file results.csv
  sudo ./tc_fluct.py --preset all_test --log-file all_results.csv
  sudo ./tc_fluct.py --reset
""",
    )

    parser.add_argument("--preset", choices=list(SCENARIOS.keys()),
                        help="预设场景名称")
    parser.add_argument("--dev", default=None,
                        help="网络接口 (默认自动检测)")
    parser.add_argument("--log-file", default=None, metavar="PATH",
                        help="CSV 日志文件路径")
    parser.add_argument("--reset", action="store_true",
                        help="清除所有 tc 规则后退出")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (用于可重现测试)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用场景")

    args = parser.parse_args()

    # 列出场景
    if args.list:
        print("\n可用场景:")
        for name, sc in SCENARIOS.items():
            dur = f"{sc.duration:.0f}s" if sc.duration > 0 else "可变"
            print(f"  {name:15s}  ({dur:>6s})  {sc.description}")
        sys.exit(0)

    # 需要 root
    if os.geteuid() != 0:
        print("错误: 需要 root 权限 (sudo)")
        sys.exit(1)

    # 检测接口
    dev = args.dev or _detect_interface()
    print(f"[tc] 使用接口: {dev}")

    # 重置模式
    if args.reset:
        _tc_reset(dev)
        sys.exit(0)

    # 随机种子
    if args.seed is not None:
        random.seed(args.seed)

    # 选择场景
    preset = args.preset
    if preset is None:
        preset = _interactive_select()

    # 设置信号处理
    log_fh = None
    log_writer = None

    def cleanup(signum=None, frame=None):
        _tc_reset(dev)
        if log_fh:
            log_fh.close()
            if args.log_file:
                print(f"\n[tc] 日志已保存: {args.log_file}")
        print("\n[tc] 已恢复网络，退出")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 打开日志
    if args.log_file:
        log_fh, log_writer = _open_csv(args.log_file)
        print(f"[tc] CSV 日志: {args.log_file}")

    # 执行场景
    try:
        if preset == "all_test":
            # 综合测试: 依次运行所有场景
            t_start = 0
            for name in ["mid_drop", "gradual", "sudden", "jitter",
                         "loss_cause", "realistic", "spike"]:
                sc = SCENARIOS[name]
                run_scenario(sc, dev, log_writer, log_fh, t_start)
                t_start += sc.duration
                time.sleep(2)  # 场景间间隔
        else:
            scenario = SCENARIOS[preset]
            run_scenario(scenario, dev, log_writer, log_fh)
    finally:
        _tc_reset(dev)
        if log_fh:
            log_fh.close()
            if args.log_file:
                print(f"[tc] 日志已保存: {args.log_file}")

    print(f"\n[tc] 所有场景执行完毕")


if __name__ == "__main__":
    main()
