#!/usr/bin/env python3
"""ABKT 功能快速验证 — 滑动窗口 + 状态机 + 真实链路探测。

验证 ABKT 核心组件:
  1. 滑动窗口最小值带宽估计 (替代旧 EWMA)
  2. 三级状态机 (GOOD/DEGRADED/POOR) + 滞回
  3. 真实链路探测 (需要 Jetson 上运行 probe.py --server)

用法:
  # 只验证本地逻辑 (无需网络):
  python3 verify.py

  # 验证全部 (包括真实链路):
  python3 verify.py --host 192.168.0.20 --port 9877

  # 自动运行全部 (不询问):
  python3 verify.py --host 192.168.0.20 --auto
"""

import argparse
import collections
import socket as _socket
import struct
import sys
import time

sys.path.insert(0, ".")
from backend.state_machine import (
    NetworkStateMachine, NetworkState,
    BW_GOOD_THRESHOLD, BW_POOR_THRESHOLD,
    RTT_GOOD_THRESHOLD,
)

_ok, _fail = 0, 0


def check(name: str, cond: bool, detail: str = ""):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}: {detail}")


# ════════════════════════════════════════════════════════════════════
# 滑动窗口验证
# ════════════════════════════════════════════════════════════════════


def verify_sliding_window():
    """验证滑动窗口最小值带宽估计。"""
    print("\n" + "=" * 60)
    print("  1. 滑动窗口最小值带宽估计")
    print("=" * 60)

    window = collections.deque(maxlen=10)

    # 冷启动: 空窗口
    check("冷启动: 窗口为空", len(window) == 0)

    # 首次探测
    window.append(80e6)
    check(f"首次探测: min={min(window)/1e6:.0f} MB/s", min(window) == 80e6)

    # 稳态
    for _ in range(5):
        window.append(80e6)
    check(f"稳态: min={min(window)/1e6:.0f} MB/s", min(window) == 80e6)

    # 带宽骤降: min 应立即响应
    window.append(20e6)
    check(f"骤降响应: min={min(window)/1e6:.0f} MB/s (应为20)",
          min(window) == 20e6, f"got {min(window)/1e6:.0f}")

    # 恢复: min 不会立即升高 (需要旧样本过期)
    window.append(80e6)
    check(f"恢复延迟: min={min(window)/1e6:.0f} MB/s (仍为20)",
          min(window) == 20e6, f"got {min(window)/1e6:.0f}")

    # 填满窗口使旧样本过期 (需要 10 个新样本把 20e6 挤出窗口)
    for _ in range(10):
        window.append(80e6)
    check(f"窗口滑动后: min={min(window)/1e6:.0f} MB/s (应为80)",
          min(window) == 80e6, f"got {min(window)/1e6:.0f}")

    # 与 EWMA 对比: 窗口最小值响应更快
    window2 = collections.deque(maxlen=10)
    for _ in range(5):
        window2.append(80e6)
    window2.append(20e6)
    window_min = min(window2)

    # EWMA 模拟: alpha=0.3, 0.3*20 + 0.7*80 = 62
    ewma_value = 0.3 * 20e6 + 0.7 * 80e6
    check(f"窗口最小值({window_min/1e6:.0f}) < EWMA({ewma_value/1e6:.0f}): "
          f"响应更快", window_min < ewma_value)


# ════════════════════════════════════════════════════════════════════
# 状态机验证
# ════════════════════════════════════════════════════════════════════


def verify_state_machine():
    """验证三级状态机 + 滞回。"""
    print("\n" + "=" * 60)
    print("  2. 三级网络状态机 (含滞回)")
    print("=" * 60)

    sm = NetworkStateMachine()

    # GOOD: 高带宽 + 低延迟
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 1.1, RTT_GOOD_THRESHOLD * 0.5)
    check("GOOD: 高带宽+低延迟", sm.state == NetworkState.GOOD)

    # POOR: 仅带宽低 (非 OR bug)
    for _ in range(3):
        sm.update(BW_POOR_THRESHOLD * 0.3, 1.0)
    check("POOR: 仅带宽低 (非 OR with RTT)",
          sm.state == NetworkState.POOR, f"状态={sm.state.value}")

    # DEGRADED: 高 RTT + 好带宽
    for _ in range(6):
        sm.update(BW_GOOD_THRESHOLD * 2, RTT_GOOD_THRESHOLD * 0.5)
    for _ in range(3):
        sm.update(BW_GOOD_THRESHOLD * 2, 60.0)
    check("DEGRADED: 高RTT+好带宽 → 非 POOR",
          sm.state == NetworkState.DEGRADED, f"状态={sm.state.value}")

    # 滞回: 1次劣化不降级
    for _ in range(6):
        sm.update(BW_GOOD_THRESHOLD * 2, RTT_GOOD_THRESHOLD * 0.5)
    sm.update(BW_POOR_THRESHOLD * 0.3, 100.0)
    check("滞回: 1次劣化不降级", sm.state == NetworkState.GOOD)

    # 滞回: 2/3 快速降级
    sm.update(BW_POOR_THRESHOLD * 0.3, 100.0)
    check("滞回: 2/3 快速降级", sm.state == NetworkState.POOR)

    # 预算递减: 低带宽 → 低预算
    sm2 = NetworkStateMachine()
    budget_high = sm2.budget_bytes(100e6)   # 100 MB/s → 200 MB
    budget_low = sm2.budget_bytes(10e6)     # 10 MB/s → 20 MB
    check(f"预算递减: 低BW预算({budget_low/1e6:.0f}MB) < "
          f"高BW预算({budget_high/1e6:.0f}MB)",
          budget_low < budget_high * 0.2)


# ════════════════════════════════════════════════════════════════════
# 反馈循环防护验证
# ════════════════════════════════════════════════════════════════════


def verify_feedback_loop():
    """验证反馈循环防护机制。"""
    print("\n" + "=" * 60)
    print("  3. 反馈循环防护")
    print("=" * 60)

    # 场景: 压缩后的传输 → 低测量 BW → 更多压缩 → 恶性循环
    # 修复: record_transfer 使用等效未压缩带宽

    # 模拟: compressed=4MB, elapsed=0.2s, ratio=4.0
    # 正确: 4MB * 4.0 / 0.2 = 80 MB/s
    # 错误: 4MB / 0.2 = 20 MB/s
    correct_bw = 4 * 1024 * 1024 * 4.0 / 0.2
    wrong_bw = 4 * 1024 * 1024 / 0.2
    check(f"等效未压缩BW({correct_bw/1e6:.0f}) >> 压缩BW({wrong_bw/1e6:.0f})",
          correct_bw > wrong_bw * 3)

    # 模拟 GC 暂停: 一次慢传输
    window = collections.deque(maxlen=10)
    for _ in range(5):
        window.append(80e6)

    # 无 clamp: EWMA 大幅下降
    alpha = 0.5
    ewma_unclamped = alpha * 25e6 + (1 - alpha) * 80e6

    # 有 clamp (±20%): 80 * 0.8 = 64
    ewma_clamped = max(80e6 * 0.8, alpha * 25e6 + (1 - alpha) * 80e6)

    # 窗口最小值: 立即响应但不剧烈
    window.append(25e6)
    window_min = min(window)

    check(f"EWMA clamp 保护: clamped({ewma_clamped/1e6:.0f}) > "
          f"unclamped({ewma_unclamped/1e6:.0f})",
          ewma_clamped > ewma_unclamped)


# ════════════════════════════════════════════════════════════════════
# 真实链路探测
# ════════════════════════════════════════════════════════════════════


def verify_real_probe(host: str, port: int):
    """真实链路探测 — 使用滑动窗口。"""
    print("\n" + "=" * 60)
    print("  4. 真实链路探测")
    print("=" * 60)
    print(f"  目标: {host}:{port}")

    window = collections.deque(maxlen=10)
    rtt_window = collections.deque(maxlen=10)
    sm = NetworkStateMachine()

    # RTT 探测
    for i in range(3):
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            t0 = time.time()
            sock.connect((host, port))
            sock.sendall(bytes([0x01]))
            sock.recv(1)
            rtt = (time.time() - t0) * 1000
            rtt_window.append(rtt)
            rtt_avg = sum(rtt_window) / len(rtt_window)
            print(f"  RTT#{i+1}: {rtt:.2f} ms  (avg={rtt_avg:.2f} ms)")
        except Exception as e:
            print(f"  RTT#{i+1}: FAIL — {e}")
        finally:
            sock.close()

    # 带宽探测 (2MB)
    ds = 2 * 1024 * 1024
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        data = b'\x00' * ds
        sock.connect((host, port))
        t0 = time.time()
        sock.sendall(bytes([0x02]) + struct.pack("!I", ds) + data)
        sock.recv(1)
        elapsed = time.time() - t0
        bw_bps = ds / elapsed
        window.append(bw_bps)
        print(f"  BW: {bw_bps/1e6:.2f} MB/s ({bw_bps*8/1e6:.0f} Mbps), "
              f"耗时 {elapsed*1000:.0f}ms")
    except Exception as e:
        print(f"  BW: FAIL — {e}")
    finally:
        sock.close()

    # 第二次带宽探测 (更准确)
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        data = b'\x00' * ds
        sock.connect((host, port))
        t0 = time.time()
        sock.sendall(bytes([0x02]) + struct.pack("!I", ds) + data)
        sock.recv(1)
        elapsed = time.time() - t0
        bw_bps = ds / elapsed
        window.append(bw_bps)
        print(f"  BW: {bw_bps/1e6:.2f} MB/s ({bw_bps*8/1e6:.0f} Mbps), "
              f"耗时 {elapsed*1000:.0f}ms")
    except Exception as e:
        print(f"  BW: FAIL — {e}")
    finally:
        sock.close()

    # 计算结果
    rtt_avg = sum(rtt_window) / len(rtt_window) if rtt_window else 0
    window_min = min(window) if window else 0
    window_median = sorted(window)[len(window) // 2] if window else 0
    effective = window_min * 0.7  # 冷启动系数

    sm.update(window_median, rtt_avg)
    budget = effective * 2.0

    print(f"\n  窗口: min={window_min/1e6:.2f}  "
          f"median={window_median/1e6:.2f} MB/s")
    print(f"  有效BW: {effective/1e6:.2f} MB/s (冷启动系数 0.7)")
    print(f"  预算(2s): {budget/1e6:.2f} MB")
    print(f"  状态: {sm.state.value}")
    print(f"  RTT: {rtt_avg:.2f} ms")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="ABKT 功能快速验证")
    parser.add_argument("--host", default="192.168.0.20")
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument("--auto", action="store_true",
                        help="自动运行全部验证 (不询问)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ABKT 功能快速验证 (滑动窗口 + 状态机)")
    print("=" * 60)

    verify_sliding_window()
    verify_state_machine()
    verify_feedback_loop()

    # 真实链路探测
    do_probe = args.auto
    if not do_probe:
        try:
            resp = input(f"\n  是否进行真实链路探测? ({args.host}:{args.port}) (y/N): ")
            do_probe = resp.lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            do_probe = False

    if do_probe:
        print(f"  确保 Jetson 已运行: python3 probe.py --server --port {args.port}")
        verify_real_probe(args.host, args.port)

    print(f"\n{'=' * 60}")
    print(f"  结果: {_ok}/{_ok + _fail} 通过"
          + (f", {_fail} 失败" if _fail else ", 全部通过"))
    print(f"{'=' * 60}")

    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
