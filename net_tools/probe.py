#!/usr/bin/env python3
"""ABKT 网络探测 CLI — 滑动窗口最小值带宽估计 + 状态机。

与 ABKT 系统 (backend/network_probe.py) 使用相同的带宽估计算法:
  - 滑动窗口最小值 (最近 10 个样本)
  - 冷启动安全系数 0.7 → 0.8 (随置信度提升)
  - 状态机: GOOD/DEGRADED/POOR (滞回投票)

用法:
  # Jetson (decode) 上启动服务端:
  python3 probe.py --server --port 9877

  # x86 (prefill) 上启动探测客户端:
  python3 probe.py --client --host 192.168.0.20 --port 9877

  # 只测 RTT:
  python3 probe.py --client --host 192.168.0.20 --rtt-only

  # 大探测包 (更准确):
  python3 probe.py --client --host 192.168.0.20 --bw-size 4.0

  # 持续监控:
  python3 probe.py --client --host 192.168.0.20 --duration 120
"""

import argparse
import collections
import signal
import struct
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")
from backend.state_machine import NetworkStateMachine, NetworkState


# 与 backend/network_probe.py 保持一致
BW_WINDOW_SIZE = 10
COLD_BW = 15e6             # 15 MB/s 冷启动默认
COLD_START_MARGIN = 0.7
MIN_SAFETY_MARGIN = 0.8
CONFIDENCE_TRANSFERS = 5
TARGET_TRANSFER_TIME = 2.0


class SlidingWindowBW:
    """滑动窗口带宽估计 — 与 NetworkProbeClient 逻辑一致。"""

    def __init__(self, window_size: int = BW_WINDOW_SIZE):
        self._window = collections.deque(maxlen=window_size)
        self._transfer_count = 0

    def add_probe(self, bw_bps: float):
        self._window.append(bw_bps)

    def add_transfer(self, bw_bps: float):
        self._window.append(bw_bps)
        self._transfer_count += 1

    def get_effective_bw(self) -> float:
        """保守带宽估计: 窗口最小值 × 置信度安全系数。"""
        if not self._window:
            return COLD_BW * COLD_START_MARGIN

        window_min = min(self._window)
        confidence = min(self._transfer_count / CONFIDENCE_TRANSFERS, 1.0)
        margin = COLD_START_MARGIN + confidence * (MIN_SAFETY_MARGIN - COLD_START_MARGIN)
        return window_min * margin

    def get_window_stats(self) -> dict:
        if not self._window:
            return {"min": 0, "max": 0, "median": 0, "count": 0}
        vals = sorted(self._window)
        return {
            "min": vals[0],
            "max": vals[-1],
            "median": vals[len(vals) // 2],
            "count": len(vals),
        }


# ── Server ──


def _run_server(port: int):
    import socket as _socket
    import threading

    server = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    server.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(8)
    server.settimeout(1.0)
    print(f"[Server] 监听端口 {port} (Ctrl+C 退出)")

    def _handle(conn):
        conn.settimeout(10.0)
        try:
            while True:
                h = conn.recv(1)
                if not h:
                    break
                t = h[0]
                if t == 0x01:  # RTT
                    conn.sendall(bytes([0x01]))
                elif t == 0x02:  # BW
                    raw = conn.recv(4)
                    if len(raw) < 4:
                        break
                    dl = struct.unpack("!I", raw)[0]
                    recvd = 0
                    while recvd < dl:
                        c = conn.recv(min(dl - recvd, 65536))
                        if not c:
                            break
                        recvd += len(c)
                    conn.sendall(bytes([0x02]))
        except Exception:
            pass
        finally:
            conn.close()

    try:
        while True:
            try:
                conn, _ = server.accept()
                threading.Thread(target=_handle, args=(conn,), daemon=True).start()
            except _socket.timeout:
                continue
    except KeyboardInterrupt:
        print("\n[Server] 退出")
    finally:
        server.close()


# ── Client ──


def _run_client(host: str, port: int, interval: float, duration: int,
                bw_size_mb: float, rtt_only: bool):
    import socket as _socket

    bw_est = SlidingWindowBW()
    rtt_window = collections.deque(maxlen=10)
    sm = NetworkStateMachine()

    print("=" * 75)
    print(f"  ABKT 网络探测 (滑动窗口最小值)")
    print(f"  目标: {host}:{port}  间隔: {interval}s  时长: {duration}s")
    print(f"  带宽探测: {'禁用' if rtt_only else f'{bw_size_mb} MB'}")
    print(f"  算法: 窗口大小={BW_WINDOW_SIZE}, "
          f"冷启动系数={COLD_START_MARGIN}, 最小系数={MIN_SAFETY_MARGIN}")
    print("=" * 75)

    hdr = (f"  {'时间':>12} {'类型':>6} {'当前值':>12} "
           f"{'窗口最小':>10} {'有效BW':>10} {'RTT':>8} {'状态':>10}")
    print(hdr)
    print("  " + "-" * 70)

    start = time.time()
    count = 0

    try:
        while time.time() - start < duration:
            now = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            do_bw = not rtt_only and count % 5 == 0

            if do_bw:
                # 带宽探测
                ds = int(bw_size_mb * 1024 * 1024)
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.settimeout(10.0)
                try:
                    data = b'\x00' * ds
                    sock.connect((host, port))
                    t0 = time.time()
                    sock.sendall(bytes([0x02]) + struct.pack("!I", ds) + data)
                    sock.recv(1)
                    elapsed = time.time() - t0
                    bw_bps = ds / elapsed if elapsed > 0 else 0

                    bw_est.add_probe(bw_bps)
                    stats = bw_est.get_window_stats()
                    effective = bw_est.get_effective_bw()

                    rtt_v = rtt_window[-1] if rtt_window else 0
                    st = sm.update(stats["median"], rtt_v)

                    print(f"  {now:>12} {'BW':>6} {bw_bps/1e6:>8.2f} MB/s"
                          f" {stats['min']/1e6:>8.2f}"
                          f" {effective/1e6:>8.2f}"
                          f" {rtt_v:>6.1f}ms"
                          f" {st.value:>10}")
                except Exception as e:
                    print(f"  {now:>12} {'BW':>6} {'FAIL':>12} — {e}")
                finally:
                    sock.close()
            else:
                # RTT 探测
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.settimeout(3.0)
                try:
                    t0 = time.time()
                    sock.connect((host, port))
                    sock.sendall(bytes([0x01]))
                    sock.recv(1)
                    rtt = (time.time() - t0) * 1000

                    rtt_window.append(rtt)
                    rtt_avg = sum(rtt_window) / len(rtt_window)

                    effective = bw_est.get_effective_bw()
                    stats = bw_est.get_window_stats()
                    st = sm.update(stats["median"], rtt_avg)

                    print(f"  {now:>12} {'RTT':>6} {rtt:>8.2f} ms"
                          f" {stats['min']/1e6:>8.2f}"
                          f" {effective/1e6:>8.2f}"
                          f" {rtt_avg:>6.1f}ms"
                          f" {st.value:>10}")
                except Exception as e:
                    print(f"  {now:>12} {'RTT':>6} {'FAIL':>12} — {e}")
                finally:
                    sock.close()

            count += 1
            wait = time.time() + interval
            while time.time() < wait:
                if time.time() - start >= duration:
                    break
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[Client] 中断")

    # 汇总
    effective = bw_est.get_effective_bw()
    stats = bw_est.get_window_stats()
    budget = effective * TARGET_TRANSFER_TIME

    print(f"\n{'=' * 75}")
    print(f"  探测结束 — {count} 次 ({time.time() - start:.1f}s)")
    print(f"  窗口统计: min={stats['min']/1e6:.2f}  "
          f"median={stats['median']/1e6:.2f}  "
          f"max={stats['max']/1e6:.2f} MB/s  "
          f"samples={stats['count']}")
    print(f"  有效带宽: {effective/1e6:.2f} MB/s ({effective*8/1e6:.1f} Mbps)")
    print(f"  预算({TARGET_TRANSFER_TIME}s): {budget/1e6:.2f} MB")
    print(f"  最终状态: {sm.state.value}")
    print(f"{'=' * 75}")


# ── Entry ──


def main():
    p = argparse.ArgumentParser(description="ABKT 网络探测工具 (滑动窗口)")
    p.add_argument("--server", action="store_true", help="启动服务端 (Jetson)")
    p.add_argument("--client", action="store_true", help="启动客户端 (x86)")
    p.add_argument("--host", default="192.168.0.20")
    p.add_argument("--port", type=int, default=9877)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--bw-size", type=float, default=1.0, help="带宽探测大小 MB")
    p.add_argument("--rtt-only", action="store_true")

    args = p.parse_args()
    if not args.server and not args.client:
        p.print_help()
        sys.exit(1)

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    if args.server:
        _run_server(args.port)
    else:
        _run_client(args.host, args.port, args.interval,
                     args.duration, args.bw_size, args.rtt_only)


if __name__ == "__main__":
    main()
