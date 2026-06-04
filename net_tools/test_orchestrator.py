#!/usr/bin/env python3
"""ABKT 测试编排器 — 精确控制网络变化与 KV cache 传输的时序。

核心设计: tc 规则在 prefill_node.py 启动前施加，确保 bandwidth probe
测量到受限带宽，ABKT 做出正确的压缩决策。

原理:
  1. 在 prefill 节点 eno1 上施加 tc 带宽限制 (egress)
  2. 启动 prefill_node.py — probe 测量受限带宽，ABKT 基于此决策
  3. 监控 stdout，传输开始后可动态切换带宽 (mid_drop / oscillating)
  4. 传输结束后清除 tc 规则
  5. 收集并对比结果

用法:
  # 基本测试:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario baseline

  # 中途骤降 (OPT-2.7B 触发 ABKT):
  python3 test_orchestrator.py --model /ssd/models/opt-2.7b --scenario mid_drop

  # 对比测试:
  python3 test_orchestrator.py --model /ssd/models/opt-2.7b --compare baseline mid_drop

  # 全场景测试:
  python3 test_orchestrator.py --model /ssd/models/opt-2.7b --all

  # 自定义带宽:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario custom \\
      --pre-tc 10 --transfer-tc 5
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════

PREFILL_HOST = "192.168.0.50"
PREFILL_USER = "nvidia"
DECODE_HOST = "192.168.0.20"
DECODE_PORT = 29501
PROBE_PORT = 9877
PREFILL_IFACE = "eno1"
ABKT_DIR = "/ssd/pd/ABKT"


# ════════════════════════════════════════════════════════════════════
# 场景定义
# ════════════════════════════════════════════════════════════════════

@dataclass
class TestScenario:
    """测试场景配置。

    pre_tc_mbps: prefill 启动前施加的带宽 (Mbps), 0=不限制。
                 这决定了 bandwidth probe 测量到的带宽和 ABKT 的压缩决策。
    transfer_tc_mbps: KV 传输过程中切换到的带宽 (Mbps), 0=不变。
                      用于 mid_drop 等场景，传输中动态降速。
    """
    name: str
    description: str
    pre_tc_mbps: float = 0          # prefill 启动前施加
    transfer_tc_mbps: float = 0     # 传输中切换到 (0=不变)
    delay_ms: float = 0             # 延迟 (ms)
    loss_pct: float = 0             # 丢包 (%)
    oscillate: bool = False         # 是否振荡 (由 OscillationThread 控制)
    model_override: str = ""        # 覆盖模型路径 (空=使用 --model)


# Qwen2.5-3B KV cache 大小参考 (173 tokens):
#   ~6.4 MB FP16 → 压缩阈值 bw < 25.6 Mbps (budget = bw_mbps/8 * 2.0s)
#
# constant_low / constant_very_low: 全程受限，probe 测量受限带宽，ABKT 压缩
# mid_drop: pre-tc 高带宽 → FP16 决策，传输中骤降 → sender 检测 BW drop 降级

SCENARIOS = {
    "baseline": TestScenario(
        name="baseline",
        description="无网络限制 (对照组)",
        pre_tc_mbps=0,
    ),

    "constant_low": TestScenario(
        name="constant_low",
        description="全程低带宽: 10 Mbps → ABKT 压缩",
        pre_tc_mbps=10,
    ),

    "constant_very_low": TestScenario(
        name="constant_very_low",
        description="全程极低带宽: 2 Mbps → 极端压缩",
        pre_tc_mbps=2,
    ),

    "mid_drop": TestScenario(
        name="mid_drop",
        description="中途骤降: 100→2 Mbps → FP16 再降级",
        pre_tc_mbps=100,       # probe ~12.5 MB/s, budget=25 MB > 6.4 MB → FP16
        transfer_tc_mbps=2,    # 传输中降到 2 Mbps → sender 检测 BW drop
    ),

    "mid_drop_severe": TestScenario(
        name="mid_drop_severe",
        description="中途骤降 (严重): 100→1 Mbps",
        pre_tc_mbps=100,
        transfer_tc_mbps=1,
    ),

    "gradual": TestScenario(
        name="gradual",
        description="渐进退化: 50→10 Mbps",
        pre_tc_mbps=50,
        transfer_tc_mbps=10,
    ),

    "with_delay": TestScenario(
        name="with_delay",
        description="带宽+延迟: 20 Mbps + 20ms 延迟",
        pre_tc_mbps=20,
        delay_ms=20,
    ),

    "with_loss": TestScenario(
        name="with_loss",
        description="带宽+丢包: 30 Mbps + 5% 丢包",
        pre_tc_mbps=30,
        loss_pct=5,
    ),

    "oscillating": TestScenario(
        name="oscillating",
        description="振荡: 50→5→50→5 Mbps (每 2s 切换)",
        pre_tc_mbps=50,
        oscillate=True,
    ),
}


# ════════════════════════════════════════════════════════════════════
# SSH / tc 操作
# ════════════════════════════════════════════════════════════════════


def ssh_cmd(host: str, command: str, user: str = PREFILL_USER,
            timeout: int = 10) -> subprocess.CompletedProcess:
    """执行远程 SSH 命令。"""
    full_cmd = [
        "ssh", "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}", command,
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)


def tc_apply(host: str, bw_mbps: float, delay_ms: float = 0,
             loss_pct: float = 0, iface: str = PREFILL_IFACE) -> bool:
    """在远程节点施加 tc 规则。"""
    tc_clear(host, iface)

    if bw_mbps <= 0 and delay_ms <= 0 and loss_pct <= 0:
        return True

    cmds = []

    if bw_mbps > 0:
        rate_kbit = max(1, int(bw_mbps * 1000))
        burst = max(16, rate_kbit // 100)
        cmds.append(f"sudo tc qdisc add dev {iface} root handle 1: htb default 10")
        cmds.append(f"sudo tc class add dev {iface} parent 1: classid 1:10 "
                    f"htb rate {rate_kbit}kbit burst {burst}kbit")

        netem_parts = []
        if delay_ms > 0:
            jitter = max(1, delay_ms * 0.2)
            netem_parts.append(f"delay {delay_ms:.0f}ms {jitter:.0f}ms")
        if loss_pct > 0:
            netem_parts.append(f"loss {loss_pct:.1f}%")
        if netem_parts:
            cmds.append(f"sudo tc qdisc add dev {iface} parent 1:10 handle 10: "
                        f"netem {' '.join(netem_parts)}")
    elif delay_ms > 0 or loss_pct > 0:
        netem_parts = []
        if delay_ms > 0:
            jitter = max(1, delay_ms * 0.2)
            netem_parts.append(f"delay {delay_ms:.0f}ms {jitter:.0f}ms")
        if loss_pct > 0:
            netem_parts.append(f"loss {loss_pct:.1f}%")
        cmds.append(f"sudo tc qdisc add dev {iface} root "
                    f"netem {' '.join(netem_parts)}")

    for cmd in cmds:
        r = ssh_cmd(host, cmd)
        if r.returncode != 0:
            print(f"  [WARN] tc 命令失败: {cmd}")
            print(f"         stderr: {r.stderr.strip()}")
            return False
    return True


def tc_update_bw(host: str, bw_mbps: float, iface: str = PREFILL_IFACE) -> bool:
    """仅更新带宽 (保留 netem)。"""
    if bw_mbps <= 0:
        return tc_clear(host, iface)
    rate_kbit = max(1, int(bw_mbps * 1000))
    burst = max(16, rate_kbit // 100)
    r = ssh_cmd(host, f"sudo tc class replace dev {iface} parent 1: classid 1:10 "
                       f"htb rate {rate_kbit}kbit burst {burst}kbit")
    return r.returncode == 0


def tc_clear(host: str, iface: str = PREFILL_IFACE) -> bool:
    """清除远程节点的所有 tc 规则。"""
    ssh_cmd(host, f"sudo tc qdisc del dev {iface} root 2>/dev/null || true")
    return True


# ════════════════════════════════════════════════════════════════════
# OscillationThread — 传输中振荡带宽
# ════════════════════════════════════════════════════════════════════


class OscillationThread(threading.Thread):
    """在传输过程中周期性切换带宽。

    用法:
        osc = OscillationThread("192.168.0.50", 50, 5, interval=2.0)
        osc.start()
        # ... 传输进行中 ...
        osc.stop()
    """

    def __init__(self, host: str, bw_high: float, bw_low: float,
                 interval: float = 2.0, iface: str = PREFILL_IFACE):
        super().__init__(daemon=True)
        self.host = host
        self.bw_high = bw_high
        self.bw_low = bw_low
        self.interval = interval
        self.iface = iface
        self._stop_event = threading.Event()
        self.phase = 0

    def run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if self._stop_event.is_set():
                break
            self.phase += 1
            new_bw = self.bw_low if self.phase % 2 == 1 else self.bw_high
            print(f"\n  >>> 振荡: 切换到 {new_bw} Mbps (phase {self.phase})")
            tc_update_bw(self.host, new_bw, self.iface)

    def stop(self):
        self._stop_event.set()


# ════════════════════════════════════════════════════════════════════
# verify_decode_node — 检查 decode 节点是否就绪
# ════════════════════════════════════════════════════════════════════


def verify_decode_node(host: str, port: int) -> bool:
    """检查 decode 节点是否在监听。"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        s.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def restart_decode_node(host: str, model: str, port: int = DECODE_PORT,
                        user: str = PREFILL_USER, timeout: int = 60) -> bool:
    """重启 decode 节点，加载指定模型。"""
    print(f"  [decode] 重启 decode 节点: {model} (port {port})...")
    ssh_cmd(host, "pkill -f 'python3.*decode_node' 2>/dev/null || true", user=user)
    time.sleep(1)

    ssh_cmd(host,
        f"nohup python3 /ssd/pd/ABKT/decode_node.py "
        f"--model-name {model} --port {port} "
        f"> /tmp/decode_node.log 2>&1 &",
        user=user)

    # 等待 decode 节点就绪
    t0 = time.time()
    while time.time() - t0 < timeout:
        if verify_decode_node(host, port):
            print(f"  [decode] 就绪 ({time.time()-t0:.1f}s)")
            return True
        time.sleep(2)
    print(f"  [decode] 超时 ({timeout}s)")
    return False


# ════════════════════════════════════════════════════════════════════
# 结果解析
# ════════════════════════════════════════════════════════════════════

@dataclass
class TransferResult:
    """单次传输的结果。"""
    scenario: str = ""
    prefill_time: float = 0
    chunk_send_time: float = 0
    decode_time: float = 0
    total_time: float = 0
    num_tokens: int = 0
    tok_per_sec: float = 0
    generated_text: str = ""
    abkt_state: str = ""
    abkt_bw: float = 0          # MB/s
    abkt_budget: float = 0      # MB
    abkt_compression: float = 0
    abkt_avg_bits: float = 0
    abkt_bytes: float = 0       # MB
    abkt_decision: str = ""     # ABKT DECISION 诊断行
    num_layers: int = 0
    fp16_layers: int = 0
    fp8_layers: int = 0
    int4_layers: int = 0
    int2_layers: int = 0
    raw_output: str = ""


def parse_result(output: str, scenario: str = "") -> TransferResult:
    """从 prefill_node.py 的 stdout 中解析结果。"""
    r = TransferResult(scenario=scenario, raw_output=output)

    # 提取 generated_text from [RESULT] dict line (most reliable)
    result_dict_match = re.search(r"\[RESULT\]\s*(\{.*\})", output)
    if result_dict_match:
        dict_str = result_dict_match.group(1)
        gt = re.search(r"'generated_text':\s*'(.*?)'\s*'", dict_str, re.DOTALL)
        if gt:
            r.generated_text = gt.group(1).replace("\\n", "\n").strip()

    # Fallback: 提取 [RESULT] 和 === 之间的内容
    if not r.generated_text:
        result_match = re.search(
            r"\[RESULT\]\s*Generated text:\s*\n=+\s*\n(.*?)\n=+",
            output, re.DOTALL
        )
        if result_match:
            raw = result_match.group(1).strip()
            inner = re.search(r"'generated_text':\s*'(.*?)'", raw, re.DOTALL)
            if inner:
                r.generated_text = inner.group(1).replace("\\n", "\n").strip()
            else:
                r.generated_text = raw

    for line in output.split("\n"):
        line_s = line.strip()

        # 预填充时间
        m = re.search(r"Prefill complete in ([\d.]+)s", line_s)
        if m:
            r.prefill_time = float(m.group(1))

        # chunk 发送时间
        m = re.search(r"All chunks sent in ([\d.]+)s", line_s)
        if m:
            r.chunk_send_time = float(m.group(1))

        # ABKT 状态
        m = re.search(r"state=(\w+)\s+bw=([\d.]+)\s+MB/s\s+budget=([\d.]+)\s+MB", line_s)
        if m:
            r.abkt_state = m.group(1)
            r.abkt_bw = float(m.group(2))
            r.abkt_budget = float(m.group(3))

        # ABKT 压缩信息
        m = re.search(r"avg_bits=([\d.]+)\s+compression=([\d.]+)x\s+bytes=([\d.]+)\s+MB", line_s)
        if m:
            r.abkt_avg_bits = float(m.group(1))
            r.abkt_compression = float(m.group(2))
            r.abkt_bytes = float(m.group(3))

        # ABKT DECISION 诊断行
        if "ABKT DECISION" in line_s:
            r.abkt_decision = line_s

        # Generated N tokens in Xs (Y tok/s) — from prefill result section
        m = re.search(r"Generated (\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s", line_s)
        if m:
            r.num_tokens = int(m.group(1))
            r.decode_time = float(m.group(2))
            r.tok_per_sec = float(m.group(3))

        # 解码速度 (from decode_node output)
        m = re.search(r"(\d+)\s+tokens?\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s", line_s)
        if m:
            r.num_tokens = int(m.group(1))
            r.decode_time = float(m.group(2))
            r.tok_per_sec = float(m.group(3))

        # 解码信息 (from result dict)
        m = re.search(r"'num_tokens':\s*(\d+)", line_s)
        if m:
            r.num_tokens = int(m.group(1))
        m = re.search(r"'time':\s*([\d.]+)", line_s)
        if m:
            r.decode_time = float(m.group(1))

        # 传输总时间
        m = re.search(r"KV transfer: ([\d.]+)s, decode: ([\d.]+)s, total: ([\d.]+)s", line_s)
        if m:
            r.chunk_send_time = float(m.group(1))
            r.decode_time = float(m.group(2))
            r.total_time = float(m.group(3))
        else:
            m = re.search(r"Transfer complete in ([\d.]+)s", line_s)
            if m:
                r.total_time = float(m.group(1))

        # 层数
        m = re.search(r"num_layers[=:]\s*(\d+)", line_s)
        if m:
            r.num_layers = int(m.group(1))

    # 从 chunk 大小推断精度分布
    chunk_sizes = re.findall(r"(\d+\.\d+)KB", output)
    for sz in chunk_sizes:
        sz_f = float(sz)
        if sz_f > 12:
            r.fp16_layers += 1
        elif sz_f > 6:
            r.fp8_layers += 1
        elif sz_f > 3:
            r.int4_layers += 1
        else:
            r.int2_layers += 1

    return r


def print_result(result: TransferResult) -> None:
    """格式化打印结果。"""
    if result.tok_per_sec == 0 and result.num_tokens > 0 and result.decode_time > 0:
        result.tok_per_sec = result.num_tokens / result.decode_time

    print(f"\n{'─' * 60}")
    print(f"  场景: {result.scenario}")
    print(f"{'─' * 60}")
    print(f"  预填充:      {result.prefill_time:.3f}s")
    print(f"  KV 传输:     {result.chunk_send_time:.3f}s")
    print(f"  解码:        {result.decode_time:.1f}s  "
          f"({result.num_tokens} tokens, {result.tok_per_sec:.1f} tok/s)")
    print(f"  总传输:      {result.total_time:.2f}s")
    if result.abkt_decision:
        print(f"  {result.abkt_decision}")
    if result.abkt_state:
        print(f"  ABKT 状态:   {result.abkt_state}")
        print(f"  ABKT 带宽:   {result.abkt_bw:.1f} MB/s")
        print(f"  ABKT 预算:   {result.abkt_budget:.1f} MB")
        print(f"  ABKT 压缩:   {result.abkt_compression:.1f}x "
              f"(avg {result.abkt_avg_bits:.1f} bits)")
        print(f"  实际传输:    {result.abkt_bytes:.1f} MB")
    if result.fp16_layers or result.fp8_layers:
        parts = []
        if result.fp16_layers:
            parts.append(f"FP16={result.fp16_layers}")
        if result.fp8_layers:
            parts.append(f"FP8={result.fp8_layers}")
        if result.int4_layers:
            parts.append(f"INT4={result.int4_layers}")
        if result.int2_layers:
            parts.append(f"INT2={result.int2_layers}")
        print(f"  精度分布:    {' '.join(parts)}")
    if result.generated_text:
        print(f"  {'─' * 56}")
        print(f"  生成文本:")
        for text_line in result.generated_text.split("\n"):
            print(f"    {text_line}")
    print(f"{'─' * 60}")


def print_comparison(baseline: Optional[TransferResult],
                     test: TransferResult) -> None:
    """打印对比结果。"""
    if baseline is None:
        return

    print(f"\n{'═' * 60}")
    print(f"  对比: baseline vs {test.scenario}")
    print(f"{'═' * 60}")

    def _delta(a, b, lower_better=True):
        if a == 0:
            return ""
        diff = (b - a) / a * 100
        arrow = "↓" if (diff < 0 and lower_better) or (diff > 0 and not lower_better) else "↑"
        return f" ({arrow}{abs(diff):.0f}%)"

    print(f"  {'指标':<16} {'baseline':>12} {test.scenario:>12}  {'变化':>10}")
    print(f"  {'─' * 52}")

    bv, tv = baseline.chunk_send_time, test.chunk_send_time
    print(f"  {'KV传输时间':<14} {bv:>10.3f}s {tv:>10.3f}s  {_delta(bv, tv)}")

    bv, tv = baseline.total_time, test.total_time
    print(f"  {'总传输时间':<14} {bv:>10.2f}s {tv:>10.2f}s  {_delta(bv, tv)}")

    bv, tv = baseline.tok_per_sec, test.tok_per_sec
    print(f"  {'解码速度':<14} {bv:>10.1f}  {tv:>10.1f}   {_delta(bv, tv, lower_better=False)}")

    bv, tv = baseline.abkt_compression, test.abkt_compression
    print(f"  {'压缩比':<14} {bv:>10.1f}x {tv:>10.1f}x  {_delta(bv, tv, lower_better=False)}")

    bv, tv = baseline.abkt_avg_bits, test.abkt_avg_bits
    print(f"  {'平均精度':<14} {bv:>10.1f}b {tv:>10.1f}b  {_delta(bv, tv, lower_better=False)}")

    bv, tv = baseline.abkt_bytes, test.abkt_bytes
    print(f"  {'传输字节':<14} {bv:>10.1f}MB {tv:>10.1f}MB  {_delta(bv, tv)}")

    print(f"{'═' * 60}")


# ════════════════════════════════════════════════════════════════════
# 核心: 运行单次测试
# ════════════════════════════════════════════════════════════════════


def run_single_test(
    model: str,
    scenario: TestScenario,
    prompt: str = "The capital of France is",
    max_tokens: int = 128,
    sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    decode_host: str = DECODE_HOST,
    decode_port: int = DECODE_PORT,
) -> TransferResult:
    """运行单次 ABKT 测试。

    关键: tc 规则在 prefill_node.py 启动前施加，确保 bandwidth probe
    测量到受限带宽。
    """

    actual_model = scenario.model_override if scenario.model_override else model

    print(f"\n{'═' * 60}")
    print(f"  测试场景: {scenario.name}")
    print(f"  描述: {scenario.description}")
    print(f"  模型: {actual_model}")
    if scenario.pre_tc_mbps > 0:
        print(f"  预施加带宽: {scenario.pre_tc_mbps} Mbps")
    if scenario.transfer_tc_mbps > 0:
        print(f"  传输中带宽: {scenario.transfer_tc_mbps} Mbps")
    print(f"{'═' * 60}")

    # ── Step 1: 清除旧 tc 规则 ──
    print("  [1/5] 清除 prefill 节点 tc 规则...")
    tc_clear(PREFILL_HOST)
    time.sleep(0.3)

    # ── Step 2: 检查/重启 decode 节点 ──
    print(f"  [2/5] 检查 decode 节点 {decode_host}:{decode_port}...")
    if scenario.model_override:
        # 模型不同时需要重启 decode 节点
        if not restart_decode_node(decode_host, actual_model, decode_port):
            return TransferResult(scenario=scenario.name,
                                  raw_output="ERROR: decode node restart failed")
    elif not verify_decode_node(decode_host, decode_port):
        print(f"  [ERROR] decode 节点 {decode_host}:{decode_port} 未就绪!")
        print(f"          请先在 decode 节点运行: python3 {ABKT_DIR}/decode_node.py")
        return TransferResult(scenario=scenario.name,
                              raw_output="ERROR: decode node not ready")

    # ── Step 3: 施加 tc 规则 (在 prefill 启动前!) ──
    if scenario.pre_tc_mbps > 0 or scenario.delay_ms > 0 or scenario.loss_pct > 0:
        print(f"  [3/5] 施加 tc: {scenario.pre_tc_mbps} Mbps"
              f" + {scenario.delay_ms}ms delay + {scenario.loss_pct}% loss")
        tc_apply(PREFILL_HOST, scenario.pre_tc_mbps,
                 scenario.delay_ms, scenario.loss_pct)
    else:
        print("  [3/5] 无带宽限制")

    # ── Step 4: 构建 prefill 命令 ──
    sample_args = ""
    if sample:
        sample_args = f" --do-sample --temperature {temperature}"
        if top_k > 0:
            sample_args += f" --top-k {top_k}"
        if top_p < 1.0:
            sample_args += f" --top-p {top_p}"

    cmd = (
        f"cd {ABKT_DIR} && python3 prefill_node.py"
        f" --model-name {actual_model}"
        f" --decode-host {decode_host}"
        f" --decode-port {decode_port}"
        f" --prompt '{prompt}'"
        f" --max-new-tokens {max_tokens}"
        f"{sample_args}"
    )

    # ── Step 5: 启动 prefill，实时监控输出 ──
    print(f"  [4/5] 启动 prefill 节点...")
    print(f"  命令: {cmd[:80]}...")

    ssh_process = subprocess.Popen(
        ["ssh", "-o", "ConnectTimeout=3",
         "-o", "StrictHostKeyChecking=no",
         f"{PREFILL_USER}@{PREFILL_HOST}", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    transfer_started = False
    transfer_ended = False
    oscillation_thread: Optional[OscillationThread] = None

    try:
        for line in iter(ssh_process.stdout.readline, ""):
            output_lines.append(line)
            line_s = line.strip()

            # 实时打印关键行
            skip_keywords = ["Loading weights", "Materializing", "it/s]",
                             "━", "╸", "| "]
            if line_s and not any(kw in line_s for kw in skip_keywords):
                print(f"  │ {line_s}")

            # ── 检测 KV 传输开始 ──
            if not transfer_started and (
                "Starting chunked transfer" in line_s
                or "using legacy FP16 transfer" in line_s
                or "Budget ample" in line_s
            ):
                transfer_started = True
                print(f"\n  >>> 检测到传输开始!")

                # 施加传输中带宽限制
                if scenario.transfer_tc_mbps > 0:
                    print(f"  >>> 切换带宽: {scenario.transfer_tc_mbps} Mbps")
                    tc_update_bw(PREFILL_HOST, scenario.transfer_tc_mbps)

                # 启动振荡线程
                if scenario.oscillate:
                    oscillation_thread = OscillationThread(
                        PREFILL_HOST, scenario.pre_tc_mbps, 5.0, interval=2.0
                    )
                    oscillation_thread.start()

            # ── 检测传输结束 ──
            if "Transfer complete" in line_s:
                transfer_ended = True
                if oscillation_thread:
                    oscillation_thread.stop()
                    oscillation_thread = None
                print("\n  >>> 传输结束, 清除 tc 规则")
                tc_clear(PREFILL_HOST)

        ssh_process.wait()

    except KeyboardInterrupt:
        print("\n  [INTERRUPT] 中断测试...")
        ssh_process.kill()
    finally:
        if oscillation_thread:
            oscillation_thread.stop()
        tc_clear(PREFILL_HOST)

    # ── 解析结果 ──
    print(f"  [5/5] 解析结果...")
    full_output = "".join(output_lines)
    result = parse_result(full_output, scenario.name)
    print_result(result)

    return result


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ABKT 测试编排器 — 在 prefill 启动前施加 tc，精确控制带宽",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
场景说明:
  baseline           无限制 (对照组)
  constant_low       全程 10 Mbps → ABKT 压缩
  constant_very_low  全程 2 Mbps → 极端压缩
  mid_drop           100→2 Mbps (中途骤降)
  mid_drop_severe    100→1 Mbps (严重骤降)
  gradual            50→10 Mbps
  with_delay         20 Mbps + 20ms 延迟
  with_loss          30 Mbps + 5% 丢包
  oscillating        50→5→50→5 Mbps (每 2s 振荡)
  custom             自定义 (需指定 --pre-tc / --transfer-tc)

Qwen2.5-3B KV cache (173 tokens): ~6.4 MB
  压缩阈值: bw < 25.6 Mbps (budget = bw_mbps/8 * 2.0s)

示例:
  # 基线测试:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario baseline

  # 低带宽压缩:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario constant_low

  # 中途骤降:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario mid_drop

  # 对比:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --compare baseline constant_low

  # 从文件读取 prompt:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario constant_low \\
      --txt-prompt /tmp/prompt_2k.txt

  # 自定义:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario custom \\
      --pre-tc 10 --transfer-tc 5
""",
    )

    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--scenario", default=None,
                        choices=list(SCENARIOS.keys()),
                        help="测试场景")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "TEST"),
                        help="对比两个场景")
    parser.add_argument("--all", action="store_true",
                        help="运行所有场景并对比")
    parser.add_argument("--prompt", default=None,
                        help="输入 prompt")
    parser.add_argument("--prompt-file", "--txt-prompt", default=None,
                        help="从 txt 文件读取 prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--decode-host", default=DECODE_HOST)
    parser.add_argument("--decode-port", type=int, default=DECODE_PORT)

    # 自定义场景参数
    parser.add_argument("--pre-tc", type=float, default=0,
                        help="自定义: prefill 启动前带宽 Mbps")
    parser.add_argument("--transfer-tc", type=float, default=0,
                        help="自定义: 传输中带宽 Mbps")
    parser.add_argument("--delay", type=float, default=0,
                        help="自定义: 延迟 ms")
    parser.add_argument("--loss", type=float, default=0,
                        help="自定义: 丢包百分比")

    args = parser.parse_args()

    # 信号处理
    def cleanup(signum=None, frame=None):
        print("\n[CLEANUP] 清除 tc 规则...")
        tc_clear(PREFILL_HOST)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 检查 SSH 连通性
    print("[CHECK] 测试 SSH 连接到 prefill 节点...")
    r = ssh_cmd(PREFILL_HOST, "echo ok", timeout=5)
    if r.returncode != 0 or "ok" not in r.stdout:
        print(f"[ERROR] 无法连接到 {PREFILL_USER}@{PREFILL_HOST}")
        sys.exit(1)
    print(f"[OK] {PREFILL_HOST} 可达")

    # 处理 prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read().strip()
        print(f"[INFO] 从文件读取 prompt: {args.prompt_file} ({len(prompt)} chars)")
    elif prompt is None:
        prompt = (
            "请详细撰写一篇关于人工智能发展历史的长文，涵盖以下内容："
            "1) 1950年代图灵测试和达特茅斯会议的起源；"
            "2) 1960-1970年代专家系统的兴起与第一次AI寒冬；"
            "3) 1980年代神经网络的复兴与反向传播算法；"
            "4) 1990年代支持向量机和统计学习方法；"
            "5) 2000年代深度学习的突破，包括卷积神经网络和循环神经网络；"
            "6) 2010年代Transformer架构和大语言模型的革命；"
            "7) 2020年代GPT、Claude等模型的能力与挑战；"
            "8) 未来人工智能的发展方向和伦理考量。"
            "请尽可能详细，每个时代至少写三段。"
        )
    print(f"[INFO] Prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}' "
          f"({len(prompt)} chars)")

    test_kwargs = dict(
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        decode_host=args.decode_host,
        decode_port=args.decode_port,
    )

    # ── 对比模式 ──
    if args.compare:
        s1_name, s2_name = args.compare
        s1 = SCENARIOS[s1_name]
        s2 = SCENARIOS[s2_name]

        print(f"\n{'═' * 60}")
        print(f"  对比测试: {s1_name} vs {s2_name}")
        print(f"{'═' * 60}")

        r1 = run_single_test(scenario=s1, **test_kwargs)
        time.sleep(2)
        r2 = run_single_test(scenario=s2, **test_kwargs)
        print_comparison(r1, r2)
        return

    # ── 全场景测试 ──
    if args.all:
        results = {}
        for name in ["baseline", "constant_low", "mid_drop",
                      "gradual", "with_loss", "oscillating"]:
            sc = SCENARIOS[name]
            r = run_single_test(scenario=sc, **test_kwargs)
            results[name] = r
            time.sleep(3)

        print(f"\n{'═' * 70}")
        print(f"  汇总对比")
        print(f"{'═' * 70}")
        print(f"  {'场景':<18} {'传输时间':>10} {'压缩比':>8} {'精度':>8} "
              f"{'状态':>10} {'tok/s':>8}")
        print(f"  {'─' * 62}")
        for name, r in results.items():
            print(f"  {name:<18} {r.chunk_send_time:>8.3f}s "
                  f"{r.abkt_compression:>7.1f}x {r.abkt_avg_bits:>7.1f}b "
                  f"{r.abkt_state:>10} {r.tok_per_sec:>7.1f}")
        print(f"{'═' * 70}")
        return

    # ── 单场景测试 ──
    if args.scenario is None:
        print("[ERROR] 请指定 --scenario, --compare, 或 --all")
        parser.print_help()
        sys.exit(1)

    scenario = SCENARIOS[args.scenario]

    # 自定义参数覆盖
    if args.scenario == "custom":
        scenario = TestScenario(
            name="custom",
            description=f"自定义: pre={args.pre_tc}Mbps, "
                        f"transfer={args.transfer_tc}Mbps, "
                        f"delay={args.delay}ms, loss={args.loss}%",
            pre_tc_mbps=args.pre_tc,
            transfer_tc_mbps=args.transfer_tc,
            delay_ms=args.delay,
            loss_pct=args.loss,
        )

    run_single_test(scenario=scenario, **test_kwargs)


if __name__ == "__main__":
    main()
