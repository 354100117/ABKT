#!/usr/bin/env python3
"""ABKT 测试编排器 — 精确控制网络变化与 KV cache 传输的时序。

解决的核心问题: tc 网络模拟和 ABKT 推理独立运行，很难让带宽变化
恰好发生在 KV cache 传输过程中。

原理:
  1. 通过 SSH 在 prefill 节点 (192.168.0.50) 上运行 prefill_node.py
  2. 实时监控 prefill 的 stdout，检测 KV 传输的开始/结束
  3. 在传输开始的瞬间通过 SSH 施加 tc 带宽限制
  4. 传输结束后清除 tc 规则
  5. 收集并对比结果

关键: tc 规则施加在 prefill 节点的 eno1 上 (egress)，直接限制
KV cache 数据发往 decode 节点的带宽。

用法:
  # 基本测试 (中途骤降):
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b

  # 指定场景:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario mid_drop
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario constant_low
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario gradual

  # 自定义带宽:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario custom \\
      --bw-before 100 --bw-during 5 --bw-after 100

  # 完整 ABKT 测试 (含采样):
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b \\
      --prompt "请详细解释量子计算的基本原理" --max-tokens 300 --sample

  # 对比测试 (无网络限制):
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario baseline
"""

import argparse
import os
import re
import signal
import subprocess
import sys
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
    """测试场景配置。"""
    name: str
    description: str
    # 传输前带宽 (Mbps), 0=不限制
    bw_before_mbps: float = 0
    # 传输中带宽 (Mbps), 0=不限制
    bw_during_mbps: float = 0
    # 传输后带宽 (Mbps), 0=不限制
    bw_after_mbps: float = 0
    # 延迟 (ms)
    delay_ms: float = 0
    # 丢包 (%)
    loss_pct: float = 0
    # 是否等待传输中再降速 (False=提前施加)
    wait_for_transfer: bool = True


SCENARIOS = {
    "baseline": TestScenario(
        name="baseline",
        description="无网络限制 (对照组)",
        bw_before_mbps=0, bw_during_mbps=0, bw_after_mbps=0,
        wait_for_transfer=False,
    ),

    "mid_drop": TestScenario(
        name="mid_drop",
        description="中途骤降: 正常→传输中降至 5Mbps→恢复",
        bw_before_mbps=0,      # 不限速 (让 prefill 和 probe 正常运行)
        bw_during_mbps=5,      # KV 传输时降到 5 Mbps
        bw_after_mbps=0,       # 恢复
        wait_for_transfer=True,  # 等传输开始再降速
    ),

    "mid_drop_severe": TestScenario(
        name="mid_drop_severe",
        description="中途骤降 (严重): 正常→传输中降至 1Mbps",
        bw_before_mbps=0,
        bw_during_mbps=1,
        bw_after_mbps=0,
        wait_for_transfer=True,
    ),

    "constant_low": TestScenario(
        name="constant_low",
        description="全程低带宽: 10 Mbps",
        bw_before_mbps=10,
        bw_during_mbps=10,
        bw_after_mbps=10,
        wait_for_transfer=False,
    ),

    "constant_very_low": TestScenario(
        name="constant_very_low",
        description="全程极低带宽: 2 Mbps",
        bw_before_mbps=2,
        bw_during_mbps=2,
        bw_after_mbps=2,
        wait_for_transfer=False,
    ),

    "gradual": TestScenario(
        name="gradual",
        description="渐进退化: 传输前 50→传输中 10→传输后恢复",
        bw_before_mbps=50,
        bw_during_mbps=10,
        bw_after_mbps=0,
        wait_for_transfer=True,
    ),

    "with_delay": TestScenario(
        name="with_delay",
        description="带宽+延迟: 20 Mbps + 20ms 延迟",
        bw_before_mbps=0,
        bw_during_mbps=20,
        bw_after_mbps=0,
        delay_ms=20,
        wait_for_transfer=True,
    ),

    "with_loss": TestScenario(
        name="with_loss",
        description="带宽+丢包: 30 Mbps + 5% 丢包",
        bw_before_mbps=0,
        bw_during_mbps=30,
        bw_after_mbps=0,
        loss_pct=5,
        wait_for_transfer=True,
    ),

    "oscillating": TestScenario(
        name="oscillating",
        description="传输中振荡: 50→5→50→5 Mbps (每 2s 切换)",
        bw_before_mbps=0,
        bw_during_mbps=50,  # 初始值，实际由 oscillate 控制
        bw_after_mbps=0,
        wait_for_transfer=True,
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
    # 先清除
    ssh_cmd(host, f"sudo tc qdisc del dev {iface} root 2>/dev/null || true")

    if bw_mbps <= 0 and delay_ms <= 0 and loss_pct <= 0:
        return True  # 不需要限制

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
        # 无带宽限制，只有延迟/丢包
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
    r = ssh_cmd(host, f"sudo tc qdisc del dev {iface} root 2>/dev/null || true")
    return True


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
    num_layers: int = 0
    fp16_layers: int = 0
    fp8_layers: int = 0
    int4_layers: int = 0
    int2_layers: int = 0
    raw_output: str = ""


def parse_result(output: str, scenario: str = "") -> TransferResult:
    """从 prefill_node.py 的 stdout 中解析结果。"""
    r = TransferResult(scenario=scenario, raw_output=output)

    # 提取生成文本 (在 [RESULT] 和 === 之间的内容)
    result_match = re.search(
        r"\[RESULT\]\s*Generated text:\s*\n=+\s*\n(.*?)\n=+",
        output, re.DOTALL
    )
    if result_match:
        r.generated_text = result_match.group(1).strip()

    for line in output.split("\n"):
        line = line.strip()

        # 预填充时间
        m = re.search(r"Prefill complete in ([\d.]+)s", line)
        if m:
            r.prefill_time = float(m.group(1))

        # chunk 发送时间
        m = re.search(r"All chunks sent in ([\d.]+)s", line)
        if m:
            r.chunk_send_time = float(m.group(1))

        # ABKT 状态
        m = re.search(r"state=(\w+)\s+bw=([\d.]+)\s+MB/s\s+budget=([\d.]+)\s+MB", line)
        if m:
            r.abkt_state = m.group(1)
            r.abkt_bw = float(m.group(2))
            r.abkt_budget = float(m.group(3))

        # ABKT 压缩信息
        m = re.search(r"avg_bits=([\d.]+)\s+compression=([\d.]+)x\s+bytes=([\d.]+)\s+MB", line)
        if m:
            r.abkt_avg_bits = float(m.group(1))
            r.abkt_compression = float(m.group(2))
            r.abkt_bytes = float(m.group(3))

        # 解码速度 (from decode_node output)
        m = re.search(r"(\d+)\s+tokens?\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s", line)
        if m:
            r.num_tokens = int(m.group(1))
            r.decode_time = float(m.group(2))
            r.tok_per_sec = float(m.group(3))

        # 解码信息 (from result dict: 'num_tokens': N, 'time': X.XX)
        m = re.search(r"'num_tokens':\s*(\d+)", line)
        if m:
            r.num_tokens = int(m.group(1))
        m = re.search(r"'time':\s*([\d.]+)", line)
        if m:
            r.decode_time = float(m.group(1))

        # 传输总时间 (支持新旧两种格式)
        m = re.search(r"KV transfer: ([\d.]+)s, decode: ([\d.]+)s, total: ([\d.]+)s", line)
        if m:
            r.chunk_send_time = float(m.group(1))  # KV-only transfer time
            r.decode_time = float(m.group(2))       # decode time from report
            r.total_time = float(m.group(3))
        else:
            m = re.search(r"Transfer complete in ([\d.]+)s", line)
            if m:
                r.total_time = float(m.group(1))

        # 层数
        m = re.search(r"num_layers[=:]\s*(\d+)", line)
        if m:
            r.num_layers = int(m.group(1))

    # 从 chunk 大小推断精度分布
    fp16 = len(re.findall(r"\d+KB\b.*last=", output))  # rough
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
    # Auto-calculate tok_per_sec if not set
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

    def _delta(a, b, unit="", lower_better=True):
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
    """运行单次 ABKT 测试，精确控制网络时序。"""

    print(f"\n{'═' * 60}")
    print(f"  测试场景: {scenario.name}")
    print(f"  描述: {scenario.description}")
    print(f"{'═' * 60}")

    # ── Step 0: 确保 prefill 节点无 tc 规则 ──
    print("  [1/5] 清除 prefill 节点 tc 规则...")
    tc_clear(PREFILL_HOST)
    time.sleep(0.5)

    # ── Step 1: 如果需要提前施加带宽限制 ──
    if not scenario.wait_for_transfer and scenario.bw_before_mbps > 0:
        print(f"  [2/5] 预施加带宽限制: {scenario.bw_before_mbps} Mbps")
        tc_apply(PREFILL_HOST, scenario.bw_before_mbps,
                 scenario.delay_ms, scenario.loss_pct)
    else:
        print("  [2/5] 无预限制 (等待传输开始)")

    # ── Step 2: 构建 prefill 命令 ──
    sample_args = ""
    if sample:
        sample_args = f" --do-sample --temperature {temperature}"
        if top_k > 0:
            sample_args += f" --top-k {top_k}"
        if top_p < 1.0:
            sample_args += f" --top-p {top_p}"

    cmd = (
        f"cd {ABKT_DIR} && python3 prefill_node.py"
        f" --model-name {model}"
        f" --decode-host {decode_host}"
        f" --decode-port {decode_port}"
        f" --prompt '{prompt}'"
        f" --max-new-tokens {max_tokens}"
        f"{sample_args}"
    )

    # ── Step 3: 启动 prefill，实时监控输出 ──
    print(f"  [3/5] 启动 prefill 节点...")
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
    oscillate_phase = 0
    oscillate_timer = time.time()

    try:
        for line in iter(ssh_process.stdout.readline, ""):
            output_lines.append(line)
            line_stripped = line.strip()

            # 实时打印关键行 (跳过进度条和空行)
            skip_keywords = ["Loading weights", "Materializing", "it/s]",
                             "━", "╸", "| "]
            if line_stripped and not any(kw in line_stripped for kw in skip_keywords):
                print(f"  │ {line_stripped}")

            # ── 检测 KV 传输开始 ──
            # ABKT 路径: "Starting chunked transfer"
            # Legacy 路径: "Budget ample" 或 "using legacy FP16 transfer"
            # 通用: "Connected to" (socket 连接建立 = 传输即将开始)
            if ("Starting chunked transfer" in line_stripped
                    or "using legacy FP16 transfer" in line_stripped
                    or "Budget ample" in line_stripped
                    or "Connected to" in line_stripped):
                if not transfer_started and scenario.wait_for_transfer:
                    transfer_started = True
                    if scenario.bw_during_mbps > 0:
                        print(f"\n  >>> 检测到传输开始! 施加带宽限制: "
                              f"{scenario.bw_during_mbps} Mbps")
                        tc_apply(PREFILL_HOST, scenario.bw_during_mbps,
                                 scenario.delay_ms, scenario.loss_pct)
                        oscillate_timer = time.time()

            # ── 振荡场景: 传输中交替带宽 ──
            if (scenario.name == "oscillating" and transfer_started
                    and not transfer_ended):
                if time.time() - oscillate_timer >= 2.0:
                    oscillate_phase += 1
                    new_bw = 50 if oscillate_phase % 2 == 0 else 5
                    print(f"\n  >>> 振荡: 切换到 {new_bw} Mbps")
                    tc_update_bw(PREFILL_HOST, new_bw)
                    oscillate_timer = time.time()

            # ── 检测传输结束 ──
            if "Transfer complete" in line_stripped:
                transfer_ended = True
                if scenario.wait_for_transfer and scenario.bw_after_mbps > 0:
                    print(f"\n  >>> 传输结束, 恢复带宽: {scenario.bw_after_mbps} Mbps")
                    tc_apply(PREFILL_HOST, scenario.bw_after_mbps)
                elif scenario.wait_for_transfer:
                    print("\n  >>> 传输结束, 清除 tc 规则")
                    tc_clear(PREFILL_HOST)

        ssh_process.wait()

    except KeyboardInterrupt:
        print("\n  [INTERRUPT] 中断测试...")
        ssh_process.kill()
    finally:
        # 确保清除 tc 规则
        tc_clear(PREFILL_HOST)

    # ── Step 4: 解析结果 ──
    full_output = "".join(output_lines)
    result = parse_result(full_output, scenario.name)
    print_result(result)

    return result


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ABKT 测试编排器 — 精确控制网络变化与 KV 传输时序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
场景说明:
  baseline          无限制 (对照组)
  mid_drop          中途骤降: 正常→传输中 5Mbps→恢复
  mid_drop_severe   中途骤降 (严重): 传输中 1Mbps
  constant_low      全程 10 Mbps
  constant_very_low 全程 2 Mbps
  gradual           渐进退化: 50→10→恢复
  with_delay        带宽+延迟: 20 Mbps + 20ms
  with_loss         带宽+丢包: 30 Mbps + 5%
  oscillating       传输中振荡: 50→5→50→5 (每2s)
  custom            自定义 (需指定 --bw-during)

示例:
  # 中途骤降测试:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario mid_drop

  # 对比测试 (baseline vs mid_drop):
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --compare baseline mid_drop

  # 自定义带宽:
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --scenario custom \\
      --bw-during 3

  # 完整测试 (所有场景):
  python3 test_orchestrator.py --model /ssd/models/qwen2.5-3b --all
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
                        help="输入 prompt (默认使用内置长 prompt)")
    parser.add_argument("--prompt-file", default=None,
                        help="从文件读取 prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--decode-host", default=DECODE_HOST)
    parser.add_argument("--decode-port", type=int, default=DECODE_PORT)

    # 自定义场景参数
    parser.add_argument("--bw-before", type=float, default=0,
                        help="自定义: 传输前带宽 Mbps")
    parser.add_argument("--bw-during", type=float, default=0,
                        help="自定义: 传输中带宽 Mbps")
    parser.add_argument("--bw-after", type=float, default=0,
                        help="自定义: 传输后带宽 Mbps")
    parser.add_argument("--delay", type=float, default=0,
                        help="自定义: 延迟 ms")
    parser.add_argument("--loss", type=float, default=0,
                        help="自定义: 丢包百分比")

    args = parser.parse_args()

    # 信号处理: 确保清除 tc
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
        print(f"        请确认 SSH 免密登录已配置")
        sys.exit(1)
    print(f"[OK] {PREFILL_HOST} 可达")

    # 检查 decode 节点
    print(f"[CHECK] 测试 decode 节点 {args.decode_host}:{args.decode_port}...")
    # (由 prefill_node.py 内部检查，这里只做提示)

    # 处理 prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read().strip()
        print(f"[INFO] 从文件读取 prompt: {args.prompt_file} ({len(prompt)} chars)")
    elif prompt is None:
        # 默认: 足够长的 prompt 以产生大的 KV cache (触发 ABKT 压缩)
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
        for name in ["baseline", "mid_drop", "mid_drop_severe",
                      "constant_low", "gradual", "with_loss"]:
            sc = SCENARIOS[name]
            r = run_single_test(scenario=sc, **test_kwargs)
            results[name] = r
            time.sleep(3)

        # 打印汇总
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
            description=f"自定义: during={args.bw_during}Mbps, "
                        f"delay={args.delay}ms, loss={args.loss}%",
            bw_before_mbps=args.bw_before,
            bw_during_mbps=args.bw_during,
            bw_after_mbps=args.bw_after,
            delay_ms=args.delay,
            loss_pct=args.loss,
            wait_for_transfer=(args.bw_during > 0),
        )

    run_single_test(scenario=scenario, **test_kwargs)


if __name__ == "__main__":
    main()
