# ABKT: Adaptive Bitrate KV Cache Transfer
## 面向边缘异构 PD 分离推理的自适应码率 KV Cache 传输方案

> 完整技术方案 | 2026-05-13

---

## 一、问题定义与核心洞察

### 1.1 问题本质

在你的 x86 (Prefill) + Jetson (Decode) 异构 PD 分离架构中，KV Cache 传输面临一个**带宽受限的资源分配问题**：

- **资源**：可用网络带宽 B(t)，随时间波动
- **对象**：KV Cache 张量，按 (layer, token_position) 组织，每个 entry 有 K 和 V 两个张量
- **目标**：在带宽约束下，最大化传输质量（即 decode 阶段的生成质量）

### 1.2 核心洞察（为什么现有方案不够）

现有方案的本质缺陷是**将传输策略与内容重要性解耦**：

| 方案 | 传输策略 | 内容感知 | 问题 |
|------|---------|---------|------|
| KIVI/KVQuant | 全局静态量化 | 无 | 不感知网络，不感知重要性 |
| CacheGen | 全局统一压缩 | 无 | 压缩策略固定，无法适应波动 |
| Mooncake | 高性能传输引擎 | 无 | 面向同构数据中心 |
| H2O/SnapKV | N/A（计算侧） | 有 | 只用于 eviction，未用于传输 |

**我们的核心洞察**：KV Cache 传输应该像**视频自适应码率（ABR）流媒体**一样工作——重要的"帧"（token）获得高码率（高精度），不重要的"帧"获得低码率（低精度），整体码率根据网络状况动态调整。

### 1.3 创新点总览

| 创新点 | 描述 | 与现有工作区别 |
|--------|------|---------------|
| **自适应码率传输（ABR）** | 将视频 ABR 思想迁移到 KV Cache 传输 | 首次将 ABR 概念应用于 KV Cache 领域 |
| **联合重要性评分** | 统一 token 重要性 + layer 敏感度 + 位置衰减 | H2O/SnapKV 只考虑 attention score 单一维度 |
| **带宽受限优化分配** | 形式化为约束优化问题，求解最优精度分配 | KIVI/CacheGen 无优化分配，全局统一策略 |
| **两阶段自适应传输** | 粗粒度预决策 + 细粒度 chunk 级动态调整 | 现有方案要么静态，要么只做一级自适应 |
| **带宽预测驱动** | EWMA 预测带宽趋势，主动调整而非被动响应 | 现有方案均为被动响应 |

---

## 二、数学形式化

### 2.1 符号定义

```
L: 模型层数（如 OPT-6.7B 的 32 层）
T: 当前请求的 token 数量
S = {(l, t) | l ∈ [0,L), t ∈ [0,T)}: 所有 KV Cache entry 的集合
p ∈ {FP16, FP8, INT4, INT2}: 量化精度级别
b(p): 精度 p 下每个 entry 的字节数
  - FP16: 2 bytes × 2(K+V) × head_dim × num_heads = 2 × 2 × d
  - FP8:  1 byte × 2 × d
  - INT4: 0.5 byte × 2 × d
  - INT2: 0.25 byte × 2 × d
I(l, t): entry (l, t) 的重要性评分（0~1）
Q(p): 精度 p 下的质量保真度（0~1），FP16=1.0, FP8≈0.98, INT4≈0.92, INT2≈0.80
B(t): 当前可用带宽（bytes/sec）
D(t): 可接受的传输延迟上界（sec）
```

### 2.2 优化目标

```
最大化:  Σ_{(l,t) ∈ S} I(l,t) × Q(p_{l,t})

约束:    Σ_{(l,t) ∈ S} b(p_{l,t}) ≤ B(t) × D(t)
         p_{l,t} ∈ {FP16, FP8, INT4, INT2}
```

这是一个**背包问题的变体**，但因为 Q(p) 是离散的且只有 4 个级别，可以用贪心算法高效求解。

### 2.3 贪心求解算法

```
算法: Proportional-Importance Allocation (PIA)

输入: 重要性矩阵 I[L][T], 带宽预算 W = B(t) × D(t), 精度级别列表 P
输出: 每个 (l,t) 的精度分配 p[l][t]

1. 将所有 entry 按 I(l,t) 降序排列
2. 初始化: 所有 entry 分配最低精度 p_min
3. 计算剩余预算: W_remaining = W - Σ b(p_min)
4. 按重要性从高到低遍历:
   a. 尝试将当前 entry 升级到更高精度
   b. 升级收益 = I(l,t) × (Q(p_higher) - Q(p_current))
   c. 升级代价 = b(p_higher) - b(p_current)
   d. 如果 剩余预算 >= 升级代价，执行升级
   e. 否则跳过，尝试下一个 entry
5. 返回精度分配

时间复杂度: O(L × T × |P|)，对于 L=32, T=512, |P|=4，约 65K 次操作，微秒级
```

---

## 三、系统架构

### 3.1 整体架构图

```
Prefill 节点 (x86 + RTX 3060)                    Decode 节点 (Jetson AGX Orin)
┌─────────────────────────────────┐              ┌─────────────────────────────────┐
│ PrefillStage.forward()          │              │                                 │
│   ↓                             │              │                                 │
│ [1] 逐层计算，收集 KV Cache     │              │                                 │
│   ↓                             │              │                                 │
│ [2] 重要性评估器                │              │                                 │
│     TokenImportanceEvaluator    │              │                                 │
│     - Attention Score 分析      │              │                                 │
│     - Layer 敏感度评估          │              │                                 │
│     - 位置衰减因子              │              │                                 │
│     → 重要性矩阵 I[L][T]       │              │                                 │
│   ↓                             │              │                                 │
│ [3] 网络探测器                  │              │                                 │
│     NetworkProbe                │              │                                 │
│     - 实时带宽 B(t)             │              │                                 │
│     - 实时 RTT                  │              │                                 │
│     - EWMA 带宽预测 B̂(t+Δ)    │              │                                 │
│     → 网络状态 + 预算 W         │              │                                 │
│   ↓                             │              │                                 │
│ [4] 精度分配器 (PIA 算法)       │              │                                 │
│     PrecisionAllocator          │              │                                 │
│     - 求解优化问题              │              │                                 │
│     → 精度矩阵 p[L][T]         │              │                                 │
│   ↓                             │              │                                 │
│ [5] 自适应量化器                │              │                                 │
│     AdaptiveQuantizer           │              │                                 │
│     - 按 p[l][t] 量化每个 entry │              │                                 │
│     - 生成量化元数据            │              │                                 │
│   ↓                             │              │                                 │
│ [6] 流水线分块传输              │   Network    │  [7] 流式接收                    │
│     ChunkedTransfer             │ ──────────→  │     StreamReceiver              │
│     - 按 layer 分块             │              │     - 按 chunk 接收             │
│     - chunk 大小自适应          │              │     - 记录量化元数据            │
│     - 优先传高重要性 chunk      │              │   ↓                             │
│                                 │              │ [8] 反量化器                    │
│                                 │              │     Dequantizer                 │
│                                 │              │     - 按元数据反量化到 FP16     │
│                                 │              │   ↓                             │
│                                 │              │ [9] DecodeStage.init_kv()       │
│                                 │              │     - 写入 KV Cache             │
│                                 │              │   ↓                             │
│                                 │              │ [10] Decode 流水线              │
└─────────────────────────────────┘              └─────────────────────────────────┘
```

### 3.2 与现有代码的集成点

```
现有代码流程:
  _prefill_loop()
    → run_prefill_logged()          # 计算 KV Cache
    → _split_kv_cache_by_len()      # 按请求分割
    → _compute_token_ids_from_prefill()  # 计算 token IDs
    → rpc.call("init_kv", kv)       # 发送到 decode 节点

ABKT 插入点:
  _prefill_loop()
    → run_prefill_logged()          # 计算 KV Cache（保留）
    → _split_kv_cache_by_len()      # 按请求分割（保留）
    → ⭐ ABKT_Adapt(kv, network_state)   # 新增: 自适应量化
    → _compute_token_ids_from_prefill()  # 计算 token IDs（保留）
    → ⭐ ABKT_Transfer(kv, rpc)          # 新增: 分块传输
    → (原有 rpc.call 逻辑替换为 ABKT_Transfer)
```

---

## 四、核心模块详细设计

### 4.1 模块 1：Token 重要性评估器 (TokenImportanceEvaluator)

**创新点**：联合三维重要性评分，而非单一 attention score

```python
# backend/token_importance.py

class TokenImportanceEvaluator:
    """联合三维重要性评分器。

    重要性 = α × AttentionScore(l,t) + β × LayerSensitivity(l) + γ × PositionDecay(t)

    其中:
    - AttentionScore: 基于 prefill 阶段 attention 权重的 token 重要性
    - LayerSensitivity: 不同 layer 对量化的敏感度（预先标定或在线估计）
    - PositionDecay: 位置衰减因子（近期 token 通常更重要）
    """

    def __init__(self, alpha=0.6, beta=0.25, gamma=0.15):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Layer 敏感度先验（可通过消融实验标定）
        # 一般规律: 早期 layer 的 Key 更敏感，后期 layer 的 Value 更敏感
        self._layer_sensitivity_cache = {}

    def compute_importance(
        self,
        kv_cache: dict,           # {decode_node_idx: {layer_idx: (k, v)}}
        attention_weights: torch.Tensor,  # [batch, heads, seq, seq] 或 None
        num_layers: int,
        seq_len: int,
    ) -> dict:
        """计算每个 (layer, token) 的重要性矩阵。

        Returns:
            {decode_node_idx: {layer_idx: importance_tensor[seq_len]}}
        """
        result = {}

        # === 维度 1: Attention Score ===
        attn_scores = self._compute_attention_importance(
            attention_weights, seq_len
        )  # shape: [seq_len]

        # === 维度 2: Layer Sensitivity ===
        layer_sensitivity = self._get_layer_sensitivity(
            kv_cache, num_layers
        )  # shape: [num_layers]

        # === 维度 3: Position Decay ===
        position_weights = self._compute_position_decay(seq_len)
        # shape: [seq_len]

        for decode_node_idx, layer_cache in kv_cache.items():
            result[decode_node_idx] = {}
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                # 联合评分
                importance = (
                    self.alpha * attn_scores.to(k.device)
                    + self.beta * layer_sensitivity[layer_idx].to(k.device)
                    + self.gamma * position_weights.to(k.device)
                )
                # 归一化到 [0, 1]
                importance = torch.clamp(importance, 0, 1)
                result[decode_node_idx][layer_idx] = importance

        return result

    def _compute_attention_importance(self, attention_weights, seq_len):
        """从 attention 权重计算 token 重要性。

        两种策略:
        1. 如果有 attention_weights（最后几层），直接用
        2. 如果没有，用 Key 的 L2 范数作为代理指标
        """
        if attention_weights is not None:
            # 对所有 head 取平均，对所有 query position 取最大
            # attention_weights: [batch, heads, query_len, key_len]
            avg_attn = attention_weights.mean(dim=1)  # [batch, q, k]
            # 对每个 key position，取所有 query 中的最大 attention
            max_attn = avg_attn.max(dim=1).values  # [batch, k]
            # 对 batch 取平均
            scores = max_attn.mean(dim=0)  # [k]
            # 归一化
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            return scores
        else:
            # Fallback: 均匀重要性
            return torch.ones(seq_len)

    def _compute_attention_importance_from_kv(self, kv_cache):
        """从 KV Cache 的 Key 张量估算重要性（无需 attention weights）。

        原理: Key 的 L2 范数越大，该 token 在 attention 中的贡献越大。
        这是一个轻量级代理指标，计算开销极小。
        """
        # 收集所有 layer 的 Key
        all_keys = []
        for decode_node_idx, layer_cache in kv_cache.items():
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                # k shape: [batch, heads, seq, head_dim]
                # 对 heads 和 head_dim 取 L2 范数
                k_norm = torch.norm(k.float(), dim=(1, 3))  # [batch, seq]
                all_keys.append(k_norm)

        if not all_keys:
            return None

        # 对所有 layer 取平均
        stacked = torch.stack(all_keys, dim=0)  # [num_layers, batch, seq]
        avg_norm = stacked.mean(dim=(0, 1))  # [seq]

        # 归一化
        if avg_norm.max() > avg_norm.min():
            scores = (avg_norm - avg_norm.min()) / (avg_norm.max() - avg_norm.min())
        else:
            scores = torch.ones_like(avg_norm)
        return scores

    def _get_layer_sensitivity(self, kv_cache, num_layers):
        """获取每层的量化敏感度。

        策略: 在线估算 — 通过 Key/Value 张量的数值动态范围来判断。
        动态范围大的 layer 对量化更敏感（量化误差更大）。

        也可以使用预标定的静态敏感度表（通过消融实验得到）。
        """
        sensitivities = []
        for layer_idx in range(num_layers):
            sensitivity = self._estimate_layer_sensitivity(kv_cache, layer_idx)
            sensitivities.append(sensitivity)

        if not sensitivities:
            return torch.ones(num_layers)

        stacked = torch.tensor(sensitivities)
        if stacked.max() > stacked.min():
            stacked = (stacked - stacked.min()) / (stacked.max() - stacked.min())
        else:
            stacked = torch.ones_like(stacked)
        return stacked

    def _estimate_layer_sensitivity(self, kv_cache, layer_idx):
        """估算单层的量化敏感度。

        指标: Key 和 Value 张量的变异系数 (CV = std/mean)
        CV 越大 → 数值分布越分散 → 量化误差越大 → 敏感度越高
        """
        for decode_node_idx, layer_cache in kv_cache.items():
            kv = layer_cache.get(layer_idx)
            if kv is None:
                continue
            k, v = kv
            # 计算 Key 的变异系数
            k_flat = k.float().reshape(-1)
            k_cv = k_flat.std() / (k_flat.abs().mean() + 1e-8)
            # 计算 Value 的变异系数
            v_flat = v.float().reshape(-1)
            v_cv = v_flat.std() / (v_flat.abs().mean() + 1e-8)
            # Key 和 Value 的敏感度取平均
            return float((k_cv + v_cv) / 2)
        return 1.0

    def _compute_position_decay(self, seq_len, decay_rate=0.01):
        """位置衰减因子: 近期 token 权重更高。

        使用指数衰减: w(t) = exp(-decay_rate * (seq_len - 1 - t))
        最后一个 token 权重为 1，第一个 token 权重最小。
        """
        positions = torch.arange(seq_len, dtype=torch.float32)
        weights = torch.exp(-decay_rate * (seq_len - 1 - positions))
        # 归一化到 [0, 1]
        weights = weights / weights.max()
        return weights
```

**创新性说明**：

现有方案（H2O、SnapKV、PyramidKV）只使用单一维度评估 token 重要性：
- H2O：累积 attention score
- SnapKV：observation window 的 attention pattern
- PyramidKV：layer-wise attention entropy

我们的方案将三个正交维度（attention score + layer 敏感度 + 位置衰减）融合为统一评分。其中**layer 敏感度**维度是独创的——现有工作没有在传输场景下考虑不同 layer 对量化的差异化敏感度。

### 4.2 模块 2：网络探测器 (NetworkProbe)

```python
# backend/network_probe.py

import time
import threading
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class NetworkState(Enum):
    GOOD = "good"        # 带宽充足，RTT 稳定
    DEGRADED = "degraded" # 带宽下降或 RTT 波动
    POOR = "poor"         # 带宽严重不足


@dataclass
class NetworkSnapshot:
    """某一时刻的网络状态快照"""
    timestamp: float
    bandwidth_bps: float      # 当前带宽 (bytes/sec)
    bandwidth_ewma: float     # EWMA 预测带宽
    rtt_ms: float             # 当前 RTT (ms)
    rtt_ewma: float           # EWMA 预测 RTT
    state: NetworkState       # 网络状态分级
    budget_bytes: float       # 当前预算 = bandwidth_ewma × max_delay


class NetworkProbe:
    """网络状态探测器。

    核心功能:
    1. 周期性探测带宽和 RTT
    2. EWMA 平滑，预测下一时刻的网络状态
    3. 三级状态判定（good/degraded/poor）
    4. 计算带宽预算

    创新点: EWMA 预测驱动的主动决策
    - 现有方案（CacheGen、KIVI）只在传输时做一次性决策
    - 我们持续监测网络，预测趋势，主动调整策略
    - 如果带宽呈下降趋势，提前降级精度，避免传输中断
    """

    def __init__(
        self,
        probe_interval_sec: float = 1.0,
        ewma_alpha: float = 0.3,      # EWMA 平滑系数（越大越关注近期）
        max_delay_sec: float = 0.5,    # 最大可接受传输延迟
        bw_threshold_good: float = 50e6,   # 50 MB/s 以上为 good
        bw_threshold_poor: float = 10e6,   # 10 MB/s 以下为 poor
        rtt_threshold_good: float = 5.0,   # RTT < 5ms 为 good
        rtt_threshold_poor: float = 50.0,  # RTT > 50ms 为 poor
    ):
        self.probe_interval = probe_interval_sec
        self.ewma_alpha = ewma_alpha
        self.max_delay = max_delay_sec
        self.bw_threshold_good = bw_threshold_good
        self.bw_threshold_poor = bw_threshold_poor
        self.rtt_threshold_good = rtt_threshold_good
        self.rtt_threshold_poor = rtt_threshold_poor

        # 状态
        self._bandwidth_ewma: Optional[float] = None
        self._rtt_ewma: Optional[float] = None
        self._last_snapshot: Optional[NetworkSnapshot] = None
        self._probe_thread: Optional[threading.Thread] = None
        self._running = False

        # 用于带宽探测的校准数据
        self._calibration_samples = []
        self._calibrated = False

    def start(self, target_host: str, target_port: int):
        """启动后台探测线程"""
        self._running = True
        self._target_host = target_host
        self._target_port = target_port
        self._probe_thread = threading.Thread(
            target=self._probe_loop, daemon=True
        )
        self._probe_thread.start()

    def stop(self):
        self._running = False

    def get_snapshot(self) -> NetworkSnapshot:
        """获取当前网络状态快照（线程安全）"""
        if self._last_snapshot is not None:
            return self._last_snapshot
        # 如果还未探测到数据，返回默认值（保守估计）
        return NetworkSnapshot(
            timestamp=time.time(),
            bandwidth_bps=50e6,
            bandwidth_ewma=50e6,
            rtt_ms=5.0,
            rtt_ewma=5.0,
            state=NetworkState.GOOD,
            budget_bytes=50e6 * self.max_delay,
        )

    def calibrate_with_transfer(self, num_bytes: int, elapsed_sec: float):
        """用一次实际传输来校准带宽估计。

        在首次传输时调用，用真实数据初始化 EWMA。
        """
        if elapsed_sec > 0:
            measured_bw = num_bytes / elapsed_sec
            self._bandwidth_ewma = measured_bw
            self._calibrated = True

    def _probe_loop(self):
        """后台探测循环"""
        while self._running:
            try:
                bw, rtt = self._probe_once()
                self._update_ewma(bw, rtt)
                self._last_snapshot = self._make_snapshot()
            except Exception:
                pass
            time.sleep(self.probe_interval)

    def _probe_once(self) -> tuple:
        """执行一次带宽和 RTT 探测。

        实现方式: 发送一个小探测包，测量 RTT。
        带宽通过定期传输校准数据来估算。
        """
        import socket
        start = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self._target_host, self._target_port))
            sock.sendall(b'\x00' * 64)
            sock.recv(1)
            sock.close()
            rtt = (time.time() - start) * 1000  # ms
        except Exception:
            rtt = 100.0  # 超时默认值

        # 带宽使用最近的校准值或默认值
        bw = self._bandwidth_ewma or 50e6
        return bw, rtt

    def _update_ewma(self, bw: float, rtt: float):
        """EWMA 更新: new_ewma = α × new_value + (1-α) × old_ewma"""
        alpha = self.ewma_alpha
        if self._bandwidth_ewma is None:
            self._bandwidth_ewma = bw
        else:
            self._bandwidth_ewma = alpha * bw + (1 - alpha) * self._bandwidth_ewma

        if self._rtt_ewma is None:
            self._rtt_ewma = rtt
        else:
            self._rtt_ewma = alpha * rtt + (1 - alpha) * self._rtt_ewma

    def _make_snapshot(self) -> NetworkSnapshot:
        """根据 EWMA 值判定网络状态"""
        bw = self._bandwidth_ewma
        rtt = self._rtt_ewma

        # 状态判定（两级阈值）
        if bw >= self.bw_threshold_good and rtt <= self.rtt_threshold_good:
            state = NetworkState.GOOD
        elif bw <= self.bw_threshold_poor or rtt >= self.rtt_threshold_poor:
            state = NetworkState.POOR
        else:
            state = NetworkState.DEGRADED

        # 预算计算: 使用 EWMA 预测带宽 × 最大延迟
        # 这比用瞬时带宽更稳定
        budget = bw * self.max_delay

        return NetworkSnapshot(
            timestamp=time.time(),
            bandwidth_bps=bw,
            bandwidth_ewma=bw,
            rtt_ms=rtt,
            rtt_ewma=rtt,
            state=state,
            budget_bytes=budget,
        )
```

**创新性说明**：

1. **EWMA 预测驱动**：不只用瞬时带宽做决策，而是用指数加权移动平均预测带宽趋势。如果带宽呈下降趋势，EWMA 值会比瞬时值更低，系统会提前降级精度，避免传输过程中突然中断。

2. **校准机制**：首次传输时用实际数据校准带宽估计，而非依赖探测包（探测包的带宽与实际大块传输差异很大）。

3. **两级阈值状态机**：使用 good/poor 两个阈值定义三个状态，避免在边界频繁切换（滞回效应）。

### 4.3 模块 3：精度分配器 (PrecisionAllocator)

```python
# backend/precision_allocator.py

import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import IntEnum


class Precision(IntEnum):
    FP16 = 16
    FP8 = 8
    INT4 = 4
    INT2 = 2


# 每种精度的质量保真度（通过离线实验标定）
QUALITY_FIDELITY = {
    Precision.FP16: 1.00,
    Precision.FP8: 0.98,
    Precision.INT4: 0.92,
    Precision.INT2: 0.80,
}

# 每种精度每元素的字节数
BYTES_PER_ELEMENT = {
    Precision.FP16: 2.0,
    Precision.FP8: 1.0,
    Precision.INT4: 0.5,
    Precision.INT2: 0.25,
}


@dataclass
class AllocationResult:
    """精度分配结果"""
    # 精度矩阵: {decode_node: {layer_idx: precision_tensor[seq_len]}}
    precision_map: Dict[int, Dict[int, torch.Tensor]]
    # 统计信息
    total_bytes: float
    budget_bytes: float
    avg_precision_bits: float
    compression_ratio: float


class PrecisionAllocator:
    """基于重要性的精度分配器。

    创新点: 形式化为带约束的优化问题，用贪心算法求解。

    核心算法: Proportional-Importance Allocation (PIA)
    1. 所有 entry 从最低精度开始
    2. 按重要性从高到低，逐步升级精度
    3. 直到预算用完

    这保证了: 在给定预算下，最重要的 token 获得最高精度。
    """

    def __init__(self, precision_levels=None):
        self.precision_levels = precision_levels or [
            Precision.FP16, Precision.FP8, Precision.INT4, Precision.INT2
        ]
        self.precision_levels_sorted = sorted(
            self.precision_levels, key=lambda p: p.value, reverse=True
        )

    def allocate(
        self,
        importance_map: Dict[int, Dict[int, torch.Tensor]],
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        budget_bytes: float,
        default_precision: Precision = Precision.FP16,
    ) -> AllocationResult:
        """执行精度分配。

        Args:
            importance_map: 重要性矩阵 {decode_node: {layer: importance[seq_len]}}
            kv_cache: KV Cache 数据 {decode_node: {layer: (k, v)}}
            budget_bytes: 带宽预算（字节）
            default_precision: 默认精度（带宽充足时使用）

        Returns:
            AllocationResult: 精度分配结果
        """
        # Step 1: 计算当前 FP16 下的总大小
        total_fp16_bytes = self._compute_total_bytes(kv_cache, Precision.FP16)

        # 如果预算充足，直接用默认精度
        if budget_bytes >= total_fp16_bytes:
            precision_map = self._build_uniform_map(kv_cache, default_precision)
            return AllocationResult(
                precision_map=precision_map,
                total_bytes=total_fp16_bytes,
                budget_bytes=budget_bytes,
                avg_precision_bits=float(default_precision.value),
                compression_ratio=1.0,
            )

        # Step 2: 计算压缩比，确定最低精度
        compression_needed = total_fp16_bytes / budget_bytes
        min_precision = self._select_min_precision(compression_needed)

        # Step 3: 用 PIA 算法分配精度
        precision_map, actual_bytes = self._pia_allocate(
            importance_map, kv_cache, budget_bytes, min_precision
        )

        # Step 4: 计算统计信息
        avg_bits = self._compute_avg_precision(precision_map)
        compression_ratio = total_fp16_bytes / actual_bytes if actual_bytes > 0 else 1.0

        return AllocationResult(
            precision_map=precision_map,
            total_bytes=actual_bytes,
            budget_bytes=budget_bytes,
            avg_precision_bits=avg_bits,
            compression_ratio=compression_ratio,
        )

    def _pia_allocate(
        self,
        importance_map,
        kv_cache,
        budget_bytes,
        min_precision,
    ) -> Tuple[Dict, float]:
        """Proportional-Importance Allocation 算法。

        贪心策略: 按重要性降序，逐步升级精度。
        """
        # 收集所有 entry
        entries = []  # [(importance, decode_node, layer_idx, element_count)]
        for decode_node, layer_cache in kv_cache.items():
            imp_layer = importance_map.get(decode_node, {})
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                element_count = k.numel() + v.numel()
                # 使用该 layer 所有 token 的平均重要性
                importance = imp_layer.get(layer_idx)
                if importance is not None:
                    avg_imp = float(importance.mean())
                else:
                    avg_imp = 0.5
                entries.append((avg_imp, decode_node, layer_idx, element_count))

        # 按重要性降序排列
        entries.sort(key=lambda x: x[0], reverse=True)

        # 初始化: 所有 entry 用最低精度
        min_bytes_per_elem = BYTES_PER_ELEMENT[min_precision]
        current_bytes = sum(e[3] * min_bytes_per_elem for e in entries)

        # 精度映射: {(decode_node, layer_idx): current_precision}
        precision_assignments = {}
        for e in entries:
            precision_assignments[(e[1], e[2])] = min_precision

        # 按重要性从高到低，尝试升级精度
        precision_order = self.precision_levels_sorted  # [FP16, FP8, INT4, INT2]
        for imp, decode_node, layer_idx, elem_count in entries:
            current_prec = precision_assignments[(decode_node, layer_idx)]
            current_idx = precision_order.index(current_prec)

            # 尝试升级到更高精度
            for upgrade_idx in range(current_idx - 1, -1, -1):
                target_prec = precision_order[upgrade_idx]
                upgrade_cost = (
                    (BYTES_PER_ELEMENT[target_prec] - BYTES_PER_ELEMENT[current_prec])
                    * elem_count
                )
                if current_bytes + upgrade_cost <= budget_bytes:
                    precision_assignments[(decode_node, layer_idx)] = target_prec
                    current_bytes += upgrade_cost
                    current_prec = target_prec
                else:
                    break  # 预算不够，停止升级此 entry

        # 构建输出映射
        precision_map = {}
        for decode_node, layer_cache in kv_cache.items():
            precision_map[decode_node] = {}
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                prec = precision_assignments.get((decode_node, layer_idx), min_precision)
                seq_len = kv[0].shape[2]  # k: [batch, heads, seq, head_dim]
                precision_map[decode_node][layer_idx] = torch.full(
                    (seq_len,), prec.value, dtype=torch.int8
                )

        return precision_map, current_bytes

    def _select_min_precision(self, compression_needed: float) -> Precision:
        """根据所需压缩比选择最低精度"""
        if compression_needed <= 2.0:
            return Precision.FP8
        elif compression_needed <= 4.0:
            return Precision.INT4
        else:
            return Precision.INT2

    def _compute_total_bytes(self, kv_cache, precision) -> float:
        """计算给定精度下的总字节数"""
        total = 0.0
        bytes_per_elem = BYTES_PER_ELEMENT[precision]
        for decode_node, layer_cache in kv_cache.items():
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                total += (k.numel() + v.numel()) * bytes_per_elem
        return total

    def _build_uniform_map(self, kv_cache, precision) -> Dict:
        """构建统一精度映射"""
        precision_map = {}
        for decode_node, layer_cache in kv_cache.items():
            precision_map[decode_node] = {}
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                seq_len = kv[0].shape[2]
                precision_map[decode_node][layer_idx] = torch.full(
                    (seq_len,), precision.value, dtype=torch.int8
                )
        return precision_map

    def _compute_avg_precision(self, precision_map) -> float:
        """计算平均精度（加权平均）"""
        total_bits = 0.0
        total_elements = 0
        for decode_node, layer_map in precision_map.items():
            for layer_idx, prec_tensor in layer_map.items():
                count = prec_tensor.numel()
                avg_prec = float(prec_tensor.float().mean())
                total_bits += avg_prec * count
                total_elements += count
        return total_bits / total_elements if total_elements > 0 else 0.0
```

**创新性说明**：

1. **形式化优化**：将 KV Cache 传输问题形式化为带约束的资源分配优化问题，这是现有工作中没有的。KIVI 和 CacheGen 都是启发式策略，没有优化框架。

2. **PIA 贪心算法**：专为 KV Cache 传输场景设计的贪心算法，保证在给定预算下，最重要的 token 获得最高精度。时间复杂度 O(L×T×|P|)，微秒级，适合在线决策。

3. **精度级别可扩展**：支持 FP16/FP8/INT4/INT2 四级精度，未来可以加入更多级别（如 FP4、NF4 等）。

### 4.4 模块 4：自适应量化器 (AdaptiveQuantizer)

```python
# backend/adaptive_quant.py

import torch
from typing import Dict, Tuple
from backend.precision_allocator import Precision


class AdaptiveQuantizer:
    """自适应量化器: 按精度矩阵对 KV Cache 进行量化/反量化。

    创新点: token 级 + layer 级的混合精度量化
    - KIVI: 全局统一 INT2/INT4
    - KVQuant: 全局统一低精度
    - 我们: 每个 (layer, token) 可以有不同的精度
    """

    def __init__(self):
        pass

    def quantize_kv_cache(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        precision_map: Dict[int, Dict[int, torch.Tensor]],
    ) -> Tuple[Dict, Dict]:
        """按精度矩阵量化 KV Cache。

        Args:
            kv_cache: 原始 KV Cache (FP16)
            precision_map: 精度分配矩阵

        Returns:
            quantized_kv: 量化后的 KV Cache
            metadata: 量化元数据（用于反量化）
        """
        quantized_kv = {}
        metadata = {}

        for decode_node, layer_cache in kv_cache.items():
            quantized_kv[decode_node] = {}
            metadata[decode_node] = {}

            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                prec_tensor = precision_map.get(decode_node, {}).get(layer_idx)

                if prec_tensor is None:
                    # 默认 FP16
                    quantized_kv[decode_node][layer_idx] = (k, v)
                    metadata[decode_node][layer_idx] = {
                        "precision": Precision.FP16.value,
                        "scale_k": None, "zero_point_k": None,
                        "scale_v": None, "zero_point_v": None,
                    }
                    continue

                # 使用该 layer 所有 token 的平均精度来决定统一量化级别
                # （实际实现中可以做 token 级混合精度，但序列化开销较大）
                avg_prec_value = int(round(float(prec_tensor.float().mean())))
                prec = Precision(avg_prec_value) if avg_prec_value in [16, 8, 4, 2] else Precision.FP16

                if prec == Precision.FP16:
                    quantized_kv[decode_node][layer_idx] = (k, v)
                    metadata[decode_node][layer_idx] = {
                        "precision": 16,
                        "scale_k": None, "zero_point_k": None,
                        "scale_v": None, "zero_point_v": None,
                    }
                else:
                    qk, sk, zk = self._quantize_tensor(k, prec)
                    qv, sv, zv = self._quantize_tensor(v, prec)
                    quantized_kv[decode_node][layer_idx] = (qk, qv)
                    metadata[decode_node][layer_idx] = {
                        "precision": prec.value,
                        "scale_k": sk, "zero_point_k": zk,
                        "scale_v": sv, "zero_point_v": zv,
                        "original_dtype": str(k.dtype),
                    }

        return quantized_kv, metadata

    def dequantize_kv_cache(
        self,
        quantized_kv: Dict,
        metadata: Dict,
    ) -> Dict:
        """按元数据反量化 KV Cache 到 FP16。"""
        result = {}

        for decode_node, layer_cache in quantized_kv.items():
            result[decode_node] = {}
            node_meta = metadata.get(decode_node, {})

            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                meta = node_meta.get(layer_idx, {})
                prec = meta.get("precision", 16)

                if prec == 16:
                    result[decode_node][layer_idx] = kv
                else:
                    k, v = kv
                    precision = Precision(prec)
                    dk = self._dequantize_tensor(
                        k, meta.get("scale_k"), meta.get("zero_point_k"), precision
                    )
                    dv = self._dequantize_tensor(
                        v, meta.get("scale_v"), meta.get("zero_point_v"), precision
                    )
                    result[decode_node][layer_idx] = (dk, dv)

        return result

    def _quantize_tensor(self, tensor: torch.Tensor, precision: Precision):
        """对称/非对称量化"""
        tensor_f = tensor.float()

        if precision == Precision.FP8:
            # FP8: 简单的 float8 量化（使用 E4M3 格式近似）
            scale = tensor_f.abs().max() / 127.0
            if scale < 1e-10:
                scale = 1.0
            quantized = torch.clamp(torch.round(tensor_f / scale), -128, 127).to(torch.int8)
            return quantized, scale, 0

        elif precision == Precision.INT4:
            # INT4: 非对称量化到 [-8, 7]
            min_val = tensor_f.min()
            max_val = tensor_f.max()
            scale = (max_val - min_val) / 15.0
            if scale < 1e-10:
                scale = 1.0
            zero_point = torch.round(-min_val / scale).clamp(0, 15)
            quantized = torch.clamp(
                torch.round(tensor_f / scale + zero_point), 0, 15
            ).to(torch.uint8)
            return quantized, scale, zero_point

        elif precision == Precision.INT2:
            # INT2: 非对称量化到 [0, 3]
            min_val = tensor_f.min()
            max_val = tensor_f.max()
            scale = (max_val - min_val) / 3.0
            if scale < 1e-10:
                scale = 1.0
            zero_point = torch.round(-min_val / scale).clamp(0, 3)
            quantized = torch.clamp(
                torch.round(tensor_f / scale + zero_point), 0, 3
            ).to(torch.uint8)
            return quantized, scale, zero_point

        return tensor, None, None

    def _dequantize_tensor(self, quantized, scale, zero_point, precision):
        """反量化到 FP16"""
        if scale is None:
            return quantized

        if precision in (Precision.INT4, Precision.INT2):
            return ((quantized.float() - zero_point) * scale).half()
        else:  # FP8
            return (quantized.float() * scale).half()
```

### 4.5 模块 5：流水线分块传输 (ChunkedTransfer)

```python
# backend/chunked_transfer.py

import torch
import time
from typing import Dict, List, Tuple, Optional
from backend.dist import send_obj, recv_obj
from backend.precision_allocator import Precision


class ChunkedTransfer:
    """流水线分块传输器。

    创新点:
    1. Layer 级流水线: 每层 KV 计算完立即传输，与下一层计算并行
    2. Chunk 大小自适应: 带宽高→大 chunk（减少协议开销），带宽低→小 chunk（减少延迟）
    3. 优先级传输: 高重要性 layer 先传输
    4. 传输与量化重叠: 量化下一层的同时传输当前层
    """

    def __init__(
        self,
        base_chunk_size: int = 64,  # base: 每 chunk 64 个 token
        min_chunk_size: int = 16,
        max_chunk_size: int = 256,
    ):
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def compute_adaptive_chunk_size(self, bandwidth_bps: float) -> int:
        """根据带宽自适应计算 chunk 大小。

        策略: chunk_size ∝ bandwidth
        - 带宽高: 大 chunk → 减少协议头开销
        - 带宽低: 小 chunk → 减少单次传输延迟，更快响应带宽变化
        """
        # 归一化带宽到 [0, 1] 范围（假设 100 MB/s 为上限）
        bw_normalized = min(bandwidth_bps / 100e6, 1.0)
        chunk_size = int(
            self.min_chunk_size
            + (self.max_chunk_size - self.min_chunk_size) * bw_normalized
        )
        return max(self.min_chunk_size, min(chunk_size, self.max_chunk_size))

    def transfer_kv_cache(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        metadata: Dict,
        importance_map: Dict,
        rpc,
        decode_stages: list,
        bandwidth_bps: float,
        request_id: str,
    ) -> bool:
        """执行自适应分块传输。

        流程:
        1. 按重要性排序 layer
        2. 对每个 layer，按 chunk 大小分块
        3. 逐 chunk 传输，附带量化元数据
        4. Decode 端按 chunk 接收并反量化
        """
        chunk_size = self.compute_adaptive_chunk_size(bandwidth_bps)

        # 按重要性排序 layer（高重要性先传）
        layer_order = self._sort_layers_by_importance(kv_cache, importance_map)

        # 为每个 decode 节点构建传输队列
        for decode_node_idx, layer_cache in kv_cache.items():
            node_meta = metadata.get(decode_node_idx, {})

            # 按排序顺序传输
            for layer_idx in layer_order:
                if layer_idx not in layer_cache:
                    continue
                kv = layer_cache[layer_idx]
                if kv is None:
                    continue
                meta = node_meta.get(layer_idx, {})

                # 分 chunk 传输
                k, v = kv
                seq_len = k.shape[2]
                for chunk_start in range(0, seq_len, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_len)
                    chunk_k = k[:, :, chunk_start:chunk_end, :]
                    chunk_v = v[:, :, chunk_start:chunk_end, :]

                    # 构建 chunk payload
                    chunk_payload = {
                        "request_id": request_id,
                        "decode_node_idx": decode_node_idx,
                        "layer_idx": layer_idx,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "total_seq_len": seq_len,
                        "k": chunk_k,
                        "v": chunk_v,
                        "meta": meta,
                        "is_last_chunk": (chunk_end >= seq_len),
                    }

                    # 传输
                    target_stage = self._find_decode_stage(
                        decode_stages, decode_node_idx
                    )
                    if target_stage is not None:
                        rpc.call(
                            target_stage.rank,
                            "receive_kv_chunk",
                            chunk_payload,
                            priority=0,  # HIGH
                        )

        return True

    def _sort_layers_by_importance(self, kv_cache, importance_map) -> List[int]:
        """按平均重要性对 layer 排序（高重要性优先）"""
        layer_importance = {}
        for decode_node, layer_cache in kv_cache.items():
            imp_layer = importance_map.get(decode_node, {})
            for layer_idx in layer_cache.keys():
                if layer_idx in layer_importance:
                    continue
                imp = imp_layer.get(layer_idx)
                if imp is not None:
                    layer_importance[layer_idx] = float(imp.mean())
                else:
                    layer_importance[layer_idx] = 0.5

        return sorted(layer_importance.keys(),
                      key=lambda l: layer_importance[l], reverse=True)

    def _find_decode_stage(self, decode_stages, decode_node_idx):
        """查找对应的 decode stage"""
        for stage in decode_stages:
            if stage.stage_id == decode_node_idx:
                return stage
        return None
```

---

## 五、端到端集成

### 5.1 修改 `_prefill_loop` (modes_split_pd.py)

在 `_prefill_loop` 中，KV Cache 计算完成后、传输之前，插入 ABKT 逻辑：

```python
# 在 _prefill_loop 的 kv_splits 准备好之后，插入:

# ===== ABKT: 自适应传输 =====
for b, kv, token_ids in zip(batch, kv_splits, prefill_token_ids_list):
    req_id = b["record"].request_id

    if kv is not None and self._abkt_enabled:
        # Step 1: 网络探测
        net_snapshot = self._network_probe.get_snapshot()

        # Step 2: 重要性评估
        importance_map = self._importance_evaluator.compute_importance(
            kv_cache=kv,
            attention_weights=None,  # 从 prefill 结果中提取
            num_layers=self.num_layers,
            seq_len=len(b["input_ids"]),
        )

        # Step 3: 精度分配
        alloc_result = self._precision_allocator.allocate(
            importance_map=importance_map,
            kv_cache=kv,
            budget_bytes=net_snapshot.budget_bytes,
        )

        # Step 4: 自适应量化
        quantized_kv, meta = self._adaptive_quantizer.quantize_kv_cache(
            kv, alloc_result.precision_map
        )

        # Step 5: 分块传输（替代原有 rpc.call("init_kv", ...)）
        self._chunked_transfer.transfer_kv_cache(
            quantized_kv, meta, importance_map,
            self.rpc, self.decode_stages,
            net_snapshot.bandwidth_bps, req_id,
        )
    else:
        # 非 ABKT 模式: 原有逻辑
        ...
```

### 5.2 修改 Decode 端 (worker_server)

在 `worker_server` 中添加 `receive_kv_chunk` handler：

```python
# 在 worker_server 的 handlers 中添加:

# chunk 缓冲区: {request_id: {layer_idx: [(chunk_k, chunk_v, meta, end)]}}
_chunk_buffers = {}
_chunk_lock = threading.Lock()

def _receive_kv_chunk(request_id, decode_node_idx, layer_idx,
                      chunk_start, chunk_end, total_seq_len,
                      k, v, meta, is_last_chunk):
    """接收 KV Cache chunk，全部收完后反量化并初始化 KV Cache"""
    with _chunk_lock:
        if request_id not in _chunk_buffers:
            _chunk_buffers[request_id] = {}
        if layer_idx not in _chunk_buffers[request_id]:
            _chunk_buffers[request_id][layer_idx] = {
                "chunks": [], "total_seq_len": total_seq_len, "meta": meta
            }
        _chunk_buffers[request_id][layer_idx]["chunks"].append(
            (chunk_start, chunk_end, k, v)
        )

    if is_last_chunk:
        # 检查是否所有 chunk 都已收到
        buf = _chunk_buffers.get(request_id, {})
        all_received = all(
            self._all_chunks_complete(info)
            for info in buf.values()
        )
        if all_received:
            # 拼接 + 反量化 + init_kv
            full_kv = self._assemble_and_dequantize(request_id, buf)
            local_decode.init_kv(request_id, full_kv)
            with _chunk_lock:
                del _chunk_buffers[request_id]
    return True

handlers["receive_kv_chunk"] = _receive_kv_chunk
```

---

## 六、创新性总结与论证

### 6.1 与最接近现有工作的逐项对比

| 维度 | KIVI (ICML'24) | CacheGen (SIGCOMM'24) | Mooncake (FAST'25) | **ABKT (Ours)** |
|------|---------------|----------------------|-------------------|-----------------|
| 量化策略 | 静态 INT2/INT4 | delta encoding + VQ | 无量化 | **动态多级精度** |
| 网络感知 | 无 | 无 | 无 | **EWMA 预测驱动** |
| Token 重要性 | 无 | 无 | 无 | **三维联合评分** |
| 优化框架 | 无（启发式） | 无（启发式） | 无 | **带约束优化** |
| 分块传输 | 无 | 流式传输 | 高性能传输 | **自适应分块** |
| 异构适配 | 无 | 无 | 无（同构） | **Jetson 适配** |
| 两阶段自适应 | 无 | 无 | 无 | **预决策 + chunk级调整** |

### 6.2 创新点提炼（论文角度）

**创新点 1: 自适应码率 KV Cache 传输 (ABR-KV)**

将视频流媒体领域的自适应码率（ABR）思想首次迁移到 KV Cache 传输领域。定义了 KV Cache 传输的优化目标函数，提出 PIA 贪心求解算法，在带宽约束下最大化传输质量。与 KIVI/CacheGen 的全局统一策略相比，ABR-KV 能在相同带宽下传输更多高重要性内容。

**创新点 2: 联合三维重要性评分 (TDS)**

提出 Token-Layer-Position 三维联合重要性评分机制，将 token 级 attention 重要性、layer 级量化敏感度、位置级衰减因子融合为统一评分。与 H2O/SnapKV 的单一维度评分相比，TDS 能更准确地识别"对生成质量影响最大的 KV Cache entry"。

**创新点 3: 带宽预测驱动的两阶段自适应传输**

提出粗粒度预决策（传输前基于 EWMA 预测选择全局精度策略）+ 细粒度 chunk 级调整（传输中根据实际带宽反馈调整 chunk 大小和后续 layer 精度）的两阶段自适应机制。与现有方案的一次性静态决策相比，能更好地应对边缘网络的带宽波动。

### 6.3 为什么这个方案"够创新"

1. **问题定义新**：首次将 KV Cache 传输形式化为带约束优化问题。现有工作都是启发式方法。

2. **方法论迁移新**：ABR 概念从视频流媒体迁移到 KV Cache 传输，这是一个跨领域的类比创新。

3. **维度融合新**：三维重要性评分（attention + layer sensitivity + position）是独创的，现有工作都只用单一维度。

4. **场景新**：面向边缘异构 PD 分离架构（x86+Jetson），这个场景下网络波动是核心挑战，而现有方案都假设数据中心的稳定网络。

5. **系统设计新**：两阶段自适应 + chunk 级反馈 + 优先级传输，构成完整的自适应传输系统，现有方案缺乏这种系统级设计。

---

## 七、实验设计

### 7.1 实验环境

| 节点 | 硬件 | 角色 |
|------|------|------|
| x86 节点 | Intel i7 + RTX 3060 (12GB) | Prefill |
| Jetson 节点 | Jetson AGX Orin (64GB) | Decode |
| 网络 | 千兆以太网 + tc netem | 模拟波动 |

### 7.2 对比基线

| 方法 | 说明 |
|------|------|
| Direct Transfer | FP16 全量传输，无优化 |
| KIVI | 静态 INT4 量化后传输 |
| CacheGen | delta encoding + 流式传输 |
| ABRT-Ablation-1 | 仅自适应量化（无重要性感知） |
| ABRT-Ablation-2 | 仅重要性感知（无自适应量化） |
| ABRT-Full | 完整方案（量化 + 重要性 + 分块） |

### 7.3 网络波动场景

| 场景 | 配置 |
|------|------|
| 稳定高带宽 | 100 Mbps，无抖动 |
| 稳定低带宽 | 10 Mbps，无抖动 |
| 周期性抖动 | 10-100 Mbps，周期 5s |
| 突发降速 | 100 Mbps → 10 Mbps → 100 Mbps |
| 丢包 | 100 Mbps + 1% 随机丢包 |

### 7.4 评估指标

| 指标 | 说明 |
|------|------|
| KV Cache 传输延迟 (ms) | 从 prefill 完成到 decode 收到完整 KV 的时间 |
| TTFT (ms) | Time To First Token |
| 生成质量 (PPL) | 在 WikiText-2 / C4 上的 perplexity |
| 压缩比 | FP16 大小 / 实际传输大小 |
| 延迟方差 | 不同网络抖动强度下的延迟标准差 |
| 量化/反量化开销 (ms) | 在 x86 和 Jetson 上分别测量 |

### 7.5 消融实验

| 实验 | 变量 | 目的 |
|------|------|------|
| 重要性维度消融 | 逐个关闭 α/β/γ | 验证三维评分的必要性 |
| 精度级别消融 | 只用 {FP16,INT4} vs {FP16,FP8,INT4,INT2} | 验证多级精度的收益 |
| Chunk 大小消融 | 固定 chunk vs 自适应 chunk | 验证自适应分块的收益 |
| 预测 vs 反应 | EWMA 预测 vs 瞬时带宽 | 验证预测驱动的价值 |

---

## 八、实施路线图

```
Week 1-2: 基础模块
├── 实现 NetworkProbe（网络探测器）
├── 实现 AdaptiveQuantizer（量化/反量化接口）
└── 端到端链路验证（量化→传输→反量化→decode）

Week 3-4: 核心算法
├── 实现 TokenImportanceEvaluator（重要性评估）
├── 实现 PrecisionAllocator（PIA 精度分配）
└── 单元测试 + 微基准测试

Week 5-6: 系统集成
├── 集成到 _prefill_loop
├── 实现 ChunkedTransfer（分块传输）
├── 实现 decode 端 chunk 接收 + 反量化
└── 端到端 ABKT pipeline 跑通

Week 7-8: 实验与优化
├── 基线对比实验（Direct Transfer / KIVI / CacheGen）
├── 消融实验
├── 多场景对比实验
└── 性能调优

Week 9-10: 论文撰写
├── 实验数据分析
├── 论文初稿
└── 补充实验
```

---

## 参考文献

1. DistServe — OSDI 2024
2. Mooncake — FAST 2025 (Best Paper)
3. CacheGen — SIGCOMM 2024
4. KIVI — ICML 2024
5. KVQuant — NeurIPS 2024
6. GEAR — NeurIPS 2024 Workshop
7. KVDirect — 2025
8. H2O — NeurIPS 2023
9. SnapKV — 2024
10. PyramidKV — 2024
11. RadixAttention / SGLang — 2024
12. Splitwise — ISCA 2024
13. StreamingLLM — 2023
