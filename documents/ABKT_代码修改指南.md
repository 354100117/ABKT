# ABKT 代码修改指南

> 逐步实施手册 | 每一步标明修改位置、依赖关系、代码内容

---

## 总览：修改层次与依赖关系

```
层次 0（无依赖，先创建）：
  ├── backend/network_probe.py      ← 新建
  ├── backend/token_importance.py   ← 新建
  ├── backend/precision_allocator.py ← 新建
  └── backend/adaptive_quant.py     ← 新建

层次 1（依赖层次 0）：
  └── backend/chunked_transfer.py   ← 新建

层次 2（依赖层次 0+1，修改现有文件）：
  ├── backend/dist.py               ← 小改：添加量化感知的发送/接收
  ├── backend/modes_split_pd.py     ← 核心改：_prefill_loop 中插入 ABKT 逻辑
  └── backend/pipeline_runner.py    ← 核心改：decode 端 chunk 接收 + 反量化

层次 3（依赖层次 2）：
  └── serve.py                      ← 小改：添加 ABKT 配置项和启动参数
```

**原则：先创建新文件（不破坏现有代码），再修改现有文件。每一步完成后可以单独测试。**

---

## 第一步：创建 backend/network_probe.py

**目的**：网络带宽和 RTT 探测，提供实时网络状态

**依赖**：无（纯 Python，不依赖项目其他模块）

**文件路径**：`E:\EdgePD\222222\whole_process_fixedkv\backend\network_probe.py`

```python
"""网络状态探测器 — ABKT 模块 1

功能：
1. 后台线程周期性探测 RTT
2. 用 EWMA 平滑带宽和 RTT 估计
3. 三级状态判定：GOOD / DEGRADED / POOR
4. 提供带宽预算 = 预测带宽 × 最大延迟

使用方式：
    probe = NetworkProbe()
    probe.start(target_host="192.168.0.11", target_port=29500)
    snapshot = probe.get_snapshot()
    # snapshot.budget_bytes, snapshot.state, snapshot.bandwidth_ewma
"""

import socket
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NetworkState(Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


@dataclass
class NetworkSnapshot:
    timestamp: float
    bandwidth_ewma: float      # bytes/sec
    rtt_ewma: float             # ms
    state: NetworkState
    budget_bytes: float         # bandwidth_ewma × max_delay


class NetworkProbe:
    def __init__(
        self,
        probe_interval_sec: float = 2.0,
        ewma_alpha: float = 0.3,
        max_delay_sec: float = 0.5,
        bw_good: float = 50e6,
        bw_poor: float = 10e6,
        rtt_good: float = 5.0,
        rtt_poor: float = 50.0,
    ):
        self.probe_interval = probe_interval_sec
        self.alpha = ewma_alpha
        self.max_delay = max_delay_sec
        self.bw_good = bw_good
        self.bw_poor = bw_poor
        self.rtt_good = rtt_good
        self.rtt_poor = rtt_poor

        self._bw_ewma: Optional[float] = None
        self._rtt_ewma: Optional[float] = None
        self._snapshot = NetworkSnapshot(
            timestamp=0, bandwidth_ewma=50e6, rtt_ewma=5.0,
            state=NetworkState.GOOD, budget_bytes=50e6 * 0.5,
        )
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, target_host: str, probe_port: int = 29501):
        self._running = True
        self._target_host = target_host
        self._probe_port = probe_port
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_snapshot(self) -> NetworkSnapshot:
        return self._snapshot

    def update_bandwidth(self, num_bytes: int, elapsed_sec: float):
        """用一次实际传输校准带宽（在首次传输后调用）"""
        if elapsed_sec > 0:
            measured = num_bytes / elapsed_sec
            if self._bw_ewma is None:
                self._bw_ewma = measured
            else:
                self._bw_ewma = self.alpha * measured + (1 - self.alpha) * self._bw_ewma
            self._snapshot = self._make_snapshot()

    def _loop(self):
        while self._running:
            try:
                rtt = self._probe_rtt()
                if self._rtt_ewma is None:
                    self._rtt_ewma = rtt
                else:
                    self._rtt_ewma = self.alpha * rtt + (1 - self.alpha) * self._rtt_ewma
                self._snapshot = self._make_snapshot()
            except Exception:
                pass
            time.sleep(self.probe_interval)

    def _probe_rtt(self) -> float:
        """通过 TCP 连接测量 RTT"""
        start = time.time()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self._target_host, self._probe_port))
            s.sendall(b'\x00' * 64)
            s.recv(1)
            s.close()
            return (time.time() - start) * 1000
        except Exception:
            return 100.0  # 超时默认值

    def _make_snapshot(self) -> NetworkSnapshot:
        bw = self._bw_ewma or 50e6
        rtt = self._rtt_ewma or 5.0

        if bw >= self.bw_good and rtt <= self.rtt_good:
            state = NetworkState.GOOD
        elif bw <= self.bw_poor or rtt >= self.rtt_poor:
            state = NetworkState.POOR
        else:
            state = NetworkState.DEGRADED

        return NetworkSnapshot(
            timestamp=time.time(),
            bandwidth_ewma=bw,
            rtt_ewma=rtt,
            state=state,
            budget_bytes=bw * self.max_delay,
        )
```

**测试方法**：独立运行，不依赖其他模块。可以在 Python 中 `from backend.network_probe import NetworkProbe` 验证导入。

---

## 第二步：创建 backend/token_importance.py

**目的**：基于 KV Cache 的 Key 张量计算每个 token 位置的重要性分数

**依赖**：无（只依赖 torch）

**文件路径**：`E:\EdgePD\222222\whole_process_fixedkv\backend\token_importance.py`

```python
"""Token 重要性评估器 — ABKT 模块 2

核心思路：
  用 Key 张量的 L2 范数作为 token 重要性的代理指标。
  Key 范数越大 → 该 token 在 attention 中的 dot-product 贡献越大 → 越重要

输入：kv_cache dict（与你项目中的格式一致）
输出：importance dict（同样结构，值变为 importance score tensor）
"""

import torch
from typing import Dict, Tuple


class TokenImportanceEvaluator:
    def __init__(self, alpha=0.7, gamma=0.3):
        """
        alpha: Key L2 范数的权重
        gamma: 位置衰减的权重
        """
        self.alpha = alpha
        self.gamma = gamma

    def compute(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Args:
            kv_cache: {decode_node_idx: {layer_idx: (k, v)}}
                k shape: [1, num_heads, seq_len, head_dim]

        Returns:
            importance: {decode_node_idx: {layer_idx: tensor[seq_len]}}
                每个值在 [0, 1] 之间
        """
        result = {}

        # 先收集所有 layer 的 Key L2 范数，用于全局归一化
        all_k_norms = []
        for decode_node, layer_cache in kv_cache.items():
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                # k: [1, heads, seq, head_dim] → 对 heads 和 head_dim 求 L2 → [seq]
                k_norm = torch.norm(k.float(), dim=(0, 1, 3))  # [seq]
                all_k_norms.append(k_norm)

        if not all_k_norms:
            return result

        # 全局归一化：对所有 layer 取平均，再归一化到 [0, 1]
        stacked = torch.stack(all_k_norms, dim=0)  # [num_layers, seq]
        avg_norm = stacked.mean(dim=0)  # [seq]
        if avg_norm.max() > avg_norm.min():
            k_scores = (avg_norm - avg_norm.min()) / (avg_norm.max() - avg_norm.min())
        else:
            k_scores = torch.ones_like(avg_norm)

        # 位置衰减：最后一个 token 权重最高，第一个最低
        seq_len = len(k_scores)
        pos = torch.arange(seq_len, dtype=torch.float32)
        pos_weights = torch.exp(-0.01 * (seq_len - 1 - pos))
        pos_weights = pos_weights / pos_weights.max()

        # 联合评分
        combined = self.alpha * k_scores + self.gamma * pos_weights
        combined = torch.clamp(combined, 0, 1)

        # 为每个 decode_node 和 layer 复制同样的 importance
        for decode_node, layer_cache in kv_cache.items():
            result[decode_node] = {}
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                result[decode_node][layer_idx] = combined.clone()

        return result
```

**测试方法**：
```python
from backend.token_importance import TokenImportanceEvaluator
evaluator = TokenImportanceEvaluator()
# 构造假数据测试
fake_kv = {0: {0: (torch.randn(1, 32, 128, 128), torch.randn(1, 32, 128, 128))}}
importance = evaluator.compute(fake_kv)
print(importance[0][0].shape)  # torch.Size([128])
```

---

## 第三步：创建 backend/precision_allocator.py

**目的**：根据重要性评分和带宽预算，为每个 KV Cache entry 分配量化精度

**依赖**：无（只依赖 torch）

**文件路径**：`E:\EdgePD\222222\whole_process_fixedkv\backend\precision_allocator.py`

```python
"""精度分配器 — ABKT 模块 3

PIA (Proportional-Importance Allocation) 算法：
  1. 所有 entry 从最低精度开始
  2. 按重要性从高到低，逐步升级精度
  3. 直到预算用完

输出：每个 (decode_node, layer_idx) 的目标精度
"""

import torch
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import IntEnum


class Precision(IntEnum):
    FP16 = 16
    FP8 = 8
    INT4 = 4
    INT2 = 2


BYTES_PER_ELEM = {
    Precision.FP16: 2.0,
    Precision.FP8: 1.0,
    Precision.INT4: 0.5,
    Precision.INT2: 0.25,
}


@dataclass
class AllocationResult:
    # {decode_node: {layer_idx: Precision}}
    precision_map: Dict[int, Dict[int, Precision]]
    total_bytes: float
    budget_bytes: float
    compression_ratio: float


class PrecisionAllocator:
    def __init__(self, levels=None):
        self.levels = levels or [Precision.FP16, Precision.FP8, Precision.INT4, Precision.INT2]
        self.levels_desc = sorted(self.levels, key=lambda p: p.value, reverse=True)

    def allocate(
        self,
        importance: Dict[int, Dict[int, torch.Tensor]],
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        budget_bytes: float,
    ) -> AllocationResult:
        # 计算 FP16 总大小
        total_fp16 = 0.0
        entries = []  # [(avg_importance, decode_node, layer_idx, elem_count)]

        for decode_node, layer_cache in kv_cache.items():
            imp_node = importance.get(decode_node, {})
            for layer_idx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                elem_count = k.numel() + v.numel()
                total_fp16 += elem_count * BYTES_PER_ELEM[Precision.FP16]

                imp = imp_node.get(layer_idx)
                avg_imp = float(imp.mean()) if imp is not None else 0.5
                entries.append((avg_imp, decode_node, layer_idx, elem_count))

        # 预算充足 → 全部 FP16
        if budget_bytes >= total_fp16:
            pm = self._uniform_map(kv_cache, Precision.FP16)
            return AllocationResult(pm, total_fp16, budget_bytes, 1.0)

        # 确定最低精度
        ratio = total_fp16 / budget_bytes
        if ratio <= 2.0:
            min_prec = Precision.FP8
        elif ratio <= 4.0:
            min_prec = Precision.INT4
        else:
            min_prec = Precision.INT2

        # PIA 贪心分配
        entries.sort(key=lambda x: x[0], reverse=True)
        min_bpe = BYTES_PER_ELEM[min_prec]
        current_bytes = sum(e[3] * min_bpe for e in entries)
        assignments = {(e[1], e[2]): min_prec for e in entries}

        for imp, dn, li, ec in entries:
            cur = assignments[(dn, li)]
            cur_idx = self.levels_desc.index(cur)
            for up_idx in range(cur_idx - 1, -1, -1):
                target = self.levels_desc[up_idx]
                cost = (BYTES_PER_ELEM[target] - BYTES_PER_ELEM[cur]) * ec
                if current_bytes + cost <= budget_bytes:
                    assignments[(dn, li)] = target
                    current_bytes += cost
                    cur = target
                else:
                    break

        # 构建输出
        pm = {}
        for decode_node, layer_cache in kv_cache.items():
            pm[decode_node] = {}
            for layer_idx in layer_cache:
                if (decode_node, layer_idx) in assignments:
                    pm[decode_node][layer_idx] = assignments[(decode_node, layer_idx)]

        comp_ratio = total_fp16 / current_bytes if current_bytes > 0 else 1.0
        return AllocationResult(pm, current_bytes, budget_bytes, comp_ratio)

    def _uniform_map(self, kv_cache, prec):
        pm = {}
        for dn, lc in kv_cache.items():
            pm[dn] = {li: prec for li, kv in lc.items() if kv is not None}
        return pm
```

---

## 第四步：创建 backend/adaptive_quant.py

**目的**：按精度矩阵对 KV Cache 做量化/反量化

**依赖**：`backend/precision_allocator.py`（Precision 枚举）

**文件路径**：`E:\EdgePD\222222\whole_process_fixedkv\backend\adaptive_quant.py`

```python
"""自适应量化器 — ABKT 模块 4

功能：
  - quantize(): 按 precision_map 量化 KV Cache
  - dequantize(): 按 metadata 反量化回 FP16

量化方式：
  - FP16: 不处理
  - FP8:  对称量化到 int8，scale = max/127
  - INT4: 非对称量化到 uint8 [0, 15]
  - INT2: 非对称量化到 uint8 [0, 3]
"""

import torch
from typing import Dict, Tuple
from backend.precision_allocator import Precision


def quantize_kv(
    kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    precision_map: Dict[int, Dict[int, Precision]],
) -> Tuple[Dict, Dict]:
    """
    Returns:
        quantized_kv: 同结构，(k, v) 可能是量化后的 int8/uint8 张量
        metadata: {decode_node: {layer_idx: {"precision": int, "scale_k":..., ...}}}
    """
    qkv = {}
    meta = {}

    for dn, layer_cache in kv_cache.items():
        qkv[dn] = {}
        meta[dn] = {}
        for li, kv in layer_cache.items():
            if kv is None:
                continue
            k, v = kv
            prec = precision_map.get(dn, {}).get(li, Precision.FP16)

            if prec == Precision.FP16:
                qkv[dn][li] = (k, v)
                meta[dn][li] = {"precision": 16}
            else:
                qk, sk = _quantize(k, prec)
                qv, sv = _quantize(v, prec)
                qkv[dn][li] = (qk, qv)
                meta[dn][li] = {
                    "precision": prec.value,
                    "scale_k": sk, "scale_v": sv,
                    "orig_dtype": str(k.dtype),
                }

    return qkv, meta


def dequantize_kv(
    quantized_kv: Dict,
    metadata: Dict,
) -> Dict:
    """反量化到 FP16，返回与原 kv_cache 同结构"""
    result = {}
    for dn, layer_cache in quantized_kv.items():
        result[dn] = {}
        dn_meta = metadata.get(dn, {})
        for li, kv in layer_cache.items():
            if kv is None:
                continue
            m = dn_meta.get(li, {})
            prec = m.get("precision", 16)
            if prec == 16:
                result[dn][li] = kv
            else:
                k, v = kv
                result[dn][li] = (
                    _dequantize(k, m.get("scale_k"), Precision(prec)),
                    _dequantize(v, m.get("scale_v"), Precision(prec)),
                )
    return result


def _quantize(tensor: torch.Tensor, prec: Precision):
    t = tensor.float()
    if prec == Precision.FP8:
        scale = t.abs().max() / 127.0
        scale = max(float(scale), 1e-10)
        q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
        return q, scale
    elif prec == Precision.INT4:
        mn, mx = t.min(), t.max()
        scale = float((mx - mn) / 15.0)
        scale = max(scale, 1e-10)
        zp = torch.round(-mn / scale).clamp(0, 15)
        q = torch.clamp(torch.round(t / scale + zp), 0, 15).to(torch.uint8)
        return q, (float(scale), float(zp))
    elif prec == Precision.INT2:
        mn, mx = t.min(), t.max()
        scale = float((mx - mn) / 3.0)
        scale = max(scale, 1e-10)
        zp = torch.round(-mn / scale).clamp(0, 3)
        q = torch.clamp(torch.round(t / scale + zp), 0, 3).to(torch.uint8)
        return q, (float(scale), float(zp))
    return tensor, None


def _dequantize(quantized, scale, prec: Precision):
    if scale is None:
        return quantized
    if prec == Precision.FP8:
        return (quantized.float() * scale).half()
    elif prec in (Precision.INT4, Precision.INT2):
        s, zp = scale
        return ((quantized.float() - zp) * s).half()
    return quantized
```

---

## 第五步：创建 backend/chunked_transfer.py

**目的**：自适应分块传输，按重要性排序 layer 后分 chunk 发送

**依赖**：
- `backend/dist.py`（send_obj / recv_obj）
- `backend/precision_allocator.py`（Precision）
- `backend/adaptive_quant.py`（quantize_kv / dequantize_kv）
- `backend/token_importance.py`（TokenImportanceEvaluator）
- `backend/pipeline_runner.py`（StageHandle）

**文件路径**：`E:\EdgePD\222222\whole_process_fixedkv\backend\chunked_transfer.py`

```python
"""分块传输器 — ABKT 模块 5

功能：
  1. 按重要性排序 layer（高重要性先传）
  2. 自适应 chunk 大小（带宽高→大 chunk，带宽低→小 chunk）
  3. 每个 chunk 附带量化元数据
  4. 传输完成后通知 decode 端反量化并初始化 KV Cache
"""

import torch
from typing import Dict, List, Tuple
from backend.dist import send_obj, RPC_PRIORITY_HIGH
from backend.precision_allocator import Precision


class ChunkedTransfer:
    def __init__(self, base_chunk_tokens=64, min_chunk=16, max_chunk=256):
        self.base_chunk = base_chunk_tokens
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk

    def compute_chunk_size(self, bandwidth_bps: float) -> int:
        """根据带宽自适应计算 chunk 大小（token 数）"""
        bw_norm = min(bandwidth_bps / 100e6, 1.0)
        size = int(self.min_chunk + (self.max_chunk - self.min_chunk) * bw_norm)
        return max(self.min_chunk, min(size, self.max_chunk))

    def transfer(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        metadata: Dict,
        importance: Dict,
        rpc,
        decode_stage_ranks: Dict[int, int],  # {decode_node_idx: rank}
        bandwidth_bps: float,
        request_id: str,
    ):
        """
        执行分块传输。

        Args:
            kv_cache: 量化后的 KV Cache
            metadata: 量化元数据
            importance: 重要性矩阵
            rpc: DistRpc 实例
            decode_stage_ranks: {decode_node_idx: rank}
            bandwidth_bps: 当前带宽
            request_id: 请求 ID
        """
        chunk_size = self.compute_chunk_size(bandwidth_bps)

        # 按平均重要性对 layer 排序（高重要性先传）
        layer_order = self._sort_layers(kv_cache, importance)

        for dn, layer_cache in kv_cache.items():
            rank = decode_stage_ranks.get(dn)
            if rank is None:
                continue
            dn_meta = metadata.get(dn, {})

            for layer_idx in layer_order:
                if layer_idx not in layer_cache:
                    continue
                kv = layer_cache[layer_idx]
                if kv is None:
                    continue
                k, v = kv
                seq_len = k.shape[2]
                meta = dn_meta.get(layer_idx, {})

                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    is_last = (end >= seq_len)

                    chunk_payload = {
                        "request_id": request_id,
                        "decode_node_idx": dn,
                        "layer_idx": layer_idx,
                        "chunk_start": start,
                        "chunk_end": end,
                        "total_seq_len": seq_len,
                        "k": k[:, :, start:end, :].contiguous(),
                        "v": v[:, :, start:end, :].contiguous(),
                        "meta": meta,
                        "is_last_chunk": is_last,
                    }

                    rpc.call(rank, "receive_kv_chunk", chunk_payload, priority=RPC_PRIORITY_HIGH)

    def _sort_layers(self, kv_cache, importance) -> List[int]:
        """按平均重要性降序排列 layer_idx"""
        layer_imp = {}
        for dn, layer_cache in kv_cache.items():
            imp_node = importance.get(dn, {})
            for li in layer_cache:
                if li in layer_imp:
                    continue
                imp = imp_node.get(li)
                layer_imp[li] = float(imp.mean()) if imp is not None else 0.5
        return sorted(layer_imp.keys(), key=lambda l: layer_imp[l], reverse=True)
```

---

## 第六步：修改 backend/dist.py（小改）

**目的**：添加 probe 端口支持（用于 NetworkProbe 的 RTT 探测）

**修改位置**：文件末尾添加一个轻量级 probe server

**具体修改**：在 `dist.py` 末尾追加以下代码

```python
# ===== ABKT: 轻量级 RTT 探测服务 =====

def start_probe_server(port: int = 29501):
    """在 Decode 节点启动 RTT 探测服务（后台线程）。

    NetworkProbe 发送 64 字节探测包，此服务立即回复，用于测量 RTT。
    """
    import socket
    import threading

    def _serve():
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", port))
            srv.listen(5)
            while True:
                conn, _ = srv.accept()
                try:
                    conn.recv(64)
                    conn.sendall(b'\x01')
                except Exception:
                    pass
                finally:
                    conn.close()
        except Exception as e:
            print(f"[PROBE_SERVER] bind failed on port {port}: {e}")

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t
```

---

## 第七步：修改 backend/modes_split_pd.py（核心改动）

**目的**：在 `_prefill_loop` 中插入 ABKT 传输逻辑

**这是最关键的一步。需要修改 3 个地方。**

### 7.1 修改 `__init__`：添加 ABKT 组件初始化

**位置**：`PDSplitExperiment.__init__()` 方法末尾（`self._running = True` 之后，worker 线程启动之前）

**插入代码**：

```python
        # ===== ABKT 组件初始化 =====
        self._abkt_enabled = os.environ.get("ABKT_ENABLED", "0") == "1"
        if self._abkt_enabled:
            from backend.network_probe import NetworkProbe
            from backend.token_importance import TokenImportanceEvaluator
            from backend.precision_allocator import PrecisionAllocator
            from backend.adaptive_quant import quantize_kv, dequantize_kv
            from backend.chunked_transfer import ChunkedTransfer

            self._network_probe = NetworkProbe(
                max_delay_sec=float(os.environ.get("ABKT_MAX_DELAY", "0.5")),
            )
            self._importance_eval = TokenImportanceEvaluator()
            self._precision_alloc = PrecisionAllocator()
            self._quantize_fn = quantize_kv
            self._chunked_transfer = ChunkedTransfer()
```

### 7.2 添加 `_abkt_transfer` 方法

**位置**：在 `PDSplitExperiment` 类中，`_prefill_loop` 方法之前添加新方法

```python
    def _abkt_transfer(
        self, kv_cache, request_id, seq_len, record
    ):
        """ABKT 自适应传输：重要性评估 → 精度分配 → 量化 → 分块传输"""
        import time as _time

        # 1. 获取网络状态
        net = self._network_probe.get_snapshot()

        # 2. 重要性评估
        importance = self._importance_eval.compute(kv_cache)

        # 3. 精度分配
        alloc = self._precision_alloc.allocate(importance, kv_cache, net.budget_bytes)

        # 4. 量化
        t_q0 = _time.time()
        quantized_kv, meta = self._quantize_fn(kv_cache, alloc.precision_map)
        t_q1 = _time.time()

        # 5. 构建 decode_stage_ranks 映射
        decode_ranks = {}
        for stage in self.decode_stages:
            decode_ranks[stage.stage_id] = stage.rank

        # 6. 分块传输
        t_t0 = _time.time()
        self._chunked_transfer.transfer(
            quantized_kv, meta, importance,
            self.rpc, decode_ranks,
            net.bandwidth_bps, request_id,
        )
        t_t1 = _time.time()

        # 7. 用传输数据校准带宽
        transfer_bytes = alloc.total_bytes
        transfer_elapsed = t_t1 - t_t0
        if transfer_elapsed > 0:
            self._network_probe.update_bandwidth(transfer_bytes, transfer_elapsed)

        # 日志
        self.tracker.log_info(
            f"[ABKT] req={request_id} state={net.state.value} "
            f"budget={net.budget_bytes/1e6:.1f}MB actual={alloc.total_bytes/1e6:.1f}MB "
            f"compression={alloc.compression_ratio:.2f}x "
            f"quantize_ms={(t_q1-t_q0)*1000:.1f} transfer_ms={transfer_elapsed*1000:.1f}"
        )
```

### 7.3 修改 `_prefill_loop`：替换 KV Cache 发送逻辑

**位置**：`_prefill_loop` 方法中，第 296-317 行（`with self._cond:` 块中，发送 kv 和 token_ids 的部分）

**原始代码**（第 296-317 行附近）：
```python
                with self._cond:
                    for b, kv, token_ids in zip(batch, kv_splits, prefill_token_ids_list):
                        req_id = b["record"].request_id
                        self._kv_ready[req_id] = kv
                        self._prefill_token_ids[req_id] = token_ids
                        self.tracker.decode_queue_len += 1

                        # 异步发送 token_ids 到 decode 首节点（不等待响应）
                        if decode_first_rank is not None and decode_first_rank != self.dist_ctx.rank:
                            _dprint(f"[DEBUG PRE] RPC send_token_ids req={req_id} token_ids={token_ids} -> decode_first_rank={decode_first_rank}")
                            try:
                                self.rpc.call(
                                    decode_first_rank,
                                    "send_token_ids",
                                    {"request_id": req_id, "token_ids": token_ids},
                                    priority=RPC_PRIORITY_HIGH,
                                )
                            except Exception as e:
                                print(f"[WARN] send_token_ids failed for req {req_id}: {e}")
                        else:
                            _dprint(f"[DEBUG PRE] local send_token_ids req={req_id} token_ids={token_ids} (decode_first_rank={decode_first_rank})")
                    self._cond.notify_all()
```

**替换为**：
```python
                with self._cond:
                    for b, kv, token_ids in zip(batch, kv_splits, prefill_token_ids_list):
                        req_id = b["record"].request_id
                        seq_len = len(b["input_ids"])

                        # ===== ABKT: 自适应传输 或 原有逻辑 =====
                        if self._abkt_enabled and kv is not None:
                            # ABKT 路径：量化 + 分块传输（替代原有 init_kv）
                            self._abkt_transfer(kv, req_id, seq_len, b["record"])
                            # 标记 KV 已通过 ABKT 传输，decode 端会自行初始化
                            self._kv_ready[req_id] = "__abkt_transmitted__"
                        else:
                            # 原有路径：直接存入 _kv_ready
                            self._kv_ready[req_id] = kv

                        self._prefill_token_ids[req_id] = token_ids
                        self.tracker.decode_queue_len += 1

                        # 发送 token_ids 到 decode 首节点（原有逻辑，ABKT 下也需要）
                        if decode_first_rank is not None and decode_first_rank != self.dist_ctx.rank:
                            _dprint(f"[DEBUG PRE] RPC send_token_ids req={req_id} token_ids={token_ids} -> decode_first_rank={decode_first_rank}")
                            try:
                                self.rpc.call(
                                    decode_first_rank,
                                    "send_token_ids",
                                    {"request_id": req_id, "token_ids": token_ids},
                                    priority=RPC_PRIORITY_HIGH,
                                )
                            except Exception as e:
                                print(f"[WARN] send_token_ids failed for req {req_id}: {e}")
                        else:
                            _dprint(f"[DEBUG PRE] local send_token_ids req={req_id} token_ids={token_ids} (decode_first_rank={decode_first_rank})")
                    self._cond.notify_all()
```

**注意**：ABKT 模式下 `_kv_ready[req_id]` 存的是 `"__abkt_transmitted__"` 字符串标记，而不是实际的 KV Cache。decode 端会通过 `receive_kv_chunk` 接口自行接收和组装 KV Cache。

### 7.4 修改 `worker_server`：添加 chunk 接收 handler

**位置**：`worker_server` 方法中，`handlers` 字典构建处（`modes_split_pd.py` 第 651-663 行）

**在 `handlers = {` 之前添加 chunk 缓冲区和处理函数**：

```python
        # ===== ABKT: chunk 接收缓冲区 =====
        _chunk_buffers = {}  # {req_id: {layer_idx: {"chunks": [], "total": N, "meta": {}}}}
        _chunk_lock = threading.Lock()

        def _receive_kv_chunk(request_id, decode_node_idx, layer_idx,
                              chunk_start, chunk_end, total_seq_len,
                              k, v, meta, is_last_chunk):
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

            # 如果是最后一个 chunk，检查是否所有 layer 都收齐了
            if is_last_chunk:
                buf = _chunk_buffers.get(request_id)
                if buf is None:
                    return True

                # 检查每个 layer 的所有 chunk 是否收齐
                all_complete = True
                for li, info in buf.items():
                    received_end = max(c[1] for c in info["chunks"])
                    if received_end < info["total_seq_len"]:
                        all_complete = False
                        break

                if all_complete:
                    # 拼接所有 chunk，反量化，初始化 KV Cache
                    from backend.adaptive_quant import dequantize_kv

                    assembled_kv = [None] * len(local_decode.layers) if local_decode else []
                    assembled_meta = {}

                    for li, info in buf.items():
                        # 按 chunk_start 排序后拼接
                        sorted_chunks = sorted(info["chunks"], key=lambda c: c[0])
                        full_k = torch.cat([c[2] for c in sorted_chunks], dim=2)
                        full_v = torch.cat([c[3] for c in sorted_chunks], dim=2)
                        if li < len(assembled_kv):
                            assembled_kv[li] = (full_k, full_v)
                        assembled_meta[li] = info["meta"]

                    # 反量化
                    if local_decode is not None:
                        # 构造与 dequantize_kv 兼容的格式
                        qkv = {0: {li: kv for li, kv in enumerate(assembled_kv) if kv is not None}}
                        mmeta = {0: assembled_meta}
                        dequantized = dequantize_kv(qkv, mmeta)
                        final_kv = [None] * len(local_decode.layers)
                        for li2, kv2 in dequantized.get(0, {}).items():
                            if li2 < len(final_kv):
                                final_kv[li2] = kv2
                        local_decode.init_kv(request_id, final_kv)

                    with _chunk_lock:
                        del _chunk_buffers[request_id]

            return True
```

**然后在 `handlers` 字典中添加**：
```python
        handlers = {
            # ... 原有 handlers ...
            "receive_kv_chunk": _receive_kv_chunk,  # ← 新增
        }
```

---

## 第八步：修改 backend/pipeline_runner.py（小改）

**目的**：decode 端识别 ABKT 传输模式，跳过原有的 `init_kv` 调用

**修改位置**：`run_decode_logged` 函数中，`_call_init_kv` 调用处

**原始代码**（约第 286-295 行）：
```python
        for stage in stages:
            stage_kv = _normalize_stage_kv_for_decode(stage, kv_cache)
            ...
            _call_init_kv(stage, rpc, req_id, stage_kv)
```

**替换为**：
```python
        for stage in stages:
            stage_kv = _normalize_stage_kv_for_decode(stage, kv_cache)
            # ABKT 模式：KV Cache 已通过 chunk 接收器初始化，跳过 init_kv
            if isinstance(kv_cache, str) and kv_cache == "__abkt_transmitted__":
                _dprint(f"[DEBUG RUN_DEC] ABKT mode, skipping init_kv for stage={stage.stage_id}")
                continue
            ...
            _call_init_kv(stage, rpc, req_id, stage_kv)
```

**同样修改** `run_decode_logged_batch` 中的对应位置。

---

## 第九步：修改 serve.py（小改）

**目的**：添加 ABKT 相关启动参数

**修改位置**：`serve.py` 的 argparse 部分，添加以下参数

```python
    parser.add_argument("--abkt", action="store_true",
                        help="Enable ABKT adaptive transfer")
    parser.add_argument("--abkt_max_delay", type=float, default=0.5,
                        help="Max transfer delay budget (seconds)")
```

**在环境变量设置处**：
```python
    if args.abkt:
        os.environ["ABKT_ENABLED"] = "1"
        os.environ["ABKT_MAX_DELAY"] = str(args.abkt_max_delay)
```

**在 Decode 节点启动处，添加 probe server**：
```python
    # 启动 RTT 探测服务（仅 Decode 节点）
    if args.abkt and rank != 0:  # rank 0 是 Prefill，其他是 Decode
        from backend.dist import start_probe_server
        start_probe_server(port=29501)
```

**在 Prefill 节点（rank 0）启动 NetworkProbe**：
```python
    # 启动网络探测（仅 Prefill 节点）
    if args.abkt and rank == 0:
        # probe 目标是第一个 decode 节点
        decode_host = cluster_config.get(decode_nodes[0], {}).get("ip", "192.168.0.11")
        experiment._network_probe.start(decode_host, probe_port=29501)
```

---

## 修改顺序总结

```
执行顺序（严格按此顺序）：

Step 1: 创建 backend/network_probe.py       ← 独立，可单独测试
Step 2: 创建 backend/token_importance.py    ← 独立，可单独测试
Step 3: 创建 backend/precision_allocator.py ← 独立，可单独测试
Step 4: 创建 backend/adaptive_quant.py      ← 依赖 Step 3
Step 5: 创建 backend/chunked_transfer.py    ← 依赖 Step 3, 4

Step 6: 修改 backend/dist.py               ← 添加 probe server（小改，追加代码）
Step 7: 修改 backend/modes_split_pd.py     ← 核心改动，4 处修改
  7.1: __init__ 添加 ABKT 组件初始化
  7.2: 添加 _abkt_transfer 方法
  7.3: _prefill_loop 中替换 KV 发送逻辑
  7.4: worker_server 中添加 receive_kv_chunk handler

Step 8: 修改 backend/pipeline_runner.py     ← 小改，ABKT 模式跳过 init_kv
Step 9: 修改 serve.py                       ← 添加启动参数
```

---

## 验证方案

**每步验证**：

```
Step 1-5 完成后：
  python -c "from backend.network_probe import NetworkProbe; print('OK')"
  python -c "from backend.token_importance import TokenImportanceEvaluator; print('OK')"
  python -c "from backend.precision_allocator import PrecisionAllocator; print('OK')"
  python -c "from backend.adaptive_quant import quantize_kv; print('OK')"
  python -c "from backend.chunked_transfer import ChunkedTransfer; print('OK')"

Step 6-9 完成后：
  # 不启用 ABKT，验证原有功能不受影响
  bash scripts/run_experiment.sh --mode pd_split --model 6.7b --samples 5

  # 启用 ABKT，验证新功能
  ABKT_ENABLED=1 bash scripts/run_experiment.sh --mode pd_split --model 6.7b --samples 5
```

---

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ABKT_ENABLED` | `0` | 是否启用 ABKT（`1` 启用） |
| `ABKT_MAX_DELAY` | `0.5` | 最大传输延迟预算（秒） |
| `ABKT_PROBE_PORT` | `29501` | RTT 探测服务端口 |
| `ABKT_CHUNK_MIN` | `16` | 最小 chunk 大小（token 数） |
| `ABKT_CHUNK_MAX` | `256` | 最大 chunk 大小（token 数） |
