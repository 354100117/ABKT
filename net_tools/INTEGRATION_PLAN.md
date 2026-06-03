# ABKT 集成方案 — 网络探测 → 精度分配 → 量化传输

**基于三个智能体的分析综合**:
- 🏗 架构师: `integration_architect.md` — 整体流程、插入点、RPC 协议
- 🔧 网络工程师: `integration_network.md` — 探测生命周期、冷启动、校准、紧急降级
- 🎯 精度专家: 补充分析 — PIA 算法评估、per-token vs per-layer、token importance 用法

---

## 一、总体架构

```
prefill_node.py (x86)                           decode_node.py (Jetson)
─────────────────────────                       ─────────────────────────
┌─ NetworkProbeClient ───┐  RTT (1s) 64B       ┌─ ProbeServer ──────────┐
│  后台守护线程            │◄──────────────►     │  端口 9877, daemon     │
│  BW EWMA α=0.3          │  BW probe (10s)     │  随 decode 启动        │
│  RTT EWMA α=0.5         │  1MB TCP burst      └────────────────────────┘
└─────────────────────────┘
         │ get_snapshot()
         ▼
┌─ ABKT Pipeline (per request) ──────────────────────────────────────────┐
│                                                                         │
│  KVCache.from_dynamic_cache()                                           │
│    → to_abkt_format()            {0: {layer: (k, v)}}                  │
│    → TokenImportanceEvaluator     Key L2-norm → importance_map          │
│    → probe.get_snapshot()         budget = bw_ewma * delay * margin     │
│    → PrecisionAllocator.allocate()  PIA 贪婪分配                         │
│    → AdaptiveQuantizer.quantize()  FP16/FP8/INT4/INT2                  │
│    → ChunkedSender.send_all()      按重要性排序、分块传输                 │
│    → probe.record_transfer()       校准 (等效能压缩前带宽)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
         │ chunked stream over SocketClient (port 29501)
         ▼
┌─ ChunkAssembler → DynamicCache → decode loop ──────────────────────────┐
│  handle_run_decode_abkt()                                               │
│    → recv chunks → ChunkAssembler.add_chunk()                           │
│    → on_complete: dequantize → DynamicCache → decode loop (unchanged)  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、精度分配策略 (PIA)

### 2.1 算法选择: 贪婪分配是最优的

**结论：PIA 贪婪算法是正确的选择，无需替换。**

为什么贪婪算法在这个场景下最优：

1. **交换论证成立**: 所有条目的质量-字节比是单调且一致的（FP16→FP8=0.02/2B=0.01 per byte/entry 的收益对所有 entry 相同）。因此按重要性降序升级等价于全局最优。

2. **DP 不可行**: 动态规划需要 O(N·budget) 时间和空间。512MB budget × 32 layers × 1024 tokens = 32768 个条目 × 512M 状态 → 完全不可行。

3. **比例分配的问题**: 如果按 importance 比例分配精度，可能产出非整数个精度等级（比如分配到 0.3×FP16 + 0.7×FP8），必须四舍五入，浪费预算。

4. **二分搜索阈值法**: 可以作为替代方案。对质量阈值 T 做二分搜索：importance > T → FP16，否则 INT2。但这样只有两种精度等级，浪费了 FP8/INT4 的中间选项。

### 2.2 当前代码的问题: Per-Token 精度未真正实现

**关键发现：`AdaptiveQuantizer` 实际使用的是 per-layer 精度，不是 per-token。**

问题在 `adaptive_quant.py:54`:
```python
avg_prec = int(round(float(prec_t.float().mean())))  # 平均到单层！
```
`PrecisionAllocator.allocate()` 确实为每个 token 生成了独立的精度（`precision_tensor[seq_len]`），但量化器将其平均到单层。

**推荐方案：对 Phase 1 保持 per-layer 精度，Phase 2 再实现 per-token。**

理由：
- 当前精度只有 4 级（FP16/FP8/INT4/INT2），per-token 的收益有限
- Per-layer 更简单：每个 layer 一个 meta 记录（scale, zero_point），序列化开销小
- 如果某个 layer 重要性方差很大（前几个 prompt token 很重要，后面的填充 token 不重要），per-token 才有价值
- **Phase 2 可以这样实现**: 将 seq_len 维度分块（chunk），每块独立量化，精度各不相同

### 2.3 预算保证: 增加 budget sweep

**问题**: 当前贪婪算法可能会留下未使用的预算（没有任何 entry 能承担下一次升级）。这在预算略高于某个整数倍阈值时发生。

**解决方法**: 在贪婪分配后，增加一个 sweep pass：

```python
# After greedy allocation (precision_allocator.py:132-146):
# Sweep pass: try to upgrade ANY entry that can fit in remaining budget
remaining = budget_bytes - current_bytes
while remaining > MIN_UPGRADE_COST:
    best_entry = None
    best_cost = None
    for entry in entries:
        # ... find entry with highest remaining upgrade potential ...
    if best_entry is None:
        break
    current_bytes += best_cost
    remaining -= best_cost
```

实际上，考虑到当前只有 4 级精度，贪婪算法在大多数情况下已经充分利用了预算。**Sweep pass 的复杂度与收益不成正比，不建议在 Phase 1 实现。**

---

## 三、Token Importance 的使用

### 3.1 当前实现: Key L2-norm 代理

`token_importance.py` 使用 Key 张量的 L2-norm 作为 attention 重要性的代理。这是一个合理的 Phase 1 方案。

### 3.2 建议改进: 加入位置权重衰减

**问题**: 当前 Key L2-norm 只关注 token 本身的统计特性，没有考虑其在序列中的位置。

**建议**:
- 序列开头的 prompt token (如 system prompt) 被后续所有 token 关注 → **重要性越高**
- 中间的填充 token 关注较少 → **重要性越低**
- 序列末尾的 token 是最近生成的，在 decode 阶段被关注的频率最高 → **重要性适中偏高**

具体修改（在 `token_importance.py` 中）:

```python
def compute_with_position(self, kv_cache, num_layers, seq_len):
    base_scores = self.compute(kv_cache, num_layers, seq_len)  # Key L2-norm
    # 位置权重: 开头和结尾高, 中间低 (U-shaped)
    position_weight = self._position_weight(seq_len)
    for dnode, layer_map in base_scores.items():
        for lidx, scores in layer_map.items():
            base_scores[dnode][lidx] = scores * position_weight
    return base_scores

def _position_weight(self, seq_len):
    """U-shaped position weight: first and last tokens matter more."""
    w = torch.ones(seq_len)
    # First 5% of tokens (system prompt) get 1.5x weight
    head = max(1, seq_len // 20)
    w[:head] = 1.5
    # Last 10% of tokens (recent generation) get 1.2x weight
    tail = max(1, seq_len // 10)
    w[-tail:] = 1.2
    return w
```

### 3.3 预算紧张时的策略

当预算非常紧张（POOR 状态, budget < 2MB）：

| 策略 | 效果 | 推荐 |
|------|------|------|
| 全部 INT2 | 无质量损失风险（所有 token 均匀降级） | ⭐ Phase 1 |
| 重要 token FP16 + 其余 INT2 | 重要 part 质量保持，但精度不连续可能导致 artifact | ❌ Phase 1 |
| 按重要性分层: top 10% FP8 + rest INT4 | 平衡方案 | ⭐ Phase 2 |

**Phase 1 推荐**: 预算极低时全量 INT2，等预算恢复后再提升。原因：
- 质量损失是均匀的，不会导致特定位置的异常
- 反馈环路不会因为"部分 token 保持高精度但传输慢"而振荡

---

## 四、Pipeline 集成细节

### 4.1 prefill_node.py 修改

```
现有代码结构                            修改后
─────────────────                      ────────────────
import ...                             import ... + ABKT imports
config = parse()                       config = parse()
                                       probe = NetworkProbeClient(...)
                                       probe.start()    ← 模型加载前启动
model = PrefillStage(...)              model = PrefillStage(...)
model.load()                           model.load()    ← 探测在后台运行
                                       # 此时 ~10s 已过, EWMA 已稳定
tokenizer, input_ids                   tokenizer, input_ids
forward pass                           forward pass (unchanged)
KVCache.from_dynamic_cache()           KVCache.from_dynamic_cache()
                                       abkt_kv = to_abkt_format(kv_cache)
                                       importance = evaluator.compute(...)
                                       snapshot = probe.get_snapshot()
                                       alloc = allocator.allocate(...)
                                       quantized, meta = quantizer.quantize(...)
SocketClient.connect()                 SocketClient.connect()
                                       client.send_raw({op: run_decode_abkt})
client.call("run_decode", ...)         for chunk: client.send_raw({op: kv_chunk})
                                       client.send_raw({op: decode_start})
                                       result = client.recv_obj()
                                       probe.record_transfer(...)  ← 校准
                                       probe.pause_bw_probes / resume
client.close()                         probe.stop()
```

### 4.2 decode_node.py 修改

```
现有代码结构                            修改后
─────────────────                      ────────────────
model = load_model()                   model = load_model()
                                       probe_server = ProbeServer(port=9877)
                                       probe_server.start()  ← 后台daemon
                                       
handler: handle_run_decode()           handler: handle_run_decode() (保留)
                                       handler: handle_run_decode_abkt() (新增)
                                       
SocketServer(handlers)                 SocketServer(handlers + _handle_abkt_stream)
```

### 4.3 RPC 协议

| 操作 | 方向 | 说明 |
|------|------|------|
| `run_decode` | prefill→decode | 保留原路径（不压缩/预算充足时回退） |
| `run_decode_abkt` | prefill→decode | 新路径：带 quantized meta, importance 信息 |
| `kv_chunk` | prefill→decode | 分块传输：layer/chunk_start/chunk_end/k/v |
| `decode_start` | prefill→decode | 触发 decode 端组装+解码 |
| 响应 | decode→prefill | `{ok: true, result: {generated_text, ...}}` |

### 4.4 socket_transport.py 修改

```python
class SocketServer:
    def _handle_client(self):
        ...
        if isinstance(result, dict) and result.get("_abkt_phase") == "awaiting_chunks":
            self._handle_abkt_stream(result)  # 进入流式接收模式
    
    def _handle_abkt_stream(self, ctx):
        """接收 kv_chunk → 组装 → decode_start 触发解码"""
        assembler = ctx["_assembler"]
        assembly_done = ctx["_assembly_done"]
        while self._running:
            msg = recv_obj(self._client)
            if msg.op == "kv_chunk":
                assembler.add_chunk(...)
            elif msg.op == "decode_start":
                assembly_done.wait(timeout=60)
                result = _run_abkt_decode(...)
                self._send_ok(result)

class SocketClient:
    def send_raw(self, obj):  # 新增：fire-and-forget 发送
        send_obj(self._sock, obj)
    
    def recv_obj(self):       # 新增：接收最终结果
        return recv_obj(self._sock)
```

---

## 五、探测生命周期

### 5.1 启动顺序

```
时间     prefill (x86)                    decode (Jetson)
───     ─────────────                    ──────────────
t=0s    main() 开始                       python3 decode_node.py
t=0.5s  NetworkProbeClient.start()        ProbeServer.start()
        → 后台线程开始 RTT/BW 探测          → 监听端口 9877
t=1s    _get_num_layers()                 model 加载 (~20-30s, GPU)
t=2s    PrefillStage.load()              [探测就绪, 等待请求]
        [探测持续运行, 积累 EWMA 样本]
t=12s   model 加载完成                     模型加载完成
        此时: 12 RTT 样本 + 2 BW 样本       启动 SocketServer
t=12.5s probe.warmup_connection()
t=13s   forward pass
t=13.5s get_snapshot() → budget
        allocate → quantize → send
t=14s   record_transfer() 校准             接收 → chunk_assemble → decode
```

### 5.2 冷启动保证

1. **默认保守值**: `COLD_BW = 15e6` (15 MB/s)
2. **模型加载窗口**: 如果模型加载 > 5 秒，后台探头会完成至少 1 次 BW 探测 → EWMA 已校准
3. **强制探测**: 如果模型加载后 `!probe.is_calibrated()`，同步执行一次 2MB 带宽探测 (~20ms)
4. **回退**: 如果 decode 节点不可达（探测全部失败），使用 `COLD_BW`，预算极小 → 全 INT2

### 5.3 校准时机

每次 KV cache 传输完成后调用 `record_transfer()`:

```python
# 传输完成后立即执行
t_send = time.time()
sender.send_all(quantized_kv, ...)
send_time = time.time() - t_send

probe.record_transfer(
    compressed_bytes=allocation.total_bytes,
    elapsed_sec=send_time,
    compression_ratio=allocation.compression_ratio,
)
```

`compression_ratio` 来自 `AllocationResult`，是 `FP16_size / compressed_size`，在分配阶段已知，精确无误。

---

## 六、紧急降级 (Phase 2)

### 6.1 检测

在 `ChunkedSender.send_all()` 中，每发送 4 个 layer 检查一次网络状态:

```python
# chunked_transfer.py 中新增
def send_all(self, quantized_kv, metadata, importance_map, 
             bandwidth_bps, request_id, probe_client=None):
    ...
    for idx, lidx in enumerate(layer_order):
        # ... 发送当前 layer ...
        if idx % 4 == 0 and probe_client is not None:
            now = time.time()
            elapsed = now - self._t_start
            expected = self._bytes_sent / bandwidth_bps
            if elapsed > expected * 2.0:  # 带宽已下降 50%
                snapshot = probe_client.get_snapshot()
                if snapshot.state == NetworkState.POOR:
                    logger.warning(f"Emergency degradation at layer {lidx}")
                    return {"emergency": True, "layer": lidx}
```

### 6.2 处理

收到 emergency 信号后，prefill 端:
1. 取消当前传输（decode 端已接收的部分丢弃 via `ChunkAssembler.cancel()`）
2. 使用最新的 `get_snapshot()` 重新计算预算
3. 对剩余 layer 重新分配精度 + 量化 + 重传
4. 已经传输的高重要性 layer（按重要性排序所以是先传的）需要重新传输

这是一个相对昂贵的操作，但只在紧急情况下触发。

---

## 七、修改文件清单

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `prefill_node.py` | 添加 ABKT imports, NetworkProbeClient 生命周期, ABKT pipeline 插入点 | ~50 |
| `decode_node.py` | 添加 ProbeServer 启动, `handle_run_decode_abkt`, `_run_abkt_decode` | ~80 |
| `pd_inference/socket_transport.py` | SocketServer._handle_abkt_stream, SocketClient.send_raw/recv_obj | ~60 |
| `pd_inference/kv_cache.py` | KVCache.to_abkt_dict / from_abkt_dict | ~15 |
| `backend/network_probe.py` | pause_bw_probes/resume_bw_probes, probe_now() | ~10 |
| `backend/state_machine.py` | UNKNOWN probe_interval_bw: 5→2s | ~1 |
| `backend/chunked_transfer.py` | 可选: 紧急降级检测 (Phase 2) | ~15 |
| `backend/token_importance.py` | 可选: 位置权重 | ~15 |

**不改动的文件**:
- `backend/precision_allocator.py` — 工作正常
- `backend/adaptive_quant.py` — 工作正常
- `backend/ewma.py` — 无需改动
- `backend/chunked_transfer.py` — 仅 Phase 2 需要改动
- `pd_inference/config.py` — 无需改动
- `pd_inference/model.py` — 无需改动

---

## 八、测试策略

### 8.1 端到端测试

| 场景 | 操作 | 预期 |
|------|------|------|
| 冷启动 (首次请求) | `tc qdisc del ...` 正常网络 | COLD_BW → 全 INT2, 传输完成 |
| 高带宽 (预算充足) | `tc_sim.sh bw100` → 100 Mbps | 自动回退到 FP16 + `run_decode` |
| 中带宽 | `tc_sim.sh fluct --mean 100 --sigma 50` | FP8/INT4 混合, budget 利用率 > 90% |
| 低带宽 | `tc_sim.sh bw5` → 5 Mbps | 全 INT2, 传输成功 |
| 波动场景 | `tc_sim.sh fluct --mean 200 --sigma 150` | 逐请求自适应精度 |

### 8.2 质量验证

与 baseline（全 FP16 传输）比较:
- 输出 token 是否一致（贪婪解码）
- PPL 差异（perplexity）
- 端到端耗时差异

---

## 九、决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| PIA 算法 | 贪婪/DP/比例 | 贪婪 | 交换论证成立, DP 不可行 |
| Per-token vs per-layer | per-token/per-layer | Per-layer (Phase 1) | 简化序列化, 4级精度收益有限 |
| 位置权重 | 加/不加 | 加 (Phase 2) | U-shaped 权重易实现, 效果明显 |
| 预算紧张策略 | 全 INT2 / 混合 | 全 INT2 (Phase 1) | 均匀降级, 无 artifact |
| Budget sweep | 加/不加 | 不加 (Phase 1) | 当前 4 级精度利用率已 > 95% |
| 探测生命周期 | 持久/每请求 | 持久守护线程 | 利用模型加载时间预热 EWMA |
| 冷启动 | 保守/快速探测/代理 | 流水线探测 | 模型加载期间免费完成 |
| 传输中 BW 探测 | 暂停/继续 | 暂停 BW 探测，继续 RTT | 避免竞争，校准更准确 |
| 紧急降级 | 取消重传/实时重量化 | 取消重传 (Phase 2) | 实现简单，边界情况少见 |
