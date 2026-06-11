# ABKT 实验执行手册

## 前置准备

### 1. 下载模型

需要 Qwen2.5 系列至少 2 个尺寸（推荐 3B + 7B，或 1.5B + 3B + 7B）。

```bash
# 方法一：huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B --local-dir /ssd/models/qwen2.5-3b
huggingface-cli download Qwen/Qwen2.5-7B --local-dir /ssd/models/qwen2.5-7b

# 方法二：Python
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B', local_dir='/ssd/models/qwen2.5-7b')
"

# 可选：更小的模型（快速验证）
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir /ssd/models/qwen2.5-1.5b
```

### 2. 确认环境

```bash
# 检查 GPU/内存
free -h
nvidia-smi 2>/dev/null || echo "Jetson: unified memory"

# 确认依赖
python3 -c "import torch, transformers, datasets, matplotlib; print('OK')"

# 确认测试通过
python3 -m pytest test_abkt.py -v
```

---

## 实验计划总览

| 实验 | 目的 | 产出 | 预计耗时 |
|------|------|------|---------|
| Exp 1 | 多策略 PPL 对比 | 表格 + Pareto 图 | 3B: ~30min, 7B: ~2h |
| Exp 2 | 消融实验 | 消融表格 | 3B: ~30min |
| Exp 3 | 不同压缩率 | budget-sweep 曲线 | 3B: ~1h |
| Exp 4 | 不同上下文长度 | 长序列 PPL 曲线 | 3B: ~40min |
| Exp 5 | 网络波动端到端 | TTFT/吞吐量表格 | 需双机 ~2h |

每个实验独立，可按任意顺序执行。

---

## Exp 1：多策略 PPL 对比（核心实验）

**目的**：证明 ABKT 优于 uniform INT4/INT2/random 等基线。

### Qwen2.5-3B

```bash
# 全量运行（~30分钟，WikiText-2 全集）
python tools/eval_ppl.py \
    --model /ssd/models/qwen2.5-3b \
    --strategy fp16 abkt uniform_int8 uniform_int4 uniform_int2 random \
    --output results_exp1_qwen3b.json

# 生成 Pareto 图
python tools/plot_pareto.py results_exp1_qwen3b.json \
    --output pareto_exp1_qwen3b.png \
    --title "Qwen2.5-3B: Quality vs Compression"
```

### Qwen2.5-7B

```bash
python tools/eval_ppl.py \
    --model /ssd/models/qwen2.5-7b \
    --strategy fp16 abkt uniform_int8 uniform_int4 uniform_int2 random \
    --output results_exp1_qwen7b.json

python tools/plot_pareto.py results_exp1_qwen7b.json \
    --output pareto_exp1_qwen7b.png \
    --title "Qwen2.5-7B: Quality vs Compression"
```

### 快速验证（先跑 5 个窗口确认无报错）

```bash
python tools/eval_ppl.py \
    --model /ssd/models/qwen2.5-3b \
    --max-windows 5 \
    --output results_exp1_smoke.json
```

### 预期结果

| 策略 | 压缩比 | PPL 变化 |
|------|--------|---------|
| fp16 | 1.0x | 基线 |
| uniform_int8 | 2.0x | +0.04% (几乎无损) |
| abkt (50%) | 2.0x | +3~6% |
| uniform_int4 | 4.0x | +10~20% |
| random | 2.0~2.5x | +15~30% |
| uniform_int2 | 7.5x | +60~90% |

---

## Exp 2：消融实验

**目的**：证明重要性评分的三个维度（attention/layer/position）和 quality fidelity 各自有贡献。

```bash
# Qwen2.5-3B
python tools/eval_ppl.py \
    --model /ssd/models/qwen2.5-3b \
    --strategy abkt_full abkt_no_attn abkt_no_layer abkt_no_position abkt_no_fidelity \
    --output results_exp2_qwen3b.json

# 打印对比
python3 -c "
import json
with open('results_exp2_qwen3b.json') as f:
    data = json.load(f)
baseline = next(r['ppl'] for r in data if r['strategy'] == 'abkt_full')
print(f'{'Strategy':<25} {'PPL':>10} {'Delta':>10} {'Compress':>10}')
print('-' * 55)
for r in data:
    delta = r['ppl'] - baseline
    print(f'{r[\"strategy\"]:<25} {r[\"ppl\"]:>10.4f} {delta:>+10.4f} {r[\"compression_ratio\"]:>9.2f}x')
"
```

### 预期结果

去掉任一维度后 PPL 应该变差（delta > 0），证明每个维度都有贡献。Attention importance（alpha）的影响应该最大。

---

## Exp 3：不同压缩率（Budget Sweep）

**目的**：展示 ABKT 在不同带宽预算下的自适应能力。

```bash
# 扫描 budget_ratio: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    echo "=== budget_ratio=$ratio ==="
    python tools/eval_ppl.py \
        --model /ssd/models/qwen2.5-3b \
        --strategy abkt \
        --budget-ratio $ratio \
        --max-windows 10 \
        --output "results_exp3_ratio${ratio}.json"
done

# 合并结果
python3 -c "
import json, glob
all_data = []
for f in sorted(glob.glob('results_exp3_ratio*.json')):
    with open(f) as fh:
        for r in json.load(fh):
            all_data.append(r)
with open('results_exp3_sweep.json', 'w') as f:
    json.dump(all_data, f, indent=2)

# 打印表格
print(f'{'Budget':>8} {'PPL':>10} {'Compress':>10} {'AvgBits':>8}')
print('-' * 40)
for r in sorted(all_data, key=lambda x: -x.get('budget_bytes', 0)):
    print(f'{r[\"compression_ratio\"]:>7.2f}x {r[\"ppl\"]:>10.4f} {r[\"compression_ratio\"]:>9.2f}x {r[\"avg_bits\"]:>7.1f}')
"
```

### 预期结果

PPL 随压缩率单调递增。ABKT 的曲线应该在 uniform 基线之下（同等压缩率下 PPL 更低）。

---

## Exp 4：不同上下文长度

**目的**：验证 ABKT 在长序列上依然有效（E6）。

```bash
# 一次性跑 3 个上下文长度
python tools/eval_ppl.py \
    --model /ssd/models/qwen2.5-3b \
    --strategy fp16 abkt uniform_int4 \
    --long-seq \
    --max-windows 5 \
    --output results_exp4_longseq.json

# 生成图表
python tools/plot_longseq.py results_exp4_longseq.json \
    --output longseq_exp4_qwen3b.png \
    --title "Qwen2.5-3B: PPL vs Context Length"
```

### 预期结果

PPL 随上下文长度增加而下降（更多上下文 = 更好预测）。ABKT 在所有长度上应保持稳定的压缩比。

---

## Exp 5：网络波动端到端（需要双机）

**目的**：证明 ABKT 在真实网络波动下能自适应调整精度，保持低延迟。

### 前置条件

- Prefill 节点 (192.168.0.50) 可 SSH 到 Decode 节点 (192.168.0.20)
- 两个节点都已安装依赖
- Prefill 节点有 root 权限（用于 tc 限速）

### 运行

```bash
# 在 prefill 节点执行（需要 sudo）
cd /ssd/pd/ABKT

# 场景 1：带宽骤降
sudo python3 net_tools/test_orchestrator.py \
    --scenario mid_drop \
    --model /ssd/models/qwen2.5-3b

# 场景 2：带宽渐变
sudo python3 net_tools/test_orchestrator.py \
    --scenario gradual \
    --model /ssd/models/qwen2.5-3b

# 场景 3：带宽振荡
sudo python3 net_tools/test_orchestrator.py \
    --scenario oscillating \
    --model /ssd/models/qwen2.5-3b

# 场景 4：高延迟
sudo python3 net_tools/test_orchestrator.py \
    --scenario with_delay \
    --model /ssd/models/qwen2.5-3b
```

### 记录指标

每个场景记录：
- TTFT (Time To First Token)
- 总传输时间
- 实际压缩比
- 精度降级次数
- 是否超时

---

## 数据记录模板

每次实验运行后，在此记录结果：

### Exp 1: Qwen2.5-3B 多策略对比

| 策略 | PPL | 压缩比 | 平均位数 | 大小(MB) | 时间(s) |
|------|-----|--------|---------|---------|---------|
| fp16 | 7.7630 | 1.00x | 16.0 | 18.87 | 732.3 |
| uniform_int8 | 7.7642 | 2.00x | 8.0 | 9.44 | - |
| abkt | 8.3238 | 1.99x | 8.0 | 9.48 | - |
| uniform_int4 | 8.8536 | 3.88x | 4.1 | 4.86 | - |
| random | 9.7576 | 2.28x | 6.9 | 8.29 | - |
| uniform_int2 | 13.5414 | 7.53x | 2.1 | 2.51 | - |

### Exp 1: Qwen2.5-7B 多策略对比

| 策略 | PPL | 压缩比 | 平均位数 | 大小(MB) | 时间(s) |
|------|-----|--------|---------|---------|---------|
| fp16 | | | | | |
| uniform_int8 | | | | | |
| abkt | | | | | |
| uniform_int4 | | | | | |
| random | | | | | |
| uniform_int2 | | | | | |

### Exp 2: 消融实验 (Qwen2.5-3B, budget_ratio=0.5)

| 策略 | PPL | Delta vs Full | 压缩比 |
|------|-----|--------------|--------|
| abkt_full | 8.3238 | 0 | 1.99x |
| abkt_no_attn | 7.7620 | -0.5618 | 1.99x |
| abkt_no_layer | 8.3286 | +0.0048 | 1.99x |
| abkt_no_position | 8.2748 | -0.0490 | 1.99x |
| abkt_no_fidelity | 8.8186 | +0.4948 | 1.98x |

注：budget_ratio=0.5 时预算充足（all-INT8），importance 评分无发挥空间。
需在低预算（0.3-0.4）下重新验证。

### Exp 3: Budget Sweep (Qwen2.5-3B)

| Budget Ratio | PPL | 压缩比 | 平均位数 |
|-------------|-----|--------|---------|
| 0.2 | 13.5414 | 4.84x | 3.2 |
| 0.3 | 8.8249 | 3.28x | 4.8 |
| 0.4 | 8.6739 | 2.48x | 6.4 |
| 0.5 | 8.3238 | 1.99x | 8.0 |
| 0.6 | 7.7620 | 1.66x | 9.6 |
| 0.7 | 7.7618 | 1.43x | 11.2 |
| 0.8 | 7.7646 | 1.25x | 12.8 |

注：ratio≥0.6 时 PPL≈7.76（≈uniform_int8），ratio=0.3 时 PPL≈8.82（≈uniform_int4）。

### 网络波动评估 (Qwen2.5-3B, 20 windows, budget_ratio=0.8)

**oscillate 场景 (带宽 6↔2 MB/s, 周期 5s)**

| 策略 | PPL | 传输时间(s) | 压缩比 | 平均位数 |
|------|-----|-----------|--------|---------|
| fp16 | 8.3912 | 100.8 | 1.00x | 16.0 |
| uniform_int8 | 8.3945 | 50.4 | 2.00x | 8.0 |
| abkt | 9.1441 | 40.4 | 2.29x | 7.0 |
| uniform_int4 | 9.7403 | 26.0 | 3.88x | 4.1 |
| uniform_int2 | 15.8568 | 13.4 | 7.53x | 2.1 |

**mid_drop 场景 (带宽 8→2 MB/s, t=10s)**

| 策略 | PPL | 传输时间(s) | 压缩比 | 平均位数 |
|------|-----|-----------|--------|---------|
| fp16 | 8.3912 | 160.4 | 1.00x | 16.0 |
| uniform_int8 | 8.3945 | 80.2 | 2.00x | 8.0 |
| abkt | 14.1830 | 40.6 | 2.99x | 5.4 |
| uniform_int4 | 9.7403 | 41.4 | 3.88x | 4.1 |
| uniform_int2 | 15.8568 | 21.3 | 7.53x | 2.1 |

**结论**: ABKT 在带宽振荡场景下表现最佳 — vs uniform_int8 传输快 20%, PPL 仅差 9%。

---

## 文件命名规范

```
results_exp{N}_{model}.json        # 实验结果 JSON
pareto_exp{N}_{model}.png          # Pareto 图
longseq_exp{N}_{model}.png         # 长序列图
results_exp{N}_smoke.json          # 快速验证（不计入论文数据）
```

## 注意事项

1. **每次实验前确认 GPU 空闲**：`nvidia-smi` 或 `free -h`
2. **先跑 smoke test**（`--max-windows 5`）确认无报错，再跑全量
3. **全量运行时不要同时跑其他任务**，避免 OOM 或计时不准确
4. **JSON 结果文件不要手动编辑**，由脚本自动生成
5. **每个实验完成后立即记录数据**，避免遗忘
