# ABKT 问题清单

**生成日期**: 2026-06-06
**来源**: 研究背景分析、实验设计分析、技术弱点分析、论文定位分析

---

## 一、致命问题（论文无法投稿）

| # | 问题 | 现状 | 影响 |
|---|------|------|------|
| **F1** | 零生成质量评估 | 没有 PPL、ROUGE、Exact Match 等任何量化指标 | 无法证明压缩后的输出质量可接受 |
| **F2** | 零基线对比 | 没有与 uniform INT4、uniform FP8、random precision、KIVI 等方法对比 | 无法证明 ABKT 优于简单方案 |
| **F3** | 零消融实验 | 没有验证各组件（重要性评分、自适应、中途降级）的独立贡献 | 无法证明每个设计的必要性 |
| **F4** | 单模型单 prompt | 只在 Qwen2.5-3B 上测试，只有一个 prompt | 无法证明方法的通用性 |

---

## 二、实现-设计脱节

| # | 问题 | 代码现状 | 设计预期 |
|---|------|---------|---------|
| **S1** | per-token 精度坍缩为 per-layer | `quantize()` 取 `avg_prec = round(mean())`，全层一个精度 | 每个 token 应有独立精度 |
| **S2** | 三维重要性评分未实现 | 只有 Key L2-norm（65 行代码） | attention score + layer sensitivity + position decay |
| **S3** | `QUALITY_FIDELITY` 定义但未使用 | FP8=0.98, INT4=0.92, INT2=0.80 硬编码，PIA 算法完全不引用 | 应作为贪心权重的一部分 |
| **S4** | `compression_ratio` 参数未使用 | `_min_precision_for_layer` 接收但忽略 | 应在高压缩需求时收紧最小精度 |

---

## 三、算法/理论缺陷

| # | 问题 | 详情 |
|---|------|------|
| **T1** | PIA 最优性声称无证明 | 代码注释说"proved by greedy exchange argument"，实际是 MCKP 近似算法，不保证最优 |
| **T2** | 质量可加性假设不成立 | `SUM I(l,t) * Q(p)` 假设各 entry 质量贡献独立，忽略了层间误差传播 |
| **T3** | `QUALITY_FIDELITY` 无标定方法 | INT4=0.92 的 8% 损失如何度量？PPL delta？BLEU？无说明 |
| **T4** | per-layer 最小精度约束无实证 | 底部 1/3 FP8、中部 INT4、顶部 INT2 的切分无量化敏感度实验支撑 |

---

## 四、带宽估计问题

| # | 问题 | 位置 | 修复难度 |
|---|------|------|---------|
| **B1** | `update_clamped` 从未被调用 | `record_transfer` 直接用 `update`，±20% 保护失效 | 一行改动 |
| **B2** | 探测用零字节载荷 | `_probe_bandwidth` 发送 `b'\x00' * data_size`，TCP 可能压缩 | 低 |
| **B3** | 冷启动过于保守 | 一次 4MB 探测后 confidence 仍为 0，实际用 10.5 MB/s | 低 |
| **B4** | 双 EWMA 置信度无时间衰减 | 5 次传输 1 小时前完成，confidence 仍为 1.0 | 低 |
| **B5** | 滑动窗口取 min 恢复极慢 | 网络恢复后需等满 10 个好样本 | 低 |

---

## 五、系统设计问题

| # | 问题 | 影响 |
|---|------|------|
| **Y1** | 无容错/重传机制 | 任何 chunk 丢失导致整个请求失败 |
| **Y2** | chunk size 与 BDP 无关 | 只看带宽不看延迟，管道利用率低 |
| **Y3** | 中途降级重量化所有层 | 包括已发送层，浪费计算 |
| **Y4** | FP8 命名不准确 | 实际是 INT8 symmetric，不是真正的 FP8 E4M3 |
| **Y5** | 单 decode 节点无扩展 | 生产部署受限 |
| **Y6** | 状态机阈值全部硬编码 | 不同网络环境需手动调参 |

---

## 六、实验基础设施缺失

| # | 缺失项 |
|---|--------|
| **E1** | 没有 perplexity 计算脚本 |
| **E2** | 没有下游 benchmark 评估脚本 |
| **E3** | 没有 Pareto frontier 绘图脚本 |
| **E4** | 没有 baseline 对比模式（uniform INT4、random precision） |
| **E5** | 没有多模型测试配置 |
| **E6** | 没有长序列（4K+ tokens）测试用例 |

---

## 统计

| 类别 | 数量 | 说明 |
|------|------|------|
| 致命问题（F） | 4 | 决定能否投稿 |
| 设计脱节（S） | 4 | 决定贡献 claim 是否成立 |
| 理论缺陷（T） | 4 | 决定 reviewer 是否信服 |
| 带宽估计（B） | 5 | 影响系统可靠性 |
| 系统设计（Y） | 6 | 影响工程完整性 |
| 实验缺失（E） | 6 | 影响论文可验证性 |
| **总计** | **29** | |

---

## 改进优先级

### 第一阶段：修复实现-设计脱节（1-2 周）
1. 实现三维重要性评分（S2）
2. 修复 per-token → per-layer 坍缩（S1）
3. 调用 `update_clamped`（B1，一行改动）
4. 集成 `QUALITY_FIDELITY` 到 PIA 算法（S3）

### 第二阶段：补充核心实验（2-4 周）
5. 生成质量评估：PPL + 下游 benchmark（F1, E1, E2）
6. Pareto frontier 图（E3）
7. 消融实验（F3）
8. 基线对比：uniform INT4、uniform FP8、random precision（F2, E4）

### 第三阶段：扩展验证（1-2 月）
9. 多模型验证（F4, E5）
10. 长序列实验（E6）
11. 网络波动场景完整测试
12. PIA 近似比理论分析（T1）

---

## 投稿路径

1. **短期**：MLSys Workshop（短文，积累 feedback）
2. **中期**：MLSys / ATC / EuroSys（长文，完整实验）
3. **核心叙事**：ABR for KV Cache — 边缘异构 PD 分离推理中的带宽自适应 KV Cache 传输
4. **Hero metric**：端到端 TTFT under bandwidth constraints（Pareto frontier）
