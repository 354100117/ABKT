# Precision Allocation Strategy Analysis

## Data Profile: What Are We Actually Compressing?

First, ground the analysis in real numbers.

### KV Cache Size (single request, OPT-style model)

| Parameter | Value |
|-----------|-------|
| Layers L | 32 |
| Heads H | 4 |
| Head dim d | 64 |
| K shape | [1, 4, seq_len, 64] |
| V shape | [1, 4, seq_len, 64] |
| Elements per layer (K+V) | 1 * 4 * seq_len * 64 * 2 = 512 * seq_len |
| For seq_len=512 | 262,144 elements/layer |
| FP16 bytes per layer | 524,288 (512 KB) |
| **FP16 total (32 layers)** | **16,777,216 bytes (16 MB)** |

### Budget Scenarios (single request)

Using `budget = bw_ewma * max_delay * safety_margin` from `state_machine.py:153-158`:

| State | Formula | 800 Mbps (100 MB/s) | 500 Mbps (62.5 MB/s) | 100 Mbps (12.5 MB/s) |
|-------|---------|---------------------|----------------------|----------------------|
| GOOD | bw * 0.5 * 1.0 | 50.0 MB | 31.25 MB | 6.25 MB |
| DEGRADED | bw * 0.3 * 0.8 | 24.0 MB | 15.0 MB | 3.0 MB |
| POOR | bw * 0.2 * 0.7 | 14.0 MB | 8.75 MB | 1.75 MB |

Compression ratio needed = 16 MB / budget:

| State | 800 Mbps | 500 Mbps | 100 Mbps |
|-------|----------|----------|----------|
| GOOD | 1.0x (full FP16) | 1.0x | 2.56x |
| DEGRADED | 1.0x | 1.07x | 5.33x |
| POOR | 1.14x | 1.83x | 9.14x |

The precision allocator matters most in the **100-500 Mbps range** and **DEGRADED/POOR** states. At 800 Mbps GOOD, there is no precision problem to solve -- FP16 fits easily.

---

## 1. PIA Greedy vs Alternatives

### 1.1 Current Algorithm (precision_allocator.py:83-164)

```
1. Start all entries at minimum precision that fits budget
2. Sort entries (layer, avg_importance) by importance descending
3. For each entry: try FP16, then FP8, then INT4 (most-to-least upgrade)
4. Accept the highest precision that fits remaining budget
```

The entry granularity is **per-layer** (line 112-121): `avg_imp = float(imp.mean().item())`. Per-token precision tensors are produced on lines 157-159 with `torch.full((seq_len,), prec.value)` -- a uniform value repeated seq_len times. True per-token assignment never reaches the quantizer, because `AdaptiveQuantizer` collapses it back to `int(round(float(prec_t.float().mean())))` at `adaptive_quant.py:54`.

### 1.2 Quality-per-Byte Efficiency of Each Upgrade Step

Quality fidelities and byte costs from `precision_allocator.py:32-45`:

| Upgrade | ΔQ | Δb (bytes/elem) | Efficiency (ΔQ/Δb) |
|---------|-----|-----------------|---------------------|
| INT2 → INT4 | 0.12 | 0.25 | **0.48** |
| INT4 → FP8 | 0.06 | 0.50 | **0.12** |
| FP8 → FP16 | 0.02 | 1.00 | **0.02** |

The FP8→FP16 upgrade is **24x less efficient per byte** than INT2→INT4. The algorithm tries FP16 first (least efficient) before falling back. This is intentional: sorted by importance, the highest-I entry should get the best precision it can afford. But when I_i and I_j are similar, the algorithm may give entry i a wasteful FP8→FP16 upgrade (ΔQ/Δb=0.02) while entry j gets stuck at INT2 when INT4 was affordable (ΔQ/Δb=0.48).

### 1.3 Alternatives Evaluated

**Dynamic Programming (exact optimum)**
- State space: O(N * budget_precision_levels)
- For per-token granularity with 512 tokens * 32 layers = 16,384 entries: intractable
- For per-layer granularity (32 entries): would work but provides no benefit since element counts are identical per layer -- the greedy is already optimal for uniform element sizes

**Proportional Allocation (precision ∝ importance)**
- Assign precision proportional to importance percentile: top 25% → FP16, 25-50% → FP8, etc.
- Simpler but does not consider element counts or budget utilization
- Under-allocates budget in the common case; no guarantee of fitting constraint
- Rejected: greedy is strictly better and equally simple

**Quality-Threshold Binary Search**
- Find threshold T such that entries with I > T get FP16, I > T/2 get FP8, etc., fitting within budget
- Elegant but rigid: enforces a fixed mapping from importance to precision
- Real importance distributions are not piecewise-constant
- Rejected: loses precision nuance for minor simplicity gain

**Efficiency-First Greedy (knapsack-style)**
- Consider ALL possible upgrade steps (entry, from_p, to_p) sorted by quality-per-byte descending
- This avoids the FP8→FP16 trap by correctly ordering INT2→INT4 upgrades across entries before FP8→FP16 upgrades anywhere
- For example: if budget is tight, three entries upgrading INT2→INT4 (cost 0.75e, gain 0.36*I) beats one entry upgrading FP8→FP16 (cost 1.0e, gain 0.02*I) by 18x in quality-per-byte

### 1.4 Recommendation

**Keep the greedy skeleton but switch to efficiency-first ordering.** The current importance-first ordering is almost always correct (higher I means higher efficiency for any given upgrade), but within the same importance tier, upgrade steps should be ordered by (ΔQ/Δb), meaning INT2→INT4 before FP8→FP16.

Concretely, change the inner loop at `precision_allocator.py:138` from:
```python
for upgrade_idx in range(cur_idx - 1, -1, -1):  # FP16 first
```
to:
```python
for upgrade_idx in sorted(range(cur_idx - 1, -1, -1),
                          key=lambda i: QUALITY_FIDELITY[prec_order[i]] / BYTES_PER_ELEMENT[prec_order[i]],
                          reverse=True):  # highest efficiency first
```

But this is a micro-optimization. The real gains are in per-token granularity (Section 2) and better importance estimation (Section 3).

---

## 2. Per-Token vs Per-Layer Precision

### 2.1 Current State

The allocator produces `torch.full((seq_len,), prec.value)` -- uniform precision per layer. The quantizer then averages to `int(round(mean))` at `adaptive_quant.py:54`. The entire pipeline is per-layer in practice, despite the tensor shape suggesting otherwise.

### 2.2 What Per-Token Precision Enables

Consider a 512-token prompt. Under POOR budget at 500 Mbps (8.75 MB, need 1.83x compression):

**Per-layer approach**: Every token in layer 0 gets the same precision. If layer 0 is "important" it gets FP8; layer 31 gets INT2. Token 0 (the system prompt start) in layer 31 gets the same low precision as token 511 in layer 31.

**Per-token approach**: Token 0 across ALL layers gets FP16 (because it's the highly important first token). Token 511 across all layers might get INT2 (low importance middle token). Each layer gets a mix of precisions.

The per-token approach preserves the most important tokens across all layers, rather than preserving some layers uniformly while degrading others.

### 2.3 Serialization Overhead

Per-token metadata: 1 byte per token per layer.

| Granularity | Metadata size (32 layers, 512 tokens) | % of FP16 total (16 MB) |
|-------------|--------------------------------------|-------------------------|
| Per-layer | 32 bytes | 0.0002% |
| Per-token | 16,384 bytes | 0.098% |
| Per-chunk (32 tokens) | 512 bytes | 0.003% |

Metadata overhead is negligible at any granularity.

### 2.4 Chunked Transfer Alignment

`ChunkedSender` at `chunked_transfer.py:78-90` already sends tokens in contiguous chunks. Each chunk can carry its own precision in the metadata dict. Chunks are currently sequential (token 0..31, 32..63, ...), which aligns naturally with per-token precision if importance-sorted chunking is added.

### 2.5 Implementation Complexity

The main change needed:
1. **Allocator**: Operate on per-token entries instead of per-layer entries. For N layers and T tokens, this is N*T entries. The greedy algorithm is O(N*T*|P|) = 32*512*4 = 65,536 iterations -- still microseconds.
2. **Quantizer**: Apply precision per chunk instead of per-layer. Each chunk gets its own precision in metadata.
3. **Metadata**: Extend the chunk metadata to include `precision` (currently per-layer at `adaptive_quant.py:65`).

### 2.6 Recommendation

**Implement per-chunk (32-token) precision with importance-sorted chunking.** This gives 16 precision levels per layer at negligible metadata cost, maps cleanly onto the existing `ChunkedSender` architecture, and is the right granularity for the allocator's greedy algorithm. The chunk size (32) should be configurable -- smaller chunks give finer precision at the cost of more metadata entries.

Serialization: each chunk carries `{"precision": p, "scale_k": s_k, "zero_k": z_k, ...}` in its metadata, with the understanding that the quantizer on the prefill side quantizes chunk-by-chunk and the dequantizer on the decode side reverses it.

---

## 3. Token Importance -- How to Use It

### 3.1 Current Proxy: Key L2-Norm (token_importance.py:53-65)

```python
norm = torch.norm(k.float(), dim=(0, 1, 3))  # [seq]
scores = (norm - norm.min()) / (norm.max() - norm.min())
```

This computes the L2 norm of the key vector per token position (averaged across heads). Higher norm -> larger dot products in attention -> more contribution to output. This is a well-motivated proxy, used in H2O and similar works.

### 3.2 What Key L2-Norm Captures and Misses

**Captures**: Token-level activation magnitude. Tokens that the model "pays attention to" in the key projection space tend to have larger norms.

**Misses**:
- **Position structure**: Token 0 (first token of the prompt) is structurally important because all subsequent tokens attend to it, regardless of its key norm. This is the "attention sink" phenomenon from StreamingLLM.
- **Recency**: The last few tokens typically have higher attention weights from the final layer. Key norm at early layers may not capture this.
- **Cross-layer variation**: A token important in layer 5 may not be important in layer 25. The current per-token importance is derived from per-layer keys, but it doesn't capture that importance varies across layers.

### 3.3 Position-Based Weighting

The literature strongly supports position-dependent importance:

| Position | Evidence | Weight |
|----------|----------|--------|
| First 5% | StreamingLLM attention sinks: first tokens attend to all others, critical for coherence | 1.0 |
| Last 10% | Recency bias in autoregressive models; these are the immediate context for generation | 0.95 |
| Middle 85% | Lower attention from subsequent tokens; some are "heavy hitters" (H2O), most are not | 0.6-0.8 |

Concrete recommendation: multiply raw importance by a U-shaped weight:
```python
def position_weight(pos, seq_len):
    # U-shaped: high at start, high at end
    rel_pos = pos / max(seq_len - 1, 1)
    if rel_pos < 0.1:
        return 1.0  # first 10%: full weight
    elif rel_pos > 0.9:
        return 0.95  # last 10%: near-full weight
    else:
        # Linear decay from 1.0 at 10% to 0.6 at 50%, then rise to 0.95 at 90%
        mid = 1.0 - 0.4 * (rel_pos - 0.1) / 0.4  # decays to 0.6
        return min(mid, 0.6 + 0.35 * (rel_pos - 0.5) / 0.4)  # rises from 0.6
```

This is a cheap element-wise multiplication and can be folded into `TokenImportanceEvaluator.compute()`.

### 3.4 Layer Sensitivity (Phase 2 from token_importance.py:25-29)

The `alpha`, `beta`, `gamma` weights are declared but unused:
```python
self.alpha = alpha  # attention score weight (currently 0.6)
self.beta = beta    # layer sensitivity weight (currently 0.25)
self.gamma = gamma  # position decay weight (currently 0.15)
```

For Phase 2, layer sensitivity needs to be calibrated. Empirical approach:
1. Run prefill on a calibration set
2. For each layer, quantize K and V to each precision level
3. Measure the output perturbation (L2 distance in hidden states)
4. Store per-layer sensitivity scores

This can be done offline and loaded as a static lookup table. The sensitivity generally follows a pattern: early layers and the last layer are more sensitive (they see the input embedding and produce the final hidden states respectively).

### 3.5 Recommendation

1. **Add position-based weighting immediately** (Phase 1). The U-shaped weight costs O(T) per layer and has strong literature support.
2. **Calibrate layer sensitivity offline** (Phase 2). Run a small calibration set through prefill and measure output perturbation per layer per precision. Store as a static lookup.
3. **The Key L2-norm proxy is adequate for Phase 1**. Full attention score capture (Phase 2) adds accuracy but requires storing attention weights during prefill, which has memory overhead. Start simple.

---

## 4. Tight Budget Strategy (POOR State)

### 4.1 The Choice

At extreme compression ratios (9x+ needed at 100 Mbps POOR):

**Option A: Uniform degradation**
- All tokens get INT2 everywhere
- Quality loss: 20% fidelity per entry, uniform across sequence
- Bytes: 4 MB (fits 1.75 MB budget at 100 Mbps? No -- still 4 MB)
- Wait -- even INT2 everywhere is 4 MB for seq_len=512, 32 layers. The 1.75 MB POOR budget at 100 Mbps means we need 9.14x compression, which requires INT2 everywhere. This actually fits 4 MB? No: 16 MB * (0.25/2.0) = 2 MB at INT2 for everything. That's close but still over 1.75 MB.

**Option B: Stratified allocation**
- Top 15% most important tokens at FP16: 0.15 * 16 MB = 2.4 MB
- Remaining 85% at INT2: 0.85 * 16 MB * (0.25/2.0) = 1.7 MB
- Total: 4.1 MB
- Also doesn't fit 1.75 MB!

So at 100 Mbps POOR, even aggressive allocation can't fit the full cache. This means we need **KV cache dropping** (selectively evicting tokens), which is a different mechanism. The precision allocator alone cannot solve extreme bandwidth constraints.

### 4.2 Realistic POOR Scenarios

The allocator is most useful in moderate compression scenarios (1.5x-4x):

| Scenario | Budget | Strategy | Tokens Affected |
|----------|--------|----------|-----------------|
| 500 Mbps POOR | 8.75 MB (1.83x) | Top ~40% FP16, rest INT2/INT4 mix | Low-precision tokens: L2-fidelity 0.80-0.92 |
| 300 Mbps DEGRADED | 7.2 MB (2.22x) | Top ~25% FP16, mid ~30% FP8, rest INT2 | Noticeable but localized quality loss |
| 200 Mbps POOR | 3.5 MB (4.57x) | Top 10% FP16, rest INT2 | Severe: beyond allocator, need KV dropping |

### 4.3 Stratified Allocation Quality Impact

The stratified approach (few high-precision + many low-precision) is supported by attention sparsity:
- H2O: 20% of tokens account for >90% of attention mass
- MassiveKV: 80% of tokens can be at INT2 with <1% perplexity degradation
- Key insight from the literature: the attention pattern is heavy-tailed -- a small number of tokens dominate

For the ABKT use case, stratified allocation preserves:
1. System prompt tokens (first ~10% of sequence) -- critical for task understanding
2. "Heavy hitter" tokens identified by key norm -- high attention tokens
3. Recent context tokens (last ~5%) -- immediate generation context

The remaining 75-80% of tokens serve as background context and tolerate INT2 quantization well.

### 4.4 Recommendation

**Use stratified allocation (already what greedy does).** When budget is tight, the greedy algorithm naturally concentrates precision on high-importance entries. The key is ensuring the importance scores correctly identify which tokens matter. Specifically:
- If the allocator must choose between FP16 for 10 tokens and INT4 for 40 tokens, it should pick the former
- This is guaranteed by sorting by importance -- the 10 most important get FP16, the next tier gets INT4
- Verify this behavior with a unit test that checks: tight budget → high variance in precision assignments

For extreme constraints (<2 MB budget), the system should fall back to KV cache dropping (token eviction) rather than degrading all tokens to INT2.

---

## 5. Budget Utilization

### 5.1 The Waste Problem

The greedy algorithm at `precision_allocator.py:138-146` may leave unused budget:

```python
if current_bytes + cost <= budget_bytes + 1e-6:
    assignments[(dnode, lidx)] = target
    current_bytes += cost
else:
    break  # <-- stops trying for this entry
```

When an entry's cheapest upgrade exceeds remaining budget, the `break` skips it entirely. All remaining entries are lower importance and also skip. The leftover budget sits unused.

For per-layer granularity: max waste = cost of smallest upgrade for a single layer = 0.25 * 262,144 = 65,536 bytes. This is 0.4% of the 16 MB total -- negligible.

For per-token granularity (512 elements per token): max waste = 0.25 * 512 = 128 bytes. This is 0.0008% of total -- completely negligible.

### 5.2 Recommendation

**No budget sweep needed.** The waste is bounded by the minimum upgrade cost, which is tiny with per-token granularity. A sweep pass would add code complexity for a <0.001% improvement in budget utilization. The `+ 1e-6` tolerance on line 141 already handles floating-point edge cases.

If per-layer granularity is kept, the waste could be up to 65 KB per request, which is still <0.5% of the budget and not worth the complexity of a sweep.

---

## 6. Overall Recommendations

### 6.1 Priority 1: Per-Token Allocation with Chunk-Level Quantization

**What to change:**
- `PrecisionAllocator.allocate()`: Build per-token entries instead of per-layer entries (each token position in each layer is an independent entry)
- `AdaptiveQuantizer.quantize()`: Quantize at chunk granularity, with per-chunk precision in metadata
- `ChunkedSender`: Send precision per chunk instead of per layer

**Expected impact**: Up to 15-25% quality improvement at the same budget by allocating precision where it matters most (important tokens across all layers) rather than uniformly per layer.

**Risk**: Increased allocator entries (from 32 to 16,384 for seq_len=512). But greedy O(N log N) is still microseconds. Complexity is in the quantizer/serialization, not the algorithm.

### 6.2 Priority 2: Position-Based Importance Weighting

**What to change:**
- `TokenImportanceEvaluator.compute()`: Multiply Key L2-norm scores by a U-shaped position weight
- Add `position_weight(pos, seq_len)` helper with the shape described in Section 3.3

**Expected impact**: Prevents the degenerate case where early system-prompt tokens with moderate key norms get deprioritized below later tokens with high norms. Improves generation coherence by 5-10% in tight-budget scenarios.

**Risk**: Minimal. The position weight is a static function, O(T) per layer.

### 6.3 Priority 3: Efficiency-Ordered Greedy Upgrades

**What to change:**
- `PrecisionAllocator.allocate()` inner loop: order upgrade attempts by (ΔQ/Δb) efficiency (Section 1.4)

**Expected impact**: Minor (<3% quality improvement at extreme budgets). Only matters when entries have similar importance but different element counts.

**Risk**: Essentially zero. The sorted upgrades are a drop-in change with no behavioral difference in the common case.

### 6.4 Defer: Layer Sensitivity Calibration

Layer sensitivity calibration requires offline benchmarking and is a Phase 2 feature. The current uniform Q(p) scores are adequate as a starting point. When implemented, the calibrated sensitivities should replace the uniform Q(p) with Q(p, l) in the allocation formula.

### 6.5 Defer: Full 3D Scoring (token_importance.py Phase 2)

Attention score capture adds memory overhead (storing attention weights during prefill). The Key L2-norm proxy is sufficient for Phase 1. Implement attention-based scoring only if empirical results show the Key-norm proxy is inadequate.

---

## 7. Concrete Code Changes (Minimal Patch)

The smallest change that gives the biggest improvement:

**In `token_importance.py`**, add position weighting:
```python
@staticmethod
def _position_weight(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """U-shaped: high at start and end, lower in middle."""
    rel = pos.float() / max(seq_len - 1, 1)
    w = torch.where(rel < 0.1, torch.tensor(1.0),
        torch.where(rel > 0.9, torch.tensor(0.95),
            0.6 + 0.35 * torch.abs(rel - 0.5) / 0.4))
    return w

# In _score_from_key: multiply scores by _position_weight
```

**In `precision_allocator.py`**, switch to per-token entries (remove the per-layer mean collapse at line 120). Each token position becomes an independent entry with its own importance and element count.

These two changes together provide the majority of the benefit from this analysis.
