# PIA Algorithm: Theoretical Analysis

## Problem Statement

Given a KV cache with $N$ entries (layer, group pairs), each with importance score $I_i \in [0,1]$, assign a precision level $p_i \in \{FP16, INT8, INT4, INT2\}$ to each entry to maximize total weighted quality:

$$\max \sum_{i=1}^{N} I_i \cdot Q(p_i)$$

subject to a byte budget constraint:

$$\sum_{i=1}^{N} s_i \cdot B(p_i) \leq \text{budget}$$

where $Q(p)$ is the quality fidelity at precision $p$, $B(p)$ is bytes per element, and $s_i$ is the element count for entry $i$.

## Reduction to Multiple-Choice Knapsack Problem (MCKP)

Each entry $i$ has 4 "choices" (precision levels) with:
- **Profit**: $I_i \cdot Q(p)$
- **Cost**: $s_i \cdot B(p)$

This is exactly the [Multiple-Choice Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem#Multiple-choice_constraints) (MCKP), where we pick exactly one item from each group.

## Greedy Algorithm

PIA uses a greedy exchange algorithm:

1. Initialize all entries at the minimum precision that fits within budget
2. Sort entries by importance descending
3. For each entry, try upgrading from current precision to the next higher level if budget allows

### Greedy Exchange Argument

At each step, the algorithm selects the upgrade with the highest quality-per-byte ratio:

$$\text{benefit}(i, p \to p') = I_i \cdot (Q(p') - Q(p)) / (s_i \cdot (B(p') - B(p)))$$

Since $Q(p)$ is the same for all entries (global constants), and $B(p') - B(p)$ is also constant for a given upgrade step, the ordering is determined by $I_i / s_i$ for a given upgrade type. Among upgrades of the same type, higher-importance entries are upgraded first.

Across upgrade types, the algorithm implicitly prefers:
- INT2→INT4: $\Delta Q = 0.12$, $\Delta B = 0.25$, ratio = 0.48 per byte
- INT4→INT8: $\Delta Q = 0.06$, $\Delta B = 0.50$, ratio = 0.12 per byte
- INT8→FP16: $\Delta Q = 0.02$, $\Delta B = 1.00$, ratio = 0.02 per byte

This is the correct ordering: INT2→INT4 upgrades are 24x more efficient per byte than INT8→FP16 upgrades.

## Approximation Ratio

The greedy algorithm for MCKP achieves a **1/2-approximation** in the worst case (Kellerer, Pferschy, and Pisinger, "Knapsack Problems", 2004, Theorem 3.4). This means:

$$\text{Quality}_{\text{greedy}} \geq \frac{1}{2} \cdot \text{Quality}_{\text{optimal}}$$

In practice, the approximation is much tighter because:
1. Quality fidelity values have a narrow range (0.80–1.00)
2. Importance scores are bounded in [0, 1]
3. The monotonic ordering of precision levels means the greedy choice is rarely suboptimal

## Limitations

### Additive Quality Assumption (T2)

The objective $\sum I_i \cdot Q(p_i)$ assumes each entry's quality contribution is independent. In reality:
- Quantization error at layer $l$ propagates through subsequent transformer layers
- Early-layer errors compound more than late-layer errors
- The interaction between K and V quantization errors within a layer is not captured

This is a practical engineering approximation. The calibration script (`tools/calibrate_fidelity.py`) provides empirical validation that the additive model produces reasonable results.

### Static Quality Fidelity (T3)

The $Q(p)$ values are global constants, not per-model or per-layer. Different models may have different sensitivity profiles. The calibration script can produce model-specific values.

## References

- Kellerer, H., Pferschy, U., and Pisinger, D. (2004). *Knapsack Problems*. Springer. Chapter 3: MCKP.
- Martello, S. and Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.
