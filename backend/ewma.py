"""Exponentially Weighted Moving Average — stateless computation.

Usage:
    alpha = 0.3
    ewma = EWMA(alpha)

    bw = 100e6
    ewma.update(bw)      # first call seeds the EWMA
    ewma.update(80e6)    # subsequent calls smooth
    assert ewma.value == 86e6  # 0.3*80 + 0.7*100

The effective window is ~1/alpha samples (≈3 at α=0.3).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EWMA:
    """Exponentially Weighted Moving Average.

    Attributes:
        alpha: Smoothing factor in (0, 1]. Higher = more weight on recent values.
        value: Current EWMA value. None before first update.

    Thread-safe only if the caller holds a lock around update/value reads.
    """

    alpha: float
    value: Optional[float] = None

    def update(self, x: float) -> float:
        """Update with a new sample and return the new EWMA value.

        On first call, the EWMA is seeded with x (cold start).
        """
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def update_clamped(self, x: float, max_change: float = 0.20) -> float:
        """Update with a cap on single-step change to prevent oscillation.

        |new_value - old_value| / old_value <= max_change
        Used for bandwidth EWMA to prevent feedback-loop instability.
        """
        if self.value is None:
            self.value = x
            return self.value
        raw = self.alpha * x + (1 - self.alpha) * self.value
        delta = abs(raw - self.value)
        if delta > max_change * abs(self.value):
            direction = 1.0 if raw > self.value else -1.0
            self.value += direction * max_change * abs(self.value)
        else:
            self.value = raw
        return self.value

    def reset(self) -> None:
        """Clear the EWMA state (next update will cold-start)."""
        self.value = None

    @property
    def valid(self) -> bool:
        """True if at least one sample has been seen."""
        return self.value is not None
