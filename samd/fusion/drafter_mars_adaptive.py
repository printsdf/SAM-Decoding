"""Adaptive knobs for the drafter-MARS gate (pure, no torch).

Phase A of task drafter-mars-adaptive-gates. Deleting this module (plus its
config/CLI/hook lines) restores the fixed-theta, fixed-budget, single-graft
behavior exactly.
"""

from typing import Optional

BUDGET_MODES = ("fixed", "depth", "ratio")


class AdaptiveThetaController:
    """Per-request integral controller holding the trigger rate near a target.

    Updated only on steps that reach the ratio gate (match-length SAM steps do
    not count). Observed rate above target raises theta (stricter gate).
    """

    def __init__(
        self,
        theta_init: float,
        target_rate: float,
        step: float,
        momentum: float = 0.05,
        theta_min: float = 0.50,
        theta_max: float = 0.995,
    ) -> None:
        self._theta = float(theta_init)
        self.target_rate = float(target_rate)
        self.step = float(step)
        self.momentum = float(momentum)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.ema = float(target_rate)
        self._theta = min(max(self._theta, self.theta_min), self.theta_max)

    @property
    def theta(self) -> float:
        return self._theta

    def update(self, triggered: bool) -> float:
        self.ema = (1.0 - self.momentum) * self.ema + self.momentum * float(bool(triggered))
        self._theta = min(
            max(self._theta + self.step * (self.ema - self.target_rate), self.theta_min),
            self.theta_max,
        )
        return self._theta


def resolve_graft_budget(
    mode: str,
    base: int,
    depth: int,
    ratio: Optional[float],
    theta: float,
    min_nodes: int = 2,
) -> int:
    """Per-graft node budget; "fixed" returns base (current behavior)."""
    base = int(base)
    if mode == "fixed":
        return base
    min_nodes = min(int(min_nodes), base)
    if mode == "depth":
        return max(min_nodes, base - max(int(depth), 0))
    if mode == "ratio":
        if ratio is None or theta >= 1.0:
            return min_nodes
        excess = (float(ratio) - float(theta)) / (1.0 - float(theta))
        excess = min(max(excess, 0.0), 1.0)
        return max(min_nodes, min(base, int(round(base * excess))))
    raise ValueError("unsupported budget mode: {}".format(mode))
