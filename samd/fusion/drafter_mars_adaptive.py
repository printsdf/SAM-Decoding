"""Adaptive theta controller for the drafter-MARS gate (pure, no torch).

The update rule is the Adaptive Conformal Inference form (Gibbs & Candes,
2021): theta tracks a target trigger rate online. Grafts themselves use the
author's SAM n_predicts horizon — no budget schedules (falsified in Phase A).
"""


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
