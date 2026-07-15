import importlib.util
from pathlib import Path

import pytest

# Load pure modules directly: samd/__init__.py imports torch, which is
# unavailable in the local (server-only model) environment.
_BASE = Path(__file__).resolve().parent.parent / "samd" / "fusion"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _BASE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_adaptive = _load("drafter_mars_adaptive")
_gate = _load("drafter_mars_gate")

AdaptiveThetaController = _adaptive.AdaptiveThetaController
resolve_graft_budget = _adaptive.resolve_graft_budget
all_top_path_triggers = _gate.all_top_path_triggers
earliest_top_path_trigger = _gate.earliest_top_path_trigger

# Tree: root 0 -> {1, 2}; 1 -> 3; 3 -> 4 (greedy path 0 -> 1 -> 3 -> 4).
PARENTS = [-1, 0, 0, 1, 3]
LOGPROBS = [0.0, -0.1, -2.0, -0.3, -0.2]
# raw_pairs[i] = (z1, z2) of node i's parent expansion (child-index contract).
RAW_HIGH_HIGH = [None, (2.0, 1.9), (2.0, 1.9), (3.0, 2.9), (1.0, 0.99)]


def test_controller_rises_when_over_target():
    ctl = AdaptiveThetaController(theta_init=0.86, target_rate=0.5, step=0.02)
    for _ in range(50):
        ctl.update(True)
    assert ctl.theta > 0.86
    assert ctl.ema > 0.5


def test_controller_falls_when_under_target():
    ctl = AdaptiveThetaController(theta_init=0.86, target_rate=0.5, step=0.02)
    for _ in range(50):
        ctl.update(False)
    assert ctl.theta < 0.86
    assert ctl.ema < 0.5


def test_controller_clips_at_bounds():
    ctl = AdaptiveThetaController(
        theta_init=0.86, target_rate=0.1, step=0.5, theta_min=0.5, theta_max=0.995
    )
    for _ in range(200):
        ctl.update(True)
    assert ctl.theta == 0.995
    for _ in range(400):
        ctl.update(False)
    assert ctl.theta == 0.5


def test_controller_ema_starts_at_target():
    ctl = AdaptiveThetaController(theta_init=0.86, target_rate=0.75, step=0.02)
    assert ctl.ema == 0.75
    assert ctl.theta == 0.86


def test_budget_fixed_is_base():
    assert resolve_graft_budget("fixed", 8, 5, 0.99, 0.86) == 8


def test_budget_depth_monotone_with_floor():
    budgets = [resolve_graft_budget("depth", 8, d, None, 0.86) for d in range(9)]
    assert budgets[0] == 8
    assert all(a >= b for a, b in zip(budgets, budgets[1:]))
    assert budgets[-1] == 2


def test_budget_ratio_monotone_with_floor_and_cap():
    theta = 0.86
    budgets = [
        resolve_graft_budget("ratio", 8, 0, r, theta)
        for r in (0.86, 0.90, 0.95, 1.0, 1.5)
    ]
    assert all(a <= b for a, b in zip(budgets, budgets[1:]))
    assert budgets[0] == 2  # floor at zero excess
    assert budgets[3] == 8  # full excess reaches base
    assert budgets[4] == 8  # capped at base
    assert resolve_graft_budget("ratio", 8, 0, None, theta) == 2


def test_budget_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_graft_budget("bogus", 8, 0, 0.9, 0.86)


def test_all_triggers_ascending_depth_and_consistent_with_earliest():
    triggers = all_top_path_triggers(PARENTS, LOGPROBS, RAW_HIGH_HIGH, 0.90)
    assert [t.parent_depth for t in triggers] == sorted(t.parent_depth for t in triggers)
    assert len(triggers) >= 2
    earliest = earliest_top_path_trigger(PARENTS, LOGPROBS, RAW_HIGH_HIGH, 0.90)
    assert triggers[0] == earliest


def test_all_triggers_empty_cases():
    assert all_top_path_triggers(PARENTS, LOGPROBS, RAW_HIGH_HIGH, 1.5) == []
    assert all_top_path_triggers(PARENTS, LOGPROBS, None, 0.5) == []
    assert all_top_path_triggers(PARENTS, LOGPROBS, [None] * 5, 0.5) == []


requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


@requires_torch
def test_config_validation_new_fields():
    from samd.samd_config import SamdConfig

    def _config(**kwargs):
        return SamdConfig(
            tree_method="eagle3",
            fusion_mode="drafter_mars",
            tree=[[0]],
            tree_config={},
            **kwargs,
        )

    _config(
        drafter_mars_adaptive_theta=True,
        drafter_mars_target_trigger_rate=0.6,
        drafter_mars_theta_step=0.01,
        drafter_mars_budget_mode="depth",
        drafter_mars_max_grafts=3,
        drafter_mars_total_graft_nodes=24,
    )
    for bad in (
        {"drafter_mars_adaptive_theta": 1},
        {"drafter_mars_target_trigger_rate": 0.0},
        {"drafter_mars_target_trigger_rate": 1.0},
        {"drafter_mars_theta_step": 0.0},
        {"drafter_mars_budget_mode": "bogus"},
        {"drafter_mars_max_grafts": 0},
        {"drafter_mars_max_grafts": True},
        {"drafter_mars_total_graft_nodes": 0},
    ):
        with pytest.raises(ValueError):
            _config(**bad)
