import importlib.util
from pathlib import Path

import pytest

# Load the pure gate module directly: samd/__init__.py imports torch, which is
# unavailable in the local (server-only model) environment.
_GATE_PATH = Path(__file__).resolve().parent.parent / "samd" / "fusion" / "drafter_mars_gate.py"
_spec = importlib.util.spec_from_file_location("drafter_mars_gate", _GATE_PATH)
_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate)

greedy_top_path_parents = _gate.greedy_top_path_parents
top_path_ratio_trigger = _gate.top_path_ratio_trigger

# Tree: root 0 -> {1, 2}; 1 -> 3 (greedy path: 0 -> 1 -> 3)
PARENTS = [-1, 0, 0, 1]
LOGPROBS = [0.0, -0.1, -2.0, -0.3]


def test_greedy_top_path_parents():
    assert greedy_top_path_parents(PARENTS, LOGPROBS) == [0, 1]
    assert greedy_top_path_parents([-1], [0.0]) == []
    assert greedy_top_path_parents(PARENTS, None) == []
    assert greedy_top_path_parents(PARENTS, [0.0]) == []


def test_ratio_direction_and_trigger():
    # Capture contract: raw_pairs[i] holds the (z1, z2) of node i's parent.
    # Node 1 carries the root's pair; node 3 carries node 1's pair.
    raw_pairs = [(None, None), (10.0, 9.5), (10.0, 9.5), (10.0, 5.0)]
    triggered, max_ratio = top_path_ratio_trigger(PARENTS, LOGPROBS, raw_pairs, 0.90)
    assert triggered
    assert abs(max_ratio - 0.95) < 1e-6
    # Stricter theta triggers less or equal
    triggered_strict, _ = top_path_ratio_trigger(PARENTS, LOGPROBS, raw_pairs, 0.98)
    assert not triggered_strict


def test_deepest_top_path_parent_ratio_is_evaluated():
    # Only the leaf's parent (node 1, pair stored at leaf node 3) is uncertain.
    raw_pairs = [(None, None), (10.0, 5.0), (10.0, 5.0), (10.0, 9.5)]
    triggered, max_ratio = top_path_ratio_trigger(PARENTS, LOGPROBS, raw_pairs, 0.90)
    assert triggered
    assert abs(max_ratio - 0.95) < 1e-6


def test_off_path_high_ratio_does_not_trigger():
    # Node 4's pair belongs to off-path parent 2; the greedy path is 0 -> 1 -> 3.
    parents = [-1, 0, 0, 1, 2]
    logprobs = [0.0, -0.1, -2.0, -0.3, -0.5]
    raw_pairs = [(None, None), (10.0, 5.0), (10.0, 5.0), (10.0, 5.0), (10.0, 9.9)]
    triggered, max_ratio = top_path_ratio_trigger(parents, logprobs, raw_pairs, 0.90)
    assert not triggered
    assert abs(max_ratio - 0.5) < 1e-6


def test_missing_raw_pairs_never_trigger():
    assert top_path_ratio_trigger(PARENTS, LOGPROBS, None, 0.90) == (False, None)
    assert top_path_ratio_trigger(PARENTS, LOGPROBS, [None] * 4, 0.90) == (False, None)
    assert top_path_ratio_trigger(PARENTS, LOGPROBS, [(None, None)] * 4, 0.90) == (False, None)
    # Length mismatch
    assert top_path_ratio_trigger(PARENTS, LOGPROBS, [(1.0, 0.5)], 0.90) == (False, None)


# SamdConfig imports torch; run these only where torch is available (server).
try:
    import torch  # noqa: F401
    from samd.samd_config import SamdConfig
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _config(**kwargs):
    kwargs.setdefault("tree", [[0]])
    kwargs.setdefault("tree_config", {})
    return SamdConfig(**kwargs)


@requires_torch
def test_config_drafter_mars_valid():
    config = _config(tree_method="eagle3", fusion_mode="drafter_mars")
    assert config.drafter_mars_theta == 0.90
    assert config.drafter_mars_variant == "top_path"
    assert config.fusion_config.mode == "naive"


@requires_torch
def test_config_drafter_mars_requires_eagle3():
    with pytest.raises(ValueError):
        _config(tree_method="token_recycle", fusion_mode="drafter_mars")


@requires_torch
def test_config_theta_validation():
    with pytest.raises(ValueError):
        _config(tree_method="eagle3", fusion_mode="drafter_mars", drafter_mars_theta=0.0)
    with pytest.raises(ValueError):
        _config(tree_method="eagle3", fusion_mode="drafter_mars", drafter_mars_theta=-1.0)
    with pytest.raises(ValueError):
        _config(tree_method="eagle3", fusion_mode="drafter_mars", drafter_mars_variant="reachable")


@requires_torch
def test_config_existing_modes_still_valid():
    assert _config(tree_method="eagle3", fusion_mode="none").fusion_config is None
    assert _config(tree_method="eagle3", fusion_mode="naive").fusion_config.mode == "naive"
