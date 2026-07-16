"""Torch-free tests for subtree graft and build_sam_tree semantics."""

import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _ensure_pkg(name, rel):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(_ROOT / rel)]
        sys.modules[name] = module


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Register stub packages so tree_draft's relative import resolves without
# executing samd/__init__.py (which needs torch).
_ensure_pkg("samd", "samd")
_ensure_pkg("samd.tree_model", "samd/tree_model")
_ensure_pkg("samd.sam", "samd/sam")
_fusion = _load("samd.tree_model.fusion", "samd/tree_model/fusion.py")
_tree_draft = _load("samd.sam.tree_draft", "samd/sam/tree_draft.py")

TreeSpec = _fusion.TreeSpec
SamTreeBudget = _tree_draft.SamTreeBudget
build_sam_tree = _tree_draft.build_sam_tree


def test_graft_tree_at_appends_and_reuses():
    # base: root(1) -> 2 -> 3 ; anchor at node index 1 (token 2)
    base = TreeSpec(tokens=[1, 2, 3], parents=[-1, 0, 1])
    # sam tree rooted at anchor token 2: children 3 (exists) and 4 (new); 3 -> 5
    sam = TreeSpec(tokens=[2, 3, 4, 5], parents=[-1, 0, 0, 1])

    fused, stats = base.graft_tree_at(1, sam)

    assert stats["merged_nodes"] == 1          # token 3 under anchor reused
    assert stats["sam_added_nodes"] == 2       # tokens 4 and 5 appended
    assert fused.tokens == [1, 2, 3, 4, 5]
    assert fused.parents == [-1, 0, 1, 1, 2]   # 4 under anchor, 5 under reused 3


def test_graft_tree_at_root_equals_union():
    base = TreeSpec(tokens=[1, 2], parents=[-1, 0])
    sam = TreeSpec(tokens=[1, 2, 9], parents=[-1, 0, 0])
    fused_a, stats_a = base.graft_tree_at(0, sam)
    fused_b, stats_b = base.union_sam_tree(sam)
    assert fused_a.tokens == fused_b.tokens
    assert fused_a.parents == fused_b.parents
    assert stats_a == stats_b


def test_build_sam_tree_branches_by_frequency():
    class State:
        def __init__(self, next=None, cnt=0):
            self.next = next or {}
            self.cnt_endpos = cnt

    # state 0 --a(10)--> 1, --b(11)--> 2 ; 1 --c(12)--> 3
    states = [
        State({10: 1, 11: 2}),
        State({12: 3}, cnt=3),
        State({}, cnt=1),
        State({}, cnt=2),
    ]
    budget = SamTreeBudget(max_nodes=8, top_k=4, alpha=8.0, max_depth=None)
    spec, stats = build_sam_tree(states, 0, match_length=2, start_token=99, budget=budget)

    assert spec.tokens[0] == 99
    assert 10 in spec.tokens and 11 in spec.tokens  # both branches expanded
    assert 12 in spec.tokens                        # deeper continuation reached
    assert stats["node_count"] == len(spec.tokens)
