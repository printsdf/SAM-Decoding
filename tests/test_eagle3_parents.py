"""Server-side (torch) equivalence tests for the hot-path parent derivation."""

import pytest

torch = pytest.importorskip("torch")

from samd.tree_model.fusion import (
    TreeSpec,
    eagle3_parents_from_buffers,
    tree_buffers_from_parents,
)


TREES = [
    [-1, 0],
    [-1, 0, 0, 1, 1, 3],
    [-1, 0, 0, 2, 2, 4, 4, 6],
]


@pytest.mark.parametrize("parents", TREES)
def test_parents_match_from_eagle3_buffers(parents):
    # n==1 (root-only) is excluded: _squeeze_singletons collapses a 1x1 mask,
    # so from_eagle3_buffers itself cannot round-trip it; real EAGLE trees
    # always have >= 2 nodes.
    tokens = list(range(100, 100 + len(parents)))
    spec = TreeSpec(tokens=tokens, parents=parents)
    buffers = spec.to_buffers()
    mask = buffers["tree_attn_mask"].float()  # eagle3 passes a float mask
    pos = buffers["tree_position_ids"]

    derived = eagle3_parents_from_buffers(mask, pos)

    assert derived == parents
    assert TreeSpec.from_eagle3_buffers(tokens, mask, pos).parents == parents


@pytest.mark.parametrize("parents", TREES)
def test_tree_buffers_match_to_buffers(parents):
    tokens = list(range(200, 200 + len(parents)))
    spec = TreeSpec(tokens=tokens, parents=parents)
    expected = spec.to_buffers(mask_dtype=torch.float32)
    got = tree_buffers_from_parents(tokens, parents, mask_dtype=torch.float32)
    for key in expected:
        assert torch.equal(expected[key], got[key]), key


def test_gather_matches_vocab_topk():
    from samd.tree_model.eagle3.eagle3_model import Eagle3Model

    headout = torch.randn(4, 32)
    log_p = torch.log_softmax(headout, dim=-1)
    topk_index = torch.topk(log_p, 5, dim=-1).indices
    expected = torch.topk(headout, 2, dim=-1).values

    gathered = Eagle3Model._gather_parent_top2_raw_logits(
        object.__new__(Eagle3Model), headout, topk_index
    )

    assert gathered.shape == (4 * 5, 2)
    assert torch.allclose(gathered.view(4, 5, 2)[:, 0, :], expected)
