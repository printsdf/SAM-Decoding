import torch

from samd.draft import CandidateType, DraftModel
from samd.fusion.naive_fusion import (
    build_tree_buffers,
    candidate_trace_records,
    fuse_eagle_sam_naive,
    merge_and_dedup,
    parse_eagle_tree,
    parse_sam_sequence,
    truncate_with_ancestors,
)
from samd.fusion.types import CandidateNode, FusionConfig
from samd.samd_config import SamdConfig
from samd.samd_model import SamdModel
from samd.tree_model.fusion import TreeSpec
from samd.utils import SamdGenerationConfig, gen_candidates


def _eagle_tree(tokens=None, parents=None, logprobs=None):
    spec = TreeSpec(
        tokens=tokens or [10, 11, 12, 13],
        parents=parents or [-1, 0, 1, 0],
    )
    buffers = spec.to_buffers(device=torch.device("cpu"), mask_dtype=torch.float32)
    tree = {
        "tokens": torch.tensor([spec.tokens], dtype=torch.long),
        "tree_attn_mask": buffers["tree_attn_mask"],
        "tree_position_ids": buffers["tree_position_ids"],
        "tree_retrieve_indices": buffers["tree_retrieve_indices"],
    }
    if logprobs is not None:
        tree["logprobs"] = logprobs
    return tree


def test_parse_eagle_tree_uses_tree_mask_parents_and_logprobs():
    nodes = parse_eagle_tree(
        _eagle_tree(logprobs=[0.0, -0.2, -1.4, -0.7]),
        start_token=10,
    )

    assert [(node.path, node.token, node.depth) for node in nodes] == [
        ([10], 11, 1),
        ([10, 11], 12, 2),
        ([10], 13, 1),
    ]
    assert nodes[0].source == "eagle"
    assert [node.score for node in nodes] == [-0.2, -1.4, -0.7]
    assert nodes[0].metadata["tree_index"] == 1
    assert nodes[0].metadata["parent_index"] == 0
    assert nodes[0].metadata["rank_among_siblings"] == 1
    assert nodes[0].metadata["sibling_margin"] == 0.5
    assert nodes[1].metadata["tree_index"] == 2
    assert nodes[1].metadata["parent_index"] == 1
    assert nodes[1].metadata["sibling_margin"] is None
    assert abs(nodes[1].metadata["cumulative_path_logprob"] + 1.6) < 1e-9


def test_candidate_trace_records_exports_enriched_eagle_fields():
    nodes = parse_eagle_tree(
        _eagle_tree(logprobs=[0.0, -0.2, -1.4, -0.7]),
        start_token=10,
    )

    records = candidate_trace_records(nodes)

    assert records[0]["tree_index"] == 1
    assert records[0]["parent_index"] == 0
    assert records[0]["path_tokens"] == [10, 11]
    assert records[0]["local_logprob"] == -0.2
    assert records[0]["rank_among_siblings"] == 1
    assert records[0]["sibling_margin"] == 0.5
    assert records[0]["cumulative_path_logprob"] == -0.2


def test_parse_eagle_tree_validates_logprob_alignment():
    try:
        parse_eagle_tree(_eagle_tree(logprobs=[0.0, -0.2]), start_token=10)
    except ValueError as exc:
        assert "logprobs length" in str(exc)
    else:
        raise AssertionError("expected logprob length mismatch to fail")


def test_parse_sam_sequence_scores_by_match_length():
    nodes = parse_sam_sequence([10, 11, 12, 14], start_token=10, match_length=5)

    assert [(node.path, node.token, node.score) for node in nodes] == [
        ([10], 11, 4.0),
        ([10, 11], 12, 3.0),
        ([10, 11, 12], 14, 2.0),
    ]


def test_merge_and_dedup_marks_agreement():
    eagle_nodes = parse_eagle_tree(_eagle_tree(), start_token=10)
    sam_nodes = parse_sam_sequence([10, 11, 12, 14], start_token=10, match_length=4)

    merged = merge_and_dedup(eagle_nodes, sam_nodes, strategy="max_score")
    by_key = {node.key: node for node in merged}

    assert len(merged) == 4
    assert by_key[((10,), 11)].source == "both"
    assert by_key[((10, 11), 12)].source == "both"
    assert by_key[((10, 11, 12), 14)].source == "sam"


def test_truncate_with_ancestors_preserves_prefix_closure():
    root_child = CandidateNode(
        token=1,
        source="eagle",
        score=0.1,
        normalized_score=0.1,
        depth=1,
        path=[0],
    )
    deep_child = CandidateNode(
        token=2,
        source="sam",
        score=1.0,
        normalized_score=1.0,
        depth=2,
        path=[0, 1],
    )
    selected = truncate_with_ancestors([deep_child, root_child], max_tokens=2)

    assert set(node.token_path for node in selected) == {(0, 1), (0, 1, 2)}


def test_build_tree_buffers_exports_treespec_compatible_buffers():
    nodes = parse_sam_sequence([10, 11, 12], start_token=10, match_length=3)
    fused = build_tree_buffers(nodes, start_token=10, device=torch.device("cpu"))

    assert fused.tokens.tolist() == [10, 11, 12]
    assert fused.tree_mask.shape == (1, 1, 3, 3)
    assert fused.position_ids.tolist() == [[0, 1, 2]]
    assert fused.retrieve_indices.tolist() == [[0, 1, 2]]
    assert [node.parent_idx for node in fused.nodes] == [0, 1]


def test_fuse_eagle_sam_naive_returns_tree_and_metadata():
    eagle_logprobs = [0.0, -0.2, -1.4, -0.7]
    fused = fuse_eagle_sam_naive(
        eagle_tree=_eagle_tree(),
        sam_candidates=[10, 11, 12, 14],
        start_token=10,
        config=FusionConfig(max_draft_tokens=4),
        sam_match_length=4,
        eagle_logprobs=eagle_logprobs,
    )

    assert fused.tokens[0].item() == 10
    assert fused.metadata["eagle_nodes"] == 3
    assert fused.metadata["sam_nodes"] == 3
    assert fused.metadata["eagle_avg_score"] == sum(eagle_logprobs[1:]) / 3
    assert fused.metadata["dedup_count"] == 2
    assert fused.metadata["final_nodes"] <= 4
    assert fused.metadata["node_sources"][0] == "root"
    assert len(fused.metadata["node_sources"]) == fused.tokens.shape[0]
    assert fused.buffers_kwargs["tree_attn_mask"].shape[-1] == fused.tokens.shape[0]
    assert "oracle_candidates" not in fused.metadata


def test_fuse_eagle_sam_naive_debug_records_raw_oracle_candidates():
    fused = fuse_eagle_sam_naive(
        eagle_tree=_eagle_tree(logprobs=[0.0, -0.2, -1.4, -0.7]),
        sam_candidates=[10, 11, 12, 14],
        start_token=10,
        config=FusionConfig(max_draft_tokens=4, debug=True),
        sam_match_length=4,
    )

    records = fused.metadata["oracle_candidates"]
    assert [record["source"] for record in records].count("eagle") == 3
    assert [record["source"] for record in records].count("sam") == 3
    assert records[0]["token_path"] == [10, 11]
    assert records[0]["tree_index"] == 1
    assert records[0]["parent_index"] == 0
    assert records[0]["sibling_margin"] == 0.5
    assert records[-1]["sam_match_length"] == 4
    assert "accepted" not in records[0]


def test_samd_config_fusion_validation():
    config = SamdConfig(
        tree_method="eagle3",
        tree_model_path="/tmp/nonexistent-eagle3",
        tree=[],
        tree_config={},
        fusion_mode="naive",
        fusion_max_draft_tokens=8,
    )

    assert config.fusion_config.max_draft_tokens == 8

    try:
        SamdConfig(
            tree_method="eagle3",
            tree_model_path="/tmp/nonexistent-eagle3",
            tree=[],
            tree_config={},
            tree_fusion="sam_sequence_graft",
            fusion_mode="naive",
        )
    except ValueError as exc:
        assert "cannot both be enabled" in str(exc)
    else:
        raise AssertionError("expected fusion_mode/tree_fusion conflict to fail")


class _MockTreeModel:
    def gen_draft(self, start_token, return_logprobs=False):
        tree = _eagle_tree(tokens=[start_token, 11, 12], parents=[-1, 0, 1])
        pred_ids = tree["tokens"].view(-1).tolist()
        buffers = {
            "tree_attn_mask": tree["tree_attn_mask"],
            "tree_position_ids": tree["tree_position_ids"],
            "tree_retrieve_indices": tree["tree_retrieve_indices"],
        }
        if return_logprobs:
            return pred_ids, buffers, [0.0, -0.1, -0.4]
        return pred_ids, buffers


class _MockSam:
    def __init__(self, match_length, sequence):
        self.match_length = match_length
        self.sequence = sequence
        self.generated = False

    def lookup(self, start_token):
        return 0, self.match_length

    def gen_draft_raw(self, index, start_token, max_len):
        self.generated = True
        return [start_token] + self.sequence[: max_len - 1]


class _MockDraft:
    def __init__(self):
        self.tree_model = _MockTreeModel()
        self.sam_dyn = _MockSam(3, [11, 13])
        self.sam_static = _MockSam(0, [])
        self.len_bias = 0
        self.metadata = None

    def record_naive_fusion(self, metadata):
        self.metadata = metadata


def test_gen_candidates_naive_fusion_smoke():
    config = SamdConfig(
        tree_method="eagle3",
        tree_model_path="/tmp/nonexistent-eagle3",
        tree=[],
        tree_config={},
        fusion_mode="naive",
        fusion_max_draft_tokens=6,
    )
    draft = _MockDraft()
    sample_p = torch.zeros(1, 20)
    sample_p[0, 10] = 1.0

    candidates = gen_candidates(
        sample_p=sample_p,
        tree_retrieve_indices=torch.zeros(1, 1, dtype=torch.long),
        draft=draft,
        samd_config=config,
        gen_config=SamdGenerationConfig(),
        device=torch.device("cpu"),
    )

    assert candidates.type == CandidateType.tree
    assert candidates.tokens.shape[0] == 1
    assert candidates.buffers_kwargs["tree_retrieve_indices"].shape[0] >= 1
    assert draft.metadata["mode"] == "naive"


def test_gen_candidates_naive_quality_gate_falls_back_to_eagle():
    config = SamdConfig(
        tree_method="eagle3",
        tree_model_path="/tmp/nonexistent-eagle3",
        tree=[],
        tree_config={},
        fusion_mode="naive",
        fusion_max_draft_tokens=6,
        len_threshold=5,
    )
    draft = _MockDraft()
    draft.sam_dyn = _MockSam(2, [11, 13])
    draft.sam_static = _MockSam(0, [])
    sample_p = torch.zeros(1, 20)
    sample_p[0, 10] = 1.0

    candidates = gen_candidates(
        sample_p=sample_p,
        tree_retrieve_indices=torch.zeros(1, 1, dtype=torch.long),
        draft=draft,
        samd_config=config,
        gen_config=SamdGenerationConfig(),
        device=torch.device("cpu"),
    )

    assert candidates.type == CandidateType.tree
    assert candidates.tokens.tolist() == [[10, 11, 12]]
    assert candidates.candidate_tokens.tolist() == [[10, 11, 12]]
    assert draft.metadata["sam_skipped"] is True
    assert draft.metadata["sam_match_quality"] == 2
    assert draft.metadata["threshold"] == 5
    assert draft.metadata["sam_nodes"] == 0
    assert draft.metadata["selected_eagle"] == 2
    assert draft.metadata["node_sources"] == ["root", "eagle", "eagle"]
    assert draft.sam_dyn.generated is False
    assert draft.sam_static.generated is False


def test_record_naive_acceptance_uses_accepted_tree_indices():
    draft = type("DummyDraft", (), {})()
    draft.fusion_stats = {
        "enabled": True,
        "mode": "naive",
        "last_step": None,
        "accept_rates": [],
    }
    fusion_meta = {
        "selected_eagle": 1,
        "selected_sam": 1,
        "selected_both": 1,
        "node_sources": ["root", "eagle", "both", "sam"],
    }

    DraftModel.record_naive_acceptance(
        draft,
        accepted_tokens=3,
        fusion_meta=fusion_meta,
        accepted_indices=torch.tensor([0, 2, 3]),
    )

    rate = draft.fusion_stats["accept_rates"][0]
    assert rate["root_accepted"] == 1
    assert rate["both_accepted"] == 1
    assert rate["eagle_accepted"] == 1
    assert rate["sam_accepted"] == 2
    assert rate["accepted_nonroot"] == 2
    assert rate["eagle_rate"] == 0.5
    assert rate["sam_rate"] == 1.0


def test_record_naive_fusion_updates_stats_on_real_draft_class():
    draft = type("DummyDraft", (), {})()
    draft.fusion_stats = {
        "enabled": True,
        "mode": "naive",
        "steps": 0,
        "matched_steps": 0,
        "node_count_sum": 0,
        "sam_added_nodes_sum": 0,
        "merged_nodes_sum": 0,
        "eagle_nodes_sum": 0,
        "sam_nodes_sum": 0,
        "final_nodes_sum": 0,
        "merged_nodes_before_truncation_sum": 0,
        "eagle_score_sum": 0.0,
        "sam_score_sum": 0.0,
        "eagle_depth_sum": 0.0,
        "sam_depth_sum": 0.0,
        "sam_match_length_sum": 0.0,
        "sam_match_length_max": 0,
        "both_proposed_sum": 0,
        "truncated_count_sum": 0,
        "selected_eagle_sum": 0,
        "selected_sam_sum": 0,
        "selected_both_sum": 0,
        "sam_skipped_count": 0,
        "last_step": None,
        "last": None,
    }
    metadata = {
        "eagle_nodes": 2,
        "sam_nodes": 1,
        "merged_nodes": 3,
        "final_nodes": 3,
        "sam_contribution": 1,
        "dedup_count": 0,
        "selected_eagle": 2,
        "selected_sam": 1,
        "selected_both": 0,
        "sam_skipped": False,
    }

    DraftModel.record_naive_fusion(draft, metadata)

    assert draft.fusion_stats["steps"] == 1
    assert draft.fusion_stats["matched_steps"] == 1
    assert draft.fusion_stats["eagle_nodes_sum"] == 2
    assert draft.fusion_stats["sam_nodes_sum"] == 1
    assert draft.fusion_stats["final_nodes_sum"] == 3
    assert draft.fusion_stats["last_step"] == metadata


def test_rejection_boundary_labels_mark_parent_without_fake_tree_index():
    labels = SamdModel._build_rejection_boundary_labels(
        candidates=[
            {
                "source": "eagle",
                "token": 1,
                "depth": 1,
                "token_path": [0, 1],
                "tree_index": 1,
                "parent_index": 0,
            },
            {
                "source": "eagle",
                "token": 8,
                "depth": 2,
                "token_path": [0, 1, 8],
                "tree_index": 3,
                "parent_index": 1,
            },
        ],
        accepted_path=[1, 9],
    )

    assert labels["first_rejected_depth"] == 2
    assert labels["first_rejected_parent_index"] == 1
    assert labels["first_rejected_parent_path"] == [1]
    assert labels["first_rejected_tree_index"] is None
