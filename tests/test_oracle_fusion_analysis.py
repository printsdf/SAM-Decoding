import json

from evaluation.oracle import analyze_steps, load_decode_traces


def test_oracle_analysis_loads_profiler_json_and_counts_root_mat(tmp_path):
    trace_path = tmp_path / "profile.json"
    trace_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "candidates": [
                            {
                                "source": "eagle",
                                "token": 11,
                                "depth": 1,
                                "score": -0.2,
                                "accepted": True,
                            },
                            {
                                "source": "sam",
                                "token": 12,
                                "depth": 1,
                                "score": 3.0,
                                "accepted": False,
                            },
                        ],
                        "acceptance_path": [11],
                        "stats": {"eagle_nodes": 1, "sam_nodes": 1},
                        "metadata": {"question_index": 0},
                        "first_rejected_depth": None,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    steps = load_decode_traces([trace_path])
    results = analyze_steps(
        steps,
        top_k=2,
        node_budget=None,
        gap_baseline="eagle3_only",
    )

    by_method = {result.method: result for result in results}
    assert by_method["eagle3_only"].mat == 2.0
    assert by_method["sam_sequence_graft"].mat == 2.0
    assert by_method["perfect"].oracle_gap == 0.0


def test_oracle_analysis_preserves_enriched_boundary_fields(tmp_path):
    trace_path = tmp_path / "profile.json"
    trace_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "metadata": {"question_index": 7},
                        "candidates": [
                            {
                                "source": "eagle",
                                "token": 11,
                                "depth": 1,
                                "score": -0.2,
                                "accepted": True,
                                "token_path": [10, 11],
                                "tree_index": 1,
                                "parent_index": 0,
                                "local_logprob": -0.2,
                                "rank_among_siblings": 1,
                                "sibling_margin": 0.5,
                                "cumulative_path_logprob": -0.2,
                            }
                        ],
                        "acceptance_path": [11, 99],
                        "first_rejected_depth": 2,
                        "first_rejected_parent_index": 1,
                        "first_rejected_parent_path": [11],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    steps = load_decode_traces([trace_path])

    assert steps[0].metadata["question_index"] == 7
    assert steps[0].has_rejection_boundary_label is True
    assert steps[0].first_rejected_depth == 2
    assert steps[0].first_rejected_parent_index == 1
    assert steps[0].first_rejected_parent_path == (11,)
    assert steps[0].candidates[0].tree_index == 1
    assert steps[0].candidates[0].parent_index == 0
    assert steps[0].candidates[0].sibling_margin == 0.5


def test_oracle_candidate_preserves_parent_raw_logits(tmp_path):
    trace_path = tmp_path / "profile.json"
    trace_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "metadata": {"question_index": 0},
                        "candidates": [
                            {
                                "source": "eagle",
                                "token": 11,
                                "depth": 1,
                                "score": -0.2,
                                "accepted": True,
                                "token_path": [10, 11],
                                "tree_index": 1,
                                "parent_index": 0,
                                "local_logprob": -0.2,
                                "sibling_margin": 0.5,
                                "cumulative_path_logprob": -0.2,
                                "parent_top1_logit": 3.0,
                                "parent_top2_logit": 2.7,
                            }
                        ],
                        "acceptance_path": [11],
                        "first_rejected_depth": None,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    steps = load_decode_traces([trace_path])

    assert steps[0].candidates[0].parent_top1_logit == 3.0
    assert steps[0].candidates[0].parent_top2_logit == 2.7


def test_oracle_candidate_loads_old_traces_without_parent_raw_logits(tmp_path):
    trace_path = tmp_path / "profile.json"
    trace_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "metadata": {"question_index": 0},
                        "candidates": [
                            {
                                "source": "eagle",
                                "token": 11,
                                "depth": 1,
                                "score": -0.2,
                                "accepted": True,
                                "token_path": [10, 11],
                                "tree_index": 1,
                                "parent_index": 0,
                                "local_logprob": -0.2,
                            }
                        ],
                        "acceptance_path": [11],
                        "first_rejected_depth": None,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    steps = load_decode_traces([trace_path])

    assert steps[0].candidates[0].parent_top1_logit is None
    assert steps[0].candidates[0].parent_top2_logit is None


def test_oracle_candidate_tolerates_malformed_parent_raw_logits(tmp_path):
    # Malformed raw-logit values must not break the generic loader.
    trace_path = tmp_path / "profile.json"
    trace_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "metadata": {"question_index": 0},
                        "candidates": [
                            {
                                "source": "eagle",
                                "token": 11,
                                "depth": 1,
                                "score": -0.2,
                                "accepted": True,
                                "token_path": [10, 11],
                                "tree_index": 1,
                                "parent_index": 0,
                                "local_logprob": -0.2,
                                "parent_top1_logit": "not-a-number",
                                "parent_top2_logit": None,
                            }
                        ],
                        "acceptance_path": [11],
                        "first_rejected_depth": None,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    steps = load_decode_traces([trace_path])

    assert steps[0].candidates[0].parent_top1_logit is None
    assert steps[0].candidates[0].parent_top2_logit is None
