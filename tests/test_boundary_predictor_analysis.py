import pytest

from evaluation.analyze_boundary_predictor import (
    _choose_best,
    _threshold_grid,
    evaluate_predictions,
    predict_depth_only_low_confidence,
    predict_high_margin_node,
    predict_low_margin_node,
    require_calibration_schema,
    validate_probability_threshold,
)
from evaluation.oracle_fusion_analysis import DecodeStep, OracleCandidate


def _candidate(
    source,
    token,
    depth,
    token_path,
    tree_index=None,
    parent_index=None,
    sibling_margin=None,
    local_logprob=None,
    cumulative_path_logprob=None,
    accepted=False,
):
    return OracleCandidate.from_dict(
        {
            "source": source,
            "token": token,
            "depth": depth,
            "score": local_logprob if local_logprob is not None else 0.0,
            "accepted": accepted,
            "token_path": token_path,
            "tree_index": tree_index,
            "parent_index": parent_index,
            "sibling_margin": sibling_margin,
            "local_logprob": local_logprob,
            "rank_among_siblings": 1,
            "cumulative_path_logprob": cumulative_path_logprob,
        }
    )


def _step(question_index=0):
    return DecodeStep(
        step=0,
        metadata={"question_index": question_index},
        candidates=[
            _candidate(
                "eagle",
                token=1,
                depth=1,
                token_path=[0, 1],
                tree_index=1,
                parent_index=0,
                sibling_margin=0.5,
                local_logprob=-0.1,
                cumulative_path_logprob=-0.1,
                accepted=True,
            ),
            # First depth-2 node is on the wrong parent, so depth-only chooses
            # it but node-level low-margin should skip it for this threshold.
            _candidate(
                "eagle",
                token=7,
                depth=2,
                token_path=[0, 2, 7],
                tree_index=2,
                parent_index=4,
                sibling_margin=0.10,
                local_logprob=-0.4,
                cumulative_path_logprob=-0.5,
                accepted=False,
            ),
            _candidate(
                "eagle",
                token=8,
                depth=2,
                token_path=[0, 1, 8],
                tree_index=3,
                parent_index=1,
                sibling_margin=0.05,
                local_logprob=-1.0,
                cumulative_path_logprob=-1.1,
                accepted=False,
            ),
            _candidate(
                "sam",
                token=9,
                depth=2,
                token_path=[0, 1, 9],
                accepted=True,
            ),
        ],
        acceptance_path=[1, 9],
        stats={"eagle_nodes": 3, "sam_nodes": 1},
        first_rejected_depth=2,
        first_rejected_parent_index=1,
        first_rejected_parent_path=(1,),
        has_rejection_boundary_label=True,
    )


def test_margin_predictors_use_opposite_directions():
    step = _step()

    low = predict_low_margin_node(step, threshold=0.08)
    high = predict_high_margin_node(step, threshold=0.08)

    assert low.candidate.tree_index == 3
    assert low.score == 0.05
    assert high.candidate.tree_index == 1
    assert high.score == 0.5


def test_probability_threshold_validation_rejects_wrong_scale():
    assert validate_probability_threshold(0.0) == 0.0
    assert validate_probability_threshold(1.0) == 1.0
    with pytest.raises(ValueError):
        validate_probability_threshold(1.5)


def test_threshold_grid_is_bounded_for_dense_float_values():
    values = [index * 0.001 for index in range(1000)]

    grid = _threshold_grid(values, max_thresholds=32)

    assert len(grid) == 32
    assert grid[0] < min(values)
    assert grid[-1] > max(values)


def test_node_level_prediction_beats_depth_only_wrong_branch():
    steps = [_step()]
    low_prediction = predict_low_margin_node(steps[0], threshold=0.08)
    depth_prediction = predict_depth_only_low_confidence(steps[0], threshold=0.08)

    low_metrics = evaluate_predictions(
        steps,
        [low_prediction],
        method="low_margin_node",
        threshold=0.08,
    )
    depth_metrics = evaluate_predictions(
        steps,
        [depth_prediction],
        method="depth_only_low_confidence",
        threshold=0.08,
    )

    assert low_metrics["boundary_precision"] == 1.0
    assert depth_metrics["boundary_precision"] == 0.0
    assert (
        low_metrics["predicted_boundary_oracle_mat"]
        > depth_metrics["predicted_boundary_oracle_mat"]
    )


def test_predicted_boundary_mat_respects_perfect_node_budget():
    steps = [_step()]
    low_prediction = predict_low_margin_node(steps[0], threshold=0.08)

    metrics = evaluate_predictions(
        steps,
        [low_prediction],
        method="low_margin_node",
        threshold=0.08,
        node_budget=1,
    )

    assert metrics["node_budget"] == 1
    assert metrics["predicted_boundary_oracle_mat"] <= metrics["perfect_mat"]


def test_choose_best_reports_constraint_selection_mode():
    rows = [
        {
            "threshold": 0.1,
            "trigger_rate": 0.99,
            "boundary_precision": 0.90,
            "boundary_recall": 0.90,
            "predicted_boundary_oracle_mat": 10.0,
        },
        {
            "threshold": 0.2,
            "trigger_rate": 0.10,
            "boundary_precision": 0.25,
            "boundary_recall": 0.40,
            "predicted_boundary_oracle_mat": 5.0,
        },
    ]

    selected = _choose_best(rows)

    assert selected["threshold"] == 0.2
    assert selected["selection_constraints_met"] is True
    assert selected["selected_by"] == "constrained"
    assert selected["best_unconstrained_threshold"] == 0.1


def test_choose_best_marks_unconstrained_diagnostic_when_no_row_passes():
    rows = [
        {
            "threshold": 0.1,
            "trigger_rate": 0.99,
            "boundary_precision": 0.90,
            "boundary_recall": 0.90,
            "predicted_boundary_oracle_mat": 10.0,
        }
    ]

    selected = _choose_best(rows)

    assert selected["threshold"] == 0.1
    assert selected["selection_constraints_met"] is False
    assert selected["selected_by"] == "unconstrained_diagnostic"


def test_calibration_schema_rejects_old_trace_shape():
    old_step = DecodeStep(
        step=0,
        candidates=[
            OracleCandidate.from_dict(
                {
                    "source": "eagle",
                    "token": 1,
                    "depth": 1,
                    "score": -0.1,
                    "accepted": True,
                }
            )
        ],
        acceptance_path=[1],
        stats={"eagle_nodes": 1, "sam_nodes": 0},
    )

    with pytest.raises(ValueError):
        require_calibration_schema([old_step])
