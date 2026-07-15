import pytest

from evaluation.boundary import (
    choose_best as _choose_best,
    evaluate_predictions,
    predict_low_margin_node,
    require_calibration_schema,
    threshold_grid as _threshold_grid,
    validate_probability_threshold,
)
from evaluation.drafter_mars import (
    DRAFT_MARS_THETA_GRID,
    predict_draft_delta_top_path,
    predict_draft_mars_reachable,
    predict_draft_mars_top_path,
    raw_logit_diagnostics,
    schema_report as drafter_mars_schema_report,
)
from evaluation.oracle import DecodeStep, OracleCandidate


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
    parent_top1_logit=None,
    parent_top2_logit=None,
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
            "parent_top1_logit": parent_top1_logit,
            "parent_top2_logit": parent_top2_logit,
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
                # Root parent (start token) expansion raw logits. Ratio 0.6
                # stays below the test's theta=0.80 so the top-path gate fires
                # at depth 2 (parent_index=1, ratio 0.9967) instead.
                parent_top1_logit=5.0,
                parent_top2_logit=3.0,
                accepted=True,
            ),
            # First depth-2 node is on the wrong parent (off-path low-margin
            # branch) so the top-path Drafter-MARS gate must ignore it.
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
                parent_top1_logit=2.0,
                parent_top2_logit=1.99,
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
                parent_top1_logit=3.0,
                parent_top2_logit=2.99,
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


def test_low_margin_node_predictor_selects_low_margin_child():
    step = _step()

    low = predict_low_margin_node(step, threshold=0.08)

    assert low.candidate.tree_index == 3
    assert low.score == 0.05


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


def test_choose_best_forbids_unconstrained_fallback_when_no_row_passes():
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

    # Q2 guard: no threshold meets the bar, so selected_by is "none", no
    # threshold is selected, and the unconstrained best is a diagnostic only.
    assert selected["selection_constraints_met"] is False
    assert selected["selected_by"] == "none"
    assert selected["threshold"] is None
    assert selected["best_unconstrained_threshold"] == 0.1
    assert selected["best_unconstrained_oracle_mat"] == 10.0


def test_drafter_mars_ratio_uses_raw_logits_not_logprobs():
    step = _step()
    # The depth-2 top-path parent (parent_index=1) has z1=3.0, z2=2.99, ratio
    # ~0.9967. A theta below that triggers; a theta above does not. The ratio
    # is derived from parent_top1/top2_logit, not local_logprob.
    pred_low_theta = predict_draft_mars_top_path(step, theta=0.80)
    pred_high_theta = predict_draft_mars_top_path(step, theta=0.999)

    assert pred_low_theta.depth == 2
    assert pred_high_theta.depth is None


def test_drafter_mars_larger_theta_is_stricter():
    step = _step()
    triggered = []
    for theta in DRAFT_MARS_THETA_GRID:
        pred = predict_draft_mars_top_path(step, theta=float(theta))
        triggered.append(1 if pred.depth is not None else 0)

    # Larger theta -> fewer triggers (monotonic non-increasing).
    assert all(triggered[i] >= triggered[i + 1] for i in range(len(triggered) - 1))


def test_drafter_mars_top_path_ignores_off_path_low_margin_branch():
    step = _step()
    # The off-path branch (parent_index=4) has a near-saturated ratio too, but
    # it is not on the top path. The top-path predictor must skip it and only
    # fire on the on-path parent (parent_index=1).
    pred = predict_draft_mars_top_path(step, theta=0.80)

    assert pred.depth == 2
    assert pred.candidate.parent_index == 1


def test_drafter_mars_reports_nonpositive_and_both_negative_rates():
    step = _step()
    diagnostics = raw_logit_diagnostics([step])

    # Three parents carry raw logits (root parent + two depth-2 parents), all
    # positive top-1; both_negative_rate is 0.
    assert diagnostics["raw_logit_parent_count"] == 3
    assert diagnostics["top1_nonpositive_rate"] == 0.0
    assert diagnostics["both_negative_rate"] == 0.0


def test_drafter_mars_schema_report_marks_raw_logits_available():
    step = _step()
    report = drafter_mars_schema_report([step])

    assert report["raw_logits_available"] is True
    assert report["schema_ok"] is True


def test_drafter_mars_same_budget_predicted_mat_le_perfect_mat():
    from evaluation.drafter_mars import calibrate_drafter_mars

    train_step = _step(question_index=0)
    valid_step = _step(question_index=1)
    result = calibrate_drafter_mars(
        [train_step],
        [valid_step],
        node_budget=2,
        max_trigger_rate=1.0,
        min_boundary_precision=0.0,
    )

    for method in ("draft_mars_top_path", "draft_delta_top_path", "draft_mars_reachable"):
        held = result["heldout_metrics"][method]
        assert held["predicted_boundary_oracle_mat"] <= held["perfect_mat"] + 1e-9


def test_drafter_mars_unconstrained_fallback_guard():
    from evaluation.drafter_mars import calibrate_drafter_mars

    train_step = _step(question_index=0)
    valid_step = _step(question_index=1)
    # An unreachable bar: trigger <= 0% is impossible when the predictor fires.
    result = calibrate_drafter_mars(
        [train_step],
        [valid_step],
        node_budget=2,
        max_trigger_rate=0.0,
        min_boundary_precision=1.0,
    )

    for method in ("draft_mars_top_path", "draft_delta_top_path", "draft_mars_reachable"):
        selected = result["selected_thresholds"][method]
        held = result["heldout_metrics"][method]
        assert selected["selected_by"] == "none"
        assert selected["selection_constraints_met"] is False
        assert selected["threshold"] is None
        # Hard fail: predicted MAT falls back to eagle MAT, cannot exceed it.
        assert held["predicted_boundary_oracle_mat"] == held["eagle_mat"]
        # Unconstrained best is recorded as a diagnostic only.
        assert selected["best_unconstrained_oracle_mat"] is not None


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
