import json

from evaluation.oracle_depth_decoupled import (
    DepthStratifiedOracle,
    find_optimal_split,
    main,
    simulate_leaf_extension,
    write_gap_plot,
)
from evaluation.oracle_fusion_analysis import (
    DecodeStep,
    OracleCandidate,
    candidate_matches_acceptance,
    candidate_nonroot_path,
)


def _synthetic_trace():
    return {
        "steps": [
            {
                "candidates": [
                    {
                        "source": "eagle",
                        "token": 10,
                        "depth": 1,
                        "score": -0.1,
                        "accepted": True,
                        "token_path": [0, 10],
                    },
                    {
                        "source": "eagle",
                        "token": 20,
                        "depth": 2,
                        "score": -0.2,
                        "accepted": True,
                        "token_path": [0, 10, 20],
                    },
                    {
                        "source": "sam",
                        "token": 30,
                        "depth": 3,
                        "score": 4.0,
                        "accepted": True,
                        "token_path": [0, 10, 20, 30],
                    },
                ],
                "acceptance_path": [10, 20, 30],
                "stats": {"eagle_nodes": 2, "sam_nodes": 1},
            }
        ]
    }


def test_depth_decoupled_simulates_leaf_extension():
    result = simulate_leaf_extension(_synthetic_trace(), d_split=2)

    assert result["mat"] == 4.0
    assert result["accepted_nonroot"] == 3.0
    assert result["gap"] == 1.0 / 3.0
    assert result["eagle_nodes"] == 2.0
    assert result["sam_nodes"] == 1.0
    assert result["prefix_success_rate"] == 1.0


def test_depth_decoupled_finds_best_split_and_reports_strata(tmp_path):
    trace_path = tmp_path / "profile.json"
    trace_path.write_text(json.dumps(_synthetic_trace()) + "\n", encoding="utf-8")

    oracle = DepthStratifiedOracle.from_files([trace_path])
    analysis = oracle.analyze()

    assert find_optimal_split(_synthetic_trace()) == 2
    assert analysis["optimal_d_split"] == 2
    assert analysis["oracle_results"]["eagle_only"]["mat"] == 3.0
    assert analysis["oracle_results"]["depth_decoupled_d2"]["mat"] == 4.0
    assert set(analysis["oracle_results"]) >= {
        "depth_decoupled_d{}".format(index)
        for index in range(1, 11)
    }
    assert [row["depth"] for row in analysis["depth_strata"]] == ["0-2", "3-10"]
    assert analysis["depth_strata"][0]["owner"] == "eagle"
    assert analysis["depth_strata"][0]["mat_contribution"] == 3.0
    assert analysis["depth_strata"][1]["owner"] == "sam"
    assert analysis["depth_strata"][1]["mat_contribution"] == 1.0
    assert analysis["depth_details"]["depth_3"]["sam_matches"] == 1.0


def test_write_gap_plot_outputs_svg(tmp_path):
    analysis = DepthStratifiedOracle(_synthetic_trace()).analyze(max_split=3)
    plot_path = write_gap_plot(analysis, tmp_path / "gap.svg")

    assert plot_path == tmp_path / "gap.svg"
    svg = plot_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Oracle gap vs D_split" in svg
    assert "D_split 2" in svg
    assert "+33.33%" in svg


def test_cli_writes_default_gap_plot_and_json_field(tmp_path, monkeypatch, capsys):
    trace_path = tmp_path / "profile.json"
    output_path = tmp_path / "depth_oracle.json"
    expected_plot_path = tmp_path / "depth_oracle_gap.svg"
    trace_path.write_text(json.dumps(_synthetic_trace()) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "oracle_depth_decoupled.py",
            "--trace-file",
            str(trace_path),
            "--output",
            str(output_path),
            "--max-split",
            "3",
        ],
    )

    main()

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["plot_file"] == str(expected_plot_path)
    assert expected_plot_path.exists()
    assert "<svg" in expected_plot_path.read_text(encoding="utf-8")
    assert "plot_svg: {}".format(expected_plot_path) in capsys.readouterr().out


def test_candidate_nonroot_path_uses_token_path_without_root():
    candidate = OracleCandidate.from_dict(
        {
            "source": "sam",
            "token": 20,
            "depth": 2,
            "accepted": True,
            "token_path": [0, 10, 20],
        }
    )
    step = DecodeStep(
        step=0,
        candidates=[candidate],
        acceptance_path=[10, 20],
        stats={},
    )

    assert candidate_nonroot_path(candidate) == (10, 20)
    assert candidate_matches_acceptance(candidate, step) is True


def test_candidate_nonroot_path_fallback_path_excludes_root():
    depth_one = OracleCandidate.from_dict(
        {
            "source": "sam",
            "token": 10,
            "depth": 1,
            "accepted": True,
            "path": [0],
        }
    )
    depth_two = OracleCandidate.from_dict(
        {
            "source": "sam",
            "token": 20,
            "depth": 2,
            "accepted": True,
            "path": [0, 10],
        }
    )

    assert candidate_nonroot_path(depth_one) == (10,)
    assert candidate_matches_acceptance(
        depth_one,
        DecodeStep(step=0, candidates=[depth_one], acceptance_path=[10], stats={}),
    ) is True
    assert candidate_nonroot_path(depth_two) == (10, 20)
    assert candidate_matches_acceptance(
        depth_two,
        DecodeStep(step=0, candidates=[depth_two], acceptance_path=[10, 20], stats={}),
    ) is True
