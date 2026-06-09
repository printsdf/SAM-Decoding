import json

from evaluation.oracle_fusion_analysis import analyze_steps, load_decode_traces


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
