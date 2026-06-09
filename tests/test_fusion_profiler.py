import json

from samd.profiling import FusionProfiler


def test_fusion_profiler_records_oracle_trace_fields(tmp_path):
    trace_path = tmp_path / "fusion_profile.json"
    summary_path = tmp_path / "fusion_profile.txt"

    with FusionProfiler(
        trace_path=str(trace_path),
        summary_path=str(summary_path),
        metadata={"fusion_mode": "naive"},
    ) as profiler:
        profiler.start_step({"sample": 1})
        profiler.add_step_trace(
            candidates=[
                {
                    "source": "eagle",
                    "token": 11,
                    "depth": 1,
                    "score": -0.2,
                    "accepted": True,
                }
            ],
            acceptance_path=[11],
            stats={"mat": 2, "eagle_nodes": 1, "sam_nodes": 0},
        )
        profiler.finish_step({"accepted_tokens": 2})

    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["summary"]["steps"] == 1
    assert data["steps"][0]["candidates"][0]["accepted"] is True
    assert data["steps"][0]["acceptance_path"] == [11]
    assert data["steps"][0]["stats"]["mat"] == 2
    assert summary_path.exists()
