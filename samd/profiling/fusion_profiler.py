from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


FUSION_PROFILE_PHASES = ("draft_eagle", "draft_sam", "fusion_logic", "verify", "total")


def _cuda_memory_allocated() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
    except Exception:
        return 0
    return 0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class ProfileStats:
    steps: int
    timings: Dict[str, float]
    avg_step_timings: Dict[str, float]
    timing_pct: Dict[str, float]
    fusion_overhead_pct: float
    memory_alloc: int
    peak_memory_alloc: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "timings": self.timings,
            "avg_step_timings": self.avg_step_timings,
            "timing_pct": self.timing_pct,
            "fusion_overhead_pct": self.fusion_overhead_pct,
            "memory_alloc": self.memory_alloc,
            "peak_memory_alloc": self.peak_memory_alloc,
        }


class FusionProfiler:
    """Opt-in per-step profiler for dual-draft fusion diagnosis."""

    def __init__(
        self,
        enabled: bool = True,
        trace_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.trace_path = trace_path
        self.summary_path = summary_path
        self.metadata = dict(metadata or {})
        self._steps: List[Dict[str, Any]] = []
        self._totals: Dict[str, float] = {phase: 0.0 for phase in FUSION_PROFILE_PHASES}
        self._current_step: Optional[Dict[str, Any]] = None
        self._step_context: Dict[str, Any] = {}
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._memory_start = 0
        self._memory_end = 0
        self._peak_memory = 0

    def __enter__(self) -> "FusionProfiler":
        self.start_run()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish_run()
        if self.trace_path:
            self.write_json(self.trace_path)
        if self.summary_path:
            self.write_summary(self.summary_path)

    @classmethod
    def disabled(cls) -> "FusionProfiler":
        return cls(enabled=False)

    @property
    def active(self) -> bool:
        return self.enabled

    @property
    def steps(self) -> List[Dict[str, Any]]:
        return list(self._steps)

    def reset(self) -> None:
        self._steps = []
        self._totals = {phase: 0.0 for phase in FUSION_PROFILE_PHASES}
        self._current_step = None
        self._step_context = {}
        self._started_at = None
        self._finished_at = None
        self._memory_start = 0
        self._memory_end = 0
        self._peak_memory = 0

    def start_run(self) -> None:
        if not self.enabled or self._started_at is not None:
            return
        self._started_at = time.time()
        self._memory_start = _cuda_memory_allocated()
        self._memory_end = self._memory_start
        self._peak_memory = self._memory_start

    def finish_run(self) -> None:
        if not self.enabled:
            return
        if self._current_step is not None:
            self.finish_step()
        if self._started_at is None:
            self.start_run()
        self._finished_at = time.time()
        self._memory_end = _cuda_memory_allocated()
        self._peak_memory = max(self._peak_memory, self._memory_end)

    def start_step(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        if self._started_at is None:
            self.start_run()
        if self._current_step is not None:
            raise RuntimeError("fusion profiler step already active")
        now = time.perf_counter()
        memory_start = _cuda_memory_allocated()
        self._peak_memory = max(self._peak_memory, memory_start)
        step_metadata = dict(self._step_context)
        step_metadata.update(metadata or {})
        self._current_step = {
            "step": len(self._steps),
            "timings": {phase: 0.0 for phase in FUSION_PROFILE_PHASES},
            "metadata": _json_safe(step_metadata),
            "memory_start": memory_start,
            "_start": now,
        }

    def finish_step(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._current_step is None:
            return
        now = time.perf_counter()
        step = self._current_step
        timings = step["timings"]
        timings["total"] += now - float(step["_start"])
        if metadata:
            step["metadata"].update(_json_safe(metadata))
        memory_end = _cuda_memory_allocated()
        self._peak_memory = max(self._peak_memory, memory_end)
        step["memory_end"] = memory_end
        step["memory_alloc"] = memory_end - int(step["memory_start"])
        step.pop("_start", None)
        for phase in FUSION_PROFILE_PHASES:
            self._totals[phase] += float(timings.get(phase, 0.0))
        self._steps.append(step)
        self._current_step = None

    def abort_step(self, error: str) -> None:
        if not self.enabled or self._current_step is None:
            return
        self.finish_step({"error": str(error)})

    def start_section(self, name: str) -> float:
        if not self.enabled or self._current_step is None:
            return 0.0
        if name not in FUSION_PROFILE_PHASES:
            raise ValueError("unsupported fusion profile phase: {}".format(name))
        return time.perf_counter()

    def end_section(self, name: str, started_at: float) -> None:
        if not self.enabled or self._current_step is None or not started_at:
            return
        self._current_step["timings"][name] += time.perf_counter() - started_at

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        started_at = self.start_section(name)
        try:
            yield
        finally:
            self.end_section(name, started_at)

    def add_step_metadata(self, metadata: Dict[str, Any]) -> None:
        if not self.enabled or self._current_step is None:
            return
        self._current_step["metadata"].update(_json_safe(metadata))

    def set_step_context(self, metadata: Dict[str, Any]) -> None:
        """Set metadata copied into every subsequently started step."""
        if not self.enabled:
            return
        self._step_context = _json_safe(metadata or {})

    def clear_step_context(self) -> None:
        if not self.enabled:
            return
        self._step_context = {}

    def add_step_trace(
        self,
        candidates: Optional[List[Dict[str, Any]]] = None,
        acceptance_path: Optional[List[int]] = None,
        stats: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach oracle-analysis trace fields to the active decode step."""
        if not self.enabled or self._current_step is None:
            return
        if candidates is not None:
            self._current_step["candidates"] = _json_safe(candidates)
        if acceptance_path is not None:
            self._current_step["acceptance_path"] = _json_safe(acceptance_path)
        if stats is not None:
            self._current_step["stats"] = _json_safe(stats)
        if labels is not None:
            for key, value in labels.items():
                self._current_step[str(key)] = _json_safe(value)

    def summary(self) -> ProfileStats:
        steps = len(self._steps)
        total_time = float(self._totals.get("total", 0.0))
        avg_step_timings = {
            phase: (float(seconds) / steps if steps else 0.0)
            for phase, seconds in self._totals.items()
        }
        timing_pct = {
            phase: (float(seconds) / total_time * 100.0 if total_time > 0 else 0.0)
            for phase, seconds in self._totals.items()
        }
        fusion_overhead = float(self._totals.get("draft_sam", 0.0)) + float(
            self._totals.get("fusion_logic", 0.0)
        )
        memory_end = self._memory_end if self._finished_at is not None else _cuda_memory_allocated()
        peak_memory = max(self._peak_memory, memory_end)
        return ProfileStats(
            steps=steps,
            timings={phase: float(self._totals.get(phase, 0.0)) for phase in FUSION_PROFILE_PHASES},
            avg_step_timings=avg_step_timings,
            timing_pct=timing_pct,
            fusion_overhead_pct=(fusion_overhead / total_time * 100.0 if total_time > 0 else 0.0),
            memory_alloc=memory_end - self._memory_start,
            peak_memory_alloc=peak_memory - self._memory_start,
        )

    def to_json(self) -> Dict[str, Any]:
        stats = self.summary()
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "metadata": _json_safe(self.metadata),
            "summary": stats.to_dict(),
            "steps": _json_safe(self._steps),
        }

    def render_summary(self) -> str:
        stats = self.summary()
        lines = [
            "Fusion Profile Summary",
            "======================",
            "steps: {}".format(stats.steps),
            "fusion_overhead_pct: {:.2f}%".format(stats.fusion_overhead_pct),
            "memory_alloc: {} bytes ({:.2f} MiB)".format(
                stats.memory_alloc,
                stats.memory_alloc / (1024.0 * 1024.0),
            ),
            "peak_memory_alloc: {} bytes ({:.2f} MiB)".format(
                stats.peak_memory_alloc,
                stats.peak_memory_alloc / (1024.0 * 1024.0),
            ),
            "",
            "{:<16} {:>12} {:>12} {:>12}".format("phase", "seconds", "pct_total", "avg_ms"),
        ]
        for phase in FUSION_PROFILE_PHASES:
            lines.append(
                "{:<16} {:>12.6f} {:>11.2f}% {:>12.3f}".format(
                    phase,
                    stats.timings.get(phase, 0.0),
                    stats.timing_pct.get(phase, 0.0),
                    stats.avg_step_timings.get(phase, 0.0) * 1000.0,
                )
            )
        if self.trace_path:
            lines.extend(["", "trace_json: {}".format(self.trace_path)])
        if self.summary_path:
            lines.append("summary_txt: {}".format(self.summary_path))
        return "\n".join(lines)

    def write_json(self, path: Optional[str] = None) -> Path:
        output_path = Path(path or self.trace_path or "fusion_profile.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path

    def write_summary(self, path: Optional[str] = None) -> Path:
        output_path = Path(path or self.summary_path or "fusion_profile_summary.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render_summary() + "\n", encoding="utf-8")
        return output_path
