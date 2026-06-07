from typing import Dict, Iterable

from .types import CandidateNode


def source_counts(nodes: Iterable[CandidateNode]) -> Dict[str, int]:
    counts = {"eagle": 0, "sam": 0, "both": 0}
    for node in nodes:
        if node.source in ("eagle", "both"):
            counts["eagle"] += 1
        if node.source in ("sam", "both"):
            counts["sam"] += 1
        if node.source == "both":
            counts["both"] += 1
    return counts
