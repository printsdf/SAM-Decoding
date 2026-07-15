"""Same-trace oracle types, loaders, selectors, and analysis."""

from evaluation.oracle.analyze import analyze_steps, render_table
from evaluation.oracle.load import coerce_decode_steps, load_decode_traces, warn
from evaluation.oracle.select import (
    candidate_matches_acceptance,
    candidate_nonroot_path,
    default_budget,
    select_budgeted,
    select_eagle3_only,
    select_perfect,
    select_sam_sequence_graft,
    select_source_balanced,
    selected_mat,
)
from evaluation.oracle.types import DecodeStep, MethodResult, OracleCandidate

__all__ = [
    "DecodeStep",
    "MethodResult",
    "OracleCandidate",
    "analyze_steps",
    "candidate_matches_acceptance",
    "candidate_nonroot_path",
    "coerce_decode_steps",
    "default_budget",
    "load_decode_traces",
    "render_table",
    "select_budgeted",
    "select_eagle3_only",
    "select_perfect",
    "select_sam_sequence_graft",
    "select_source_balanced",
    "selected_mat",
    "warn",
]
