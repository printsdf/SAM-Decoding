"""Offline analysis for the Per-Prompt Linear Alignment probe.

Two subcommands:

* ``ridge-fit`` consumes the ``.pt`` dumps emitted by ``SamdModel.generate``
  when ``collect_alignment_probe`` is enabled (paired EAGLE3/base hiddens),
  fits a ridge projection P with several lambdas, and reports the explained
  variance ratio of the centered linear fit ``h_b ≈ h_d @ P`` (equivalent to
  fitting an affine model and absorbing the bias into the per-feature mean).

* ``oov-stats`` walks the diagnosis traces in an answer JSONL file, restricts
  to steps where ``verifier_target_reachable`` is ``False`` (an OOV accept),
  and partitions them by ``verifier_target_in_prompt``.

Decision thresholds for both probes live in
``.codestable/compound/2026-05-26-explore-linear-alignment-probe-feasibility.md``.
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import torch


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError("answer file not found: {}".format(path))
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{}:{} not valid JSON: {}".format(path, line_no, exc)
                ) from exc


def _iter_trace_steps(records: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for rec in records:
        for choice in rec.get("choices", []):
            for turn_trace in choice.get("diagnosis_traces") or []:
                for step in turn_trace:
                    if isinstance(step, dict):
                        yield step


def _load_dumps(dump_dir: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (H_d, H_b, num_dumps) after stacking all .pt files.

    Each .pt file is expected to hold ``h_d`` / ``h_b`` tensors of shape
    ``[S, H]`` (or ``[1, S, H]``; squeezed) with matching shapes.
    """
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError("no .pt dumps found under {}".format(dump_dir))

    h_d_parts: List[torch.Tensor] = []
    h_b_parts: List[torch.Tensor] = []
    for path in paths:
        rec = torch.load(path, map_location="cpu")
        h_d = rec["h_d"].detach().float()
        h_b = rec["h_b"].detach().float()
        if h_d.ndim == 3 and h_d.shape[0] == 1:
            h_d = h_d.squeeze(0)
        if h_b.ndim == 3 and h_b.shape[0] == 1:
            h_b = h_b.squeeze(0)
        if h_d.ndim != 2 or h_b.ndim != 2:
            raise ValueError(
                "{} expected [S, H] tensors, got h_d={} h_b={}".format(
                    path, tuple(h_d.shape), tuple(h_b.shape)
                )
            )
        if h_d.shape != h_b.shape:
            raise ValueError(
                "{} hidden shape mismatch: h_d={} h_b={}".format(
                    path, tuple(h_d.shape), tuple(h_b.shape)
                )
            )
        h_d_parts.append(h_d)
        h_b_parts.append(h_b)

    return (
        torch.cat(h_d_parts, dim=0),
        torch.cat(h_b_parts, dim=0),
        len(paths),
    )


def _parse_lambdas(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one lambda is required")
    return values


def _format_ratio(numer: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return "{:.1%}".format(numer / denom)


def run_ridge_fit(args: argparse.Namespace) -> None:
    h_d, h_b, n_dumps = _load_dumps(args.dump_dir)
    n_pairs, hidden_size = h_d.shape

    # Train / test split. The ridge solver fits SS_res≈0 in-sample whenever
    # n_train < hidden_size (underdetermined), so reporting only the training
    # EVR is uninformative. We split 50/50 with a fixed seed, center both
    # halves using the TRAIN mean (no leakage), fit P on train, and report
    # EVR on both halves. The test column is the actual go/no-go signal.
    gen = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(n_pairs, generator=gen)
    n_train = max(1, int(round(n_pairs * args.train_ratio)))
    n_train = min(n_train, n_pairs - 1)  # keep at least one test sample
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    h_d_train, h_b_train = h_d[train_idx], h_b[train_idx]
    h_d_test, h_b_test = h_d[test_idx], h_b[test_idx]

    h_d_mean = h_d_train.mean(dim=0, keepdim=True)
    h_b_mean = h_b_train.mean(dim=0, keepdim=True)
    h_d_train_c = h_d_train - h_d_mean
    h_b_train_c = h_b_train - h_b_mean
    h_d_test_c = h_d_test - h_d_mean
    h_b_test_c = h_b_test - h_b_mean

    gram = h_d_train_c.T.matmul(h_d_train_c)
    rhs = h_d_train_c.T.matmul(h_b_train_c)
    eye = torch.eye(hidden_size, dtype=gram.dtype)
    ss_tot_train = float((h_b_train_c * h_b_train_c).sum().item())
    ss_tot_test = float((h_b_test_c * h_b_test_c).sum().item())

    regime = "well-determined" if n_train >= hidden_size else "UNDERDETERMINED (n_train < H)"
    print(
        "Linear alignment probe ridge fit: {} dump(s), {} paired positions, "
        "hidden_size={}, split={}/{} ({}), seed={}".format(
            n_dumps, n_pairs, hidden_size, n_train, n_pairs - n_train,
            regime, args.split_seed,
        )
    )
    print("| lambda | train_evr | test_evr |")
    print("|---:|---:|---:|")
    for lam in _parse_lambdas(args.lambdas):
        projection = torch.linalg.solve(gram + lam * eye, rhs)

        resid_train = h_b_train_c - h_d_train_c.matmul(projection)
        ss_res_train = float((resid_train * resid_train).sum().item())
        train_evr = (
            "nan" if ss_tot_train == 0.0 else "{:.4f}".format(1.0 - ss_res_train / ss_tot_train)
        )

        resid_test = h_b_test_c - h_d_test_c.matmul(projection)
        ss_res_test = float((resid_test * resid_test).sum().item())
        test_evr = (
            "nan" if ss_tot_test == 0.0 else "{:.4f}".format(1.0 - ss_res_test / ss_tot_test)
        )

        print("| {} | {} | {} |".format(lam, train_evr, test_evr))


def run_oov_stats(args: argparse.Namespace) -> None:
    counts = {True: 0, False: 0, None: 0}
    total = 0
    for step in _iter_trace_steps(_iter_jsonl(args.answer_file)):
        if step.get("verifier_target_reachable") is not False:
            continue
        in_prompt = step.get("verifier_target_in_prompt")
        key = in_prompt if in_prompt in (True, False) else None
        counts[key] += 1
        total += 1

    print("OOV accept partition by prompt membership (from {})".format(args.answer_file))
    print("| verifier_target_in_prompt | count | ratio |")
    print("|---|---:|---:|")
    for label, key in (("true", True), ("false", False), ("null", None)):
        print("| {} | {} | {} |".format(label, counts[key], _format_ratio(counts[key], total)))
    print("| total_oov | {} | {} |".format(total, "100.0%" if total else "N/A"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ridge = subparsers.add_parser("ridge-fit", help="Fit ridge P on .pt dumps and report EVR.")
    ridge.add_argument("--dump-dir", required=True)
    ridge.add_argument(
        "--lambdas",
        default="1e-3,1e-2,1e-1,1.0",
        help="Comma-separated ridge regularizers to scan.",
    )
    ridge.add_argument(
        "--train-ratio",
        type=float,
        default=0.5,
        help="Fraction of paired positions used for fitting P. The complement is held out for EVR evaluation.",
    )
    ridge.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for the train/test permutation. Fixed by default so the report is reproducible across runs.",
    )

    oov = subparsers.add_parser("oov-stats", help="Partition accepted OOV tokens by prompt membership.")
    oov.add_argument("--answer-file", required=True)

    args = parser.parse_args()
    if args.command == "ridge-fit":
        run_ridge_fit(args)
    elif args.command == "oov-stats":
        run_oov_stats(args)
    else:  # pragma: no cover — argparse enforces required=True
        parser.error("unknown command: {}".format(args.command))


if __name__ == "__main__":
    main()
