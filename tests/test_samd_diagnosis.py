"""Smoke test for the samd diagnosis trace pipeline.

Modes:
  - unit         pure-Python verification of make_trace_step (no model required).
  - integration  constructs SamdModel(tree_method=eagle3), runs generate() twice
                 (collect_diagnosis_trace=True vs False), asserts output is
                 byte-equal across both modes and trace shape is correct.
  - both         runs unit then integration (default).

Integration mode requires --model_path + --tree_model_path (a base Llama and
matching EAGLE3 weights). Per .trellis/spec/backend/runtime-constraints.md, the
torch / transformers / safetensors deps must already be installed.
"""
import argparse
from typing import Set

import torch

from samd.diagnosis import make_trace_step

_EXPECTED_KEYS: Set[str] = {
    "step_idx", "path_type", "accept_length",
    "first_rejected_token_id", "first_rejected_reachable",
    "verifier_target_token_id", "verifier_target_reachable",
}


def unit_test() -> None:
    print("=== UNIT: make_trace_step ===")

    # case 1: sequence path - only step_idx / path_type / accept_length filled
    d = make_trace_step(0, "sequence", 3)
    assert set(d.keys()) == _EXPECTED_KEYS, f"keys: {set(d.keys())}"
    assert d["step_idx"] == 0
    assert d["path_type"] == "sequence"
    assert d["accept_length"] == 3
    for k in ("first_rejected_token_id", "first_rejected_reachable",
              "verifier_target_token_id", "verifier_target_reachable"):
        assert d[k] is None, f"{k} should be None"
    print("  case 1 (sequence path) PASS")

    # case 2: tree path, accept-to-end (accept_length == candidates.shape[1])
    candidates = torch.tensor([[1, 2, 3]], dtype=torch.long)
    tree_logits = torch.zeros(1, 3, 100)
    t2d = torch.zeros(100, dtype=torch.bool)
    d = make_trace_step(1, "tree", 3,
                        best_candidate=0, candidates=candidates,
                        tree_logits=tree_logits, t2d_buffer=t2d)
    assert d["first_rejected_token_id"] is None, "accept-to-end → first_rejected None"
    assert d["first_rejected_reachable"] is None
    assert d["verifier_target_token_id"] is None
    assert d["verifier_target_reachable"] is None
    print("  case 2 (tree accept-to-end) PASS")

    # case 3: tree path with rejection + t2d available
    # candidates[0, 3] = 7 is the rejected token; tree_logits[0, 2] argmax is 42
    candidates = torch.tensor([[1, 2, 3, 7]], dtype=torch.long)
    tree_logits = torch.zeros(1, 4, 100)
    tree_logits[0, 2, 42] = 5.0
    t2d = torch.zeros(100, dtype=torch.bool)
    t2d[42] = True  # verifier target reachable
    # t2d[7] stays False → rejected token unreachable
    d = make_trace_step(2, "tree", 3,
                        best_candidate=0, candidates=candidates,
                        tree_logits=tree_logits, t2d_buffer=t2d)
    assert d["first_rejected_token_id"] == 7
    assert d["first_rejected_reachable"] is False
    assert d["verifier_target_token_id"] == 42
    assert d["verifier_target_reachable"] is True
    print("  case 3 (tree reject + t2d) PASS")

    # case 4: tree path with rejection but no t2d (non-eagle3 tree_method)
    d = make_trace_step(3, "tree", 3,
                        best_candidate=0, candidates=candidates,
                        tree_logits=tree_logits, t2d_buffer=None)
    assert d["first_rejected_token_id"] == 7
    assert d["first_rejected_reachable"] is None, "no t2d → reachable None"
    assert d["verifier_target_token_id"] == 42
    assert d["verifier_target_reachable"] is None
    print("  case 4 (tree reject, no t2d) PASS")

    # case 5: out-of-range token id stays None instead of crashing
    candidates_oor = torch.tensor([[1, 2, 200]], dtype=torch.long)  # 200 >= t2d.shape[0]
    tree_logits_oor = torch.zeros(1, 3, 100)
    d = make_trace_step(4, "tree", 2,
                        best_candidate=0, candidates=candidates_oor,
                        tree_logits=tree_logits_oor, t2d_buffer=t2d)
    assert d["first_rejected_token_id"] == 200
    assert d["first_rejected_reachable"] is None, "out-of-range token id → None"
    print("  case 5 (out-of-range token id) PASS")

    print("UNIT 5/5 cases PASS")


def integration_test(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from samd import (
        DraftModel,
        SamdConfig,
        SamdGenerationConfig,
        SamdModel,
    )

    print("=== INTEGRATION: SamdModel.generate trace on/off ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )

    samd_config = SamdConfig(
        n_predicts=15,
        tree_method="eagle3",
        tree_model_path=args.tree_model_path,
        eagle3_total_token=args.eagle3_total_token,
        eagle3_depth=args.eagle3_depth,
        eagle3_top_k=args.eagle3_top_k,
    )
    draft = DraftModel(samd_config, lm=base, dtype=torch.float16, device=args.device)
    stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    samd_model = SamdModel(
        samd_config, base, draft, tokenizer.eos_token_id,
        torch.float16, args.device,
        stop_token_id=stop_token_id if isinstance(stop_token_id, int) else None,
    )
    samd_model.eval()

    messages = [
        {"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(args.device)

    base_kwargs = dict(max_new_tokens=40, max_cache_len=2048, greedy=True, temperature=0.0)

    print("  run #1: trace OFF")
    out_off = samd_model.generate(
        inputs.input_ids,
        generation_config=SamdGenerationConfig(**base_kwargs, collect_diagnosis_trace=False),
    )
    print("  run #2: trace ON")
    out_on = samd_model.generate(
        inputs.input_ids,
        generation_config=SamdGenerationConfig(**base_kwargs, collect_diagnosis_trace=True),
    )

    # Outputs schema assertions
    assert out_off.diagnosis_trace is None, "trace off should leave diagnosis_trace None"
    assert out_on.diagnosis_trace is not None, "trace on should populate diagnosis_trace"

    # Trace length aligns with decode_steps
    assert len(out_on.diagnosis_trace) == out_on.decode_steps, (
        f"trace len {len(out_on.diagnosis_trace)} != decode_steps {out_on.decode_steps}"
    )

    # Schema of every step
    for i, step in enumerate(out_on.diagnosis_trace):
        assert set(step.keys()) == _EXPECTED_KEYS, f"step {i} keys: {set(step.keys())}"
        assert step["step_idx"] == i, f"step {i} step_idx={step['step_idx']}"
        assert step["path_type"] in ("tree", "sequence"), (
            f"step {i} unexpected path_type={step['path_type']}"
        )

    # Byte-equal output_ids across trace toggle
    assert out_off.output_ids == out_on.output_ids, (
        "trace toggle changed output_ids:\n"
        f"  off={out_off.output_ids}\n"
        f"  on= {out_on.output_ids}"
    )

    tree_n = sum(1 for s in out_on.diagnosis_trace if s["path_type"] == "tree")
    seq_n = sum(1 for s in out_on.diagnosis_trace if s["path_type"] == "sequence")
    print(f"  decode_steps={out_on.decode_steps}, tree_steps={tree_n}, sequence_steps={seq_n}")
    print(f"  trace[0]={out_on.diagnosis_trace[0]}")
    if out_on.decode_steps > 1:
        print(f"  trace[-1]={out_on.diagnosis_trace[-1]}")
    print("INTEGRATION trace on/off PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("unit", "integration", "both"), default="both")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tree_model_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eagle3_total_token", type=int, default=60)
    parser.add_argument("--eagle3_depth", type=int, default=7)
    parser.add_argument("--eagle3_top_k", type=int, default=10)
    args = parser.parse_args()

    if args.mode in ("unit", "both"):
        unit_test()

    if args.mode in ("integration", "both"):
        if not args.model_path or not args.tree_model_path:
            if args.mode == "both":
                print("SKIP integration: --model_path / --tree_model_path not provided")
                return
            raise SystemExit("integration mode requires --model_path + --tree_model_path")
        integration_test(args)


if __name__ == "__main__":
    main()
