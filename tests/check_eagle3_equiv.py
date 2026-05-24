"""S2 验收脚本：EAGLE3 路径与 base greedy 逐 token 对照。

design 第 3 节场景 2 要求：
  samd_model.generate(input_ids, ...) 生成 ≥ 50 个 token，无 shape/key error；
  与 base_model.generate(input_ids, do_sample=False, ...) 逐 token 一致。

注意：先跑 base.generate 再做 samd 包装。samd 的 register_forward_patch 会用 monkey
patch 改写 model.forward 的实例方法绑定；包装后 model 的 forward 不再是原生 LLM 流程，
对 base greedy 行为不安全。
"""
import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from samd import (
    DraftModel,
    SamdConfig,
    SamdGenerationConfig,
    SamdModel,
    load_sam,
)


PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    "USER: Embrace the role of Sheldon from \"The Big Bang Theory\" as we delve into "
    "our conversation. Don't start with phrases like \"As Sheldon\". Let's kick things "
    "off with the following question: \"What is your opinion on hand dryers?\"\n\n"
    "ASSISTANT: "
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tree_model_path", type=str, required=True)
    parser.add_argument("--sam_path", type=str, default=None,
                        help="Optional StaticSAM pkl path; default None (DynSAM still on).")
    parser.add_argument("--tree_method", type=str, default="eagle3")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="design S2 requires >= 50; default 64 for headroom.")
    parser.add_argument("--max_cache_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--samd_n_predicts", type=int, default=15)
    args = parser.parse_args()
    args.dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    return args


def run_base_greedy(model, tokenizer, inputs, max_new_tokens):
    base_gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    st = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=base_gen_config)
    ed = time.perf_counter()
    input_length = inputs.input_ids.shape[-1]
    new_tokens = output[0, input_length:].tolist()
    return new_tokens, ed - st


def run_samd(args, model, tokenizer, inputs):
    sam = load_sam(args.sam_path) if args.sam_path is not None else None
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_model_path=args.tree_model_path,
    )
    draft = DraftModel(
        samd_config,
        sam_static=sam,
        lm=model,
        dtype=args.dtype,
        device=args.device,
    )
    samd_model = SamdModel(
        samd_config,
        model,
        draft,
        tokenizer.eos_token_id,
        args.dtype,
        args.device,
    )
    samd_model.eval()
    gen_config = SamdGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_cache_len=args.max_cache_len,
        greedy=True,
        temperature=0.0,
    )
    input_length = inputs.input_ids.shape[-1]
    st = time.perf_counter()
    with torch.inference_mode():
        outputs = samd_model.generate(**inputs, generation_config=gen_config)
    ed = time.perf_counter()
    new_tokens = outputs.output_ids[0][input_length:]
    return new_tokens, ed - st, outputs.decode_steps, outputs.accepet_length_per_step


def diff_tokens(base, samd, tokenizer):
    cmp_len = min(len(base), len(samd))
    base_cmp = base[:cmp_len]
    samd_cmp = samd[:cmp_len]
    matched = 0
    first_mismatch = None
    for i, (b, s) in enumerate(zip(base_cmp, samd_cmp)):
        if b == s:
            matched += 1
        else:
            if first_mismatch is None:
                first_mismatch = i
            break

    print(f"compared length        : {cmp_len}")
    print(f"matched prefix length  : {matched}")
    if first_mismatch is None:
        print(f"result                 : PASS (all {cmp_len} compared tokens match)")
        return True
    print(f"first mismatch index   : {first_mismatch}")
    b_tok = base_cmp[first_mismatch]
    s_tok = samd_cmp[first_mismatch]
    print(f"  base[{first_mismatch}] = {b_tok}  ({tokenizer.decode([b_tok])!r})")
    print(f"  samd[{first_mismatch}] = {s_tok}  ({tokenizer.decode([s_tok])!r})")
    return False


def main():
    args = parse_args()
    print(f"args: model={args.model_path} tree={args.tree_method} ({args.tree_model_path}) "
          f"sam={args.sam_path} max_new={args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.dtype,
        device_map=args.device,
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(args.device)

    # 1. Base greedy first (must run BEFORE samd wraps/patches the model).
    print("\n=== 1. Base greedy ===")
    base_tokens, base_time = run_base_greedy(model, tokenizer, inputs, args.max_new_tokens)
    print(f"base inference time    : {base_time:.3f}s")
    print(f"base generated tokens  : {len(base_tokens)}")
    print(f"base first 20 ids      : {base_tokens[:20]}")

    # 2. SAMD with EAGLE3 (applies monkey patches; do not run plain base.generate after this).
    print("\n=== 2. SAMD with EAGLE3 ===")
    samd_tokens, samd_time, decode_steps, accept_per_step = run_samd(args, model, tokenizer, inputs)
    print(f"samd inference time    : {samd_time:.3f}s")
    print(f"samd generated tokens  : {len(samd_tokens)}")
    print(f"samd decode steps      : {decode_steps}")
    print(f"samd avg accept length : {len(samd_tokens) / max(decode_steps, 1):.2f}")
    print(f"samd first 20 ids      : {samd_tokens[:20]}")

    # 3. Token-by-token diff.
    print("\n=== 3. Diff ===")
    ok = diff_tokens(base_tokens, samd_tokens, tokenizer)
    if not ok:
        print("S2 FAILED — token mismatch indicates samd[eagle3] diverges from base greedy.")
        sys.exit(1)
    print("S2 PASSED — eagle3 path matches base greedy on this prompt.")


if __name__ == "__main__":
    main()
