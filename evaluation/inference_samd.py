"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from typing import Optional
import torch
from fastchat.utils import str_to_torch_dtype
from evaluation.eval import run_evals, reorg_answer_files
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from samd import SamdConfig, SamdModel, SamdGenerationConfig, DraftModel, load_sam
from samd.profiling import FusionProfiler

def samd_forward(
    inputs,
    model: SamdModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    temperature: float = 0.0,
    do_sample: bool = False,
    diagnosis_trace_out: Optional[list] = None,
    max_cache_len: Optional[int] = None,
    fusion_profiler: Optional[FusionProfiler] = None,
):
    if max_cache_len is None:
        max_cache_len = model.lm.config.max_position_embeddings
    input_ids = inputs.input_ids
    outputs = model.generate(
        input_ids,
        generation_config=SamdGenerationConfig(
            max_new_tokens=max_new_tokens,
            max_cache_len=max_cache_len,
            greedy=not do_sample,
            temperature=temperature,
            collect_diagnosis_trace=(diagnosis_trace_out is not None),
            fusion_profiler=fusion_profiler,
        ),
    )
    if diagnosis_trace_out is not None and outputs.diagnosis_trace is not None:
        diagnosis_trace_out.extend(outputs.diagnosis_trace)
    output_ids = outputs.output_ids
    new_token = outputs.decode_tokens
    step = outputs.decode_steps
    accept_length_list = outputs.accepet_length_per_step
    return output_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=str,
        default="vicuna",
        choices=["vicuna", "llama3"]
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["vicuna", "llama3"]
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--samd_n_predicts",
        type=int,
        default=40
    )
    parser.add_argument(
        "--sam_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--samd_len_threshold",
        type=int,
        default=5
    )
    parser.add_argument(
        "--samd_len_bias",
        type=int,
        default=5
    )
    parser.add_argument(
        "--samd_tree_path",
        type=str,
        default=None
    )
    parser.add_argument("--tree_method", type=str, default="eagle2")
    parser.add_argument(
        "--tree_fusion",
        type=str,
        default="none",
        choices=[
            "none",
            "sam_sequence_graft",
            "sam_tree_union_prune",
            "eagle_prefix_sam_expand",
        ],
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="none",
        choices=["none", "naive", "rejection_boundary", "drafter_mars"],
    )
    parser.add_argument("--fusion_max_draft_tokens", type=int, default=60)
    parser.add_argument(
        "--fusion_dedup_strategy",
        type=str,
        default="max_score",
        choices=["max_score", "sum_score", "keep_both"],
    )
    parser.add_argument(
        "--fusion_truncate_strategy",
        type=str,
        default="score",
        choices=["score", "depth_first"],
    )
    parser.add_argument("--rejection_conf_threshold", type=float, default=0.5)
    parser.add_argument("--drafter_mars_theta", type=float, default=0.90)
    parser.add_argument(
        "--drafter_mars_repair",
        type=str,
        default="graft",
        choices=["graft", "naive_fuse"],
    )
    parser.add_argument("--drafter_mars_adaptive_theta", action="store_true")
    parser.add_argument("--drafter_mars_target_trigger_rate", type=float, default=0.75)
    parser.add_argument("--drafter_mars_theta_step", type=float, default=0.02)
    parser.add_argument(
        "--drafter_mars_budget_mode",
        type=str,
        default="fixed",
        choices=["fixed", "depth", "ratio"],
    )
    parser.add_argument("--drafter_mars_max_grafts", type=int, default=1)
    parser.add_argument("--drafter_mars_total_graft_nodes", type=int, default=16)
    parser.add_argument("--boundary_graft_threshold", type=float, default=1.5)
    parser.add_argument("--boundary_graft_max_sam_nodes", type=int, default=8)
    parser.add_argument("--boundary_graft_min_depth", type=int, default=3)
    parser.add_argument("--boundary_graft_max_depth", type=int, default=8)
    parser.add_argument("--sam_tree_max_nodes", type=int, default=16)
    parser.add_argument("--sam_tree_top_k", type=int, default=4)
    parser.add_argument("--sam_tree_alpha", type=float, default=4.0)
    parser.add_argument("--sam_tree_max_depth", type=int, default=6)
    parser.add_argument("--sam_prefix_max_added_nodes", type=int, default=4)
    parser.add_argument("--sam_prefix_top_k", type=int, default=2)
    parser.add_argument("--sam_prefix_min_depth", type=int, default=1)
    parser.add_argument("--sam_prefix_max_depth", type=int, default=4)
    parser.add_argument("--tree_model_path", type=str, default="/data/models/EAGLE-Vicuna-7B-v1.3")
    parser.add_argument("--eagle3_total_token", type=int, default=60)
    parser.add_argument("--eagle3_depth", type=int, default=7)
    parser.add_argument("--eagle3_top_k", type=int, default=10)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument(
        "--collect_diagnosis_trace",
        action="store_true",
        help="Collect per-step diagnosis trace (V_miss / verifier_target) into answer file. Off by default.",
    )
    parser.add_argument(
        "--profile-fusion",
        action="store_true",
        help="Collect opt-in per-step fusion timing and memory diagnostics. Off by default.",
    )
    parser.add_argument(
        "--fusion-profile-file",
        type=str,
        default=None,
        help="JSON trace path for --profile-fusion. Defaults to <answer-file>.fusion_profile.json.",
    )
    parser.add_argument(
        "--fusion-profile-summary-file",
        type=str,
        default=None,
        help="Human-readable summary path for --profile-fusion. Defaults to <answer-file>.fusion_profile.txt.",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=None,
        help="Override SamdStaticCache size. Defaults to model.max_position_embeddings (compat with old SAM-Decoding behavior); set to e.g. 4096 on Llama-3.1 (131072) to avoid OOM on consumer GPUs.",
    )
    args = parser.parse_args()

    question_file = f"evaluation/data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"evaluation/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    if args.profile_fusion and args.num_gpus_total // args.num_gpus_per_model > 1:
        raise SystemExit("--profile-fusion currently supports single-process evaluation only")

    print(f"Output to {answer_file}")
    
    print("len_bias:", args.samd_len_bias)
    print("len_threshold:", args.samd_len_threshold)
    print("tree_fusion:", args.tree_fusion)
    print("fusion_mode:", args.fusion_mode)
    print("drafter_mars_theta:", args.drafter_mars_theta)
    print("drafter_mars_repair:", args.drafter_mars_repair)
    print("drafter_mars_adaptive_theta:", args.drafter_mars_adaptive_theta)
    print("drafter_mars_target_trigger_rate:", args.drafter_mars_target_trigger_rate)
    print("drafter_mars_theta_step:", args.drafter_mars_theta_step)
    print("drafter_mars_budget_mode:", args.drafter_mars_budget_mode)
    print("drafter_mars_max_grafts:", args.drafter_mars_max_grafts)
    print("drafter_mars_total_graft_nodes:", args.drafter_mars_total_graft_nodes)
    print("fusion_max_draft_tokens:", args.fusion_max_draft_tokens)
    print("fusion_dedup_strategy:", args.fusion_dedup_strategy)
    print("fusion_truncate_strategy:", args.fusion_truncate_strategy)
    print("sam_tree_max_nodes:", args.sam_tree_max_nodes)
    print("sam_tree_top_k:", args.sam_tree_top_k)
    print("sam_tree_alpha:", args.sam_tree_alpha)
    print("sam_tree_max_depth:", args.sam_tree_max_depth)
    print("sam_prefix_max_added_nodes:", args.sam_prefix_max_added_nodes)
    print("sam_prefix_top_k:", args.sam_prefix_top_k)
    print("sam_prefix_min_depth:", args.sam_prefix_min_depth)
    print("sam_prefix_max_depth:", args.sam_prefix_max_depth)
    print("eagle3_total_token:", args.eagle3_total_token)
    print("eagle3_depth:", args.eagle3_depth)
    print("eagle3_top_k:", args.eagle3_top_k)

    fusion_profiler = None
    if args.profile_fusion:
        fusion_profile_file = args.fusion_profile_file or "{}.fusion_profile.json".format(answer_file)
        fusion_profile_summary_file = (
            args.fusion_profile_summary_file
            or "{}.fusion_profile.txt".format(answer_file)
        )
        fusion_profiler = FusionProfiler(
            enabled=True,
            trace_path=fusion_profile_file,
            summary_path=fusion_profile_summary_file,
            metadata={
                "bench_name": args.bench_name,
                "model_id": args.model_id,
                "fusion_mode": args.fusion_mode,
                "tree_fusion": args.tree_fusion,
                "samd_len_threshold": args.samd_len_threshold,
                "max_cache_len": args.max_cache_len,
            },
        )
        print("profile_fusion:", True)
        print("fusion_profile_file:", fusion_profile_file)
        print("fusion_profile_summary_file:", fusion_profile_summary_file)
    
    if args.num_gpus_total == 1:
        device_map = "cuda"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation=args.attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.model_type == "llama3":
        stop_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        assert isinstance(stop_token_id, int)
    else:
        stop_token_id = None

    device = next(model.lm_head.parameters()).device
    sam = load_sam(args.sam_path) if args.sam_path is not None else None
    samd_config = SamdConfig(
        n_predicts=args.samd_n_predicts,
        tree_method=args.tree_method,
        tree_fusion=args.tree_fusion,
        fusion_mode=args.fusion_mode,
        fusion_max_draft_tokens=args.fusion_max_draft_tokens,
        fusion_dedup_strategy=args.fusion_dedup_strategy,
        fusion_truncate_strategy=args.fusion_truncate_strategy,
        rejection_conf_threshold=args.rejection_conf_threshold,
        drafter_mars_theta=args.drafter_mars_theta,
        drafter_mars_repair=args.drafter_mars_repair,
        drafter_mars_adaptive_theta=args.drafter_mars_adaptive_theta,
        drafter_mars_target_trigger_rate=args.drafter_mars_target_trigger_rate,
        drafter_mars_theta_step=args.drafter_mars_theta_step,
        drafter_mars_budget_mode=args.drafter_mars_budget_mode,
        drafter_mars_max_grafts=args.drafter_mars_max_grafts,
        drafter_mars_total_graft_nodes=args.drafter_mars_total_graft_nodes,
        boundary_graft_threshold=args.boundary_graft_threshold,
        boundary_graft_max_sam_nodes=args.boundary_graft_max_sam_nodes,
        boundary_graft_min_depth=args.boundary_graft_min_depth,
        boundary_graft_max_depth=args.boundary_graft_max_depth,
        tree_model_path=args.tree_model_path,
        eagle3_total_token=args.eagle3_total_token,
        eagle3_depth=args.eagle3_depth,
        eagle3_top_k=args.eagle3_top_k,
        len_threshold=args.samd_len_threshold,
        len_bias=args.samd_len_bias,
        tree_path=args.samd_tree_path,
        sam_tree_max_nodes=args.sam_tree_max_nodes,
        sam_tree_top_k=args.sam_tree_top_k,
        sam_tree_alpha=args.sam_tree_alpha,
        sam_tree_max_depth=args.sam_tree_max_depth,
        sam_prefix_max_added_nodes=args.sam_prefix_max_added_nodes,
        sam_prefix_top_k=args.sam_prefix_top_k,
        sam_prefix_min_depth=args.sam_prefix_min_depth,
        sam_prefix_max_depth=args.sam_prefix_max_depth,
    )
    draft = DraftModel(
        samd_config, 
        sam_static=sam,
        lm=model,
        dtype=str_to_torch_dtype(args.dtype),
        device=device,
    )
    samd_model = SamdModel(
        samd_config, 
        model, 
        draft, 
        tokenizer.eos_token_id,
        str_to_torch_dtype(args.dtype),
        device, 
        stop_token_id=stop_token_id
    )

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_evals[args.template](
        model=samd_model,
        tokenizer=tokenizer,
        forward_func=samd_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
        collect_diagnosis_trace=args.collect_diagnosis_trace,
        max_cache_len=args.max_cache_len,
        fusion_profiler=fusion_profiler,
    )

    reorg_answer_files[args.template](answer_file)
    if fusion_profiler is not None:
        fusion_profiler.finish_run()
        fusion_profiler.write_json()
        fusion_profiler.write_summary()
        print(fusion_profiler.render_summary())
