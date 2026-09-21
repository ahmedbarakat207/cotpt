#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch

from cotpt import config
from cotpt.benchmark import (
    DEFAULT_CONDITIONS,
    format_benchmark_table,
    load_benchmark_problems,
    run_benchmark,
)
from cotpt.mixing_head import MixingHead
from cotpt.model_utils import load_model_and_tokenizer, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--dataset-name", default=None, choices=["gsm8k", "math", "mmlu", "arc", "svamp"])
    parser.add_argument("--dataset-path", default="gsm8k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--max-visible-tokens", type=int, default=config.MAX_VISIBLE_TOKENS)
    parser.add_argument("--use-mixing-head", action="store_true")
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--mixing-mode", default=config.MIXING_MODE, choices=["hidden", "logit"])
    parser.add_argument("--use-thought-tokens", action="store_true", default=config.USE_THOUGHT_TOKENS)
    parser.add_argument("--no-likelihood", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions:
        sys.exit("Error: No conditions specified.")

    device = pick_device()
    print(f"Loading {args.model_id} on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    from cotpt.model_utils import get_thought_token_ids
    start_id, end_id = get_thought_token_ids(tokenizer) if args.use_thought_tokens else (None, None)

    mixing_head = None
    if args.use_mixing_head or "cotpt_adaptive" in conditions:
        mixing_head_path = os.path.join(args.model_id, "mixing_head.pt")
        mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)
        if os.path.exists(mixing_head_path):
            mixing_head.load_state_dict(torch.load(mixing_head_path, map_location=device))
        else:
            print(f"Warning: {mixing_head_path} not found; using untrained mixing head.")

    target_dataset = args.dataset_name or args.dataset_path
    problems = load_benchmark_problems(target_dataset)
    if args.limit is not None and args.limit > 0:
        problems = problems[:args.limit]

    print(f"Evaluating {len(problems)} problems on: {', '.join(conditions)}")

    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        conditions=conditions,
        mixing_head=mixing_head,
        num_hidden_tokens=args.num_hidden_tokens,
        max_visible_tokens=args.max_visible_tokens,
        entropy_threshold=args.entropy_threshold,
        compute_likelihood=not args.no_likelihood,
        verbose=args.verbose,
        mixing_mode=args.mixing_mode,
        use_thought_tokens=args.use_thought_tokens,
        start_thought_id=start_id,
        end_thought_id=end_id,
    )

    summary = results["summary"]
    print("\n" + format_benchmark_table(summary, format_type="terminal") + "\n")

    if args.output_markdown:
        md_content = f"# Benchmark Results: Normal vs COTPT\n\n"
        md_content += f"- **Model**: `{args.model_id}`\n"
        md_content += f"- **Problems**: {summary['num_problems']}\n\n"
        md_content += format_benchmark_table(summary, format_type="markdown") + "\n"
        with open(args.output_markdown, "w", encoding="utf-8") as f:
            f.write(md_content)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
