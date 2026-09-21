#!/usr/bin/env python3
import argparse
import os
import torch

from cotpt import config
from cotpt.benchmark import load_benchmark_problems
from cotpt.evaluation import run_evaluation
from cotpt.mixing_head import MixingHead
from cotpt.model_utils import load_model_and_tokenizer, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--dataset-name",
        default="gsm8k",
        choices=["gsm8k", "math", "mmlu", "arc", "svamp"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--use-mixing-head", action="store_true")
    parser.add_argument("--mixing-mode", default=config.MIXING_MODE, choices=["hidden", "logit"])
    parser.add_argument("--use-thought-tokens", action="store_true", default=config.USE_THOUGHT_TOKENS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    from cotpt.model_utils import get_thought_token_ids
    start_id, end_id = get_thought_token_ids(tokenizer) if args.use_thought_tokens else (None, None)

    mixing_head = None
    if args.use_mixing_head:
        mixing_head_path = os.path.join(args.model_id, "mixing_head.pt")
        mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)
        if os.path.exists(mixing_head_path):
            mixing_head.load_state_dict(torch.load(mixing_head_path, map_location=device))

    path_or_name = args.dataset_path or args.dataset_name
    problems = load_benchmark_problems(path_or_name)
    if args.limit:
        problems = problems[: args.limit]

    result = run_evaluation(
        model,
        tokenizer,
        problems=problems,
        num_hidden_tokens=args.num_hidden_tokens,
        mixing_head=mixing_head,
        mixing_mode=args.mixing_mode,
        use_thought_tokens=args.use_thought_tokens,
        start_thought_id=start_id,
        end_thought_id=end_id,
    )

    if args.verbose:
        for i, r in enumerate(result["per_problem"]):
            print(f"Problem {i}: no_think={r['no_think']:.4f} visible_cot={r['visible_cot']:.4f} hidden={r['hidden_deliberation']:.4f}")
        print()

    s = result["summary"]
    print("=" * 50)
    print(f"Dataset: {path_or_name} ({s['num_problems']} problems)")
    print(f"Mean log-likelihood:")
    print(f"  no_think            : {s['no_think']:.4f}")
    print(f"  visible_cot         : {s['visible_cot']:.4f}")
    print(f"  hidden_deliberation : {s['hidden_deliberation']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
