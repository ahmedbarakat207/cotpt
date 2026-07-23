#!/usr/bin/env python3
"""Compare no-thinking, evicted hidden-deliberation, and visible chain-of-
thought on the same log-likelihood metric over a held-out problem set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-id ./checkpoints/qwen3-0.6b-cotpt
"""
import argparse

from cotpt import config
from cotpt.model_utils import pick_device, load_model_and_tokenizer
from cotpt.evaluation import run_evaluation
from cotpt.eval_data import EVAL_PROBLEMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--verbose", action="store_true", help="print per-problem results")
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {args.model_id} on {device} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    print(f"Evaluating on {len(EVAL_PROBLEMS)} held-out problems "
          f"(disjoint from the training corpus) ...\n")
    result = run_evaluation(model, tokenizer, num_hidden_tokens=args.num_hidden_tokens)

    if args.verbose:
        for i, r in enumerate(result["per_problem"]):
            print(f"--- problem {i} ---")
            print(f"  no_think            : {r['no_think']:.4f}")
            print(f"  visible_cot         : {r['visible_cot']:.4f}")
            print(f"  hidden_deliberation : {r['hidden_deliberation']:.4f}")
        print()

    s = result["summary"]
    print("=" * 60)
    print(f"Mean log-likelihood of the correct answer, averaged over {s['num_problems']} problems")
    print(f"(higher / less negative = model found the correct answer more likely)")
    print("=" * 60)
    print(f"  no_think             : {s['no_think']:.4f}")
    print(f"  visible_cot          : {s['visible_cot']:.4f}")
    print(f"  hidden_deliberation  : {s['hidden_deliberation']:.4f}")
    print("=" * 60)
    best = max(s, key=lambda k: s[k] if k != "num_problems" else float("-inf"))
    print(f"\nBest on this eval set: {best}")
    print("\nReminder: this compares conditions on THIS model's current weights. "
          "Run it before and after training to see whether training moved anything, "
          "and don't over-read a result from an untrained or lightly-trained model "
          "or from this small a held-out set.")


if __name__ == "__main__":
    main()
