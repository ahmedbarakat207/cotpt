#!/usr/bin/env python3
"""Run per-token hidden deliberation with aggressive KV-cache eviction.

Usage:
    python scripts/generate.py
    python scripts/generate.py --prompt "Q: ...\nA:" --num-hidden-tokens 8
    python scripts/generate.py --model-id ./checkpoints/qwen3-0.6b-cotpt

    # Use the trained mixing head instead of always fully committing to the
    # post-thought prediction (needs a checkpoint trained with scripts/train.py,
    # which saves mixing_head.pt alongside the model weights):
    python scripts/generate.py --model-id ./checkpoints/qwen3-0.6b-cotpt --use-mixing-head

    # Also skip thinking entirely when the model is already confident:
    python scripts/generate.py --model-id ./checkpoints/qwen3-0.6b-cotpt \
        --use-mixing-head --entropy-threshold 2.0
"""
import argparse
import os

import torch

from cotpt import config
from cotpt.data import MOCK_PROMPT
from cotpt.model_utils import pick_device, load_model_and_tokenizer
from cotpt.mixing_head import MixingHead
from cotpt.inference import generate_with_hidden_deliberation, generate_with_adaptive_deliberation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--prompt", default=MOCK_PROMPT)
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--max-visible-tokens", type=int, default=config.MAX_VISIBLE_TOKENS)
    parser.add_argument("--thinking-temperature", type=float, default=config.THINKING_TEMPERATURE)
    parser.add_argument("--real-token-temperature", type=float, default=config.REAL_TOKEN_TEMPERATURE)
    parser.add_argument("--sample-real-token", action="store_true", default=config.REAL_TOKEN_DO_SAMPLE)
    parser.add_argument("--hide-hidden-thoughts", dest="show_hidden_thoughts",
                         action="store_false", default=config.SHOW_HIDDEN_THOUGHTS)

    parser.add_argument("--use-mixing-head", action="store_true",
                         help="use generate_with_adaptive_deliberation instead of the original "
                              "always-fully-commit loop; needs mixing_head.pt in --model-id")
    parser.add_argument("--entropy-threshold", type=float, default=None,
                         help="only with --use-mixing-head: skip thinking entirely when the "
                              "no-thought prediction's entropy is already below this")
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {args.model_id} on {device} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    print("=" * 70)
    print(f"hidden thinking tokens / visible token : {args.num_hidden_tokens}")
    print(f"max visible tokens                     : {args.max_visible_tokens}")
    print("(dimmed text in \u27ea angle brackets\u27eb = hidden thoughts, shown for debugging;")
    print(" pass --hide-hidden-thoughts to hide them)")
    print("=" * 70 + "\n")

    if args.use_mixing_head:
        mixing_head_path = os.path.join(args.model_id, "mixing_head.pt")
        mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)
        if os.path.exists(mixing_head_path):
            mixing_head.load_state_dict(torch.load(mixing_head_path, map_location=device))
            print(f"Loaded trained mixing head from {mixing_head_path}\n")
        else:
            print(f"WARNING: no mixing_head.pt found at {mixing_head_path} -- using an UNTRAINED "
                  f"mixing head (random blend weights, not meaningful). Train with scripts/train.py first.\n")
        generate_with_adaptive_deliberation(
            model, tokenizer, mixing_head, args.prompt,
            max_visible_tokens=args.max_visible_tokens,
            num_hidden_tokens=args.num_hidden_tokens,
            thinking_temperature=args.thinking_temperature,
            real_temperature=args.real_token_temperature,
            real_do_sample=args.sample_real_token,
            show_hidden_thoughts=args.show_hidden_thoughts,
            entropy_threshold=args.entropy_threshold,
        )
    else:
        generate_with_hidden_deliberation(
            model, tokenizer, args.prompt,
            max_visible_tokens=args.max_visible_tokens,
            num_hidden_tokens=args.num_hidden_tokens,
            thinking_temperature=args.thinking_temperature,
            real_temperature=args.real_token_temperature,
            real_do_sample=args.sample_real_token,
            show_hidden_thoughts=args.show_hidden_thoughts,
        )


if __name__ == "__main__":
    main()
