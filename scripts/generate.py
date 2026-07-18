#!/usr/bin/env python3
"""Run per-token hidden deliberation with aggressive KV-cache eviction.

Usage:
    python scripts/generate.py
    python scripts/generate.py --prompt "Q: ..." --num-hidden-tokens 8
    python scripts/generate.py --model-id ./checkpoints/qwen3-0.6b-quiet-star
"""
import argparse

from quiet_star import config
from quiet_star.data import MOCK_PROMPT
from quiet_star.model_utils import pick_device, load_model_and_tokenizer
from quiet_star.inference import generate_with_hidden_deliberation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--prompt", default=MOCK_PROMPT)
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--max-visible-tokens", type=int, default=config.MAX_VISIBLE_TOKENS)
    parser.add_argument("--thinking-temperature", type=float, default=config.THINKING_TEMPERATURE)
    parser.add_argument("--real-token-temperature", type=float, default=config.REAL_TOKEN_TEMPERATURE)
    parser.add_argument("--sample-real-token", action="store_true", default=config.REAL_TOKEN_DO_SAMPLE)
    parser.add_argument("--hide-hidden-thoughts", dest="show_hidden_thoughts",
                         action="store_false", default=config.SHOW_HIDDEN_THOUGHTS)
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

    generate_with_hidden_deliberation(
        model,
        tokenizer,
        args.prompt,
        max_visible_tokens=args.max_visible_tokens,
        num_hidden_tokens=args.num_hidden_tokens,
        thinking_temperature=args.thinking_temperature,
        real_temperature=args.real_token_temperature,
        real_do_sample=args.sample_real_token,
        show_hidden_thoughts=args.show_hidden_thoughts,
    )


if __name__ == "__main__":
    main()
