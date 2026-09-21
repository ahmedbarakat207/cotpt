#!/usr/bin/env python3
import argparse
import os

import torch

from cotpt import config
from cotpt.data import DEFAULT_PROMPT
from cotpt.inference import generate_with_adaptive_deliberation, generate_with_hidden_deliberation
from cotpt.mixing_head import MixingHead
from cotpt.model_utils import load_model_and_tokenizer, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--max-visible-tokens", type=int, default=config.MAX_VISIBLE_TOKENS)
    parser.add_argument("--thinking-temperature", type=float, default=config.THINKING_TEMPERATURE)
    parser.add_argument("--real-token-temperature", type=float, default=config.REAL_TOKEN_TEMPERATURE)
    parser.add_argument("--sample-real-token", action="store_true", default=config.REAL_TOKEN_DO_SAMPLE)
    parser.add_argument("--hide-hidden-thoughts", dest="show_hidden_thoughts", action="store_false", default=config.SHOW_HIDDEN_THOUGHTS)
    parser.add_argument("--use-mixing-head", action="store_true")
    parser.add_argument("--entropy-threshold", type=float, default=None)
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    if args.use_mixing_head:
        mixing_head_path = os.path.join(args.model_id, "mixing_head.pt")
        mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)
        if os.path.exists(mixing_head_path):
            mixing_head.load_state_dict(torch.load(mixing_head_path, map_location=device))
        generate_with_adaptive_deliberation(
            model,
            tokenizer,
            mixing_head,
            args.prompt,
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
