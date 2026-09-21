#!/usr/bin/env python3
import argparse
import os
import sys

import torch

from cotpt import config
from cotpt.inference import (
    generate_with_adaptive_deliberation,
    generate_with_hidden_deliberation,
)
from cotpt.mixing_head import MixingHead
from cotpt.model_utils import load_model_and_tokenizer, pick_device


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Interactive COTPT Chat Interface")
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--use-mixing-head", action="store_true")
    parser.add_argument("--num-hidden-tokens", type=int, default=config.NUM_HIDDEN_THOUGHT_TOKENS)
    parser.add_argument("--max-visible-tokens", type=int, default=config.MAX_VISIBLE_TOKENS)
    parser.add_argument("--thinking-temperature", type=float, default=config.THINKING_TEMPERATURE)
    parser.add_argument("--real-token-temperature", type=float, default=config.REAL_TOKEN_TEMPERATURE)
    parser.add_argument("--sample-real-token", action="store_true", default=config.REAL_TOKEN_DO_SAMPLE)
    parser.add_argument("--hide-hidden-thoughts", dest="show_hidden_thoughts", action="store_false", default=config.SHOW_HIDDEN_THOUGHTS)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--system-prompt", default="You are a helpful, logically rigorous assistant.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Opt back into the base model's visible <think> reasoning. "
        "Default (off) disables it, since COTPT's hidden deliberation replaces visible thinking.",
    )
    return parser


def format_chat_prompt(tokenizer, messages, fallback_system="", enable_thinking=False):
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    if has_chat_template:
        try:
            # Qwen3-style templates default to visible thinking mode, which collides
            # with COTPT's per-token hidden deliberation (model tries to emit a coherent
            # <think> chain while we perturb every step with hidden tokens -> degenerate
            # repetition). Disable it by default so hidden thoughts replace visible CoT.
            # Extra kwargs are ignored by templates that don't support them.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except Exception:
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

    formatted = ""
    if fallback_system:
        formatted += f"System: {fallback_system}\n\n"
    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role_label}: {msg['content']}\n\n"
    formatted += "Assistant: "
    return formatted


def main():
    args = build_arg_parser().parse_args()
    device = pick_device()

    print(f"Loading {args.model_id} on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    mixing_head = None
    if args.use_mixing_head:
        mixing_head_path = os.path.join(args.model_id, "mixing_head.pt")
        mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)
        if os.path.exists(mixing_head_path):
            mixing_head.load_state_dict(torch.load(mixing_head_path, map_location=device))
        else:
            print(f"Notice: {mixing_head_path} not found; running with initialized mixing head.")

    show_thoughts = args.show_hidden_thoughts
    use_adaptive = args.use_mixing_head and mixing_head is not None

    print("\n" + "=" * 60)
    print("COTPT Interactive Chat Session")
    print(f"Model: {args.model_id} | Device: {device}")
    print(f"Mode: {'Adaptive Deliberation' if use_adaptive else 'Hidden Deliberation'}")
    print(f"Thoughts: {args.num_hidden_tokens} hidden tokens/step (display: {'ON' if show_thoughts else 'OFF'})")
    print("Commands: /exit (quit), /clear (reset history), /toggle (toggle thoughts)")
    print("=" * 60 + "\n")

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    while True:
        try:
            user_input = input("\033[1;34mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("Session ended.")
            break

        if cmd in ("/clear", "/reset"):
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            print("\033[2m[Conversation history cleared]\033[0m\n")
            continue

        if cmd in ("/toggle",):
            show_thoughts = not show_thoughts
            print(f"\033[2m[Hidden thoughts display: {'ON' if show_thoughts else 'OFF'}]\033[0m\n")
            continue

        if cmd in ("/mode",):
            if mixing_head is None:
                mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)
            use_adaptive = not use_adaptive
            mode_name = "Adaptive (MixingHead)" if use_adaptive else "Standard (Hidden Deliberation)"
            print(f"\033[2m[Mode switched to: {mode_name}]\033[0m\n")
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = format_chat_prompt(
            tokenizer, messages, fallback_system=args.system_prompt, enable_thinking=args.enable_thinking
        )

        print("\033[1;32mAssistant:\033[0m ", end="", flush=True)

        if use_adaptive:
            response = generate_with_adaptive_deliberation(
                model=model,
                tokenizer=tokenizer,
                mixing_head=mixing_head,
                prompt=prompt,
                max_visible_tokens=args.max_visible_tokens,
                num_hidden_tokens=args.num_hidden_tokens,
                thinking_temperature=args.thinking_temperature,
                real_temperature=args.real_token_temperature,
                real_do_sample=args.sample_real_token,
                show_hidden_thoughts=show_thoughts,
                entropy_threshold=args.entropy_threshold,
                print_prompt=False,
            )
        else:
            response = generate_with_hidden_deliberation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_visible_tokens=args.max_visible_tokens,
                num_hidden_tokens=args.num_hidden_tokens,
                thinking_temperature=args.thinking_temperature,
                real_temperature=args.real_token_temperature,
                real_do_sample=args.sample_real_token,
                show_hidden_thoughts=show_thoughts,
                print_prompt=False,
            )

        messages.append({"role": "assistant", "content": response.strip()})
        print()


if __name__ == "__main__":
    main()
