#!/usr/bin/env python3
"""Run full Quiet-STaR-style think/talk/learn training.

Usage:
    python scripts/train.py
    python scripts/train.py --num-steps 200 --output-dir ./checkpoints/my-run

See README.md for what this deliberately simplifies vs. the paper and how
it was validated before being included in this project.
"""
import argparse
import os
import random

import torch

from quiet_star import config
from quiet_star.data import TRAIN_TEXTS
from quiet_star.model_utils import pick_device, load_model_and_tokenizer
from quiet_star.mixing_head import MixingHead
from quiet_star.training import quiet_star_training_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--num-steps", type=int, default=config.NUM_TRAINING_STEPS)
    parser.add_argument("--num-rollouts", type=int, default=config.NUM_ROLLOUTS)
    parser.add_argument("--num-think-positions", type=int, default=config.NUM_THINK_POSITIONS)
    parser.add_argument("--lookahead", type=int, default=config.LOOKAHEAD)
    parser.add_argument("--base-lr", type=float, default=config.BASE_LR)
    parser.add_argument("--mix-head-lr", type=float, default=config.MIX_HEAD_LR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device()
    print(f"Loading {args.model_id} on {device} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)
    model.train()
    mixing_head = MixingHead(hidden_size=model.config.hidden_size).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.base_lr},
        {"params": mixing_head.parameters(), "lr": args.mix_head_lr},
    ])
    trainable_params = list(model.parameters()) + list(mixing_head.parameters())

    print(f"Training for {args.num_steps} steps on {len(TRAIN_TEXTS)} built-in examples ...")
    print("=" * 78)
    for step in range(args.num_steps):
        text = TRAIN_TEXTS[step % len(TRAIN_TEXTS)]
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[1] < args.lookahead + 2:
            continue

        stats = quiet_star_training_step(
            model, mixing_head, input_ids,
            num_think_positions=args.num_think_positions,
            num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
        )

        optimizer.zero_grad()
        stats["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, config.MAX_GRAD_NORM)
        optimizer.step()

        if step % config.LOG_EVERY == 0 or step == args.num_steps - 1:
            print(
                f"step {step:4d}/{args.num_steps} | total={stats['total_loss'].item():6.3f} "
                f"aux_lm={stats['aux_lm_loss']:6.3f} reinforce={stats['reinforce_loss']:7.4f} "
                f"mix={stats['mix_loss']:6.3f} mean_reward={stats['mean_reward']:7.3f} "
                f"mix_w={stats['mean_mix_weight']:.3f}"
            )
    print("=" * 78)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving fine-tuned model + tokenizer to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    mixing_head_path = os.path.join(args.output_dir, "mixing_head.pt")
    torch.save(mixing_head.state_dict(), mixing_head_path)
    print(f"Mixing head saved separately to {mixing_head_path} (standard HF loaders won't "
          f"pick this up automatically -- it's a new module, not part of the base architecture).")
    print(f"\nTo generate with these weights using the aggressive-eviction loop, run:\n"
          f"    python scripts/generate.py --model-id {args.output_dir}")


if __name__ == "__main__":
    main()
