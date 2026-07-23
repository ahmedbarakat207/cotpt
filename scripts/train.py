#!/usr/bin/env python3
"""Run full COTPT-style think/talk/learn training.

Usage:
    python scripts/train.py
    python scripts/train.py --num-steps 200 --output-dir ./checkpoints/my-run
    python scripts/train.py --use-lora --use-kl-penalty
    python scripts/train.py --data-path ./my_corpus.txt
    python scripts/train.py --resume-from ./checkpoints/my-run

See README.md for what each flag does and what this deliberately simplifies
vs. the COTPT paper, and for the validation this went through before
being included in this project.
"""
import argparse
import copy
import os
import random

import torch

from cotpt import config
from cotpt.data import TRAIN_TEXTS
from cotpt.dataset import load_training_texts
from cotpt.model_utils import pick_device, load_model_and_tokenizer
from cotpt.mixing_head import MixingHead
from cotpt.value_head import ValueHead
from cotpt.training import cotpt_training_step
from cotpt.logging_utils import ExperimentLogger
from cotpt.checkpoint_utils import save_checkpoint, load_checkpoint_for_resume


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--resume-from", default=None,
                         help="checkpoint dir written by a previous run of this script; "
                              "restores model/adapter weights, heads, optimizer state, and step count")
    parser.add_argument("--data-path", default=None, help="local .txt file, paragraphs separated by blank lines")
    parser.add_argument("--hf-dataset", default=None, help="HF datasets name, e.g. 'gsm8k' (needs `pip install datasets`)")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")

    parser.add_argument("--num-steps", type=int, default=config.NUM_TRAINING_STEPS)
    parser.add_argument("--num-rollouts", type=int, default=config.NUM_ROLLOUTS)
    parser.add_argument("--num-think-positions", type=int, default=config.NUM_THINK_POSITIONS)
    parser.add_argument("--lookahead", type=int, default=config.LOOKAHEAD)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="0 disables intermediate checkpoints")

    parser.add_argument("--base-lr", type=float, default=config.BASE_LR)
    parser.add_argument("--mix-head-lr", type=float, default=config.MIX_HEAD_LR)
    parser.add_argument("--value-head-lr", type=float, default=config.VALUE_HEAD_LR)

    parser.add_argument("--use-lora", action="store_true", default=config.USE_LORA)
    parser.add_argument("--lora-r", type=int, default=config.LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=config.LORA_ALPHA)

    parser.add_argument("--use-value-baseline", action="store_true", default=config.USE_VALUE_BASELINE)
    parser.add_argument("--no-value-baseline", dest="use_value_baseline", action="store_false")
    parser.add_argument("--use-kl-penalty", action="store_true", default=config.USE_KL_PENALTY)
    parser.add_argument("--no-kl-penalty", dest="use_kl_penalty", action="store_false")
    parser.add_argument("--kl-coeff", type=float, default=config.KL_COEFF)
    parser.add_argument("--use-entropy-bonus", action="store_true", default=config.USE_ENTROPY_BONUS)
    parser.add_argument("--no-entropy-bonus", dest="use_entropy_bonus", action="store_false")
    parser.add_argument("--entropy-coeff", type=float, default=config.ENTROPY_COEFF)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cotpt")
    parser.add_argument("--log-file", default=None, help="defaults to <output-dir>/train_log.jsonl")

    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser


def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = pick_device()

    ref_model = None
    use_disable_adapter = False
    value_head = None
    start_step = 0
    optimizer_state_dict = None

    if args.resume_from:
        print(f"Resuming from {args.resume_from} ...")
        # hidden_size is discovered from the reloaded model itself below.
        result = load_checkpoint_for_resume(
            args.resume_from, base_model_id=args.model_id, device=device,
            use_value_head=args.use_value_baseline,
        )
        model, tokenizer = result["model"], result["tokenizer"]
        mixing_head = MixingHead(model.config.hidden_size).to(device)
        mixing_head.load_state_dict(result["mixing_head"].state_dict())
        if args.use_value_baseline and result["value_head"] is not None:
            value_head = ValueHead(model.config.hidden_size).to(device)
            value_head.load_state_dict(result["value_head"].state_dict())
        elif args.use_value_baseline:
            value_head = ValueHead(model.config.hidden_size).to(device)
        optimizer_state_dict = result["optimizer_state_dict"]
        start_step = result["step"]
    else:
        print(f"Loading {args.model_id} on {device} ...")
        model, tokenizer = load_model_and_tokenizer(args.model_id, device)
        if args.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                task_type="CAUSAL_LM", target_modules=config.LORA_TARGET_MODULES,
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=config.LORA_DROPOUT,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        mixing_head = MixingHead(model.config.hidden_size).to(device)
        if args.use_value_baseline:
            value_head = ValueHead(model.config.hidden_size).to(device)

    model.train()

    if args.use_kl_penalty:
        if args.use_lora:
            use_disable_adapter = True  # nearly free: reuses the same weights via disable_adapter()
            print("KL penalty: using the LoRA base weights (via disable_adapter()) as the reference "
                  "-- no extra model copy.")
        elif config.USE_FROZEN_REFERENCE_WHEN_NO_LORA:
            print("KL penalty requested without LoRA: building a frozen reference copy of the model "
                  "(roughly doubles model memory; pass --no-kl-penalty or use --use-lora to avoid this).")
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
        else:
            print("KL penalty requested but USE_FROZEN_REFERENCE_WHEN_NO_LORA is False -- disabling KL penalty.")
            args.use_kl_penalty = False

    param_groups = [
        {"params": model.parameters(), "lr": args.base_lr},
        {"params": mixing_head.parameters(), "lr": args.mix_head_lr},
    ]
    if value_head is not None:
        param_groups.append({"params": value_head.parameters(), "lr": args.value_head_lr})
    optimizer = torch.optim.AdamW(param_groups)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    trainable_params = list(model.parameters()) + list(mixing_head.parameters())
    if value_head is not None:
        trainable_params += list(value_head.parameters())

    texts = load_training_texts(
        data_path=args.data_path, hf_dataset=args.hf_dataset,
        hf_split=args.hf_split, hf_text_field=args.hf_text_field,
    )
    if texts is None:
        texts = TRAIN_TEXTS
        print(f"No --data-path/--hf-dataset given; using the {len(texts)} built-in toy examples.")
    else:
        print(f"Loaded {len(texts)} training examples.")

    log_path = args.log_file or os.path.join(args.output_dir, "train_log.jsonl")
    logger = ExperimentLogger(log_path, use_wandb=args.use_wandb, wandb_project=args.wandb_project,
                               wandb_config=vars(args))

    print(f"Training from step {start_step} to {args.num_steps} "
          f"(gradient accumulation: {args.gradient_accumulation_steps}) ...")
    print("=" * 78)

    optimizer.zero_grad()
    accum_counter = 0
    step = start_step
    while step < args.num_steps:
        text = texts[step % len(texts)]
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[1] < args.lookahead + 2:
            step += 1
            continue

        stats = cotpt_training_step(
            model, mixing_head, input_ids,
            value_head=value_head, ref_model=ref_model, use_disable_adapter=use_disable_adapter,
            num_think_positions=args.num_think_positions, num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
            use_value_baseline=args.use_value_baseline, use_kl_penalty=args.use_kl_penalty,
            kl_coeff=args.kl_coeff, use_entropy_bonus=args.use_entropy_bonus, entropy_coeff=args.entropy_coeff,
        )
        (stats["total_loss"] / args.gradient_accumulation_steps).backward()
        accum_counter += 1

        if accum_counter >= args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(trainable_params, config.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            accum_counter = 0

        if step % config.LOG_EVERY == 0 or step == args.num_steps - 1:
            print(
                f"step {step:4d}/{args.num_steps} | total={stats['total_loss'].item():6.3f} "
                f"aux_lm={stats['aux_lm_loss']:6.3f} value={stats['value_loss']:6.3f} "
                f"mix={stats['mix_loss']:6.3f} mean_reward={stats['mean_reward']:7.3f} "
                f"kl={stats['mean_kl']:.4f} entropy={stats['mean_entropy']:.3f} "
                f"mix_w={stats['mean_mix_weight']:.3f}"
            )
            logger.log(step, **{k: v for k, v in stats.items() if k != "total_loss"},
                       total_loss=stats["total_loss"].item())

        if args.checkpoint_every and step > 0 and step % args.checkpoint_every == 0:
            print(f"  [checkpoint at step {step}]")
            save_checkpoint(args.output_dir, model, tokenizer, mixing_head, value_head, optimizer, step=step)

        step += 1

    print("=" * 78)
    print(f"\nSaving final checkpoint to {args.output_dir} ...")
    save_checkpoint(args.output_dir, model, tokenizer, mixing_head, value_head, optimizer, step=args.num_steps)
    logger.close()
    print(f"Metrics logged to {log_path}")
    print(f"\nTo generate with these weights using the aggressive-eviction loop, run:\n"
          f"    python scripts/generate.py --model-id {args.output_dir}")
    if value_head is not None or args.use_lora:
        print("(mixing_head.pt / value_head.pt are saved alongside the model weights but aren't "
              "picked up by plain from_pretrained -- see README for how generate.py --use-mixing-head loads them.)")


if __name__ == "__main__":
    main()
