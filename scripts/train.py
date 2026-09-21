#!/usr/bin/env python3
import argparse
import copy
import os
import random

import torch

from cotpt import config
from cotpt.checkpoint_utils import load_checkpoint_for_resume, save_checkpoint
from cotpt.data import load_training_texts, load_gsm8k, find_prompt_boundary
from cotpt.logging_utils import ExperimentLogger
from cotpt.mixing_head import MixingHead
from cotpt.model_utils import ensure_thought_tokens, get_thought_token_ids, load_model_and_tokenizer, pick_device
from cotpt.training import cotpt_training_step
from cotpt.value_head import ValueHead


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument(
        "--dataset-name",
        default="gsm8k",
        choices=["gsm8k", "gsm8k_direct", "math", "math_direct", "mmlu", "arc", "svamp", "fineweb_edu"],
    )
    parser.add_argument("--data-mode", default="step_by_step", choices=["step_by_step", "direct_answer"])
    parser.add_argument("--position-strategy", default="entropy", choices=["entropy", "uniform"])

    parser.add_argument("--num-steps", type=int, default=config.NUM_TRAINING_STEPS)
    parser.add_argument("--num-rollouts", type=int, default=config.NUM_ROLLOUTS)
    parser.add_argument("--num-think-positions", type=int, default=config.NUM_THINK_POSITIONS)
    parser.add_argument("--lookahead", type=int, default=config.LOOKAHEAD)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=0)

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

    parser.add_argument("--use-thought-tokens", action="store_true", default=config.USE_THOUGHT_TOKENS)
    parser.add_argument("--no-thought-tokens", dest="use_thought_tokens", action="store_false")
    parser.add_argument("--start-thought-token", default=config.START_THOUGHT_TOKEN)
    parser.add_argument("--end-thought-token", default=config.END_THOUGHT_TOKEN)
    parser.add_argument("--mixing-mode", default=config.MIXING_MODE, choices=["hidden", "logit"])
    parser.add_argument("--use-differential-reward", action="store_true", default=config.USE_DIFFERENTIAL_REWARD)
    parser.add_argument("--no-differential-reward", dest="use_differential_reward", action="store_false")
    parser.add_argument("--positive-only-reinforce", action="store_true", default=config.USE_POSITIVE_ONLY_REINFORCE)
    parser.add_argument("--no-positive-only-reinforce", dest="positive_only_reinforce", action="store_false")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cotpt")
    parser.add_argument("--log-file", default=None)

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
        result = load_checkpoint_for_resume(
            args.resume_from,
            base_model_id=args.model_id,
            device=device,
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
        model, tokenizer = load_model_and_tokenizer(
            args.model_id, device,
            use_thought_tokens=args.use_thought_tokens,
            start_thought_token=args.start_thought_token,
            end_thought_token=args.end_thought_token,
        )
        if args.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=config.LORA_TARGET_MODULES,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=config.LORA_DROPOUT,
            )
            model = get_peft_model(model, lora_cfg)
        mixing_head = MixingHead(model.config.hidden_size).to(device)
        if args.use_value_baseline:
            value_head = ValueHead(model.config.hidden_size).to(device)

    start_thought_id, end_thought_id = None, None
    if args.use_thought_tokens:
        # Resume path: tokenizer already saved with tokens; ensure model vocab matches.
        try:
            start_thought_id, end_thought_id = ensure_thought_tokens(
                tokenizer, model, args.start_thought_token, args.end_thought_token
            )
        except Exception:
            start_thought_id, end_thought_id = get_thought_token_ids(
                tokenizer, args.start_thought_token, args.end_thought_token
            )
        if start_thought_id is None or end_thought_id is None:
            print("Warning: --use-thought-tokens set but thought tokens unavailable; continuing without brackets.")

    model.train()

    if args.use_kl_penalty:
        if args.use_lora:
            use_disable_adapter = True
        elif config.USE_FROZEN_REFERENCE_WHEN_NO_LORA:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
        else:
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
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_text_field=args.hf_text_field,
        dataset_name=args.dataset_name,
        mode=args.data_mode,
    )
    if texts is None:
        texts = load_gsm8k()

    log_path = args.log_file or os.path.join(args.output_dir, "train_log.jsonl")
    logger = ExperimentLogger(
        log_path,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_config=vars(args),
    )

    optimizer.zero_grad()
    accum_counter = 0
    step = start_step
    while step < args.num_steps:
        text = texts[step % len(texts)]
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if input_ids.shape[1] < args.lookahead + 2:
            step += 1
            continue

        min_pos = find_prompt_boundary(text, tokenizer) or 0
        stats = cotpt_training_step(
            model,
            mixing_head,
            input_ids,
            value_head=value_head,
            ref_model=ref_model,
            use_disable_adapter=use_disable_adapter,
            num_think_positions=args.num_think_positions,
            num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
            use_value_baseline=args.use_value_baseline,
            use_kl_penalty=args.use_kl_penalty,
            kl_coeff=args.kl_coeff,
            use_entropy_bonus=args.use_entropy_bonus,
            entropy_coeff=args.entropy_coeff,
            position_strategy=args.position_strategy,
            min_position=min_pos,
            mixing_mode=args.mixing_mode,
            use_differential_reward=args.use_differential_reward,
            use_positive_only=args.positive_only_reinforce,
            start_thought_id=start_thought_id,
            end_thought_id=end_thought_id,
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
            logger.log(
                step,
                **{k: v for k, v in stats.items() if k != "total_loss"},
                total_loss=stats["total_loss"].item(),
            )

        if args.checkpoint_every and step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(args.output_dir, model, tokenizer, mixing_head, value_head, optimizer, step=step)

        step += 1

    save_checkpoint(args.output_dir, model, tokenizer, mixing_head, value_head, optimizer, step=args.num_steps)
    logger.close()


if __name__ == "__main__":
    main()
