"""
Full COTPT-Style Training: Think / Talk / Learn
======================================================

  THINK  -- at a handful of sampled positions in the text, generate several
            independent candidate thoughts (rollouts), each one ordinary
            autoregressive sampling that can attend to the ones before it.
  TALK   -- the MixingHead learns how much to blend the post-thought
            prediction with the plain no-thought prediction, mixing HIDDEN
            STATES (not logits) then projecting through the model's own
            lm_head.
  LEARN  -- REINFORCE on the thought tokens, with three stability additions
            on top of the bare-bones version:
              * a learned ValueHead gives each thought TOKEN its own
                baseline (state-dependent), instead of every token in a
                rollout sharing one scalar mean-of-rollouts baseline.
              * an optional KL penalty against a reference distribution
                (either a separate frozen model, or -- far cheaper -- a
                LoRA-wrapped model's own base weights via
                model.disable_adapter()) discourages the thought policy
                from drifting into reward-hacking degenerate token
                sequences.
              * an optional entropy bonus keeps some exploration alive so
                REINFORCE doesn't collapse onto one low-diversity thought
                early.
            Reward is length-normalized (mean, not sum, log-likelihood of
            the lookahead tokens) so longer lookahead windows don't just
            produce more negative rewards for unrelated reasons.

See README.md for what this deliberately simplifies vs. the paper, and for
the validation this went through before being included here.
"""

import random

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .mixing_head import MixingHead
from .value_head import ValueHead
from .config import (
    NUM_HIDDEN_THOUGHT_TOKENS,
    NUM_ROLLOUTS,
    NUM_THINK_POSITIONS,
    LOOKAHEAD,
    THINK_TEMPERATURE,
    AUX_LM_LOSS_WEIGHT,
    REINFORCE_LOSS_WEIGHT,
    MIX_LOSS_WEIGHT,
    VALUE_LOSS_WEIGHT,
    USE_VALUE_BASELINE,
    USE_KL_PENALTY,
    KL_COEFF,
    USE_ENTROPY_BONUS,
    ENTROPY_COEFF,
    NORMALIZE_REWARD,
)


def pick_think_positions(seq_len: int, num_positions: int, lookahead: int):
    """Random subset of positions with room for `lookahead` real tokens after them."""
    valid = list(range(0, seq_len - lookahead - 1))
    if not valid:
        return []
    k = min(num_positions, len(valid))
    return sorted(random.sample(valid, k))


def generate_rollout_batch(
    model,
    prefix_ids: torch.Tensor,
    num_rollouts: int,
    thought_length: int,
    temperature: float,
    value_head: ValueHead = None,
    ref_model=None,
    use_disable_adapter: bool = False,
):
    """
    Samples `num_rollouts` independent hidden thoughts from the same prefix,
    batched together (same prefix length -> no padding needed). Tracks, per
    thought-generation step: the hidden state BEFORE that token (for the
    value baseline), the policy's entropy, and -- if a reference is
    available -- KL(policy || reference).

    `ref_model`: a separate frozen model to compare against.
    `use_disable_adapter`: if True, get the reference by calling
        `model.disable_adapter()` instead (requires `model` to be a
        LoRA-wrapped peft model) -- much cheaper than a second full copy,
        since it reuses the same weights.

    Returns a dict: cache, logits_after_thought [R,V], post_thought_hidden
    [R,H], thought_logprob_total [R], thought_logprobs_per_step [R,T],
    entropy_mean (scalar), and optionally values_per_step [R,T] / kl_total [R].
    """
    batch_prefix = prefix_ids.expand(num_rollouts, -1).contiguous()
    cache = DynamicCache(config=model.config)
    out = model(input_ids=batch_prefix, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    hidden = out.hidden_states[-1][:, -1, :]

    compute_kl = ref_model is not None or use_disable_adapter
    if compute_kl:
        with torch.no_grad():
            if use_disable_adapter:
                with model.disable_adapter():
                    ref_cache = DynamicCache(config=model.config)
                    ref_out = model(input_ids=batch_prefix, past_key_values=ref_cache, use_cache=True)
            else:
                ref_cache = DynamicCache(config=ref_model.config)
                ref_out = ref_model(input_ids=batch_prefix, past_key_values=ref_cache, use_cache=True)
        ref_cache = ref_out.past_key_values
        ref_logits = ref_out.logits[:, -1, :]

    logprobs_per_step, values_per_step, entropies_per_step, kls_per_step = [], [], [], []

    for t in range(thought_length):
        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        logp = torch.log(probs.gather(-1, token).squeeze(-1) + 1e-10)
        logprobs_per_step.append(logp)

        entropies_per_step.append(-(probs * torch.log(probs + 1e-10)).sum(dim=-1))

        if value_head is not None:
            values_per_step.append(value_head(hidden))

        if compute_kl:
            with torch.no_grad():
                ref_probs = F.softmax(ref_logits, dim=-1)
            kl = (probs * (torch.log(probs + 1e-10) - torch.log(ref_probs + 1e-10))).sum(dim=-1)
            kls_per_step.append(kl)

        out = model(input_ids=token, past_key_values=cache, use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        hidden = out.hidden_states[-1][:, -1, :]

        if compute_kl:
            with torch.no_grad():
                if use_disable_adapter:
                    with model.disable_adapter():
                        ref_out = model(input_ids=token, past_key_values=ref_cache, use_cache=True)
                else:
                    ref_out = ref_model(input_ids=token, past_key_values=ref_cache, use_cache=True)
            ref_cache = ref_out.past_key_values
            ref_logits = ref_out.logits[:, -1, :]

    result = {
        "cache": cache,
        "logits_after_thought": logits,
        "post_thought_hidden": hidden,
        "thought_logprob_total": torch.stack(logprobs_per_step, dim=1).sum(dim=1),
        "thought_logprobs_per_step": torch.stack(logprobs_per_step, dim=1),  # [R,T]
        "entropy_mean": torch.stack(entropies_per_step, dim=1).mean(),
    }
    if value_head is not None:
        result["values_per_step"] = torch.stack(values_per_step, dim=1)  # [R,T]
    if compute_kl:
        result["kl_total"] = torch.stack(kls_per_step, dim=1).sum(dim=1)  # [R]
    return result


def score_future_tokens(model, cache, logits_after_thought: torch.Tensor,
                         real_future_ids: torch.Tensor, normalize: bool = NORMALIZE_REWARD) -> torch.Tensor:
    """log p(real_future_ids | prefix + thought). Summed by default is the raw
    log-likelihood; normalize=True instead averages over the lookahead window
    so the reward scale doesn't depend on LOOKAHEAD."""
    R, m = real_future_ids.shape
    if m == 0:
        return torch.zeros(R, device=real_future_ids.device)
    if m > 1:
        out = model(input_ids=real_future_ids, past_key_values=cache, use_cache=True)
        all_logits = torch.cat([logits_after_thought.unsqueeze(1), out.logits[:, :-1, :]], dim=1)
    else:
        all_logits = logits_after_thought.unsqueeze(1)
    log_probs = F.log_softmax(all_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, real_future_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean(dim=1) if normalize else token_log_probs.sum(dim=1)


def reinforce_loss_fn(rewards: torch.Tensor, thought_logprobs: torch.Tensor):
    """Original scalar-baseline REINFORCE (mean reward across rollouts).
    Kept as-is for backward compatibility / as the fallback when
    USE_VALUE_BASELINE is off. rewards, thought_logprobs: [R]."""
    baseline = rewards.mean()
    advantages = (rewards - baseline).detach()
    loss = -(advantages * thought_logprobs).mean()
    return loss, advantages


def reinforce_loss_with_value_baseline(rewards: torch.Tensor, thought_logprobs_per_step: torch.Tensor,
                                        values_per_step: torch.Tensor = None):
    """Per-token-credit REINFORCE: rewards [R], thought_logprobs_per_step [R,T],
    values_per_step [R,T] or None (falls back to the scalar mean baseline,
    broadcast across T, i.e. identical math to reinforce_loss_fn)."""
    if values_per_step is not None:
        advantages = (rewards.unsqueeze(1) - values_per_step).detach()
        value_loss = ((values_per_step - rewards.unsqueeze(1).detach()) ** 2).mean()
    else:
        baseline = rewards.mean()
        advantages = (rewards - baseline).detach().unsqueeze(1).expand_as(thought_logprobs_per_step)
        value_loss = torch.zeros((), device=rewards.device)
    policy_loss = -(advantages * thought_logprobs_per_step).mean()
    return policy_loss, value_loss, advantages


def cotpt_training_step(
    model,
    mixing_head: MixingHead,
    input_ids: torch.Tensor,   # [1, L]
    value_head: ValueHead = None,
    ref_model=None,
    use_disable_adapter: bool = False,
    num_think_positions: int = NUM_THINK_POSITIONS,
    num_rollouts: int = NUM_ROLLOUTS,
    thought_length: int = NUM_HIDDEN_THOUGHT_TOKENS,
    lookahead: int = LOOKAHEAD,
    think_temperature: float = THINK_TEMPERATURE,
    aux_lm_loss_weight: float = AUX_LM_LOSS_WEIGHT,
    reinforce_loss_weight: float = REINFORCE_LOSS_WEIGHT,
    mix_loss_weight: float = MIX_LOSS_WEIGHT,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    use_value_baseline: bool = USE_VALUE_BASELINE,
    use_kl_penalty: bool = USE_KL_PENALTY,
    kl_coeff: float = KL_COEFF,
    use_entropy_bonus: bool = USE_ENTROPY_BONUS,
    entropy_coeff: float = ENTROPY_COEFF,
) -> dict:
    seq_len = input_ids.shape[1]

    # ---- shared base ("no-thought") pass over the whole sequence ----
    base_out = model(input_ids=input_ids, output_hidden_states=True, labels=input_ids)
    base_hidden = base_out.hidden_states[-1]     # [1, L, H], post-final-norm
    aux_lm_loss = base_out.loss

    positions = pick_think_positions(seq_len, num_think_positions, lookahead)
    reinforce_losses, value_losses, mix_losses = [], [], []
    mean_rewards, mix_weights, entropies, mean_kls = [], [], [], []

    want_ref = use_kl_penalty and (ref_model is not None or use_disable_adapter)

    for i in positions:
        # ---------------------------- THINK ----------------------------
        prefix_ids = input_ids[:, : i + 1]
        real_future_ids = input_ids[:, i + 1 : i + 1 + lookahead].expand(num_rollouts, -1)
        rollout = generate_rollout_batch(
            model, prefix_ids, num_rollouts, thought_length, think_temperature,
            value_head=value_head if use_value_baseline else None,
            ref_model=ref_model if want_ref else None,
            use_disable_adapter=use_disable_adapter if want_ref else False,
        )

        # ---------------------------- LEARN ----------------------------
        raw_rewards = score_future_tokens(model, rollout["cache"], rollout["logits_after_thought"], real_future_ids)
        rewards = raw_rewards
        if want_ref and "kl_total" in rollout:
            rewards = raw_rewards - kl_coeff * rollout["kl_total"]
            mean_kls.append(rollout["kl_total"].mean().detach())

        r_loss, v_loss, _ = reinforce_loss_with_value_baseline(
            rewards, rollout["thought_logprobs_per_step"], rollout.get("values_per_step")
        )
        reinforce_losses.append(r_loss)
        value_losses.append(v_loss)
        mean_rewards.append(raw_rewards.mean().detach())
        entropies.append(rollout["entropy_mean"].detach())

        # ----------------------------- TALK -----------------------------
        # Rollout 0 stands in for "the thought actually used" at this position.
        post_thought_hidden = rollout["post_thought_hidden"][0:1, :]
        w = mixing_head(base_hidden[:, i, :], post_thought_hidden)
        mixed_hidden = (1 - w) * base_hidden[:, i, :] + w * post_thought_hidden
        mixed_logits = model.lm_head(mixed_hidden)
        next_real_token = input_ids[:, i + 1]
        mix_losses.append(F.cross_entropy(mixed_logits, next_real_token))
        mix_weights.append(w.detach().mean())

    if positions:
        reinforce_loss = torch.stack(reinforce_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        mix_loss = torch.stack(mix_losses).mean()
        mean_reward = torch.stack(mean_rewards).mean().item()
        mean_mix_weight = torch.stack(mix_weights).mean().item()
        mean_entropy = torch.stack(entropies).mean().item()
        mean_kl = torch.stack(mean_kls).mean().item() if mean_kls else float("nan")
        entropy_bonus_term = torch.stack(entropies).mean() if use_entropy_bonus else torch.zeros((), device=input_ids.device)
    else:
        reinforce_loss = torch.zeros((), device=input_ids.device)
        value_loss = torch.zeros((), device=input_ids.device)
        mix_loss = torch.zeros((), device=input_ids.device)
        mean_reward = mean_mix_weight = mean_entropy = mean_kl = float("nan")
        entropy_bonus_term = torch.zeros((), device=input_ids.device)

    total_loss = (
        aux_lm_loss_weight * aux_lm_loss
        + reinforce_loss_weight * reinforce_loss
        + mix_loss_weight * mix_loss
        + value_loss_weight * value_loss
        - entropy_coeff * entropy_bonus_term  # subtract: higher entropy -> lower loss
    )

    return {
        "total_loss": total_loss,
        "aux_lm_loss": aux_lm_loss.item(),
        "reinforce_loss": reinforce_loss.item() if positions else float("nan"),
        "value_loss": value_loss.item() if positions else float("nan"),
        "mix_loss": mix_loss.item() if positions else float("nan"),
        "mean_reward": mean_reward,
        "mean_mix_weight": mean_mix_weight,
        "mean_entropy": mean_entropy,
        "mean_kl": mean_kl,
        "num_positions": len(positions),
    }
