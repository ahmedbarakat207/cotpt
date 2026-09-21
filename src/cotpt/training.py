import random

import torch
import torch.nn.functional as F
from transformers import DynamicCache

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
    MIXING_MODE,
    USE_DIFFERENTIAL_REWARD,
    USE_POSITIVE_ONLY_REINFORCE,
)
from .mixing_head import MixingHead, blend_logits
from .value_head import ValueHead


def pick_think_positions(
    seq_len: int,
    num_positions: int,
    lookahead: int,
    base_logits: torch.Tensor = None,
    strategy: str = "entropy",
    min_position: int = 0,
):
    valid = list(range(max(0, min_position), seq_len - lookahead - 1))
    if not valid:
        valid = list(range(0, seq_len - lookahead - 1))
    if not valid:
        return []
    k = min(num_positions, len(valid))
    if base_logits is None or strategy == "uniform" or k >= len(valid):
        return sorted(random.sample(valid, k))

    with torch.no_grad():
        indices = torch.tensor(valid, device=base_logits.device)
        logits = base_logits[0, indices, :]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        weights = entropy.clamp(min=1e-6)
        probs_pos = weights / weights.sum()
        chosen_sub = torch.multinomial(probs_pos, num_samples=k, replacement=False)
        chosen = [valid[idx] for idx in chosen_sub.tolist()]
        return sorted(chosen)


def _feed_delimiter(
    model, cache, hidden, logits, ref_cache, ref_logits, token_id: int, num_rollouts: int,
    ref_model=None, use_disable_adapter: bool = False, compute_kl: bool = False,
):
    """Feed a forced delimiter (start/end thought) with no REINFORCE logprob."""
    device = cache.layers[0].keys.device if len(cache.layers) > 0 else logits.device
    tok = torch.full((num_rollouts, 1), token_id, dtype=torch.long, device=device)
    out = model(input_ids=tok, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    hidden = out.hidden_states[-1][:, -1, :]
    if compute_kl:
        with torch.no_grad():
            if use_disable_adapter:
                with model.disable_adapter():
                    ref_out = model(input_ids=tok, past_key_values=ref_cache, use_cache=True)
            else:
                ref_out = ref_model(input_ids=tok, past_key_values=ref_cache, use_cache=True)
        ref_cache = ref_out.past_key_values
        ref_logits = ref_out.logits[:, -1, :]
    return cache, hidden, logits, ref_cache, ref_logits


def generate_rollout_batch(
    model,
    prefix_ids: torch.Tensor,
    num_rollouts: int,
    thought_length: int,
    temperature: float,
    value_head: ValueHead = None,
    ref_model=None,
    use_disable_adapter: bool = False,
    start_thought_id: int = None,
    end_thought_id: int = None,
):
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
    else:
        ref_cache, ref_logits = None, None

    use_brackets = (
        thought_length > 0 and start_thought_id is not None and end_thought_id is not None
    )
    if use_brackets:
        cache, hidden, logits, ref_cache, ref_logits = _feed_delimiter(
            model, cache, hidden, logits, ref_cache, ref_logits,
            start_thought_id, num_rollouts,
            ref_model=ref_model, use_disable_adapter=use_disable_adapter, compute_kl=compute_kl,
        )

    logprobs_per_step, values_per_step, entropies_per_step, kls_per_step = [], [], [], []

    for _ in range(thought_length):
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

    if use_brackets:
        cache, hidden, logits, ref_cache, ref_logits = _feed_delimiter(
            model, cache, hidden, logits, ref_cache, ref_logits,
            end_thought_id, num_rollouts,
            ref_model=ref_model, use_disable_adapter=use_disable_adapter, compute_kl=compute_kl,
        )

    if len(logprobs_per_step) > 0:
        thought_logprob_total = torch.stack(logprobs_per_step, dim=1).sum(dim=1)
        thought_logprobs_per_step = torch.stack(logprobs_per_step, dim=1)
        entropy_mean = torch.stack(entropies_per_step, dim=1).mean()
    else:
        # thought_length == 0: degenerate to no-thought (preserves zero-token test).
        device = prefix_ids.device
        thought_logprob_total = torch.zeros(num_rollouts, device=device)
        thought_logprobs_per_step = torch.zeros(num_rollouts, 1, device=device)
        entropy_mean = torch.zeros((), device=device)
    result = {
        "cache": cache,
        "logits_after_thought": logits,
        "post_thought_hidden": hidden,
        "thought_logprob_total": thought_logprob_total,
        "thought_logprobs_per_step": thought_logprobs_per_step,
        "entropy_mean": entropy_mean,
    }
    if value_head is not None:
        if len(values_per_step) > 0:
            result["values_per_step"] = torch.stack(values_per_step, dim=1)
        else:
            result["values_per_step"] = torch.zeros(num_rollouts, 1, device=prefix_ids.device)
    if compute_kl and len(kls_per_step) > 0:
        result["kl_total"] = torch.stack(kls_per_step, dim=1).sum(dim=1)
    elif compute_kl:
        result["kl_total"] = torch.zeros(num_rollouts, device=prefix_ids.device)
    return result


def score_future_tokens(
    model,
    cache,
    logits_after_thought: torch.Tensor,
    real_future_ids: torch.Tensor,
    normalize: bool = NORMALIZE_REWARD,
) -> torch.Tensor:
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


def compute_base_future_loglik(
    base_logits: torch.Tensor,
    input_ids: torch.Tensor,
    position: int,
    lookahead: int,
    normalize: bool = NORMALIZE_REWARD,
) -> float:
    """No-thought teacher-forced loglik of future tokens from the base forward.

    base_logits: [1, S, V] from the full-sequence forward. Token at input_ids[0, p]
    is predicted by base_logits[0, p-1]. Future window is input_ids[:, i+1 : i+1+lookahead].
    Returns a python float (same for all rollouts).
    """
    S = base_logits.shape[1]
    futures = []
    for k in range(lookahead):
        pos = position + 1 + k
        if pos >= input_ids.shape[1] or (position + k) >= S:
            break
        logit = base_logits[:, position + k, :]
        target = input_ids[:, pos]
        logp = F.log_softmax(logit, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        futures.append(logp)
    if not futures:
        return 0.0
    stacked = torch.stack(futures, dim=1)
    return stacked.mean().item() if normalize else stacked.sum().item()


def reinforce_loss_fn(
    rewards: torch.Tensor,
    thought_logprobs: torch.Tensor,
    use_positive_only: bool = USE_POSITIVE_ONLY_REINFORCE,
):
    baseline = rewards.mean()
    advantages = (rewards - baseline).detach()
    if use_positive_only:
        advantages = torch.clamp(advantages, min=0)
    loss = -(advantages * thought_logprobs).mean()
    return loss, advantages


def reinforce_loss_with_value_baseline(
    rewards: torch.Tensor,
    thought_logprobs_per_step: torch.Tensor,
    values_per_step: torch.Tensor = None,
    use_positive_only: bool = USE_POSITIVE_ONLY_REINFORCE,
):
    if values_per_step is not None:
        advantages = (rewards.unsqueeze(1) - values_per_step).detach()
        value_loss = ((values_per_step - rewards.unsqueeze(1).detach()) ** 2).mean()
    else:
        baseline = rewards.mean()
        advantages = (rewards - baseline).detach().unsqueeze(1).expand_as(thought_logprobs_per_step)
        value_loss = torch.zeros((), device=rewards.device)
    if use_positive_only:
        advantages = torch.clamp(advantages, min=0)
    policy_loss = -(advantages * thought_logprobs_per_step).mean()
    return policy_loss, value_loss, advantages


def cotpt_training_step(
    model,
    mixing_head: MixingHead,
    input_ids: torch.Tensor,
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
    position_strategy: str = "entropy",
    min_position: int = 0,
    mixing_mode: str = MIXING_MODE,
    use_differential_reward: bool = USE_DIFFERENTIAL_REWARD,
    use_positive_only: bool = USE_POSITIVE_ONLY_REINFORCE,
    start_thought_id: int = None,
    end_thought_id: int = None,
) -> dict:
    seq_len = input_ids.shape[1]

    base_out = model(input_ids=input_ids, output_hidden_states=True, labels=input_ids)
    base_hidden = base_out.hidden_states[-1]
    base_logits_full = base_out.logits
    aux_lm_loss = base_out.loss

    positions = pick_think_positions(
        seq_len,
        num_think_positions,
        lookahead,
        base_logits=base_out.logits,
        strategy=position_strategy,
        min_position=min_position,
    )
    reinforce_losses, value_losses, mix_losses = [], [], []
    mean_rewards, mix_weights, entropies, mean_kls = [], [], [], []

    want_ref = use_kl_penalty and (ref_model is not None or use_disable_adapter)

    for i in positions:
        prefix_ids = input_ids[:, : i + 1]
        real_future_ids = input_ids[:, i + 1 : i + 1 + lookahead].expand(num_rollouts, -1)
        rollout = generate_rollout_batch(
            model,
            prefix_ids,
            num_rollouts,
            thought_length,
            think_temperature,
            value_head=value_head if use_value_baseline else None,
            ref_model=ref_model if want_ref else None,
            use_disable_adapter=use_disable_adapter if want_ref else False,
            start_thought_id=start_thought_id,
            end_thought_id=end_thought_id,
        )

        raw_rewards = score_future_tokens(model, rollout["cache"], rollout["logits_after_thought"], real_future_ids)
        if use_differential_reward:
            base_ll = compute_base_future_loglik(base_logits_full, input_ids, i, lookahead)
            rewards = raw_rewards - base_ll
        else:
            rewards = raw_rewards
        if want_ref and "kl_total" in rollout:
            rewards = rewards - kl_coeff * rollout["kl_total"]
            mean_kls.append(rollout["kl_total"].mean().detach())

        r_loss, v_loss, _ = reinforce_loss_with_value_baseline(
            rewards, rollout["thought_logprobs_per_step"], rollout.get("values_per_step"),
            use_positive_only=use_positive_only,
        )
        reinforce_losses.append(r_loss)
        value_losses.append(v_loss)
        mean_rewards.append(raw_rewards.mean().detach())
        entropies.append(rollout["entropy_mean"].detach())

        post_thought_hidden = rollout["post_thought_hidden"][0:1, :]
        w = mixing_head(base_hidden[:, i, :], post_thought_hidden)
        if mixing_mode == "logit":
            base_logit_i = base_logits_full[:, i, :]
            thought_logit_0 = rollout["logits_after_thought"][0:1, :]
            mixed_logits = blend_logits(w, base_logit_i, thought_logit_0)
        else:
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
        - entropy_coeff * entropy_bonus_term
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
