"""
Full Quiet-STaR-Style Training: Think / Talk / Learn
======================================================

  THINK  -- at a handful of sampled positions in the text, generate several
            independent candidate thoughts (rollouts), each one ordinary
            autoregressive sampling that can attend to the ones before it.
  TALK   -- the MixingHead learns how much to blend the post-thought
            prediction with the plain no-thought prediction, mixing HIDDEN
            STATES (not logits) then projecting through the model's own
            lm_head.
  LEARN  -- REINFORCE on the thought tokens. Each rollout's reward is how
            much its thought raised the log-likelihood of the next
            LOOKAHEAD real tokens, compared against the MEAN reward across
            that position's rollouts (Quiet-STaR's own baseline choice --
            thoughts are judged relative to their peers).

See README.md for what this deliberately simplifies vs. the paper (positions
sampled as a random subset rather than computed in parallel at every token
via the paper's custom attention-mask trick), and for the validation this
went through before being included here.
"""

import random

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .mixing_head import MixingHead
from .config import (
    NUM_HIDDEN_THOUGHT_TOKENS,
    NUM_ROLLOUTS,
    NUM_THINK_POSITIONS,
    LOOKAHEAD,
    THINK_TEMPERATURE,
    AUX_LM_LOSS_WEIGHT,
    REINFORCE_LOSS_WEIGHT,
    MIX_LOSS_WEIGHT,
)


def pick_think_positions(seq_len: int, num_positions: int, lookahead: int):
    """Random subset of positions with room for `lookahead` real tokens after them."""
    valid = list(range(0, seq_len - lookahead - 1))
    if not valid:
        return []
    k = min(num_positions, len(valid))
    return sorted(random.sample(valid, k))


def generate_rollout_batch(model, prefix_ids: torch.Tensor, num_rollouts: int,
                            thought_length: int, temperature: float):
    """
    Samples `num_rollouts` independent hidden thoughts from the same prefix,
    batched together (same prefix length -> no padding needed). Each thought
    is ordinary autoregressive KV-cache generation -- identical in spirit to
    Step B of quiet_star.inference, just with gradients left on and run
    num_rollouts-wide instead of 1-wide.

    Returns:
        cache                 -- grown KV cache, batch = num_rollouts
        logits_after_thought  -- [R, V] distribution right after the last thought token
        post_thought_hidden   -- [R, H] final hidden state there (post-final-norm)
        thought_logprob       -- [R] differentiable total log-prob of each sampled thought
    """
    batch_prefix = prefix_ids.expand(num_rollouts, -1).contiguous()
    cache = DynamicCache(config=model.config)
    out = model(input_ids=batch_prefix, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]

    thought_logprobs = []
    for t in range(thought_length):
        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, num_samples=1)                     # [R, 1]
        logp = torch.log(probs.gather(-1, token).squeeze(-1) + 1e-10)       # [R], differentiable
        thought_logprobs.append(logp)
        out = model(input_ids=token, past_key_values=cache, use_cache=True,
                     output_hidden_states=(t == thought_length - 1))
        cache = out.past_key_values
        logits = out.logits[:, -1, :]

    post_thought_hidden = out.hidden_states[-1][:, -1, :]           # [R, H]
    logits_after_thought = logits                                    # [R, V]
    thought_logprob = torch.stack(thought_logprobs, dim=1).sum(dim=1)  # [R]
    return cache, logits_after_thought, post_thought_hidden, thought_logprob


def score_future_tokens(model, cache, logits_after_thought: torch.Tensor,
                         real_future_ids: torch.Tensor) -> torch.Tensor:
    """log p(real_future_ids | prefix + thought), summed over the lookahead
    window, per rollout. This is the raw material for the REINFORCE reward."""
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
    return token_log_probs.sum(dim=1)  # [R]


def reinforce_loss_fn(rewards: torch.Tensor, thought_logprobs: torch.Tensor):
    """Reward relative to the mean reward across this position's rollouts
    (Quiet-STaR's own baseline choice) -- thoughts are judged against their
    peers, which is what keeps this low-variance enough to train with."""
    baseline = rewards.mean()
    advantages = (rewards - baseline).detach()
    loss = -(advantages * thought_logprobs).mean()
    return loss, advantages


def quiet_star_training_step(
    model,
    mixing_head: MixingHead,
    input_ids: torch.Tensor,   # [1, L]
    num_think_positions: int = NUM_THINK_POSITIONS,
    num_rollouts: int = NUM_ROLLOUTS,
    thought_length: int = NUM_HIDDEN_THOUGHT_TOKENS,
    lookahead: int = LOOKAHEAD,
    think_temperature: float = THINK_TEMPERATURE,
    aux_lm_loss_weight: float = AUX_LM_LOSS_WEIGHT,
    reinforce_loss_weight: float = REINFORCE_LOSS_WEIGHT,
    mix_loss_weight: float = MIX_LOSS_WEIGHT,
) -> dict:
    seq_len = input_ids.shape[1]

    # ---- shared base ("no-thought") pass over the whole sequence ----
    base_out = model(input_ids=input_ids, output_hidden_states=True, labels=input_ids)
    base_hidden = base_out.hidden_states[-1]     # [1, L, H], post-final-norm
    aux_lm_loss = base_out.loss

    positions = pick_think_positions(seq_len, num_think_positions, lookahead)
    reinforce_losses, mix_losses, mean_rewards, mix_weights = [], [], [], []

    for i in positions:
        # ---------------------------- THINK ----------------------------
        prefix_ids = input_ids[:, : i + 1]
        real_future_ids = input_ids[:, i + 1 : i + 1 + lookahead].expand(num_rollouts, -1)
        cache, logits_after_thought, post_thought_hidden, thought_logprobs = generate_rollout_batch(
            model, prefix_ids, num_rollouts, thought_length, think_temperature
        )

        # ---------------------------- LEARN ----------------------------
        rewards = score_future_tokens(model, cache, logits_after_thought, real_future_ids)
        r_loss, _ = reinforce_loss_fn(rewards, thought_logprobs)
        reinforce_losses.append(r_loss)
        mean_rewards.append(rewards.mean().detach())

        # ----------------------------- TALK -----------------------------
        # Rollout 0 stands in for "the thought actually used" at this position.
        w = mixing_head(base_hidden[:, i, :], post_thought_hidden[0:1, :])
        mixed_hidden = (1 - w) * base_hidden[:, i, :] + w * post_thought_hidden[0:1, :]
        mixed_logits = model.lm_head(mixed_hidden)
        next_real_token = input_ids[:, i + 1]
        mix_losses.append(F.cross_entropy(mixed_logits, next_real_token))
        mix_weights.append(w.detach().mean())

    if positions:
        reinforce_loss = torch.stack(reinforce_losses).mean()
        mix_loss = torch.stack(mix_losses).mean()
        mean_reward = torch.stack(mean_rewards).mean().item()
        mean_mix_weight = torch.stack(mix_weights).mean().item()
    else:
        reinforce_loss = torch.zeros((), device=input_ids.device)
        mix_loss = torch.zeros((), device=input_ids.device)
        mean_reward, mean_mix_weight = float("nan"), float("nan")

    total_loss = (
        aux_lm_loss_weight * aux_lm_loss
        + reinforce_loss_weight * reinforce_loss
        + mix_loss_weight * mix_loss
    )

    return {
        "total_loss": total_loss,
        "aux_lm_loss": aux_lm_loss.item(),
        "reinforce_loss": reinforce_loss.item() if positions else float("nan"),
        "mix_loss": mix_loss.item() if positions else float("nan"),
        "mean_reward": mean_reward,
        "mean_mix_weight": mean_mix_weight,
        "num_positions": len(positions),
    }
