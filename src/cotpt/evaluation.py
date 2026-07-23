"""
Evaluation: does hidden deliberation actually help?
=====================================================

Every other module in this project validates that a *mechanism* is wired
correctly. This one is different: it's meant to answer whether the
mechanism is worth having at all, by comparing the same metric -- mean
per-token log-likelihood the model assigns to a known-correct worked
solution -- under three conditions:

  no_think        -- plain teacher-forced log-likelihood, no thinking of
                      any kind.
  hidden_thinking -- at every answer-token position, run the full
                      checkpoint -> think -> evict -> commit cycle from
                      cotpt.inference, but instead of sampling the next
                      token, score the log-probability of the REAL
                      ground-truth token from the post-thought logits. Same
                      mechanism as generation, used for scoring instead of
                      sampling.
  visible_cot     -- no hidden mechanism at all; just append a "let's think
                      step by step" instruction to the visible prompt and
                      measure plain teacher-forced log-likelihood. This is
                      the baseline hidden deliberation actually needs to
                      beat to justify its extra complexity and compute --
                      if a free-form prompt does just as well, the eviction
                      machinery isn't buying you anything.

Higher (less negative) log-likelihood = the model found the correct answer
more probable = better.

WHAT THIS DOES NOT TELL YOU: whether an UNTRAINED base model, or this
project's tiny built-in toy corpus after a few dozen steps, shows a real
effect. Run this after real training on real data to get a result worth
trusting; run it before/after training on the same eval set to see whether
training moved anything at all.
"""

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .model_utils import sample_token
from .inference import forward_step
from .eval_data import EVAL_PROBLEMS, VISIBLE_COT_SUFFIX
from .config import NUM_HIDDEN_THOUGHT_TOKENS, THINKING_TEMPERATURE, THINKING_DO_SAMPLE


def _mean_log_likelihood_of_answer(model, tokenizer, question: str, answer: str) -> float:
    """Plain teacher-forced mean log p(answer token | everything before it)."""
    device = next(model.parameters()).device
    q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([q_ids, a_ids], dim=1)
    q_len = q_ids.shape[1]
    with torch.no_grad():
        out = model(input_ids=full_ids)
    logits_for_answer = out.logits[:, q_len - 1: -1, :]
    log_probs = F.log_softmax(logits_for_answer, dim=-1)
    token_log_probs = log_probs.gather(-1, a_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean().item()


def evaluate_no_think(model, tokenizer, question: str, answer: str) -> float:
    return _mean_log_likelihood_of_answer(model, tokenizer, question, answer)


def evaluate_visible_cot(model, tokenizer, question: str, answer: str,
                          cot_suffix: str = VISIBLE_COT_SUFFIX) -> float:
    return _mean_log_likelihood_of_answer(model, tokenizer, question + cot_suffix, answer)


def evaluate_hidden_deliberation(
    model, tokenizer, question: str, answer: str,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
) -> float:
    """Same eviction mechanism as generate_with_hidden_deliberation, but at
    each answer position the "chosen token" is the real ground-truth token
    (teacher forcing) instead of a sample -- this measures likelihood
    assignment under the deliberate-then-evict paradigm rather than
    generating freely."""
    device = next(model.parameters()).device
    q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(answer, return_tensors="pt").input_ids[0].to(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    log_probs_collected = []
    for target_id in a_ids:
        checkpoint_len = cache.get_seq_length()
        logits = last_logits
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits = forward_step(model, hidden_token, cache)

        log_prob = F.log_softmax(logits, dim=-1)[0, target_id.item()]
        log_probs_collected.append(log_prob.item())

        cache.crop(checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len
        cache, last_logits = forward_step(model, target_id.view(1), cache)

    return sum(log_probs_collected) / len(log_probs_collected)


def run_evaluation(model, tokenizer, problems=None, num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
                    thinking_temperature: float = THINKING_TEMPERATURE, thinking_do_sample: bool = THINKING_DO_SAMPLE):
    """Runs all three conditions over `problems` (defaults to the built-in
    held-out set) and returns per-problem results plus averages."""
    problems = problems if problems is not None else EVAL_PROBLEMS
    per_problem = []
    for p in problems:
        no_think = evaluate_no_think(model, tokenizer, p["question"], p["answer"])
        visible_cot = evaluate_visible_cot(model, tokenizer, p["question"], p["answer"])
        hidden = evaluate_hidden_deliberation(
            model, tokenizer, p["question"], p["answer"],
            num_hidden_tokens=num_hidden_tokens,
            thinking_temperature=thinking_temperature,
            thinking_do_sample=thinking_do_sample,
        )
        per_problem.append({
            "question": p["question"],
            "no_think": no_think,
            "visible_cot": visible_cot,
            "hidden_deliberation": hidden,
        })

    def _avg(key):
        return sum(r[key] for r in per_problem) / len(per_problem)

    summary = {
        "no_think": _avg("no_think"),
        "visible_cot": _avg("visible_cot"),
        "hidden_deliberation": _avg("hidden_deliberation"),
        "num_problems": len(per_problem),
    }
    return {"per_problem": per_problem, "summary": summary}
