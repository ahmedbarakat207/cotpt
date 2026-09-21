from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .config import NUM_HIDDEN_THOUGHT_TOKENS, THINKING_TEMPERATURE, THINKING_DO_SAMPLE
from .eval_data import EVAL_PROBLEMS, VISIBLE_COT_SUFFIX
from .inference import forward_step, forward_step_with_hidden
from .mixing_head import MixingHead
from .model_utils import sample_token


def _mean_log_likelihood_of_answer(model, tokenizer, question: str, answer: str) -> float:
    device = next(model.parameters()).device
    q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([q_ids, a_ids], dim=1)
    q_len = q_ids.shape[1]
    with torch.no_grad():
        out = model(input_ids=full_ids)
    logits_for_answer = out.logits[:, q_len - 1 : -1, :]
    log_probs = F.log_softmax(logits_for_answer, dim=-1)
    token_log_probs = log_probs.gather(-1, a_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean().item()


def evaluate_no_think(model, tokenizer, question: str, answer: str) -> float:
    return _mean_log_likelihood_of_answer(model, tokenizer, question, answer)


def evaluate_visible_cot(model, tokenizer, question: str, answer: str, cot_suffix: str = VISIBLE_COT_SUFFIX) -> float:
    return _mean_log_likelihood_of_answer(model, tokenizer, question + cot_suffix, answer)


def evaluate_hidden_deliberation(
    model,
    tokenizer,
    question: str,
    answer: str,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
    mixing_head: Optional[MixingHead] = None,
) -> float:
    device = next(model.parameters()).device
    q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(answer, return_tensors="pt").input_ids[0].to(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=q_ids, past_key_values=cache, use_cache=True, output_hidden_states=mixing_head is not None)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]
    last_hidden = out.hidden_states[-1][:, -1, :] if mixing_head is not None else None

    log_probs_collected = []
    for target_id in a_ids:
        checkpoint_len = cache.get_seq_length()
        logits = last_logits
        post_thought_hidden = last_hidden
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            if mixing_head is not None:
                cache, logits, post_thought_hidden = forward_step_with_hidden(model, hidden_token, cache)
            else:
                cache, logits = forward_step(model, hidden_token, cache)

        if mixing_head is not None and num_hidden_tokens > 0:
            w = mixing_head(last_hidden, post_thought_hidden)
            mixed_hidden = (1 - w) * last_hidden + w * post_thought_hidden
            logits = model.lm_head(mixed_hidden)

        log_prob = F.log_softmax(logits, dim=-1)[0, target_id.item()]
        log_probs_collected.append(log_prob.item())

        cache.crop(checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len

        if mixing_head is not None:
            cache, last_logits, last_hidden = forward_step_with_hidden(model, target_id.view(1), cache)
        else:
            cache, last_logits = forward_step(model, target_id.view(1), cache)

    return sum(log_probs_collected) / len(log_probs_collected)


def run_evaluation(
    model,
    tokenizer,
    problems: Optional[List[Dict[str, Any]]] = None,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
    mixing_head: Optional[MixingHead] = None,
):
    problems = problems if problems is not None else EVAL_PROBLEMS
    per_problem = []
    for p in problems:
        no_think = evaluate_no_think(model, tokenizer, p["question"], p["answer"])
        visible_cot = evaluate_visible_cot(model, tokenizer, p["question"], p["answer"])
        hidden = evaluate_hidden_deliberation(
            model,
            tokenizer,
            p["question"],
            p["answer"],
            num_hidden_tokens=num_hidden_tokens,
            thinking_temperature=thinking_temperature,
            thinking_do_sample=thinking_do_sample,
            mixing_head=mixing_head,
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
