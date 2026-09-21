import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import DynamicCache

from .config import (
    MAX_VISIBLE_TOKENS,
    NUM_HIDDEN_THOUGHT_TOKENS,
    REAL_TOKEN_DO_SAMPLE,
    REAL_TOKEN_TEMPERATURE,
    THINKING_DO_SAMPLE,
    THINKING_TEMPERATURE,
)
from .eval_data import BENCHMARK_PROBLEMS, VISIBLE_COT_SUFFIX
from .evaluation import (
    evaluate_hidden_deliberation,
    evaluate_no_think,
    evaluate_visible_cot,
)
from .inference import evict_hidden_tokens, forward_step, forward_step_with_hidden
from .mixing_head import MixingHead
from .model_utils import is_eos, sample_token


def extract_answer(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    hash_match = re.search(r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[A-E]\b)", cleaned)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", cleaned)
    if boxed_match:
        val = boxed_match.group(1).strip()
        num = re.search(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[A-E]\b)", val)
        return num.group(1).replace(",", "") if num else val

    mc_match = re.search(r"(?:answer\s*(?:is|=|:)?\s*\(?|\A\s*\(?)([A-E])(?:\)|\.|\s|\Z)", cleaned, re.IGNORECASE)
    if mc_match:
        return mc_match.group(1).upper()

    phrase_match = re.findall(
        r"(?:(?:the\s+)?(?:final\s+)?(?:answer|result|total)\s*(?:is|=|:)\s*|(?:which\s+is|equals|=|\bis)\s+)([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[A-E]\b)",
        cleaned,
        re.IGNORECASE,
    )
    if phrase_match:
        return phrase_match[-1].replace(",", "")

    all_numbers = re.findall(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", cleaned)
    if all_numbers:
        return all_numbers[-1].replace(",", "")

    return None


def check_answer(prediction: Optional[str], target: str) -> bool:
    if prediction is None or target is None:
        return False

    pred_clean = str(prediction).strip().lower().replace(",", "")
    tgt_clean = str(target).strip().lower().replace(",", "")

    if pred_clean == tgt_clean:
        return True

    try:
        return abs(float(pred_clean) - float(tgt_clean)) < 1e-3
    except ValueError:
        return False


def generate_normal(
    model,
    tokenizer,
    prompt: str,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    temperature: float = REAL_TOKEN_TEMPERATURE,
    do_sample: bool = REAL_TOKEN_DO_SAMPLE,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.perf_counter()
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    generated_text = ""
    num_visible = 0
    peak_kv_len = cache.get_seq_length()
    forward_steps = prompt_ids.shape[1]

    for _ in range(max_visible_tokens):
        token = sample_token(last_logits, temperature, do_sample)
        token_id = token.item()
        if is_eos(token_id, tokenizer, model):
            break

        cache, last_logits = forward_step(model, token, cache)
        forward_steps += 1
        num_visible += 1
        peak_kv_len = max(peak_kv_len, cache.get_seq_length())

        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        generated_text += token_text

    elapsed = max(time.perf_counter() - start_time, 1e-6)

    return {
        "text": generated_text,
        "visible_tokens": num_visible,
        "peak_kv_len": peak_kv_len,
        "final_kv_len": cache.get_seq_length(),
        "latency_sec": elapsed,
        "tokens_per_sec": num_visible / elapsed,
        "forward_steps": forward_steps,
    }


def generate_normal_cot(
    model,
    tokenizer,
    prompt: str,
    cot_suffix: str = VISIBLE_COT_SUFFIX,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    temperature: float = REAL_TOKEN_TEMPERATURE,
    do_sample: bool = REAL_TOKEN_DO_SAMPLE,
) -> Dict[str, Any]:
    full_prompt = prompt + cot_suffix
    result = generate_normal(
        model,
        tokenizer,
        full_prompt,
        max_visible_tokens=max_visible_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    result["prompt_used"] = full_prompt
    return result


def generate_cotpt_deliberation(
    model,
    tokenizer,
    prompt: str,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
    real_temperature: float = REAL_TOKEN_TEMPERATURE,
    real_do_sample: bool = REAL_TOKEN_DO_SAMPLE,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.perf_counter()
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    generated_text = ""
    num_visible = 0
    peak_kv_len = cache.get_seq_length()
    forward_steps = prompt_ids.shape[1]

    for _ in range(max_visible_tokens):
        checkpoint_len = cache.get_seq_length()

        logits = last_logits
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits = forward_step(model, hidden_token, cache)
            forward_steps += 1
            peak_kv_len = max(peak_kv_len, cache.get_seq_length())

        real_token = sample_token(logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()
        if is_eos(real_token_id, tokenizer, model):
            evict_hidden_tokens(cache, checkpoint_len)
            break

        evict_hidden_tokens(cache, checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len

        cache, last_logits = forward_step(model, real_token, cache)
        forward_steps += 1
        num_visible += 1
        peak_kv_len = max(peak_kv_len, cache.get_seq_length())

        token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
        generated_text += token_text

    elapsed = max(time.perf_counter() - start_time, 1e-6)

    return {
        "text": generated_text,
        "visible_tokens": num_visible,
        "peak_kv_len": peak_kv_len,
        "final_kv_len": cache.get_seq_length(),
        "latency_sec": elapsed,
        "tokens_per_sec": num_visible / elapsed,
        "forward_steps": forward_steps,
    }


def generate_cotpt_adaptive(
    model,
    tokenizer,
    mixing_head: MixingHead,
    prompt: str,
    entropy_threshold: Optional[float] = None,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
    real_temperature: float = REAL_TOKEN_TEMPERATURE,
    real_do_sample: bool = REAL_TOKEN_DO_SAMPLE,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    start_time = time.perf_counter()
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]
    last_hidden = out.hidden_states[-1][:, -1, :]

    generated_text = ""
    num_visible = 0
    num_thought_tokens = 0
    peak_kv_len = cache.get_seq_length()
    forward_steps = prompt_ids.shape[1]

    for _ in range(max_visible_tokens):
        checkpoint_len = cache.get_seq_length()
        num_visible += 1

        base_probs = torch.softmax(last_logits, dim=-1)
        entropy = -(base_probs * torch.log(base_probs + 1e-10)).sum(dim=-1).item()
        should_think = entropy_threshold is None or entropy > entropy_threshold

        if not should_think:
            real_token = sample_token(last_logits, real_temperature, real_do_sample)
            real_token_id = real_token.item()
            if is_eos(real_token_id, tokenizer, model):
                num_visible -= 1
                break
            cache, last_logits, last_hidden = forward_step_with_hidden(model, real_token, cache)
            forward_steps += 1
            peak_kv_len = max(peak_kv_len, cache.get_seq_length())
            token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
            generated_text += token_text
            continue

        num_thought_tokens += 1
        logits = last_logits
        post_thought_hidden = last_hidden
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits, post_thought_hidden = forward_step_with_hidden(model, hidden_token, cache)
            forward_steps += 1
            peak_kv_len = max(peak_kv_len, cache.get_seq_length())

        w = mixing_head(last_hidden, post_thought_hidden)
        mixed_hidden = (1 - w) * last_hidden + w * post_thought_hidden
        mixed_logits = model.lm_head(mixed_hidden)

        real_token = sample_token(mixed_logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()
        if is_eos(real_token_id, tokenizer, model):
            evict_hidden_tokens(cache, checkpoint_len)
            num_visible -= 1
            break

        evict_hidden_tokens(cache, checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len

        cache, last_logits, last_hidden = forward_step_with_hidden(model, real_token, cache)
        forward_steps += 1
        peak_kv_len = max(peak_kv_len, cache.get_seq_length())

        token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
        generated_text += token_text

    elapsed = max(time.perf_counter() - start_time, 1e-6)

    return {
        "text": generated_text,
        "visible_tokens": num_visible,
        "deliberated_positions": num_thought_tokens,
        "peak_kv_len": peak_kv_len,
        "final_kv_len": cache.get_seq_length(),
        "latency_sec": elapsed,
        "tokens_per_sec": num_visible / elapsed,
        "forward_steps": forward_steps,
    }


def load_benchmark_problems(path: Optional[str] = None) -> List[Dict[str, Any]]:
    if not path:
        return BENCHMARK_PROBLEMS

    clean = path.strip().lower()
    if clean in ("gsm8k", "openai/gsm8k"):
        local_path = "data/gsm8k_test.jsonl"
        if os.path.exists(local_path):
            normalized = []
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    q = f"Q: {row['question']}\nA:"
                    a = row["answer"]
                    target = extract_answer(a)
                    normalized.append({"question": q, "answer": a, "target_answer": target or ""})
            return normalized
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset("openai/gsm8k", "main", split="test")
        normalized = []
        for row in ds:
            q = f"Q: {row['question']}\nA:"
            a = row["answer"]
            target = extract_answer(a)
            normalized.append({"question": q, "answer": a, "target_answer": target or ""})
        return normalized

    if clean in ("mmlu", "cais/mmlu"):
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset("cais/mmlu", "college_mathematics", split="test")
        normalized = []
        for row in ds:
            choices = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(row["choices"])])
            q = f"Question: {row['question']}\n{choices}\nAnswer:"
            target = chr(65 + row["answer"])
            normalized.append({"question": q, "answer": target, "target_answer": target})
        return normalized

    if clean in ("math", "hendrycks_math", "eleutherai/hendrycks_math", "lighteval/math"):
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
        normalized = []
        for row in ds:
            q = f"Question: {row['problem']}\nAnswer:"
            a = row["solution"]
            target = extract_answer(a)
            normalized.append({"question": q, "answer": a, "target_answer": target or ""})
        return normalized

    if clean in ("arc", "arc_challenge", "ai2_arc"):
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        normalized = []
        for row in ds:
            choices_text = row["choices"]["text"]
            choices_label = row["choices"]["label"]
            choices = "\n".join([f"({lbl}) {txt}" for lbl, txt in zip(choices_label, choices_text)])
            q = f"Question: {row['question']}\n{choices}\nAnswer:"
            target = row["answerKey"].strip()
            normalized.append({"question": q, "answer": target, "target_answer": target})
        return normalized

    if clean in ("svamp", "chilled/svamp"):
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset("ChilleD/SVAMP", split="test")
        normalized = []
        for row in ds:
            body = row.get("Body", "").strip()
            question = row.get("Question", "").strip()
            q = f"Q: {body} {question}\nA:"
            target = str(row["Answer"]).strip()
            normalized.append({"question": q, "answer": target, "target_answer": target})
        return normalized

    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    problems = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                problems = data
            elif isinstance(data, dict) and "problems" in data:
                problems = data["problems"]
            else:
                raise ValueError("JSON file must be a list of problem objects or contain a 'problems' list.")

    normalized = []
    for item in problems:
        q = item.get("question") or item.get("prompt") or item.get("input")
        a = item.get("answer") or item.get("solution") or ""
        target = (
            item.get("target_answer")
            or item.get("target")
            or item.get("final_answer")
            or extract_answer(a)
        )
        if not q:
            continue
        normalized.append({
            "question": q,
            "answer": a,
            "target_answer": str(target) if target is not None else "",
        })

    if not normalized:
        raise ValueError(f"No valid problems found in {path}")

    return normalized


DEFAULT_CONDITIONS = ["normal_direct", "normal_cot", "cotpt_hidden"]


def run_benchmark(
    model,
    tokenizer,
    problems: Optional[List[Dict[str, Any]]] = None,
    conditions: Optional[List[str]] = None,
    mixing_head: Optional[MixingHead] = None,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    entropy_threshold: Optional[float] = None,
    compute_likelihood: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    problems = problems if problems is not None else BENCHMARK_PROBLEMS
    conditions = conditions if conditions is not None else DEFAULT_CONDITIONS

    if "cotpt_adaptive" in conditions and mixing_head is None:
        raise ValueError("Condition 'cotpt_adaptive' requested but mixing_head is None.")

    per_problem_results = []
    condition_aggregates = {
        cond: {
            "correct": [],
            "latency": [],
            "tokens_per_sec": [],
            "visible_tokens": [],
            "peak_kv": [],
            "final_kv": [],
            "forward_steps": [],
            "log_likelihood": [],
        }
        for cond in conditions
    }

    for idx, prob in enumerate(problems):
        q = prob["question"]
        ref_ans = prob.get("answer", "")
        target = prob.get("target_answer") or extract_answer(ref_ans)

        prob_res = {
            "index": idx,
            "question": q,
            "target_answer": target,
            "conditions": {},
        }

        for cond in conditions:
            if cond == "normal_direct":
                gen = generate_normal(model, tokenizer, q, max_visible_tokens=max_visible_tokens)
                log_lik = (
                    evaluate_no_think(model, tokenizer, q, ref_ans)
                    if compute_likelihood and ref_ans
                    else None
                )
            elif cond == "normal_cot":
                gen = generate_normal_cot(model, tokenizer, q, max_visible_tokens=max_visible_tokens)
                log_lik = (
                    evaluate_visible_cot(model, tokenizer, q, ref_ans)
                    if compute_likelihood and ref_ans
                    else None
                )
            elif cond == "cotpt_hidden":
                gen = generate_cotpt_deliberation(
                    model,
                    tokenizer,
                    q,
                    num_hidden_tokens=num_hidden_tokens,
                    max_visible_tokens=max_visible_tokens,
                )
                log_lik = (
                    evaluate_hidden_deliberation(model, tokenizer, q, ref_ans, num_hidden_tokens=num_hidden_tokens)
                    if compute_likelihood and ref_ans
                    else None
                )
            elif cond == "cotpt_adaptive":
                gen = generate_cotpt_adaptive(
                    model,
                    tokenizer,
                    mixing_head,
                    q,
                    entropy_threshold=entropy_threshold,
                    num_hidden_tokens=num_hidden_tokens,
                    max_visible_tokens=max_visible_tokens,
                )
                log_lik = None
            else:
                raise ValueError(f"Unknown condition: {cond}")

            extracted = extract_answer(gen["text"])
            is_correct = check_answer(extracted, target) if target else False

            entry = {
                "text": gen["text"],
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "latency_sec": gen["latency_sec"],
                "tokens_per_sec": gen["tokens_per_sec"],
                "visible_tokens": gen["visible_tokens"],
                "peak_kv_len": gen["peak_kv_len"],
                "final_kv_len": gen["final_kv_len"],
                "forward_steps": gen["forward_steps"],
                "log_likelihood": log_lik,
            }
            prob_res["conditions"][cond] = entry

            condition_aggregates[cond]["correct"].append(1.0 if is_correct else 0.0)
            condition_aggregates[cond]["latency"].append(gen["latency_sec"])
            condition_aggregates[cond]["tokens_per_sec"].append(gen["tokens_per_sec"])
            condition_aggregates[cond]["visible_tokens"].append(float(gen["visible_tokens"]))
            condition_aggregates[cond]["peak_kv"].append(float(gen["peak_kv_len"]))
            condition_aggregates[cond]["final_kv"].append(float(gen["final_kv_len"]))
            condition_aggregates[cond]["forward_steps"].append(float(gen["forward_steps"]))
            if log_lik is not None:
                condition_aggregates[cond]["log_likelihood"].append(log_lik)

        per_problem_results.append(prob_res)

        if verbose:
            print(f"Problem {idx + 1}/{len(problems)}: Target = {target}")
            for cond in conditions:
                c_data = prob_res["conditions"][cond]
                mark = "✓" if c_data["is_correct"] else "✗"
                print(f"  [{cond:15s}] {mark} Pred: {c_data['extracted_answer']!s:8s} | "
                      f"Peak KV: {c_data['peak_kv_len']:3d} | "
                      f"Latency: {c_data['latency_sec']:.2f}s | "
                      f"Out: {c_data['text'].strip()[:60]}...")
            print()

    summary = {
        "num_problems": len(problems),
        "conditions": {},
    }

    for cond, metrics in condition_aggregates.items():
        n = len(metrics["correct"])
        mean_acc = sum(metrics["correct"]) / n if n > 0 else 0.0
        mean_latency = sum(metrics["latency"]) / n if n > 0 else 0.0
        mean_tps = sum(metrics["tokens_per_sec"]) / n if n > 0 else 0.0
        mean_vis_tok = sum(metrics["visible_tokens"]) / n if n > 0 else 0.0
        mean_peak_kv = sum(metrics["peak_kv"]) / n if n > 0 else 0.0
        mean_final_kv = sum(metrics["final_kv"]) / n if n > 0 else 0.0
        mean_fwd_steps = sum(metrics["forward_steps"]) / n if n > 0 else 0.0
        ll_list = metrics["log_likelihood"]
        mean_ll = sum(ll_list) / len(ll_list) if ll_list else None

        summary["conditions"][cond] = {
            "accuracy": mean_acc,
            "accuracy_percent": mean_acc * 100.0,
            "mean_latency_sec": mean_latency,
            "mean_tokens_per_sec": mean_tps,
            "mean_visible_tokens": mean_vis_tok,
            "mean_peak_kv_len": mean_peak_kv,
            "mean_final_kv_len": mean_final_kv,
            "mean_forward_steps": mean_fwd_steps,
            "mean_log_likelihood": mean_ll,
        }

    return {
        "summary": summary,
        "per_problem": per_problem_results,
    }


def format_benchmark_table(summary: Dict[str, Any], format_type: str = "terminal") -> str:
    conditions = summary["conditions"]
    headers = [
        "Condition",
        "Accuracy (%)",
        "Peak KV (tok)",
        "Final KV (tok)",
        "Latency (s)",
        "Visible Tok/s",
        "Fwd Passes",
        "Mean Log-Lik",
    ]

    rows = []
    for cond_name, stats in conditions.items():
        ll_str = f"{stats['mean_log_likelihood']:.3f}" if stats['mean_log_likelihood'] is not None else "N/A"
        row = [
            cond_name,
            f"{stats['accuracy_percent']:.1f}%",
            f"{stats['mean_peak_kv_len']:.1f}",
            f"{stats['mean_final_kv_len']:.1f}",
            f"{stats['mean_latency_sec']:.2f}s",
            f"{stats['mean_tokens_per_sec']:.1f}",
            f"{stats['mean_forward_steps']:.1f}",
            ll_str,
        ]
        rows.append(row)

    if format_type == "markdown":
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(val))

    separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_row = "| " + " | ".join([h.ljust(col_widths[i]) for i, h in enumerate(headers)]) + " |"

    res_lines = [separator, header_row, separator]
    for r in rows:
        line = "| " + " | ".join([val.ljust(col_widths[i]) for i, val in enumerate(r)]) + " |"
        res_lines.append(line)
    res_lines.append(separator)

    return "\n".join(res_lines)
