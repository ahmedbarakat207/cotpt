import json
import tempfile
import pytest
import torch

from cotpt.benchmark import (
    extract_answer,
    check_answer,
    generate_normal,
    generate_normal_cot,
    generate_cotpt_deliberation,
    generate_cotpt_adaptive,
    run_benchmark,
    format_benchmark_table,
    load_benchmark_problems,
)
from cotpt.mixing_head import MixingHead


def test_extract_answer():
    assert extract_answer("The result is #### 42") == "42"
    assert extract_answer("Therefore \\boxed{306} dollars") == "306"
    assert extract_answer("Giving away 20 leaves 84 minus 20, which is 64 apples.") == "64"
    assert extract_answer("The final cost equals 1,000.") == "1000"
    assert extract_answer("No numbers here at all!") is None
    assert extract_answer("") is None
    assert extract_answer(None) is None


def test_check_answer():
    assert check_answer("42", "42") is True
    assert check_answer("42.0", "42") is True
    assert check_answer("1,000", "1000") is True
    assert check_answer("42", "43") is False
    assert check_answer(None, "42") is False


def test_generate_normal_runs_and_tracks_kv(tiny_model, mock_tokenizer):
    prompt = "Q: 2 + 2\nA:"
    res = generate_normal(tiny_model, mock_tokenizer, prompt, max_visible_tokens=5)
    assert isinstance(res["text"], str)
    assert res["visible_tokens"] > 0
    assert res["peak_kv_len"] >= res["final_kv_len"]
    assert res["latency_sec"] > 0
    assert res["tokens_per_sec"] > 0
    assert res["forward_steps"] > 0


def test_generate_cotpt_deliberation_eviction(tiny_model, mock_tokenizer):
    prompt = "Q: 2 + 2\nA:"
    num_hidden = 3
    res = generate_cotpt_deliberation(
        tiny_model,
        mock_tokenizer,
        prompt,
        num_hidden_tokens=num_hidden,
        max_visible_tokens=4,
    )
    assert isinstance(res["text"], str)
    assert res["visible_tokens"] > 0
    prompt_ids = mock_tokenizer(prompt).input_ids
    expected_final_kv = prompt_ids.shape[1] + res["visible_tokens"]
    assert res["final_kv_len"] == expected_final_kv
    assert res["peak_kv_len"] >= expected_final_kv


def test_generate_cotpt_adaptive(tiny_model, mock_tokenizer):
    head = MixingHead(hidden_size=tiny_model.config.hidden_size)
    prompt = "Q: 2 + 2\nA:"
    res = generate_cotpt_adaptive(
        tiny_model,
        mock_tokenizer,
        head,
        prompt,
        entropy_threshold=100.0,
        num_hidden_tokens=2,
        max_visible_tokens=3,
    )
    assert res["deliberated_positions"] == 0
    assert res["visible_tokens"] > 0


def test_run_benchmark_end_to_end(tiny_model, mock_tokenizer):
    head = MixingHead(hidden_size=tiny_model.config.hidden_size)
    problems = [
        {"question": "Q: 2 + 2\nA:", "answer": " 4", "target_answer": "4"},
        {"question": "Q: 3 + 3\nA:", "answer": " 6", "target_answer": "6"},
    ]
    conditions = ["normal_direct", "normal_cot", "cotpt_hidden", "cotpt_adaptive"]
    out = run_benchmark(
        tiny_model,
        mock_tokenizer,
        problems=problems,
        conditions=conditions,
        mixing_head=head,
        num_hidden_tokens=2,
        max_visible_tokens=3,
        compute_likelihood=True,
    )

    assert out["summary"]["num_problems"] == 2
    for cond in conditions:
        assert cond in out["summary"]["conditions"]
        stats = out["summary"]["conditions"][cond]
        assert "accuracy_percent" in stats
        assert "mean_peak_kv_len" in stats
        assert "mean_latency_sec" in stats
        assert "mean_tokens_per_sec" in stats

    table_ascii = format_benchmark_table(out["summary"], format_type="terminal")
    assert "Condition" in table_ascii
    assert "normal_direct" in table_ascii
    assert "cotpt_hidden" in table_ascii

    table_md = format_benchmark_table(out["summary"], format_type="markdown")
    assert "| Condition |" in table_md


def test_load_benchmark_problems_custom_json():
    test_data = [
        {"question": "What is 10 + 5?", "answer": "It is 15.", "target_answer": "15"}
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        f_path = f.name

    loaded = load_benchmark_problems(f_path)
    assert len(loaded) == 1
    assert loaded[0]["target_answer"] == "15"
    assert loaded[0]["question"] == "What is 10 + 5?"
