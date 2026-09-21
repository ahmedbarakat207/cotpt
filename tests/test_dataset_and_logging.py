import json
import os
import pytest
import torch

from cotpt.data import (
    load_text_file,
    load_training_texts,
    load_gsm8k,
    format_math,
    format_mmlu,
    format_arc,
    format_svamp,
    find_prompt_boundary,
)
from cotpt.logging_utils import ExperimentLogger
from cotpt.training import pick_think_positions


def test_load_text_file_splits_on_blank_lines(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(
        "This is the first paragraph, long enough to count.\n\n"
        "Short\n\n"
        "This is the third paragraph, also long enough to count.\n"
    )
    texts = load_text_file(str(p), min_chars=20)
    assert len(texts) == 2
    assert "first paragraph" in texts[0]
    assert "third paragraph" in texts[1]


def test_load_text_file_empty_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        load_text_file(str(p))


def test_load_text_file_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        json.dumps({"question": "What is 2+2?", "answer": "2+2=<<2+2=4>>4\n#### 4"}) + "\n"
        + json.dumps({"text": "This is educational reasoning text that has enough characters."}) + "\n"
    )
    texts = load_text_file(str(p))
    assert len(texts) == 2
    assert "What is 2+2?" in texts[0]
    assert "<<" not in texts[0]
    assert "educational reasoning" in texts[1]


def test_load_training_texts_prefers_local_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("A sufficiently long paragraph of training text goes here.\n")
    texts = load_training_texts(data_path=str(p))
    assert texts is not None and len(texts) == 1


def test_load_training_texts_returns_none_with_no_source():
    assert load_training_texts() is None


def test_load_gsm8k_from_local():
    path = "data/gsm8k_train.jsonl"
    if not os.path.exists(path):
        pytest.skip("data/gsm8k_train.jsonl not present")
    texts = load_gsm8k(split="train", limit=5)
    assert len(texts) == 5
    assert all("Q:" in t and "A:" in t for t in texts)
    assert all("<<" not in t for t in texts)


def test_load_gsm8k_direct_mode():
    path = "data/gsm8k_train.jsonl"
    if not os.path.exists(path):
        pytest.skip("data/gsm8k_train.jsonl not present")
    texts = load_gsm8k(split="train", mode="direct_answer", limit=5)
    assert len(texts) == 5
    assert all("The answer is" in t for t in texts)


def test_format_math():
    row = {"problem": "Compute 3 + 5.", "solution": "3 + 5 = \\boxed{8}."}
    assert "The answer is 8" in format_math(row, mode="direct_answer")
    assert "\\boxed{8}" in format_math(row, mode="step_by_step")


def test_format_mmlu():
    row = {"question": "What is 2+2?", "choices": ["1", "2", "3", "4"], "answer": 3}
    text = format_mmlu(row)
    assert "Answer: (D)" in text
    assert "(A) 1" in text


def test_format_arc():
    row = {
        "question": "What happens?",
        "choices": {"text": ["Cool", "Heat"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    text = format_arc(row)
    assert "Answer: (B)" in text


def test_format_svamp():
    row = {"Body": "Sam has 10 apples.", "Question": "He eats 2. How many left?", "Answer": 8}
    text = format_svamp(row)
    assert "The answer is 8" in text


def test_find_prompt_boundary(mock_tokenizer):
    text = "Q: What is 2 + 2?\nA: The answer is 4."
    boundary = find_prompt_boundary(text, mock_tokenizer)
    assert boundary is not None and boundary > 0


def test_experiment_logger_writes_jsonl(tmp_path):
    log_path = str(tmp_path / "logs" / "run.jsonl")
    logger = ExperimentLogger(log_path)
    logger.log(0, loss=1.23, reward=-4.5)
    logger.log(1, loss=1.10, reward=-4.2)
    logger.close()

    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["step"] == 0 and lines[0]["loss"] == 1.23
    assert lines[1]["reward"] == -4.2


def test_pick_think_positions_entropy():
    logits = torch.randn(1, 40, 100)
    pos = pick_think_positions(40, 4, 4, base_logits=logits, strategy="entropy")
    assert len(pos) == 4
    assert all(0 <= p < 35 for p in pos)


def test_pick_think_positions_with_min_position():
    logits = torch.randn(1, 40, 100)
    pos = pick_think_positions(40, 3, 2, base_logits=logits, min_position=20)
    assert len(pos) == 3
    assert all(20 <= p < 37 for p in pos)
