import json
import os

import pytest

from cotpt.dataset import load_text_file, load_training_texts
from cotpt.logging_utils import ExperimentLogger


def test_load_text_file_splits_on_blank_lines(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(
        "This is the first paragraph, long enough to count.\n\n"
        "Short\n\n"  # too short, should be filtered
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


def test_load_training_texts_prefers_local_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("A sufficiently long paragraph of training text goes here.\n")
    texts = load_training_texts(data_path=str(p))
    assert texts is not None and len(texts) == 1


def test_load_training_texts_returns_none_with_no_source():
    assert load_training_texts() is None


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
