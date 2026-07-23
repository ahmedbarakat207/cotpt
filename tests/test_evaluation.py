import torch

from cotpt.evaluation import evaluate_no_think, evaluate_hidden_deliberation, evaluate_visible_cot, run_evaluation


def test_hidden_deliberation_with_zero_tokens_matches_no_think(tiny_model, mock_tokenizer):
    """With num_hidden_tokens=0, the eviction mechanism does nothing at
    every position, so this MUST equal plain teacher-forced log-likelihood
    exactly. If it doesn't, the eval harness has a bug independent of
    whether hidden deliberation itself helps."""
    question = "Q: what is 2 plus 2\nA:"
    answer = " the answer is 4"

    no_think = evaluate_no_think(tiny_model, mock_tokenizer, question, answer)
    hidden_zero = evaluate_hidden_deliberation(tiny_model, mock_tokenizer, question, answer, num_hidden_tokens=0)
    assert abs(no_think - hidden_zero) < 1e-5, f"{no_think} vs {hidden_zero}"


def test_visible_cot_differs_from_no_think(tiny_model, mock_tokenizer):
    """Appending a CoT suffix changes the conditioning context, so it should
    (with essentially total probability, for a real vocab) produce a
    different likelihood than the bare question -- confirms the suffix is
    actually being applied, not silently ignored."""
    question = "Q: what is 2 plus 2\nA:"
    answer = " the answer is 4"
    no_think = evaluate_no_think(tiny_model, mock_tokenizer, question, answer)
    cot = evaluate_visible_cot(tiny_model, mock_tokenizer, question, answer)
    assert no_think != cot


def test_run_evaluation_end_to_end(tiny_model, mock_tokenizer):
    problems = [
        {"question": "Q: what is 2 plus 2\nA:", "answer": " the answer is 4"},
        {"question": "Q: what is 3 plus 3\nA:", "answer": " the answer is 6"},
    ]
    result = run_evaluation(tiny_model, mock_tokenizer, problems=problems, num_hidden_tokens=3)
    assert result["summary"]["num_problems"] == 2
    for key in ("no_think", "visible_cot", "hidden_deliberation"):
        assert isinstance(result["summary"][key], float)
    assert len(result["per_problem"]) == 2
