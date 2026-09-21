import torch

from cotpt.inference import generate_with_adaptive_deliberation, generate_with_hidden_deliberation
from cotpt.mixing_head import MixingHead, blend_logits
from cotpt.model_utils import ensure_thought_tokens, get_thought_token_ids
from cotpt.training import (
    compute_base_future_loglik,
    cotpt_training_step,
    generate_rollout_batch,
    reinforce_loss_with_value_baseline,
)
from cotpt.value_head import ValueHead


def test_blend_logits_formula():
    w = torch.tensor([[0.25]])
    base = torch.tensor([[1.0, 2.0, 3.0]])
    thought = torch.tensor([[5.0, 6.0, 7.0]])
    mixed = blend_logits(w, base, thought)
    expected = 0.75 * base + 0.25 * thought
    assert torch.allclose(mixed, expected)


def test_positive_only_clamps_negatives():
    torch.manual_seed(0)
    rewards = torch.tensor([0.0, 2.0, -1.0, 1.0])
    logprobs = torch.zeros(4, 3)
    _, _, adv_full = reinforce_loss_with_value_baseline(rewards, logprobs, None, use_positive_only=False)
    _, _, adv_pos = reinforce_loss_with_value_baseline(rewards, logprobs, None, use_positive_only=True)
    assert (adv_pos >= 0).all()
    # Positive entries unchanged where advantage was already positive, negatives become 0.
    assert (adv_pos[adv_full < 0] == 0).all()


def test_base_future_loglik_finite(tiny_model):
    tiny_model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    with torch.no_grad():
        base_out = tiny_model(input_ids=input_ids)
    ll = compute_base_future_loglik(base_out.logits, input_ids, position=3, lookahead=4)
    assert ll < 0  # log-prob
    assert ll > -20


def test_rollout_with_brackets_excludes_delimiters_from_logprobs(tiny_model):
    tiny_model.eval()
    torch.manual_seed(1)
    prefix = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    plain = generate_rollout_batch(tiny_model, prefix, num_rollouts=2, thought_length=4, temperature=1.0)
    bracketed = generate_rollout_batch(
        tiny_model, prefix, num_rollouts=2, thought_length=4, temperature=1.0,
        start_thought_id=5, end_thought_id=6,
    )
    # Same number of REINFORCE steps (delimiters forced, not sampled).
    assert bracketed["thought_logprobs_per_step"].shape == plain["thought_logprobs_per_step"].shape
    # Cache grew by 2 extra delimiter positions.
    assert bracketed["cache"].get_seq_length() == plain["cache"].get_seq_length() + 2


def test_rollout_zero_thought_ignores_brackets(tiny_model):
    tiny_model.eval()
    torch.manual_seed(1)
    prefix = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    out = generate_rollout_batch(
        tiny_model, prefix, num_rollouts=2, thought_length=0, temperature=1.0,
        start_thought_id=5, end_thought_id=6,
    )
    assert out["cache"].get_seq_length() == prefix.shape[1]


def test_training_step_logit_and_differential_modes(tiny_model):
    tiny_model.train()
    for mode in ["hidden", "logit"]:
        for diff in [False, True]:
            mixing_head = MixingHead(tiny_model.config.hidden_size)
            value_head = ValueHead(tiny_model.config.hidden_size)
            torch.manual_seed(3)
            input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 16))
            stats = cotpt_training_step(
                tiny_model, mixing_head, input_ids, value_head=value_head,
                num_think_positions=2, num_rollouts=2, thought_length=3, lookahead=2,
                mixing_mode=mode, use_differential_reward=diff,
                start_thought_id=5, end_thought_id=6,
            )
            assert stats["total_loss"].isfinite()
            stats["total_loss"].backward()


def test_training_step_positive_only_runs(tiny_model):
    tiny_model.train()
    mixing_head = MixingHead(tiny_model.config.hidden_size)
    torch.manual_seed(4)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 16))
    stats = cotpt_training_step(
        tiny_model, mixing_head, input_ids, use_value_baseline=False,
        num_think_positions=2, num_rollouts=2, thought_length=3, lookahead=2,
        use_positive_only=True,
    )
    assert stats["total_loss"].isfinite()


def test_ensure_thought_tokens_resizes(tiny_model, mock_tokenizer):
    start, end = ensure_thought_tokens(
        mock_tokenizer, tiny_model, "<|startofthought|>", "<|endofthought|>"
    )
    assert start is not None and end is not None and start != end
    assert tiny_model.get_input_embeddings().weight.shape[0] == len(mock_tokenizer)
    s2, e2 = get_thought_token_ids(mock_tokenizer, "<|startofthought|>", "<|endofthought|>")
    assert (s2, e2) == (start, end)


def test_inference_with_brackets_runs(tiny_model, mock_tokenizer):
    tiny_model.eval()
    out = generate_with_hidden_deliberation(
        tiny_model, mock_tokenizer, "a test prompt",
        max_visible_tokens=3, num_hidden_tokens=2, show_hidden_thoughts=False,
        use_thought_tokens=True, start_thought_id=5, end_thought_id=6,
    )
    assert isinstance(out, str)
    mixing_head = MixingHead(tiny_model.config.hidden_size)
    for mode in ["hidden", "logit"]:
        out2 = generate_with_adaptive_deliberation(
            tiny_model, mock_tokenizer, mixing_head, "a test prompt",
            max_visible_tokens=3, num_hidden_tokens=2, show_hidden_thoughts=False,
            mixing_mode=mode, use_thought_tokens=True,
            start_thought_id=5, end_thought_id=6,
        )
        assert isinstance(out2, str)


def test_bracket_zero_hidden_matches_no_bracket(tiny_model, mock_tokenizer):
    from cotpt.evaluation import evaluate_hidden_deliberation
    tiny_model.eval()
    torch.manual_seed(0)
    a = evaluate_hidden_deliberation(
        tiny_model, mock_tokenizer, "q one two", "a three four",
        num_hidden_tokens=0, use_thought_tokens=True, start_thought_id=5, end_thought_id=6,
    )
    torch.manual_seed(0)
    b = evaluate_hidden_deliberation(
        tiny_model, mock_tokenizer, "q one two", "a three four", num_hidden_tokens=0,
    )
    assert a == b
