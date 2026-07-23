import torch

from cotpt.mixing_head import MixingHead
from cotpt.training import cotpt_training_step


def test_gradients_flow_into_model_and_mixing_head(tiny_model):
    model = tiny_model
    model.train()
    mixing_head = MixingHead(hidden_size=model.config.hidden_size)

    torch.manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 20))

    stats = cotpt_training_step(
        model, mixing_head, input_ids,
        num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
    )
    stats["total_loss"].backward()

    model_grads = [p.grad for p in model.parameters() if p.grad is not None]
    mix_grads = [p.grad for p in mixing_head.parameters() if p.grad is not None]

    assert len(model_grads) == sum(1 for _ in model.parameters())
    assert len(mix_grads) == sum(1 for _ in mixing_head.parameters())
    assert all(torch.isfinite(g).all() for g in model_grads)
    assert all(torch.isfinite(g).all() for g in mix_grads)
    assert all(g.norm().item() > 0 for g in mix_grads)


def test_overfitting_a_fixed_example_reduces_loss(tiny_model):
    """Strongest integration check: proves rollouts -> reward -> REINFORCE ->
    mixing loss -> optimizer are wired together correctly, not just
    individually plausible."""
    model = tiny_model
    model.train()
    mixing_head = MixingHead(hidden_size=model.config.hidden_size)

    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-4},
        {"params": mixing_head.parameters(), "lr": 1e-3},
    ])

    torch.manual_seed(2)
    fixed_input_ids = torch.randint(0, model.config.vocab_size, (1, 20))

    losses, aux_losses = [], []
    for step in range(50):
        torch.manual_seed(100 + step)
        stats = cotpt_training_step(
            model, mixing_head, fixed_input_ids,
            num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
        )
        optimizer.zero_grad()
        stats["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(mixing_head.parameters()), 1.0)
        optimizer.step()
        losses.append(stats["total_loss"].item())
        aux_losses.append(stats["aux_lm_loss"])

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5, "total loss did not decrease"
    assert sum(aux_losses[-5:]) / 5 < sum(aux_losses[:5]) / 5, "plain LM loss did not decrease"


def test_short_sequence_falls_back_gracefully(tiny_model):
    """Sequences too short for any think position should still return a
    usable (aux-only) loss instead of crashing."""
    model = tiny_model
    model.train()
    mixing_head = MixingHead(hidden_size=model.config.hidden_size)

    short_ids = torch.randint(0, model.config.vocab_size, (1, 3))
    stats = cotpt_training_step(
        model, mixing_head, short_ids,
        num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
    )
    assert stats["num_positions"] == 0
    stats["total_loss"].backward()  # should not raise
