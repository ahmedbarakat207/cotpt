import torch

from cotpt.training import reinforce_loss_fn


def test_reinforce_favors_highest_reward_action():
    torch.manual_seed(0)
    logits = torch.nn.Parameter(torch.zeros(4))
    rewards = torch.tensor([0.0, 1.0, 5.0, -2.0])  # action 2 is clearly best
    opt = torch.optim.SGD([logits], lr=0.5)

    for _ in range(200):
        probs = torch.softmax(logits, dim=-1)
        logprob_of_each_action = torch.log(probs + 1e-10)
        loss, _ = reinforce_loss_fn(rewards, logprob_of_each_action)
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_probs = torch.softmax(logits, dim=-1)
    assert final_probs[2] > 0.5
    assert final_probs[2] == final_probs.max()
