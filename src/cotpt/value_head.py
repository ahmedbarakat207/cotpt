import torch
from torch import nn


class ValueHead(nn.Module):
    """Predicts expected terminal reward from the hidden state at a given
    thought-generation step. Used as a per-token baseline for REINFORCE
    instead of a single mean-of-rollouts scalar shared by every token in a
    thought -- standard actor-critic credit assignment.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state).squeeze(-1)
