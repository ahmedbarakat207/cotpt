import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state).squeeze(-1)
