import torch
from torch import nn


class MixingHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_before: torch.Tensor, hidden_after: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([hidden_before, hidden_after], dim=-1)))


def blend_logits(
    w: torch.Tensor, base_logits: torch.Tensor, thought_logits: torch.Tensor
) -> torch.Tensor:
    """Quiet-STaR-style logit mixing: (1-w) * base + w * thought.

    w is [..., 1] (sigmoid output); logits are [..., vocab]. Broadcasts.
    """
    return (1 - w) * base_logits + w * thought_logits
