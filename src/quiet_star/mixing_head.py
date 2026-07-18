import torch
from torch import nn


class MixingHead(nn.Module):
    """Quiet-STaR's 'talk head'. Learns how much weight to give the
    post-thought prediction vs. the plain no-thought prediction:
        mixed_hidden = (1 - w) * hidden_before + w * hidden_after
    Mixing happens on hidden states (post-final-norm), then the caller
    projects the blend through the model's own lm_head to get logits.
    (Verified empirically that a Qwen3 model's hidden_states[-1] is exactly
    what its lm_head consumes -- no extra norm step needed here.)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_before: torch.Tensor, hidden_after: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([hidden_before, hidden_after], dim=-1)))
