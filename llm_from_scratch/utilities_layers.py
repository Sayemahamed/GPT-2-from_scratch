import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps: float = eps
        self.scale = nn.Parameter(data=torch.ones(embed_dim))
        self.shift = nn.Parameter(data=torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * embed_dim
        self.linear1 = nn.Linear(in_features=embed_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(self.gelu(self.linear2(self.linear1(x))))
