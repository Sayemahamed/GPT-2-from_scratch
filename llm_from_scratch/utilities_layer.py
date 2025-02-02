import torch
from torch import nn
class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(data=torch.ones(embed_dim))
        self.shift=nn.Parameter(data=torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor =x.mean(dim=-1,keepdim=True)
        var =x.var(dim=-1,keepdim=True,unbiased=False)
        return self.scale*(x-mean)/torch.sqrt(var+self.eps)+self.shift