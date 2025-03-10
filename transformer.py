from torch import nn
import torch
from Multi_head_attention import MultiHeadAttention
from utilities_layers import FeedForward, LayerNorm


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        n_heads: int,
        dropout: float = 0.1,
        qkb_bias: bool = False,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            context_length=context_length,
            d_in=embed_dim,
            d_out=embed_dim,
            n_heads=n_heads,
            qkv_bias=qkb_bias,
        )
        self.feedforward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=None,
        )
        self.normalize1 = LayerNorm(embed_dim=embed_dim)
        self.normalize2 = LayerNorm(embed_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.normalize1(x)))
        return x + self.dropout(self.feedforward(self.normalize2(x)))
