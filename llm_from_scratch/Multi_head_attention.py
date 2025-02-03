from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        n_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.n_heads: int = n_heads
        self.head_dim: int = d_out // n_heads
        self.query_weight = nn.Linear(
            in_features=d_in, out_features=d_out, bias=qkv_bias
        )
        self.key_weight = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.value_weight = nn.Linear(
            in_features=d_in, out_features=d_out, bias=qkv_bias
        )
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out)
        self.msk: torch.Tensor = torch.triu(
            input=torch.ones(context_length, context_length), diagonal=1,
        )
        self.register_buffer(name="mask", tensor=self.msk.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries: torch.Tensor = self.query_weight(x)
        keys: torch.Tensor = self.key_weight(x)
        values: torch.Tensor = self.value_weight(x)

        b, n, _ = x.shape

        queries = queries.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        scores: torch.Tensor = queries @ keys.transpose(-2, -1) / (self.head_dim**0.5)
        scores = scores.masked_fill_(self.msk[:n, :n].bool(), float("-inf"))
        attention: torch.Tensor = scores.softmax(dim=-1)

        context: torch.Tensor = attention @ values
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(b, n, self.n_heads * self.head_dim)
        )
        return self.out_proj(context)
