from torch import nn
class MultiHeadAttention(nn.Module):
    def __init__(self
                 ,d_in:int,d_out:int,context_length:int,n_heads:int,qkv_bias:bool=False):
        super().__init__(  )
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.n_heads: int = n_heads
        self.head_dim: int = d_out // n_heads
        self.query_weight = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.key_weight = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.value_weight = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out)