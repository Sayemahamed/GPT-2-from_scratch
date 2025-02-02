import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Create a boolean causal mask (upper triangular part is True)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        # x: (batch_size, num_tokens, d_in)
        b, n, _ = x.shape
        
        queries = self.W_query(x)  # (b, n, d_out)
        keys    = self.W_key(x)    # (b, n, d_out)
        values  = self.W_value(x)  # (b, n, d_out)
        
        # Scaled dot-product attention
        scores = queries @ keys.transpose(-2, -1) / (keys.shape[-1] ** 0.5)  # (b, n, n)
        
        # Apply causal mask: only allow attention to current and previous tokens
        scores.masked_fill_(self.mask[:n, :n], float("-inf"))
        
        attn_weights = torch.softmax(scores, dim=-1)  # (b, n, n)
        context = attn_weights @ values  # (b, n, d_out)
        return context

# Example usage:
if __name__ == "__main__":
    torch.manual_seed(123)
    batch = torch.randn(2, 5, 8)  # Example: batch_size=2, num_tokens=5, d_in=8
    d_in, d_out = 8, 16
    ca = CausalAttention(d_in, d_out, context_length=batch.shape[1])
    context_vecs = ca(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
