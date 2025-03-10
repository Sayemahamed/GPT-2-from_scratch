import torch
import torch.nn as nn
from transformer import Transformer
from utilities_layers import LayerNorm


class Model(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        transformer_layers: int,
        n_heads: int,
        qkb_bias: bool = False,
        vocab_size: int = 50257,
    ) -> None:
        super().__init__()
        self.context_length: int = context_length
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=context_length, embedding_dim=embed_dim
        )
        self.transformer_layers = nn.Sequential(
            *[
                Transformer(
                    embed_dim=embed_dim,
                    context_length=context_length,
                    n_heads=n_heads,
                    qkb_bias=qkb_bias,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.layer_norm = LayerNorm(embed_dim=embed_dim)
        self.lm_head = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        token_embedding: torch.Tensor = self.token_embedding(x)
        positional_embedding: torch.Tensor = self.positional_embedding(
            torch.arange(t, device=x.device)
        )
        x = token_embedding + positional_embedding
        x = self.transformer_layers(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits
