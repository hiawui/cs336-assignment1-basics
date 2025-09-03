import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: str=None, dtype: torch.dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        weight_data = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight_data, mean=0.0, std=1, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight_data)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
