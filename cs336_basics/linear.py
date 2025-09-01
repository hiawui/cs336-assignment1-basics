from math import sqrt
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: str=None, dtype: torch.dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        std = sqrt((2/(d_in + d_out)))
        weight_data = torch.empty(d_out, d_in, device=self.device, dtype=self.dtype)
        nn.init.trunc_normal_(weight_data, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weight = nn.Parameter(weight_data)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
