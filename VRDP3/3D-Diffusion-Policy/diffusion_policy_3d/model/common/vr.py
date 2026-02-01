from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops.layers.torch import Rearrange

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])

    def nll(self, sample, dims=[1,]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

    
class VR(nn.Module):  
    def __init__(self, in_channel, cond_dim):
        super().__init__()

        self.bottleneck_proj = nn.Sequential(
            nn.Linear(in_channel, 4*in_channel),
            nn.GELU(),
            nn.Linear(4*in_channel, in_channel)
        )
        self.filtering_proj = nn.Sequential(
            nn.Linear(in_channel, 4*in_channel),
            nn.GELU(),
            nn.Linear(4*in_channel, 1)
       )

        self.film_proj = nn.Sequential(
            nn.Linear(cond_dim, 4*in_channel),
            nn.GELU(),
            nn.Linear(4*in_channel, 2*in_channel)
        )
    
    def forward(self, x, t):
        # x: [B, C, D], t: [B, H]
        tb = self.film_proj(t)[:, None, :]          # [B, 1, 2D]
        alpha, beta = tb.chunk(2, dim=-1)           # [B, 1, D]
        x = x * (1 + alpha) + beta                  # [B, C, D]

        x = self.bottleneck_proj(x)                 # [B, C, D]
        std = self.filtering_proj(x).expand_as(x)   # [B, C, D]
        x = torch.cat([x, std], dim=-1)             # [B, C, 2D]
        x = einops.rearrange(x, 'b c d -> b d c')   # [B, 2D, C]
        q = DiagonalGaussianDistribution(x)
        x = q.sample()                              # [B, D, C]
        x = einops.rearrange(x, 'b d c -> b c d')   # [B, C, D]
        
        return x, q
    