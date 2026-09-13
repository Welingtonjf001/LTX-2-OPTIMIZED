"""Rede de velocidade toy para Rectified Flow / Flow Matching.

Ao contrário do score model (que recebe sigma, nível de ruído), esta rede
recebe t em [0,1], a posição ao longo do caminho reto entre ruído (t=0) e
dado (t=1). Mesma observação do score_model.py se aplica: o vídeo inteiro é
um único ponto x; t não tem relação com "frame do vídeo".
"""
import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ToyVelocityNet(nn.Module):
    def __init__(self, data_dim: int, hidden: int = 512, emb_dim: int = 64):
        super().__init__()
        self.embed = TimeEmbedding(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.embed(t)
        return self.net(torch.cat([x, emb], dim=-1))
