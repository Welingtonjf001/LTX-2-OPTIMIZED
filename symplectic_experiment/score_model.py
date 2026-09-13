"""Rede de score toy: F_theta(x_in, c_noise) -> predição bruta, mesma forma de x.

A rede NUNCA vê o eixo de tempo físico do vídeo diretamente — só o nível de
ruído sigma (c_noise). O vídeo inteiro (todos os frames concatenados) é UM
ponto x no espaço de dados; é isso que a difusão realmente modela. Ver
README.md, seção "Probability Flow ODE", para a diferença em relação à
tentativa hamiltoniana anterior.
"""
import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=c_noise.device) / half)
        args = c_noise[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ToyScoreNet(nn.Module):
    def __init__(self, data_dim: int, hidden: int = 512, emb_dim: int = 64):
        super().__init__()
        self.embed = SinusoidalEmbedding(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        emb = self.embed(c_noise)
        return self.net(torch.cat([x, emb], dim=-1))
