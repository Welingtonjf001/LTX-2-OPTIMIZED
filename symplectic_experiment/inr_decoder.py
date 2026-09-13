"""Decoder INR modulado (Fourier features + FiLM/SIREN) sobre a coordenada
latente q(t). Resolução de saída é definida pela malha de coordenadas passada
em forward(), não pela arquitetura — permite renderizar em resoluções
diferentes das usadas em treino, dentro do que o V(q) treinado sabe descrever.
"""
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Mapeamento posicional log-espaçado — mitiga o spectral bias de MLPs
    (tendência a aprender só baixa frequência, produzindo imagem borrada)."""

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs) * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2] em [-1, 1]
        proj = x.unsqueeze(-1) * self.freqs  # [..., 2, num_freqs]
        return torch.cat([x, torch.sin(proj).flatten(-2), torch.cos(proj).flatten(-2)], dim=-1)


class ModulatedSineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w0: float = 30.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = gamma * out + beta
        return torch.sin(self.w0 * out)


class ContinuousVideoINRDecoder(nn.Module):
    """Decodifica q(t) em radiância RGB para uma malha (u, v) arbitrária."""

    def __init__(self, q_dim: int, hidden_dim: int = 64, num_layers: int = 3, num_freqs: int = 6):
        super().__init__()
        self.fourier = FourierFeatures(num_freqs)
        coord_dim = 2 + 2 * 2 * num_freqs
        self.mod_generator = nn.Linear(q_dim, num_layers * 2 * hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.coord_embed = nn.Linear(coord_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [ModulatedSineLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.to_rgb = nn.Linear(hidden_dim, 3)

    def forward(self, q: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # q: [B, q_dim], coords: [B, N, 2]
        b = coords.shape[0]
        mods = self.mod_generator(q).view(b, self.num_layers, 2, self.hidden_dim)

        feats = self.fourier(coords)
        h = torch.sin(30.0 * self.coord_embed(feats))
        for i, layer in enumerate(self.layers):
            gamma = mods[:, i, 0:1, :] + 1.0
            beta = mods[:, i, 1:2, :]
            h = layer(h, gamma, beta)

        return torch.sigmoid(self.to_rgb(h))  # [B, N, 3]


def make_coord_grid(height: int, width: int, batch_size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    return coords.unsqueeze(0).repeat(batch_size, 1, 1)
