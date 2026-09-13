"""Distribuição de vídeos sintéticos (disco quicando), para treinar um score
model de verdade — não um único vídeo fixo. Posição inicial, velocidade e cor
são amostradas aleatoriamente por vídeo: p(x) é a distribuição sobre TODAS as
trajetórias possíveis, que é o que um modelo de difusão precisa aprender.
"""
import torch


def _simulate(pos, vel, color, num_frames, height, width, dt):
    radius = 0.22
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    frames = []
    for _ in range(num_frames):
        for dim in range(2):
            if pos[dim] + radius > 1.0 or pos[dim] - radius < -1.0:
                vel[dim] = -vel[dim]
        pos = pos + dt * vel
        pos = pos.clamp(-1.0 + radius, 1.0 - radius)

        dist = torch.sqrt((grid_x - pos[1]) ** 2 + (grid_y - pos[0]) ** 2)
        mask = torch.sigmoid((radius - dist) * 40.0)
        bg = torch.tensor([0.08, 0.08, 0.10])
        frame = bg.view(3, 1, 1) * (1 - mask) + color.view(3, 1, 1) * mask
        frames.append(frame)
    return torch.stack(frames, dim=0)  # [T, 3, H, W]


def sample_toy_video_batch(batch_size: int, num_frames: int = 6, height: int = 16, width: int = 16, dt: float = 0.15) -> torch.Tensor:
    """Retorna [B, T, 3, H, W] com condições iniciais aleatórias por amostra."""
    videos = []
    for _ in range(batch_size):
        pos = torch.rand(2) * 1.0 - 0.5
        angle = torch.rand(1) * 2 * torch.pi
        speed = 0.5 + 0.6 * torch.rand(1)
        vel = speed * torch.stack([torch.cos(angle[0]), torch.sin(angle[0])])
        color = 0.3 + 0.6 * torch.rand(3)
        videos.append(_simulate(pos, vel, color, num_frames, height, width, dt))
    return torch.stack(videos, dim=0)


def flatten_videos(videos: torch.Tensor) -> torch.Tensor:
    return videos.reshape(videos.shape[0], -1)


def unflatten_video(flat: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
    return flat.reshape(-1, num_frames, 3, height, width)
