"""Vídeo sintético: um disco colorido quicando elasticamente numa caixa 2D.

Escolhido de propósito por ser um sistema aproximadamente conservativo (fora
dos instantes de colisão) — é o caso mais favorável possível para um
integrador hamiltoniano, e mesmo assim as colisões contra a parede são
descontinuidades que um H(q,p) suave não representa exatamente. Serve só
para testar a mecânica de código, não é um proxy de vídeo real (ver README.md).
"""
import torch


def generate_bouncing_ball_video(num_frames: int, height: int, width: int, dt: float = 0.08,
                                 radius: float = 0.45):
    """Retorna (frames, x0, v0): frames [T, 3, H, W] em [0, 1], e o estado
    físico inicial (posição, velocidade) em coordenadas normalizadas [-1, 1].

    `radius` default 0.45 (era 0.18): MEDIDO 2026-09-12 que com raio 0.18 a
    bola ocupa 4,7% dos pixels e a L1 de reconstrução é dominada pelo fundo
    estático — um preditor trivial que desenha SÓ O FUNDO atinge L1=0.0108,
    melhor do que os modelos treinados conseguiam (0.0110/0.0112). Ou seja, a
    tarefa não tinha sinal suficiente para o otimizador se importar com o
    movimento. Raio maior torna o objeto em movimento o termo dominante."""
    pos = torch.tensor([-0.5, 0.3])
    vel = torch.tensor([0.9, 0.6])
    color = torch.tensor([0.95, 0.35, 0.25])

    x0, v0 = pos.clone(), vel.clone()

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
        mask = torch.sigmoid((radius - dist) * 40.0)  # borda suave (evita gradiente zero)
        bg = torch.tensor([0.08, 0.08, 0.10])
        frame = bg.view(3, 1, 1) * (1 - mask) + color.view(3, 1, 1) * mask
        frames.append(frame)

    return torch.stack(frames, dim=0), x0, v0
