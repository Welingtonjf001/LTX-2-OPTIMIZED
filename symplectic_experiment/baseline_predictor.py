"""Baseline SEM estrutura física — a linha decisiva do experimento.

Um preditor temporal compacto que mapeia (q, p, ctx) -> (q', p') diretamente,
sem Hamiltoniano, sem integrador simplético, sem conservação. Mesma dimensão
de estado, mesmo decoder, mesmo orçamento de treino que a versão hamiltoniana.

Por que isto existe: sem este controle, um resultado bom do núcleo hamiltoniano
é ININTERPRETÁVEL — não há como separar "a estrutura física ajudou" de "uma
rede pequena já bastaria para esta tarefa". É a diferença entre uma tese
testável e uma narrativa. A comparação que importa não é
"hamiltoniano vs. difusão de 40 GB", é "hamiltoniano vs. um aluno igualmente
pequeno e sem restrições".
"""
import torch
import torch.nn as nn


class UnconstrainedPredictor(nn.Module):
    """z_{t+1} = z_t + dt * f_theta(z_t, ctx), com z = [q, p].

    A atualização é residual e escalada por dt para ter a MESMA forma de
    atualização do integrador de Verlet — o que muda é só a ausência de
    estrutura (sem separação T/V, sem simplético, sem conservação).
    """

    def __init__(self, q_dim: int, ctx_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * q_dim + ctx_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * q_dim),
        )

    def step(self, q: torch.Tensor, p: torch.Tensor, ctx: torch.Tensor, dt: float):
        dz = self.net(torch.cat([q, p, ctx], dim=-1))
        dq, dp = dz.chunk(2, dim=-1)
        return q + dt * dq, p + dt * dp
