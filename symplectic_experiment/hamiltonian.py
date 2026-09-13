"""Blocos hamiltonianos latentes: energia cinética/potencial e integrador simplético.

Ver README.md deste diretório para o que isto valida e o que NÃO valida.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KineticEnergy(nn.Module):
    """T(p) = 0.5 * sum(p_i^2 / m_i), massa diagonal estritamente positiva."""

    def __init__(self, dim: int):
        super().__init__()
        self.raw_mass = nn.Parameter(torch.zeros(dim))

    @property
    def mass(self) -> torch.Tensor:
        return F.softplus(self.raw_mass) + 1e-4

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((p ** 2) / self.mass, dim=-1)

    def velocity(self, p: torch.Tensor) -> torch.Tensor:
        return p / self.mass


class PotentialEnergy(nn.Module):
    """V(q; c). Ativações C² (SiLU) — o integrador diferencia V duas vezes
    (uma para a força, outra ao retropropagar), então ReLU quebraria o
    gradiente do integrador nas dobras."""

    def __init__(self, q_dim: int, ctx_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim + ctx_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, ctx], dim=-1)
        return self.net(x).squeeze(-1)


class SymplecticIntegrator(nn.Module):
    """Passo de Verlet explícito (meio-passo momento / passo completo posição /
    meio-passo momento), preservador de forma simplética por construção."""

    def __init__(self, kinetic: KineticEnergy, potential: PotentialEnergy):
        super().__init__()
        self.kinetic = kinetic
        self.potential = potential

    def _force(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """dV/dq PRESERVANDO o elo com o grafo de q.

        BUGFIX 2026-09-12: a versão anterior fazia `q.detach().requires_grad_(True)`
        aqui, o que cortava a dependência da força em relação aos estados
        anteriores do rollout — `create_graph=True` não restaura um elo já
        cortado. O gradiente do rollout ficava errado, com erro que cresce com
        dt, rigidez do potencial e número de passos: MEDIDO 0,10% (3 passos,
        potencial fraco) até 104% (24 passos, potencial rígido), sendo ~6% na
        configuração usada por train_toy.py (dt=0,08, 24 frames).
        """
        with torch.enable_grad():
            q_g = q if q.requires_grad else q.detach().requires_grad_(True)
            v_val = self.potential(q_g, ctx).sum()
            (grad_v,) = torch.autograd.grad(v_val, q_g, create_graph=q.requires_grad)
        return grad_v

    def step(self, q: torch.Tensor, p: torch.Tensor, ctx: torch.Tensor, dt: float):
        p_half = p - 0.5 * dt * self._force(q, ctx)
        q_next = q + dt * self.kinetic.velocity(p_half)
        p_next = p_half - 0.5 * dt * self._force(q_next, ctx)
        return q_next, p_next


def symplectic_residual_loss(integrator: SymplecticIntegrator, q, p, ctx, dt: float):
    """Resíduo da 2-forma simplética ω(u,v) = u_q·v_p − u_p·v_q sob a transição.

    Para A = DΦ(z), a condição é AᵀJA = J, cuja versão estocástica exige DOIS
    vetores independentes: ω(Au, Av) deve igualar ω(u, v).

    BUGFIX 2026-09-12: a versão anterior usava UM vetor só e media
    `sum(v_q * v_p)`, que não é invariante simplético — VERIFICADO: sob um
    mapa simplético exato (cisalhamento), essa quantidade muda de +0,3497 para
    +2,6369, enquanto ω(u,v) de dois vetores se preserva a 2,4e−07. Com um
    vetor só nem seria possível: vᵀJv ≡ 0 por identidade. Os valores erráticos
    que essa perda produzia no treino (0,45 → 1,33 → 0,64 → ...) eram ruído.
    """
    u_q, u_p = torch.randn_like(q), torch.randn_like(p)
    v_q, v_p = torch.randn_like(q), torch.randn_like(p)

    def step_fn(q_in, p_in):
        return integrator.step(q_in, p_in, ctx, dt)

    _, (au_q, au_p) = torch.autograd.functional.jvp(
        step_fn, (q, p), (u_q, u_p), create_graph=True
    )
    _, (av_q, av_p) = torch.autograd.functional.jvp(
        step_fn, (q, p), (v_q, v_p), create_graph=True
    )

    omega_in = torch.sum(u_q * v_p - u_p * v_q, dim=-1)
    omega_out = torch.sum(au_q * av_p - au_p * av_q, dim=-1)
    return F.mse_loss(omega_out, omega_in)
