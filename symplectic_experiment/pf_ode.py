"""Probability Flow ODE com precondicionamento EDM (Karras et al. 2022,
"Elucidating the Design Space of Diffusion-Based Generative Models").

Isto é a formulação determinística REAL e correta de "vídeo como trajetória
contínua" — ao contrário da tentativa hamiltoniana anterior, aqui o único
eixo de tempo é o nível de ruído sigma, e a ODE é exatamente equivalente
(mesma marginal p_sigma(x) em cada sigma) à SDE de difusão da qual deriva.
Não há conflação entre tempo de ruído e tempo de vídeo: o vídeo inteiro
(todos os frames) é um único ponto x; sigma não tem nada a ver com "frame".
"""
import torch


class EDMPrecond:
    """D_theta(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))."""

    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def _scalings(self, sigma: torch.Tensor):
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
        c_noise = 0.25 * torch.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def denoise(self, net, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self._scalings(sigma)
        shape = (-1,) + (1,) * (x.dim() - 1)
        c_skip, c_out, c_in = c_skip.view(*shape), c_out.view(*shape), c_in.view(*shape)
        f_x = net(c_in * x, c_noise)
        return c_skip * x + c_out * f_x


def edm_loss(net, precond: EDMPrecond, x: torch.Tensor, p_mean: float = -1.2, p_std: float = 1.2) -> torch.Tensor:
    b = x.shape[0]
    sigma = torch.exp(torch.randn(b, device=x.device) * p_std + p_mean)
    sd = precond.sigma_data
    weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2

    shape = (-1,) + (1,) * (x.dim() - 1)
    noise = torch.randn_like(x) * sigma.view(*shape)
    x_noisy = x + noise
    d_x = precond.denoise(net, x_noisy, sigma)
    loss = weight.view(*shape) * (d_x - x) ** 2
    return loss.mean()


def make_sigma_schedule(sigma_min: float, sigma_max: float, num_steps: int, rho: float = 7.0, device="cpu") -> torch.Tensor:
    """Sequência decrescente sigma_max -> sigma_min (escala de Karras)."""
    steps = torch.arange(num_steps, dtype=torch.float64, device=device)
    inv_rho_min, inv_rho_max = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    return (inv_rho_max + steps / (num_steps - 1) * (inv_rho_min - inv_rho_max)) ** rho


def solve_pf_ode(net, precond: EDMPrecond, x_init: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Integra dx/dsigma = (x - D(x;sigma)) / sigma via Heun (2ª ordem), ao
    longo de `sigmas` — decrescente para amostrar (ruído->dado), crescente
    para "inverter" um dado real em ruído (dado->ruído). A ODE é a mesma nos
    dois sentidos; só muda a direção em que se percorre a sequência."""
    x = x_init.double()
    b = x.shape[0]
    for i in range(len(sigmas) - 1):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        sigma_cur_b = sigma_cur.expand(b)
        d_cur = (x - precond.denoise(net, x.float(), sigma_cur_b.float()).double()) / sigma_cur
        x_next = x + (sigma_next - sigma_cur) * d_cur
        if sigma_next.item() > 1e-6:
            sigma_next_b = sigma_next.expand(b)
            d_next = (x_next - precond.denoise(net, x_next.float(), sigma_next_b.float()).double()) / sigma_next
            x_next = x + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)
        x = x_next
    return x.float()
