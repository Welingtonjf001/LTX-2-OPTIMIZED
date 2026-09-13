"""Conditional Flow Matching / Rectified Flow (Lipman et al. 2022;
Liu et al. 2022 "Flow Straight and Fast") COM precondicionamento — análogo ao
EDM (Karras et al. 2022), adaptado ao caminho reto do flow matching.

Por que o precondicionamento é necessário (achado empírico, não teórico):
a versão "vanilla" (rede prevendo a velocidade bruta x1-x0 diretamente) tem
alvo de treino cuja variância é dominada pelo ruído gaussiano x0 (variância 1
por dimensão), não pelo sinal de dado real (variância sigma_data^2, tipicamente
muito menor). Medido neste diretório: 4000 passos de treino não tiraram essa
versão do platô de loss ~0.95, e a amostra gerada continuava com a estatística
do ruído de entrada, não do dado. A rede troca de alvo (prever x1, o dado
limpo, escala pequena e bem comportada) e ganha um atalho residual que vira
identidade exata em t=1 (onde x_t já É x1) — mesma lógica do c_skip/c_out do
EDM, só que derivada para a interpolação linear do flow matching em vez da
SDE de difusão.
"""
import torch


class FlowPrecond:
    """x1_hat = c_skip(t)*x_t + c_out(t)*F_theta(c_in(t)*x_t, t).

    c_skip(t)=t: em t=1, x_t já é x1 exatamente (atalho perfeito, a rede não
    precisa aprender nada ali). c_skip(t)=0 em t=0: x_t=x0 não carrega
    informação nenhuma sobre x1, a predição é 100% da rede.
    c_in normaliza a entrada pela variância real de x_t neste t (x0~N(0,1),
    x1~dado com desvio sigma_data), mantendo a entrada em escala ~1 sempre.
    """

    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def _scalings(self, t: torch.Tensor):
        sd = self.sigma_data
        var_t = (1 - t) ** 2 + (t * sd) ** 2
        c_in = 1.0 / torch.sqrt(var_t)
        c_skip = t
        c_out = 1 - t
        return c_skip, c_out, c_in

    def predict_x1(self, net, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c_skip, c_out, c_in = self._scalings(t)
        shape = (-1,) + (1,) * (x_t.dim() - 1)
        c_skip, c_out, c_in = c_skip.view(*shape), c_out.view(*shape), c_in.view(*shape)
        f_x = net(c_in * x_t, t)
        return c_skip * x_t + c_out * f_x

    def velocity(self, net, x_t: torch.Tensor, t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        x1_hat = self.predict_x1(net, x_t, t)
        shape = (-1,) + (1,) * (x_t.dim() - 1)
        denom = (1 - t).clamp(min=eps).view(*shape)
        return (x1_hat - x_t) / denom


def sample_logit_normal_t(batch_size: int, device="cpu") -> torch.Tensor:
    """t = sigmoid(N(0,1)) — concentra amostragem em níveis intermediários,
    igual à técnica usada no SD3 (Esser et al. 2024) para flow matching. Perto
    das bordas (t~0 ou t~1) o sinal de treino é mais fraco/degenerado; Uniform
    (0,1) gasta orçamento de treino nessas regiões pouco informativas."""
    return torch.sigmoid(torch.randn(batch_size, device=device))


def fm_loss(net, precond: FlowPrecond, x1: torch.Tensor) -> torch.Tensor:
    b = x1.shape[0]
    t = sample_logit_normal_t(b, device=x1.device)
    x0 = torch.randn_like(x1)
    shape = (-1,) + (1,) * (x1.dim() - 1)
    x_t = (1 - t.view(*shape)) * x0 + t.view(*shape) * x1
    x1_hat = precond.predict_x1(net, x_t, t)
    return ((x1_hat - x1) ** 2).mean()


def make_time_schedule(num_steps: int, device="cpu") -> torch.Tensor:
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device)


def solve_flow_ode(net, precond: FlowPrecond, x_init: torch.Tensor, t_values: torch.Tensor) -> torch.Tensor:
    """Integra dx/dt = v(x,t) via Heun — crescente 0->1 para gerar dado a
    partir de ruído, decrescente 1->0 para inverter (encode)."""
    x = x_init.clone()
    b = x.shape[0]
    for i in range(len(t_values) - 1):
        t_cur, t_next = t_values[i], t_values[i + 1]
        v_cur = precond.velocity(net, x, t_cur.expand(b))
        x_euler = x + (t_next - t_cur) * v_cur
        v_next = precond.velocity(net, x_euler, t_next.expand(b))
        x = x + (t_next - t_cur) * 0.5 * (v_cur + v_next)
    return x
