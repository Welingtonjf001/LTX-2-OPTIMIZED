"""Linha 1 da reanálise: o núcleo hamiltoniano passa a valer quando os EVENTOS
DESCONTÍNUOS são tratados à parte?

`train_dynamics_only.py` mediu que o hamiltoniano puro perde por ~3 ordens de
grandeza para uma rede sem estrutura, e a causa é a colisão: potencial suave
não produz reflexão instantânea. A reformulação propõe um modelo HÍBRIDO:
fluxo contínuo entre eventos + mapa de salto nos eventos, s_{t+} = R(s_{t-}, e_t).

Quatro modelos, parâmetros pareados (~4,4k), mesmas sementes, mesmos dados:

  hamiltoniano   Verlet puro (o perdedor da rodada anterior)
  baseline       MLP residual sem estrutura (o vencedor)
  hibrido        Verlet + guarda APRENDIDA g(q) + salto estruturado: reflexão
                 elástica sobre a superfície g=0. Nada sobre a caixa é dado.
  hibrido_caixa  Verlet + salto por dimensão com limites b APRENDIDOS. Prior
                 forte (caixa alinhada aos eixos): é o TETO.

Treino em 256 trajetórias com estado inicial aleatório; teste em 16 trajetórias
NOVAS e 5x mais longas que o horizonte de treino. Métrica principal: tempo de
previsão válida, comparado a dois pisos -- congelado e inercial (reta sem
colisão, exata até a primeira parede). Modelo que não supera o inercial não
aprendeu a colisão, e as comparações com ele são suprimidas.

HISTÓRICO DA PRIMEIRA VERSÃO (2026-09-13), invalidada por dois defeitos meus:
1. Gatilho SUAVE no forward: `p*(1-2w)` com w entre 0 e 1 encolhe |p| em vez de
   refletir. O hibrido_caixa dissipou energia e empurrou os limites para FORA
   (0,70-0,75 contra 0,55) para fugir dos meio-rebotes. Agora o salto é DURO no
   forward (reflexão exata, |p| preservado) e usa o sigmoide só no backward
   (straight-through).
2. MSE de posição em horizonte longo SATURA: erro pequeno de velocidade desloca
   a fase linearmente, e em 80 passos a previsão fica tão descorrelacionada
   quanto um palpite -- o piso trivial (0,177) já estava nesse limite. Agora a
   métrica principal é o TEMPO DE PREVISÃO VÁLIDA (passos até o erro > limiar).
Achado real daquela rodada, mantido em observação: o baseline EXPLODIU em 2 de
5 sementes nas trajetórias novas (MSE ~1e13); os hamiltonianos não.
"""
import argparse
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hamiltonian import KineticEnergy, PotentialEnergy, SymplecticIntegrator

B_TRUE = 0.55  # parede em 1 - raio(0,45)


def simulate(n: int, steps: int, dt: float, seed: int):
    """Reflexão elástica exata em caixa [-B, B]^2. Devolve estado inicial e a
    trajetória [T, n, 2] de posição e velocidade."""
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(n, 2, generator=g) * 2 - 1) * (B_TRUE * 0.9)
    ang = torch.rand(n, generator=g) * 2 * math.pi
    spd = 0.5 + 0.6 * torch.rand(n, generator=g)
    vel = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * spd[:, None]
    q0, p0 = pos.clone(), vel.clone()
    qs, ps = [], []
    for _ in range(steps):
        pos = pos + dt * vel
        over = pos > B_TRUE
        pos = torch.where(over, 2 * B_TRUE - pos, pos)
        vel = torch.where(over, -vel, vel)
        under = pos < -B_TRUE
        pos = torch.where(under, -2 * B_TRUE - pos, pos)
        vel = torch.where(under, -vel, vel)
        qs.append(pos.clone())
        ps.append(vel.clone())
    return q0, p0, torch.stack(qs), torch.stack(ps)


def gatilho(g: torch.Tensor, k: float) -> torch.Tensor:
    """0/1 exato no forward; gradiente de sigmoid(k*g) no backward."""
    suave = torch.sigmoid(k * g)
    duro = (g > 0).to(g.dtype)
    return duro + (suave - suave.detach())


class Baseline(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4))

    def step(self, q, p, ctx, dt):
        dq, dp = self.net(torch.cat([q, p, ctx], -1)).chunk(2, -1)
        return q + dt * dq, p + dt * dp


class Hamiltoniano(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.flow = SymplecticIntegrator(KineticEnergy(2), PotentialEnergy(2, 1, hidden))

    def step(self, q, p, ctx, dt):
        return self.flow.step(q, p, ctx, dt)


class HibridoAprendido(nn.Module):
    """Fluxo simplético + reflexão elástica sobre a superfície aprendida g=0.

    Distância estimada à superfície: g/|∇g| (primeira ordem). Com o gatilho
    duro, p' = p - 2(p·n)n preserva |p| exatamente no forward.
    """

    def __init__(self, hidden: int, k: float):
        super().__init__()
        self.flow = SymplecticIntegrator(KineticEnergy(2), PotentialEnergy(2, 1, hidden))
        self.guard = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))
        with torch.no_grad():
            self.guard[-1].weight.mul_(0.1)
            self.guard[-1].bias.fill_(-0.1)
        self.k = k

    def step(self, q, p, ctx, dt):
        q, p = self.flow.step(q, p, ctx, dt)
        with torch.enable_grad():
            qg = q if q.requires_grad else q.detach().requires_grad_(True)
            g = self.guard(qg)
            (dg,) = torch.autograd.grad(g.sum(), qg, create_graph=q.requires_grad)
        norma = dg.norm(dim=-1, keepdim=True) + 1e-6
        n = dg / norma
        w = gatilho(g, self.k)
        q = q - w * 2 * (g / norma) * n
        p = p - w * 2 * (p * n).sum(-1, keepdim=True) * n
        return q, p


class HibridoCaixa(nn.Module):
    """Fluxo simplético + reflexão por dimensão com limites b aprendidos."""

    def __init__(self, hidden: int, k: float, b_init: float):
        super().__init__()
        self.flow = SymplecticIntegrator(KineticEnergy(2), PotentialEnergy(2, 1, hidden))
        self.raw_b = nn.Parameter(torch.full((2,), b_init))
        self.k = k

    def step(self, q, p, ctx, dt):
        q, p = self.flow.step(q, p, ctx, dt)
        g = q.abs() - self.raw_b.abs()
        w = gatilho(g, self.k)
        q = q - w * 2 * g * torch.sign(q)
        p = p * (1 - 2 * w)
        return q, p


def n_params(m):
    return sum(t.numel() for t in m.parameters())


def rollout(model, q0, p0, steps, dt):
    ctx = torch.zeros(q0.shape[0], 1, device=q0.device)
    q, p = q0, p0
    qs, ps = [], []
    for _ in range(steps):
        q, p = model.step(q, p, ctx, dt)
        qs.append(q)
        ps.append(p)
    return torch.stack(qs), torch.stack(ps)


def tempo_valido(qh: torch.Tensor, qe: torch.Tensor, limiar: float) -> float:
    """Passos até o erro de posição (L2) passar do limiar, média entre
    trajetórias. Não satura como o MSE de horizonte longo: diferencia um
    modelo que acompanha 60 passos de um que acompanha 5."""
    erro = (qh - qe).norm(dim=-1)                    # [T, n]
    erro = torch.nan_to_num(erro, nan=1e9, posinf=1e9)
    passou = erro > limiar
    T = erro.shape[0]
    primeiro = torch.where(passou.any(0), passou.float().argmax(0),
                           torch.full_like(passou[0], T, dtype=torch.long))
    return primeiro.float().mean().item()


def treinar_avaliar(model, treino, teste, args):
    q0, p0, qt, pt = treino
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    loss = torch.tensor(float("nan"))
    for _ in range(args.iters):
        opt.zero_grad()
        qh, ph = rollout(model, q0, p0, args.horizon, args.dt)
        loss = F.mse_loss(qh, qt) + args.lambda_p * F.mse_loss(ph, pt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    custo = time.time() - t0

    q0e, p0e, qe, pe = teste
    with torch.no_grad():
        qh, ph = rollout(model, q0e, p0e, qe.shape[0], args.dt)
    finito = torch.isfinite(qh).all(dim=(0, 2))       # por trajetória
    erro_q = ((qh - qe) ** 2).mean(dim=(1, 2))
    dentro = erro_q[: args.horizon].mean().item()
    desvio_vel = (ph.norm(dim=-1) - pe.norm(dim=-1)).abs()
    desvio_vel = torch.nan_to_num(desvio_vel, nan=1e9, posinf=1e9).median().item()
    return {
        "perda_treino": loss.item(),
        "dentro": dentro,
        "valido": tempo_valido(qh, qe, args.limiar),
        "vel": desvio_vel,
        "divergentes": int((~finito).sum().item()) + int(((qh.abs() > 10).any(dim=(0, 2)) & finito).sum().item()),
        "treino_s": custo,
    }


def ms(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--test-steps", type=int, default=100)
    # 256, e nao 16: com 16 trajetorias x 20 passos (320 transicoes) nenhum modelo
    # generalizou para estados iniciais novos (RMSE 0,19-0,28 dentro do horizonte,
    # MEDIDO 2026-09-13). Simular e gratis e, na GPU, o custo por iteracao e
    # dominado pelo laco sequencial do rollout, nao pelo tamanho do lote.
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-p", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=50.0, help="nitidez do sigmoide no backward")
    ap.add_argument("--b-init", type=float, default=0.7, help="limite inicial do hibrido_caixa (verdade: 0,55)")
    ap.add_argument("--limiar", type=float, default=0.1, help="erro de posição que encerra a previsão válida")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default="hamiltoniano,baseline,hibrido,hibrido_caixa")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    print(f"dispositivo: {dev}"
          f"{' — ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")

    treino = [t.to(dev) for t in simulate(args.n_train, args.horizon, args.dt, seed=123)]
    teste = [t.to(dev) for t in simulate(args.n_test, args.test_steps, args.dt, seed=456)]
    print(f"treino: {args.n_train} trajetórias x {args.horizon} passos | "
          f"teste: {args.n_test} trajetórias NOVAS x {args.test_steps} passos | limiar {args.limiar}")

    # Dois pisos. Congelado: ninguém se mexe. Inercial: reta com a velocidade
    # inicial, SEM colisão -- bate o congelado até a primeira parede. Um modelo
    # que não supera o inercial não aprendeu a colisão.
    q0e, p0e, qe, _ = teste
    congelado = q0e.unsqueeze(0).expand_as(qe)
    passos = torch.arange(1, qe.shape[0] + 1, device=dev, dtype=qe.dtype).view(-1, 1, 1)
    inercial = q0e.unsqueeze(0) + passos * args.dt * p0e.unsqueeze(0)
    pisos = {
        "congelado": (((congelado - qe) ** 2).mean(dim=(1, 2))[: args.horizon].mean().item(),
                      tempo_valido(congelado, qe, args.limiar)),
        "inercial": (((inercial - qe) ** 2).mean(dim=(1, 2))[: args.horizon].mean().item(),
                     tempo_valido(inercial, qe, args.limiar)),
    }
    for nome, (d, v) in pisos.items():
        print(f"piso {nome:>9}: MSE dentro {d:.5f} | previsão válida {v:.1f} passos")
    print()

    fabricas = {
        "hamiltoniano": lambda: Hamiltoniano(64),
        "baseline": lambda: Baseline(62),
        "hibrido": lambda: HibridoAprendido(44, args.k),
        "hibrido_caixa": lambda: HibridoCaixa(64, args.k, args.b_init),
    }
    nomes = [n for n in args.only.split(",") if n in fabricas]
    res = {n: [] for n in nomes}
    npar = {}

    for seed in range(args.seeds):
        for nome in nomes:
            torch.manual_seed(seed)
            m = fabricas[nome]().to(dev)
            npar[nome] = n_params(m)
            r = treinar_avaliar(m, treino, teste, args)
            if nome == "hibrido_caixa":
                r["b"] = m.raw_b.abs().detach().cpu().tolist()
            res[nome].append(r)
            extra = f"  b={[round(x, 3) for x in r['b']]}" if "b" in r else ""
            print(f"  semente {seed} {nome:>13}: perda_treino={r['perda_treino']:.5f} "
                  f"dentro={r['dentro']:.5f} valido={r['valido']:.1f} vel={r['vel']:.4f} "
                  f"div={r['divergentes']} ({r['treino_s']:.0f}s){extra}", flush=True)

    print("\n" + "=" * 104)
    print(f"{'modelo':>14} {'params':>7} {'perda treino':>13} {'MSE dentro (novas)':>21} "
          f"{'previsão válida':>18} {'|v| mediana':>12} {'diverg.':>8} {'treino':>7}")
    print("-" * 104)
    resumo = {}
    for nome in nomes:
        pt, _ = ms([r["perda_treino"] for r in res[nome]])
        d, dd = ms([r["dentro"] for r in res[nome]])
        v, vd = ms([r["valido"] for r in res[nome]])
        u, _ = ms([r["vel"] for r in res[nome]])
        dv = sum(r["divergentes"] for r in res[nome])
        t, _ = ms([r["treino_s"] for r in res[nome]])
        resumo[nome] = {"valido": (v, vd), "dentro": d}
        print(f"{nome:>14} {npar[nome]:>7,} {pt:>13.5f} {d:>12.5f} ±{dd:<7.5f} "
              f"{v:>9.1f} ±{vd:<6.1f} {u:>12.4f} {dv:>4}/{args.seeds * args.n_test:<3} {t:>6.0f}s")
    print("-" * 104)
    for nome, (d, v) in pisos.items():
        print(f"{'piso ' + nome:>14} {'—':>7} {'':>13} {d:>12.5f} {'':8} {v:>9.1f}")
    print("=" * 104)

    print("\nLeitura:")
    piso_valido = max(v for _, v in pisos.values())
    reprovados = set()
    for nome in nomes:
        v, _ = resumo[nome]["valido"]
        if v <= piso_valido * 1.1:
            reprovados.add(nome)
            print(f"  {nome}: previsão válida ({v:.1f}) não supera o melhor piso "
                  f"({piso_valido:.1f}) — não aprendeu a colisão.")

    def p_troca_sinal(d):
        """Teste pareado bicaudal por troca de sinal: exato até 16 sementes,
        Monte Carlo (20k) acima. Sem scipy, que o venv do Ubuntu não tem."""
        import random
        n = len(d)
        alvo = abs(sum(d) / n) - 1e-12
        if n <= 16:
            ext = 0
            for mask in range(1 << n):
                s = sum(-x if (mask >> i) & 1 else x for i, x in enumerate(d))
                ext += abs(s / n) >= alvo
            return ext / (1 << n)
        rng = random.Random(0)
        ext = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d) / n) >= alvo
                  for _ in range(20000))
        return (ext + 1) / 20001

    # PAREADO por semente. A regra anterior (diferença de médias contra o desvio
    # entre sementes) errou nos DOIS sentidos, MEDIDO 2026-09-13: com 5 sementes
    # disse "acima do ruído" (teste pareado p=0,11); com 20 disse "dentro do
    # ruído" (p=0,005). O pareamento é o que tira a variação entre sementes.
    def compara(a, b):
        if a not in res or b not in res:
            return
        if a in reprovados or b in reprovados:
            print(f"  {a} vs {b}: comparação suprimida (ao menos um não supera o piso).")
            return
        d = [ra["valido"] - rb["valido"] for ra, rb in zip(res[a], res[b])]
        n = len(d)
        media = sum(d) / n
        p = p_troca_sinal(d)
        venc = a if media > 0 else b
        vit = sum(1 for x in d if x != 0 and (x > 0) == (media > 0))
        if n < 6:
            print(f"  {a} vs {b}: com {n} sementes o teste exato não chega a p<0,05 "
                  f"(mínimo {2 / 2 ** n:.3f}); diferença média {media:+.1f} passos, "
                  f"p = {p:.3f}. Rode mais sementes.")
        elif p < 0.05:
            print(f"  {a} vs {b}: {venc} acompanha a trajetória por mais tempo, "
                  f"{abs(media):.1f} passos em média, em {vit}/{n} sementes pareadas "
                  f"(troca de sinal p = {p:.4f}).")
        else:
            print(f"  {a} vs {b}: sem efeito detectável (diferença média {media:+.1f} "
                  f"passos, {vit}/{n} sementes, p = {p:.3f}).")

    compara("hibrido", "baseline")
    compara("hibrido_caixa", "baseline")
    compara("hibrido", "hamiltoniano")
    compara("hamiltoniano", "baseline")


if __name__ == "__main__":
    main()
