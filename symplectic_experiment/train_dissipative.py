"""Linha 4 da reanálise: o ganho da estrutura sobrevive quando o sistema DISSIPA?

`train_hybrid.py` confirmou (20 sementes, p = 0,005) que o hamiltoniano puro
acompanha a trajetória +29% mais tempo que um aluno do mesmo tamanho sem
estrutura. Mas aquele sistema era conservativo, o caso mais favorável. Vídeo
real tem atrito, desaceleração e perda de energia. A reformulação propõe:

    ṗ = -∇V(q) - D·M⁻¹·p,   D ⪰ 0   (sem entrada externa u_t aqui)

Mesmo disco na caixa, agora com arrasto linear: a velocidade decai
exp(-γ·dt) por passo (γ = 0,15; em 100 passos sobra ~30% da velocidade).

Três modelos, parâmetros pareados (~4,5k), mesmas sementes e dados:

  hamiltoniano      conservativo puro; não tem como representar a perda
  baseline          MLP residual sem estrutura
  port_hamiltoniano Verlet conservativo + dissipação D diagonal aprendida,
                    por splitting de Strang: meio passo dissipativo exato
                    (p *= exp(-D/m·dt/2)), Verlet, meio passo dissipativo. O
                    núcleo conservativo continua simplético, e a dissipação
                    entra como fator exato, estável para qualquer dt.

Métricas, pisos, guarda de validade e teste pareado: os mesmos de
`train_hybrid.py` (onde foram validados). O D/m aprendido é comparado ao γ
verdadeiro.
"""
import argparse
import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from hamiltonian import KineticEnergy, PotentialEnergy, SymplecticIntegrator
from train_hybrid import B_TRUE, Baseline, Hamiltoniano, n_params, tempo_valido, treinar_avaliar


def simulate(n: int, steps: int, dt: float, seed: int, gamma: float):
    """Arrasto linear exato + reflexão elástica em [-B, B]^2."""
    g = torch.Generator().manual_seed(seed)
    pos = (torch.rand(n, 2, generator=g) * 2 - 1) * (B_TRUE * 0.9)
    ang = torch.rand(n, generator=g) * 2 * math.pi
    spd = 0.5 + 0.6 * torch.rand(n, generator=g)
    vel = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * spd[:, None]
    q0, p0 = pos.clone(), vel.clone()
    decaimento = math.exp(-gamma * dt)
    qs, ps = [], []
    for _ in range(steps):
        vel = vel * decaimento
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


class PortHamiltoniano(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.kin = KineticEnergy(2)
        self.pot = PotentialEnergy(2, 1, hidden)
        self.flow = SymplecticIntegrator(self.kin, self.pot)
        self.raw_d = nn.Parameter(torch.full((2,), -3.0))   # softplus(-3) ~ 0,049

    @property
    def d(self) -> torch.Tensor:
        return F.softplus(self.raw_d)

    def gamma_efetivo(self) -> torch.Tensor:
        return self.d / self.kin.mass

    def step(self, q, p, ctx, dt):
        fator = torch.exp(-self.gamma_efetivo() * dt / 2)
        p = p * fator
        q, p = self.flow.step(q, p, ctx, dt)
        return q, p * fator


def p_troca_sinal(d):
    """Pareado bicaudal por troca de sinal: exato até 16 sementes, Monte Carlo
    (20k) acima. Mesma lógica validada em train_hybrid.py (p=0,125 com as 5
    sementes, 0,005 com as 20)."""
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


def ms(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--test-steps", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--gamma", type=float, default=0.15, help="arrasto verdadeiro")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-p", type=float, default=0.5)
    ap.add_argument("--limiar", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default="hamiltoniano,baseline,port_hamiltoniano")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    print(f"dispositivo: {dev}"
          f"{' — ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''}")

    treino = [t.to(dev) for t in simulate(args.n_train, args.horizon, args.dt, 123, args.gamma)]
    teste = [t.to(dev) for t in simulate(args.n_test, args.test_steps, args.dt, 456, args.gamma)]
    print(f"treino: {args.n_train} trajetórias x {args.horizon} passos | teste: {args.n_test} "
          f"NOVAS x {args.test_steps} passos | gamma {args.gamma} "
          f"(sobra {math.exp(-args.gamma * args.dt * args.test_steps):.0%} da velocidade no fim)")

    q0e, p0e, qe, _ = teste
    congelado = q0e.unsqueeze(0).expand_as(qe)
    passos = torch.arange(1, qe.shape[0] + 1, device=dev, dtype=qe.dtype).view(-1, 1, 1)
    inercial = q0e.unsqueeze(0) + passos * args.dt * p0e.unsqueeze(0)
    pisos = {
        "congelado": tempo_valido(congelado, qe, args.limiar),
        "inercial": tempo_valido(inercial, qe, args.limiar),
    }
    for nome, v in pisos.items():
        print(f"piso {nome:>9}: previsão válida {v:.1f} passos")
    print()

    fabricas = {
        "hamiltoniano": lambda: Hamiltoniano(64),
        "baseline": lambda: Baseline(62),
        "port_hamiltoniano": lambda: PortHamiltoniano(64),
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
            if nome == "port_hamiltoniano":
                r["gamma"] = m.gamma_efetivo().detach().cpu().tolist()
            res[nome].append(r)
            extra = f"  gamma_aprendido={[round(x, 3) for x in r['gamma']]}" if "gamma" in r else ""
            print(f"  semente {seed} {nome:>17}: perda_treino={r['perda_treino']:.5f} "
                  f"dentro={r['dentro']:.5f} valido={r['valido']:.1f} vel={r['vel']:.4f} "
                  f"div={r['divergentes']} ({r['treino_s']:.0f}s){extra}", flush=True)

    print("\n" + "=" * 100)
    print(f"{'modelo':>18} {'params':>7} {'MSE dentro (novas)':>21} {'previsão válida':>18} "
          f"{'|v| mediana':>12} {'diverg.':>9} {'treino':>7}")
    print("-" * 100)
    for nome in nomes:
        d, dd = ms([r["dentro"] for r in res[nome]])
        v, vd = ms([r["valido"] for r in res[nome]])
        u, _ = ms([r["vel"] for r in res[nome]])
        dv = sum(r["divergentes"] for r in res[nome])
        t, _ = ms([r["treino_s"] for r in res[nome]])
        print(f"{nome:>18} {npar[nome]:>7,} {d:>12.5f} ±{dd:<7.5f} {v:>9.1f} ±{vd:<6.1f} "
              f"{u:>12.4f} {dv:>4}/{args.seeds * args.n_test:<4} {t:>6.0f}s")
    print("-" * 100)
    for nome, v in pisos.items():
        print(f"{'piso ' + nome:>18} {'—':>7} {'':>21} {v:>9.1f}")
    print("=" * 100)
    if "port_hamiltoniano" in res:
        gs = [g for r in res["port_hamiltoniano"] for g in r["gamma"]]
        print(f"gamma aprendido pelo port_hamiltoniano: média {statistics.mean(gs):.3f} "
              f"(verdadeiro {args.gamma})")

    print("\nLeitura:")
    piso_valido = max(pisos.values())
    reprovados = set()
    for nome in nomes:
        indiv = [r["valido"] for r in res[nome] if r["valido"] <= piso_valido]
        media = statistics.mean(r["valido"] for r in res[nome])
        if media <= piso_valido * 1.1:
            reprovados.add(nome)
            print(f"  {nome}: previsão válida média ({media:.1f}) não supera o piso "
                  f"({piso_valido:.1f}) — não aprendeu a dinâmica.")
        elif indiv:
            print(f"  {nome}: {len(indiv)} semente(s) individualmente abaixo do piso {indiv}.")

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
            print(f"  {a} vs {b}: com {n} sementes o teste exato não chega a p<0,05; "
                  f"diferença média {media:+.1f}, p = {p:.3f}. Rode mais sementes.")
        elif p < 0.05:
            print(f"  {a} vs {b}: {venc} acompanha por mais tempo, {abs(media):.1f} passos "
                  f"em média, {vit}/{n} sementes (troca de sinal p = {p:.4f}).")
        else:
            print(f"  {a} vs {b}: sem efeito detectável (diferença média {media:+.1f}, "
                  f"{vit}/{n} sementes, p = {p:.3f}).")

    compara("port_hamiltoniano", "baseline")
    compara("port_hamiltoniano", "hamiltoniano")
    compara("hamiltoniano", "baseline")


if __name__ == "__main__":
    main()
