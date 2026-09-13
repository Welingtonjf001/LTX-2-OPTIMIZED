"""Comparação LIMPA: núcleo hamiltoniano vs. preditor compacto sem estrutura,
em espaço de estado puro — SEM decoder, SEM aprender o espaço latente.

Por que esta versão existe: as duas tentativas anteriores (train_toy.py,
train_compare.py) aprendiam dinâmica + espaço latente + renderizador ao mesmo
tempo, e colapsavam na solução estática (decoder constante -> gradiente em q
morre -> a dinâmica nunca aprende). MEDIDO: predições com variação temporal
0,0001-0,0003 contra 0,0060 do vídeo real, e L1 pior que o piso trivial de
"desenhar só o fundo". Resultado ininterpretável.

Aqui o estado é ancorado no estado físico verdadeiro (posição, velocidade) do
simulador, e a ÚNICA coisa comparada é o propagador temporal. Sem renderização
não existe a solução degenerada. Isto segue a recomendação de não substituir
dinâmica, latente e renderizador simultaneamente.

Roda várias sementes e reporta média ± desvio: diferenças de uma corrida só
foram ruído o tempo todo neste projeto.
"""
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from hamiltonian import KineticEnergy, PotentialEnergy, SymplecticIntegrator


def true_trajectory(num_steps: int, dt: float = 0.08, radius: float = 0.45):
    """Estado verdadeiro (pos, vel) do disco quicando, mesma física do
    toy_dataset, mas devolvendo a trajetória inteira em vez de frames."""
    pos = torch.tensor([-0.3, 0.2])
    vel = torch.tensor([0.9, 0.6])
    states = []
    for _ in range(num_steps):
        for d in range(2):
            if pos[d] + radius > 1.0 or pos[d] - radius < -1.0:
                vel[d] = -vel[d]
        pos = (pos + dt * vel).clamp(-1.0 + radius, 1.0 - radius)
        states.append(torch.cat([pos, vel]).clone())
    return torch.stack(states)  # [T, 4]


class UnconstrainedDynamics(nn.Module):
    """(q,p) -> (q,p) residual, sem separação T/V, sem simplético."""

    def __init__(self, dim: int, ctx_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def step(self, q, p, ctx, dt):
        dq, dp = self.net(torch.cat([q, p, ctx], -1)).chunk(2, -1)
        return q + dt * dq, p + dt * dp


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_eval(stepper, traj, args, seed):
    torch.manual_seed(seed)
    device = traj.device
    dim = 2
    ctx = torch.zeros(1, args.ctx_dim, device=device)

    q_true = traj[:, :2].unsqueeze(1)   # [T,1,2]
    p_true = traj[:, 2:].unsqueeze(1)

    q0 = q_true[0].clone()
    p0 = p_true[0].clone()

    opt = torch.optim.Adam(stepper.parameters(), lr=args.lr)
    K = args.train_steps_horizon

    for it in range(args.iters):
        opt.zero_grad()
        q, p = q0, p0
        loss = 0.0
        for t in range(K):
            q, p = stepper.step(q, p, ctx, dt=args.dt)
            loss = loss + F.mse_loss(q, q_true[t]) + args.lambda_p * F.mse_loss(p, p_true[t])
        loss = loss / K
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stepper.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        q, p = q0, p0
        errs = []
        for t in range(traj.shape[0]):
            q, p = stepper.step(q, p, ctx, dt=args.dt)
            errs.append(F.mse_loss(q, q_true[t]).item())

    dentro = sum(errs[:K]) / K
    fora = sum(errs[K:]) / max(1, len(errs) - K)
    return dentro, fora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--train-steps-horizon", type=int, default=20)
    ap.add_argument("--total-steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-p", type=float, default=0.5)
    ap.add_argument("--ctx-dim", type=int, default=1)
    ap.add_argument("--ham-hidden", type=int, default=64)
    ap.add_argument("--base-hidden", type=int, default=62)  # pareia params (~4,5k)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    traj = true_trajectory(args.total_steps, dt=args.dt).to(device)
    print(f"trajetória: {args.total_steps} passos, treino nos primeiros "
          f"{args.train_steps_horizon}, extrapolação nos demais")

    # Piso trivial: prever que nada se move (estado congelado no inicial).
    q_true = traj[:, :2]
    piso = F.mse_loss(q_true[0].expand_as(q_true), q_true).item()
    print(f"piso trivial (estado congelado): MSE = {piso:.5f}\n")

    res = {"hamiltoniano": {"dentro": [], "fora": []},
           "baseline": {"dentro": [], "fora": []}}
    n_par = {}

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        kin = KineticEnergy(dim=2).to(device)
        pot = PotentialEnergy(q_dim=2, ctx_dim=args.ctx_dim, hidden_dim=args.ham_hidden).to(device)
        ham = SymplecticIntegrator(kin, pot).to(device)
        n_par["hamiltoniano"] = n_params(kin) + n_params(pot)
        d, f = train_eval(ham, traj, args, seed)
        res["hamiltoniano"]["dentro"].append(d)
        res["hamiltoniano"]["fora"].append(f)

        torch.manual_seed(seed)
        base = UnconstrainedDynamics(2, args.ctx_dim, args.base_hidden).to(device)
        n_par["baseline"] = n_params(base)
        d, f = train_eval(base, traj, args, seed)
        res["baseline"]["dentro"].append(d)
        res["baseline"]["fora"].append(f)
        print(f"  semente {seed}: ham_fora={res['hamiltoniano']['fora'][-1]:.5f}  "
              f"base_fora={res['baseline']['fora'][-1]:.5f}")

    print("\n" + "=" * 74)
    print(f"{'modelo':>14} {'params':>8} {'MSE dentro':>20} {'MSE extrapolação':>24}")
    print("-" * 74)
    for k in ("hamiltoniano", "baseline"):
        d, f = res[k]["dentro"], res[k]["fora"]
        print(f"{k:>14} {n_par[k]:>8,} "
              f"{statistics.mean(d):>12.5f} ±{(statistics.stdev(d) if len(d)>1 else 0):<7.5f}"
              f"{statistics.mean(f):>16.5f} ±{(statistics.stdev(f) if len(f)>1 else 0):<7.5f}")
    print("-" * 74)
    print(f"{'PISO TRIVIAL':>14} {'—':>8} {piso:>12.5f} {'':<8}{piso:>16.5f}")
    print("=" * 74)

    hm = statistics.mean(res["hamiltoniano"]["fora"])
    bm = statistics.mean(res["baseline"]["fora"])
    hs = statistics.stdev(res["hamiltoniano"]["fora"]) if args.seeds > 1 else 0.0
    bs = statistics.stdev(res["baseline"]["fora"]) if args.seeds > 1 else 0.0

    print("\nLeitura honesta:")
    if min(hm, bm) >= piso * 0.9:
        print("  INVÁLIDO: nenhum modelo bate o piso trivial — não compare.")
        return
    dif = abs(hm - bm)
    ruido = max(hs, bs)
    if dif < ruido:
        print(f"  Diferença ({dif:.5f}) MENOR que o desvio entre sementes "
              f"({ruido:.5f}):")
        print("  não há efeito detectável. A estrutura hamiltoniana não se")
        print("  justifica nesta tarefa — que é o caso MAIS favorável possível")
        print("  (sistema quase-conservativo, sem cortes, sem dissipação).")
    elif hm < bm:
        print(f"  Hamiltoniano extrapola melhor ({hm:.5f} vs {bm:.5f}), "
              f"diferença acima do ruído entre sementes.")
    else:
        print(f"  Baseline extrapola melhor ({bm:.5f} vs {hm:.5f}), "
              f"diferença acima do ruído entre sementes.")


if __name__ == "__main__":
    main()
