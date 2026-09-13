"""Compara núcleo hamiltoniano vs. preditor compacto SEM estrutura física,
sob condições idênticas (mesmo dado, mesma semente, mesmo decoder, mesmo
orçamento de treino), medindo duas coisas:

  1. reconstrução DENTRO do horizonte de treino
  2. reconstrução EXTRAPOLANDO além dele  <- o teste que discrimina

Se a estrutura física vale alguma coisa, é na extrapolação que ela aparece:
dentro do horizonte de treino, qualquer rede com capacidade suficiente decora.
A pergunta real não é "hamiltoniano é melhor que difusão", é "hamiltoniano é
melhor que um aluno igualmente pequeno e sem restrições".

Roda em GPU se houver. Dado sintético (disco quicando) — ver README.md para
o que este experimento NÃO prova.
"""
import argparse

import torch
import torch.nn.functional as F

from baseline_predictor import UnconstrainedPredictor
from hamiltonian import KineticEnergy, PotentialEnergy, SymplecticIntegrator
from inr_decoder import ContinuousVideoINRDecoder, make_coord_grid
from toy_dataset import generate_bouncing_ball_video


def n_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def build_hamiltonian(q_dim, ctx_dim, device):
    kinetic = KineticEnergy(dim=q_dim).to(device)
    potential = PotentialEnergy(q_dim=q_dim, ctx_dim=ctx_dim).to(device)
    integrator = SymplecticIntegrator(kinetic, potential).to(device)
    return integrator, n_params(kinetic, potential)


def build_baseline(q_dim, ctx_dim, device, hidden_dim):
    model = UnconstrainedPredictor(q_dim=q_dim, ctx_dim=ctx_dim, hidden_dim=hidden_dim).to(device)
    return model, n_params(model)


def run_one(name, stepper, n_par, video, args, device):
    """Treina stepper+decoder nos primeiros train_frames e avalia até o fim."""
    torch.manual_seed(args.seed)
    total_frames = video.shape[0]
    h, w = video.shape[2], video.shape[3]

    decoder = ContinuousVideoINRDecoder(q_dim=args.q_dim).to(device)
    q0 = torch.nn.Parameter(0.01 * torch.randn(1, args.q_dim, device=device))
    p0 = torch.nn.Parameter(0.01 * torch.randn(1, args.q_dim, device=device))
    ctx = torch.zeros(1, args.ctx_dim, device=device)
    grid = make_coord_grid(h, w, batch_size=1, device=device)

    params = list(stepper.parameters()) + list(decoder.parameters()) + [q0, p0]
    opt = torch.optim.Adam(params, lr=args.lr)

    for step in range(args.steps):
        opt.zero_grad()
        q, p = q0, p0
        loss = 0.0
        for t in range(args.train_frames):
            q, p = stepper.step(q, p, ctx, dt=args.dt)
            pred = decoder(q, grid).view(1, h, w, 3).permute(0, 3, 1, 2)
            loss = loss + F.l1_loss(pred, video[t].unsqueeze(0))
        loss = loss / args.train_frames
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  [{name}] step {step:4d}  l1_treino={loss.item():.4f}")

    # Avaliação: rollout completo, separando horizonte de treino de extrapolação
    with torch.no_grad():
        q, p = q0, p0
        errs = []
        for t in range(total_frames):
            q, p = stepper.step(q, p, ctx, dt=args.dt)
            pred = decoder(q, grid).view(1, h, w, 3).permute(0, 3, 1, 2)
            errs.append(F.l1_loss(pred, video[t].unsqueeze(0)).item())

    dentro = sum(errs[: args.train_frames]) / args.train_frames
    fora_n = total_frames - args.train_frames
    fora = sum(errs[args.train_frames :]) / max(1, fora_n)

    # Variação temporal das PREDIÇÕES: se ~0, o modelo esta emitindo uma imagem
    # quase estatica e nao modelou movimento nenhum, por melhor que a L1 pareca.
    with torch.no_grad():
        q, p = q0, p0
        preds = []
        for _ in range(total_frames):
            q, p = stepper.step(q, p, ctx, dt=args.dt)
            preds.append(decoder(q, grid).view(1, h, w, 3).permute(0, 3, 1, 2))
        preds = torch.cat(preds, dim=0)
        tv_pred = (preds[1:] - preds[:-1]).abs().mean().item()

    return {"nome": name, "params": n_par, "dentro": dentro, "fora": fora, "tv": tv_pred}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--train-frames", type=int, default=16)
    ap.add_argument("--total-frames", type=int, default=32)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--q-dim", type=int, default=8)
    ap.add_argument("--ctx-dim", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    # 118 iguala a contagem de parametros do par (T,V) hamiltoniano (~17,9k)
    # dentro de ~0,3%. Com o default 128 o baseline teria 16% A MAIS, o que
    # seria um confundidor: uma vitoria dele poderia ser so capacidade extra.
    ap.add_argument("--baseline-hidden", type=int, default=118)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dispositivo: {device}"
          f"{' — ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"treino nos primeiros {args.train_frames} frames, "
          f"extrapolação nos {args.total_frames - args.train_frames} seguintes\n")

    video, _, _ = generate_bouncing_ball_video(
        args.total_frames, args.height, args.width, dt=args.dt
    )
    video = video.to(device)

    resultados = []

    print("=== A) núcleo hamiltoniano (Verlet simplético, corrigido) ===")
    torch.manual_seed(args.seed)
    ham, n_ham = build_hamiltonian(args.q_dim, args.ctx_dim, device)
    resultados.append(run_one("hamiltoniano", ham, n_ham, video, args, device))

    print("\n=== B) baseline compacto SEM estrutura física ===")
    torch.manual_seed(args.seed)
    base, n_base = build_baseline(args.q_dim, args.ctx_dim, device, args.baseline_hidden)
    resultados.append(run_one("baseline", base, n_base, video, args, device))

    # PISO TRIVIAL: L1 de um preditor que desenha so o fundo, sem bola e sem
    # movimento. Qualquer modelo que nao bata isso com folga NAO APRENDEU NADA,
    # e comparar dois modelos assim entre si nao significa coisa alguma.
    bg = torch.tensor([0.08, 0.08, 0.10], device=device).view(1, 3, 1, 1).expand_as(video)
    piso = (video - bg).abs().mean().item()
    tv_real = (video[1:] - video[:-1]).abs().mean().item()

    print("\n" + "=" * 72)
    print(f"{'modelo':>14} {'params':>9} {'L1 dentro':>11} {'L1 extrapol.':>14} {'var.temporal':>14}")
    print("-" * 72)
    for r in resultados:
        print(f"{r['nome']:>14} {r['params']:>9,} {r['dentro']:>11.4f} "
              f"{r['fora']:>14.4f} {r['tv']:>14.4f}")
    print("-" * 72)
    print(f"{'PISO TRIVIAL':>14} {'—':>9} {piso:>11.4f} {piso:>14.4f} {0.0:>14.4f}")
    print(f"{'(video real)':>14} {'—':>9} {'—':>11} {'—':>14} {tv_real:>14.4f}")
    print("=" * 72)

    h_, b_ = resultados[0], resultados[1]

    # Guarda de sanidade ANTES de qualquer comparacao entre os dois.
    degenerado = [r for r in resultados if r["dentro"] >= piso * 0.9]
    if degenerado:
        print("\nRESULTADO INVALIDO — nao compare os dois modelos:")
        for r in degenerado:
            print(f"  '{r['nome']}' tem L1={r['dentro']:.4f} contra piso trivial "
                  f"{piso:.4f} (desenhar so o fundo).")
        print("  Modelo que nao bate o piso trivial colapsou na solucao estatica;")
        print("  um 'empate' entre dois modelos assim e empate em nao ter aprendido.")
        print("  Aumente o raio da bola, pondere a perda nas regioes em movimento,")
        print("  ou treine por mais passos antes de ler qualquer conclusao.")
        return

    print("\nLeitura honesta:")
    if h_["fora"] < b_["fora"] * 0.9:
        print("  Hamiltoniano extrapola MELHOR -> a estrutura física pagou por si.")
    elif b_["fora"] < h_["fora"] * 0.9:
        print("  Baseline extrapola MELHOR -> a estrutura física ATRAPALHOU aqui.")
    else:
        print("  Empate (diferença < 10%) -> a estrutura física NÃO se justifica")
        print("  nesta tarefa; o ganho seria de ter uma rede pequena, não de ser")
        print("  hamiltoniana.")
    print("\nLembrete: dado de brinquedo, um sistema quase-conservativo — o caso")
    print("MAIS favorável possível ao hamiltoniano. Vídeo real tem dissipação,")
    print("cortes e ações externas; um empate aqui é um resultado fraco para a tese.")


if __name__ == "__main__":
    main()
