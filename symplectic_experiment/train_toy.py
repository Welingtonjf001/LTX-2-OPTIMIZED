"""Sanity check: o integrador simplético + decoder INR conseguem aprender a
reconstruir um vídeo sintético de brinquedo (disco quicando)?

Isto NÃO treina sobre pesos do LTX nem usa texto/áudio condicionante — ver
README.md deste diretório para o motivo. q0/p0 são parâmetros livres
otimizados junto com a rede (auto-decoder de uma sequência só), porque não há
encoder aqui: o objetivo é só validar a mecânica matemática.
"""
import argparse

import torch
import torch.nn.functional as F

from hamiltonian import KineticEnergy, PotentialEnergy, SymplecticIntegrator, symplectic_residual_loss
from inr_decoder import ContinuousVideoINRDecoder, make_coord_grid
from toy_dataset import generate_bouncing_ball_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--q-dim", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--lambda-h", type=float, default=0.05)
    parser.add_argument("--lambda-symp", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(0)

    video, x0, v0 = generate_bouncing_ball_video(args.num_frames, args.height, args.width, dt=args.dt)
    video = video.to(device)

    ctx_dim = 1
    ctx = torch.zeros(1, ctx_dim, device=device)

    kinetic = KineticEnergy(dim=args.q_dim).to(device)
    potential = PotentialEnergy(q_dim=args.q_dim, ctx_dim=ctx_dim).to(device)
    integrator = SymplecticIntegrator(kinetic, potential).to(device)
    decoder = ContinuousVideoINRDecoder(q_dim=args.q_dim).to(device)

    # q0/p0: parâmetros livres. Inicializa as duas primeiras dimensões com o
    # estado físico real (posição/velocidade) para dar ao otimizador um ponto
    # de partida informativo; o resto é ruído pequeno.
    q0 = torch.zeros(1, args.q_dim, device=device)
    q0[0, :2] = x0
    p0 = torch.zeros(1, args.q_dim, device=device)
    p0[0, :2] = v0
    q0 = torch.nn.Parameter(q0 + 0.01 * torch.randn_like(q0))
    p0 = torch.nn.Parameter(p0 + 0.01 * torch.randn_like(p0))

    grid = make_coord_grid(args.height, args.width, batch_size=1, device=device)

    params = (
        list(kinetic.parameters())
        + list(potential.parameters())
        + list(decoder.parameters())
        + [q0, p0]
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    for step in range(args.steps):
        optimizer.zero_grad()
        q, p = q0, p0
        h0 = (kinetic(p) + potential(q, ctx)).detach()

        loss_l1 = 0.0
        loss_h = 0.0
        loss_symp = 0.0
        for t in range(args.num_frames):
            q, p = integrator.step(q, p, ctx, dt=args.dt)

            h_t = kinetic(p) + potential(q, ctx)
            loss_h = loss_h + F.mse_loss(h_t, h0)

            if t % 4 == 0:
                loss_symp = loss_symp + symplectic_residual_loss(integrator, q, p, ctx, args.dt)

            pred = decoder(q, grid).view(1, args.height, args.width, 3).permute(0, 3, 1, 2)
            target = video[t].unsqueeze(0)
            loss_l1 = loss_l1 + F.l1_loss(pred, target)

        loss_l1 = loss_l1 / args.num_frames
        loss_h = loss_h / args.num_frames
        loss_symp = loss_symp / max(1, args.num_frames // 4)

        total = loss_l1 + args.lambda_h * loss_h + args.lambda_symp * loss_symp
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if step % 20 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}  total={total.item():.4f}  "
                f"l1={loss_l1.item():.4f}  h_cons={loss_h.item():.5f}  symp={loss_symp.item():.5f}"
            )

    print("\nConcluído. l1 baixo = o integrador+decoder conseguem reconstruir a "
          "trajetória de brinquedo. Isto valida só a mecânica de código — ver "
          "README.md para os limites desta conclusão.")


if __name__ == "__main__":
    main()
