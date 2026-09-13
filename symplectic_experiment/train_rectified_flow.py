"""Treina o campo de velocidade (agora com precondicionamento — ver
rectified_flow.py) via Conditional Flow Matching sobre a mesma distribuição
de vídeos sintéticos usada em train_pf_ode.py, e roda o MESMO teste de
round-trip + diagnóstico anti-falso-positivo, para comparar de forma justa
com a PF-ODE quantos passos cada um precisa.

Histórico (ver README.md): a versão SEM precondicionamento (rede prevendo a
velocidade bruta x1-x0) não convergiu nem com 4000 passos de treino — a
amostra gerada continuava com estatística de ruído puro. Esta versão corrige
isso reparametrizando a rede para prever x1 diretamente.
"""
import argparse

import torch

from rectified_flow import FlowPrecond, fm_loss, make_time_schedule, solve_flow_ode
from velocity_model import ToyVelocityNet
from toy_video_distribution import flatten_videos, sample_toy_video_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    torch.manual_seed(0)
    data_dim = args.num_frames * 3 * args.height * args.width

    calib = flatten_videos(sample_toy_video_batch(256, args.num_frames, args.height, args.width))
    sigma_data = calib.std().item()
    print(f"sigma_data estimado = {sigma_data:.4f}  (dim={data_dim})")

    net = ToyVelocityNet(data_dim=data_dim)
    precond = FlowPrecond(sigma_data=sigma_data)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for step in range(args.train_steps):
        videos = sample_toy_video_batch(args.batch_size, args.num_frames, args.height, args.width)
        x1 = flatten_videos(videos)
        optimizer.zero_grad()
        loss = fm_loss(net, precond, x1)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.train_steps - 1:
            print(f"train step {step:4d}  fm_loss={loss.item():.4f}")

    print("\n--- Round-trip determinístico (dado -> ruído -> dado), Rectified Flow ---")
    test_videos = sample_toy_video_batch(8, args.num_frames, args.height, args.width)
    x1_real = flatten_videos(test_videos)

    for num_steps in (1, 2, 5, 10, 30):
        t_decode = make_time_schedule(num_steps)
        t_encode = torch.flip(t_decode, dims=[0])
        with torch.no_grad():
            x0_est = solve_flow_ode(net, precond, x1_real, t_encode)
            x1_recon = solve_flow_ode(net, precond, x0_est, t_decode)
        err = (x1_recon - x1_real).abs().mean().item()
        print(f"passos={num_steps:3d}  erro L1 de round-trip = {err:.4f}")

    print("\n--- Diagnóstico: x0 estimado (encode) parece ruído gaussiano? ---")
    t_encode_full = torch.flip(make_time_schedule(30), dims=[0])
    with torch.no_grad():
        x0_est_diag = solve_flow_ode(net, precond, x1_real, t_encode_full)
    print(
        f"x0 estimado (30 passos): media={x0_est_diag.mean().item():.4f} "
        f"std={x0_est_diag.std().item():.4f}  (N(0,I) real teria media~0, std~1)"
    )

    print("\n--- Diagnóstico: geração a partir de ruído puro (não round-trip) ---")
    torch.manual_seed(123)
    x0_fresh = torch.randn(8, data_dim)
    t_decode_full = make_time_schedule(30)
    with torch.no_grad():
        x1_fresh = solve_flow_ode(net, precond, x0_fresh, t_decode_full)
    print(
        f"amostra gerada:  media={x1_fresh.mean().item():.4f} std={x1_fresh.std().item():.4f} "
        f"min={x1_fresh.min().item():.4f} max={x1_fresh.max().item():.4f}"
    )
    print(
        f"dado real:       media={x1_real.mean().item():.4f} std={x1_real.std().item():.4f} "
        f"min={x1_real.min().item():.4f} max={x1_real.max().item():.4f}"
    )

    print("\nCompare com train_pf_ode.py nos mesmos passos (5/10/30). Só compare")
    print("de verdade se o diagnóstico acima também passar (std do x0 estimado")
    print("perto de 1, amostra gerada com estatística parecida com o dado real)")
    print("— round-trip baixo sem isso é o mesmo artefato degenerado já visto")
    print("na versão sem precondicionamento.")


if __name__ == "__main__":
    main()
