"""Treina um score model real (EDM denoising score matching) sobre a
distribuição de vídeos sintéticos, depois demonstra a propriedade que a
tentativa hamiltoniana prometia mas não conseguia sustentar de forma válida:
uma trajetória determinística e INVERTÍVEL entre dado e ruído (round-trip
encode->decode via a mesma PF-ODE, medindo erro de reconstrução por nº de
passos de integração).

Isto NÃO usa pesos do LTX. É a técnica real (Karras et al. 2022) em miniatura,
sobre dados de brinquedo — ver README.md.
"""
import argparse

import torch

from pf_ode import EDMPrecond, edm_loss, make_sigma_schedule, solve_pf_ode
from score_model import ToyScoreNet
from toy_video_distribution import flatten_videos, sample_toy_video_batch, unflatten_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(0)

    data_dim = args.num_frames * 3 * args.height * args.width

    # Estima sigma_data a partir de uma amostra grande da distribuição real,
    # como o EDM recomenda (não é hiperparâmetro livre).
    calib = flatten_videos(sample_toy_video_batch(256, args.num_frames, args.height, args.width))
    sigma_data = calib.std().item()
    print(f"sigma_data estimado = {sigma_data:.4f}  (dim={data_dim})")

    net = ToyScoreNet(data_dim=data_dim)
    precond = EDMPrecond(sigma_data=sigma_data)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for step in range(args.train_steps):
        videos = sample_toy_video_batch(args.batch_size, args.num_frames, args.height, args.width)
        x = flatten_videos(videos)
        optimizer.zero_grad()
        loss = edm_loss(net, precond, x)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.train_steps - 1:
            print(f"train step {step:4d}  edm_loss={loss.item():.4f}")

    print("\n--- Round-trip determinístico (dado -> ruído -> dado) ---")
    print("Ao contrário da versão hamiltoniana, esta invertibilidade é uma")
    print("propriedade matemática real da PF-ODE, não uma alegação não-verificada.\n")

    test_videos = sample_toy_video_batch(8, args.num_frames, args.height, args.width)
    x0 = flatten_videos(test_videos)

    for num_steps in (5, 10, 30):
        sigmas_decode = make_sigma_schedule(args.sigma_min, args.sigma_max, num_steps)
        sigmas_encode = torch.flip(sigmas_decode, dims=[0])
        sigmas_decode_full = torch.cat([sigmas_decode, torch.zeros(1, dtype=torch.float64)])

        with torch.no_grad():
            x_noise = solve_pf_ode(net, precond, x0, sigmas_encode)
            x_recon = solve_pf_ode(net, precond, x_noise, sigmas_decode_full)

        err = (x_recon - x0).abs().mean().item()
        print(f"passos={num_steps:3d}  erro L1 de round-trip = {err:.4f}")

    print("\nErro caindo com mais passos = a ODE é de fato determinística e")
    print("invertível nesta implementação. Continua sendo dado de brinquete,")
    print("não prova nada sobre qualidade de vídeo real gerado do zero.")

    # Mesmo diagnóstico anti-falso-positivo aplicado no experimento de
    # rectified flow: round-trip baixo também aparece de forma degenerada se
    # a rede aprendeu pouco. Confere se x_noise (encode) tem a escala certa
    # (deveria ter std ~ sigma_max) e se gerar de ruído puro dá algo plausível.
    print("\n--- Diagnóstico: x_noise estimado (encode) tem a escala certa? ---")
    sigmas_decode_full30 = make_sigma_schedule(args.sigma_min, args.sigma_max, 30)
    sigmas_encode_full30 = torch.flip(sigmas_decode_full30, dims=[0])
    with torch.no_grad():
        x_noise_diag = solve_pf_ode(net, precond, x0, sigmas_encode_full30)
    print(
        f"x_noise estimado (30 passos): media={x_noise_diag.mean().item():.4f} "
        f"std={x_noise_diag.std().item():.4f}  (esperado: std ~ sigma_max={args.sigma_max})"
    )

    print("\n--- Diagnóstico: geração a partir de ruído puro (não round-trip) ---")
    torch.manual_seed(123)
    x_pure_noise = torch.randn(8, data_dim) * args.sigma_max
    sigmas_decode_full30_ext = torch.cat([sigmas_decode_full30, torch.zeros(1, dtype=torch.float64)])
    with torch.no_grad():
        x_fresh = solve_pf_ode(net, precond, x_pure_noise, sigmas_decode_full30_ext)
    print(
        f"amostra gerada:  media={x_fresh.mean().item():.4f} std={x_fresh.std().item():.4f} "
        f"min={x_fresh.min().item():.4f} max={x_fresh.max().item():.4f}"
    )
    print(
        f"dado real:       media={x0.mean().item():.4f} std={x0.std().item():.4f} "
        f"min={x0.min().item():.4f} max={x0.max().item():.4f}"
    )


if __name__ == "__main__":
    main()
