"""Teste standalone (fora do pipeline de producao): plano longo vs plano
dividido, no MiniMax H3 -- ver conversa 2026-09-03, pergunta "limitar ou
deixar variavel". NAO integra em render_shots.py ainda; e um teste isolado
pra decidir SE vale a pena antes de mexer em codigo de producao.

METODO
------
Pega UM prompt de teste (por padrao, o mesmo plano 1 da cena Lyra/Thoren --
16,62s, o que gerou o outlier medido hoje) e gera das DUAS formas:

  A) inteiro   -- 1 chamada ao MiniMax H3, duration_seconds=16.6, como ja
                  acontece em producao hoje.
  B) dividido  -- N sub-planos de ate `--max-seconds` cada, ENCADEADOS: cada
                  sub-plano usa dois ref_images -- a character sheet (mesma
                  ancora de identidade que a rota de consistencia ja usa) E o
                  ULTIMO FRAME do sub-plano anterior (continuidade de
                  movimento/cenario, mesmo principio do continuous_chain.py
                  pro LTX -- MEMORIAL 3.47/3.49). O primeiro sub-plano usa so
                  a sheet (nao ha frame anterior).

O QUE ISTO NAO RESOLVE (de proposito, fora do escopo deste teste)
-------------------------------------------------------------------
Dividir a FALA continua em pedacos corretamente (qual trecho do dialogo cabe
em cada sub-plano) exige consciencia de roteiro que este teste nao tem --
aqui cada sub-plano recebe o MESMO prompt de video inteiro, e o que se mede
e SO: tempo de geracao e (por inspecao visual, nao automatica) se a
consistencia se sustenta atraves dos cortes. Pra dialogo de verdade dividido
directinho, a decisao de ONDE cortar a fala fica pra quem revisar o
resultado -- nao e algo que este script tenta adivinhar.

SAIDA
-----
outputs/_test_duration_cap/inteiro/clip.mp4   (opcao A)
outputs/_test_duration_cap/dividido/subN.mp4  (opcao B, cada sub-plano)
outputs/_test_duration_cap/dividido/final.mp4 (opcao B, concatenado)
outputs/_test_duration_cap/relatorio.json     (tempos de cada chamada)

CLI:
    .venv/Scripts/python.exe _test_minimax_duration_cap.py \
        --sheet-a caminho/LYRA.png --max-seconds 6 \
        --prompt "..." --duration-seconds 16.62
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_PROMPT = (
    "Cinematic high-fantasy scene, medium shot, in a circular ancient stone "
    "chamber illuminated by blue moonlight and glowing golden runes. Lyra, a "
    "young female mage with long silver hair, a dark-blue cloak and a crystal "
    "staff, touches a glowing rune, turns toward Thoren and says urgently, "
    "\"Thoren, as runas despertaram. O dragão sentiu nossa presença. Desde "
    "ontem, após o terremoto na cidade perdida, as correntes que prendiam os "
    "portais da chama intensa se quebraram. Subiram centenas deles da "
    "escuridão, famintos, incansáveis.\" Restrained orchestral music builds "
    "underneath, wind passes through the chamber, stones vibrate."
)
DEFAULT_SHEET = str(ROOT / "outputs" / "minimax_test" / "run01_lyra_thoren" /
                    "characters" / "sheets" / "LYRA.png")


def log(msg: str) -> None:
    print(f"[teste_cap] {msg}", flush=True)


def gerar_inteiro(prompt: str, duration_seconds: float, sheet: str, out_dir: Path) -> dict:
    import minimax_h3_backend as h3

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clip.mp4"
    t0 = time.time()
    h3.generate(
        prompt, str(out_path), ref_images=[sheet] if sheet else None,
        duration_seconds=duration_seconds, seed=42, turbo=True,
        log_cb=lambda m: log(f"  [A] {m}"), timeout=3600,
    )
    return {"modo": "inteiro", "segundos_pedidos": duration_seconds,
           "tempo_geracao_s": time.time() - t0, "arquivo": str(out_path)}


def extrair_ultimo_frame(video_path: str, out_path: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame))


def gerar_dividido(prompt: str, duration_seconds: float, max_seconds: float,
                   sheet: str, out_dir: Path) -> dict:
    import math
    import minimax_h3_backend as h3
    from script_pipeline.assemble_final import concat_videos

    n_partes = max(1, math.ceil(duration_seconds / max_seconds))
    seg_seconds = round(duration_seconds / n_partes, 2)
    log(f"[B] dividindo {duration_seconds}s em {n_partes} sub-plano(s) de ~{seg_seconds}s cada")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    partes = []
    tempos = []
    frame_anterior = None

    for i in range(n_partes):
        refs = [sheet] if sheet else []
        if frame_anterior:
            refs = refs + [frame_anterior] if sheet else [frame_anterior]
        refs = refs[:2] or None

        sub_path = out_dir / f"sub{i:02d}.mp4"
        t0 = time.time()
        h3.generate(
            prompt, str(sub_path), ref_images=refs,
            duration_seconds=seg_seconds, seed=42 + i, turbo=True,
            log_cb=lambda m, i=i: log(f"  [B sub{i}] {m}"), timeout=1800,
        )
        dt = time.time() - t0
        tempos.append(dt)
        partes.append(str(sub_path))
        log(f"[B] sub{i:02d} pronto em {dt:.0f}s -> {sub_path}")

        frame_path = frames_dir / f"sub{i:02d}_last.png"
        if extrair_ultimo_frame(str(sub_path), frame_path):
            frame_anterior = str(frame_path)
        else:
            log(f"[B] sub{i:02d}: nao consegui extrair ultimo frame -- proximo sub-plano perde a continuidade de movimento (mantem so a sheet).")

    final_path = out_dir / "final.mp4"
    concat_videos(partes, final_path, work_dir=out_dir / "intermediate", log=log)

    return {"modo": "dividido", "segundos_pedidos": duration_seconds,
           "n_partes": n_partes, "segundos_por_parte": seg_seconds,
           "tempo_total_s": sum(tempos), "tempos_por_parte_s": tempos,
           "arquivo": str(final_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--duration-seconds", type=float, default=16.62)
    ap.add_argument("--max-seconds", type=float, default=6.0,
                    help="teto por sub-plano na opcao dividida -- MEMORIAL 3.41 registra "
                         "10s como ja estressado; comecar conservador (6s).")
    ap.add_argument("--sheet-a", default=DEFAULT_SHEET, help="character sheet, opcao A")
    ap.add_argument("--sheet-b", default=None, help="character sheet, opcao B (default: mesma de --sheet-a)")
    ap.add_argument("--only", choices=["a", "b", "both"], default="both")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "_test_duration_cap"))
    args = ap.parse_args()

    if not Path(args.sheet_a).exists():
        log(f"AVISO: sheet {args.sheet_a} nao existe -- rodando sem ancora de identidade.")
        args.sheet_a = ""
    sheet_b = args.sheet_b if args.sheet_b is not None else args.sheet_a

    out_dir = Path(args.out_dir)
    relatorio = {}

    if args.only in ("a", "both"):
        log("=== Opcao A: plano inteiro ===")
        relatorio["inteiro"] = gerar_inteiro(
            args.prompt, args.duration_seconds, args.sheet_a, out_dir / "inteiro")

    if args.only in ("b", "both"):
        log("=== Opcao B: plano dividido e encadeado ===")
        relatorio["dividido"] = gerar_dividido(
            args.prompt, args.duration_seconds, args.max_seconds, sheet_b, out_dir / "dividido")

    (out_dir / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"relatorio -> {out_dir / 'relatorio.json'}")
    log("Comparacao de TEMPO e automatica (acima); comparacao de QUALIDADE/consistencia e visual -- revise os dois arquivos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
