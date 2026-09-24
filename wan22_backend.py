"""Wan 2.2 (TI2V 5B) via ComfyUI HTTP -- MESMO padrao de `ltx25_backend.py`/
`minimax_h3_backend.py`, MESMO servidor ComfyUI (8188, o do LTX 2.5 -- os
checkpoints do Wan ja vivem em `ComfyUI/models/`, nao precisa de instalacao
separada como o MiniMax H3/LongCat).

Reaproveita a logica ja escrita e usada em `script_pipeline/render_scenes.py`
(`render_job_wan`, caminho da screenplay_ui/porta 7810) -- aqui ela vira um
modulo `generate()` reutilizavel, no formato que `render_shots.py` (decupagem)
espera dos outros motores, para poder oferecer `--video-engine wan` do mesmo
jeito que ja oferece `minimax`/`longcat`.

⚠️ So a variante 5B (`Wan22ImageToVideoLatent`, 48 canais latentes, /16
espacial, `wan2.2_vae`, so imagem INICIAL) esta com peso baixado neste
checkout -- a 14B (`WanImageToVideo`/`WanFirstLastFrameToVideo`, permite
frame final) NAO. `checkpoint` default aponta pro 5B; passar um checkpoint
14B aqui exigiria trocar o `template_name` (ver `render_job_wan` em
render_scenes.py para a logica de roteamento por arquitetura) -- nao
replicado aqui de proposito, ja que so o 5B esta disponivel.

⚠️ Wan gera video MUDO -- ao contrario do LTX (audio_conditioning) e do
MiniMax H3 (fala embutida), nao ha nenhum branch de audio no grafo. Quem
chama isto e precisa de audio (fala ou trilha) tem que muxar depois (ver
`render_job_wan` em render_scenes.py para o padrao de mux com ffmpeg) --
este modulo so entrega o video puro, na mesma convencao "so devolve o
caminho do arquivo" do `ltx25_backend.generate()`.

⚠️ NUNCA testado ponta a ponta nesta maquina antes de 2026-09-18 (ver
`script_pipeline/SISTEMA_VIDEO.md`: "codigo escrito, modelos baixados,
nunca exercitado"). A validacao real feita nesta sessao usa este modulo.
"""
from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY_SERVER = "http://127.0.0.1:8188"
COMFY_INPUT_DIR = ROOT / "ComfyUI" / "input"
COMFYUI_OUTPUT_DIR = ROOT / "ComfyUI" / "output"
TEMPLATE_PATH = ROOT / "comfyui_workflows" / "wan22_ti2v_5b.json"

DEFAULT_CHECKPOINT = "wan2.2_ti2v_5B_fp16.safetensors"
DEFAULT_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
DEFAULT_VAE = "wan2.2_vae.safetensors"

# Mesmo texto de render_scenes.py -- Wan nao e destilado como o checkpoint LTX
# usado aqui, roda com CFG real e precisa de negative prompt de verdade.
DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall grayish, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, cluttered background, three legs, many people in the "
    "background, walking backwards"
)


def normalize_wan_frames(value: float) -> int:
    """Grade latente do Wan e 4k+1 (Wan22ImageToVideoLatent computa
    ((length-1)//4)+1 quadros latentes) -- NAO e o 8k+1 do LTX. Mesma formula
    de `script_pipeline.render_scenes.normalize_wan_frames`."""
    raw = max(5, int(round(value)))
    return 4 * max(1, round((raw - 1) / 4)) + 1


def ensure_server(log_cb=None, boot_timeout: int = 300) -> None:
    """Sobe (ou confirma que ja esta no ar) o MESMO ComfyUI do LTX 2.5, porta
    8188 -- os checkpoints do Wan ja estao em `ComfyUI/models/`, entao nao ha
    servidor proprio como o do MiniMax H3/LongCat."""
    from script_pipeline.generate_storyboards import ensure_comfyui_running
    if not ensure_comfyui_running(COMFY_SERVER, log=log_cb or print, wait_seconds=boot_timeout):
        raise RuntimeError(f"ComfyUI (Wan 2.2) nao respondeu em {boot_timeout}s.")


def _stage_input_image(path: str) -> str:
    """LoadImage le por NOME DE ARQUIVO em ComfyUI/input -- copia com nome
    unico pra nao colidir com outra chamada concorrente (mesmo padrao de
    `ltx25_backend.stage_input_image`/`minimax_h3_backend._stage_input`)."""
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = COMFY_INPUT_DIR / f"{uuid.uuid4().hex[:8]}_{Path(path).name}"
    shutil.copy2(path, dest)
    return dest.name


def _fill_template(template: dict, values: dict) -> dict:
    """Mesma logica de `generate_storyboards._fill_template` (substitui
    folhas "{{TOKEN}}" preservando o TIPO nativo do valor -- troca de texto
    ingenua deixaria numero virar string e o ComfyUI recusa o schema)."""
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and node.startswith("{{") and node.endswith("}}"):
            token = node[2:-2]
            if token not in values:
                raise KeyError(f"Workflow template referencia placeholder desconhecido {{{{{token}}}}}")
            return values[token]
        return node
    return walk(copy.deepcopy(template))


def generate(
    prompt: str,
    output_path: str,
    *,
    image_path: str,
    negative: str = DEFAULT_NEGATIVE,
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    frame_rate: float = 24.0,
    seed: int = 42,
    steps: int = 20,
    cfg: float = 5.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    checkpoint: str = DEFAULT_CHECKPOINT,
    clip_name: str = DEFAULT_CLIP,
    vae_name: str = DEFAULT_VAE,
    weight_dtype: str = "default",
    log_cb=None,
    timeout: int = 1800,
) -> str:
    """Gera um clipe Wan 2.2 (TI2V 5B) e copia para *output_path*. Retorna o
    caminho. `image_path` e OBRIGATORIO -- o grafo 5B (`Wan22ImageToVideoLatent`)
    so aceita imagem inicial, nao ha caminho T2V puro aqui (replicar T2V puro
    exigiria outro template, nao usado em producao neste projeto).

    Video SEM audio -- quem chama muxa depois se precisar (ver docstring do
    modulo). `width`/`height` devem ser multiplos de 16 (grade espacial do
    Wan, MEDIDO em render_scenes.py -- NAO e o multiplo de 64 do LTX nem o de
    32 do MiniMax H3); `num_frames` e ajustado pro 4k+1 mais proximo via
    `normalize_wan_frames`."""
    log = log_cb or print
    if width % 16 or height % 16:
        raise ValueError(f"width={width}/height={height} precisam ser multiplos de 16 (grade do Wan).")
    ensure_server(log_cb=log)

    length = normalize_wan_frames(num_frames)
    staged_image = _stage_input_image(image_path)

    values = {
        "CHECKPOINT": checkpoint, "CLIP_NAME": clip_name, "VAE_NAME": vae_name,
        "WEIGHT_DTYPE": weight_dtype, "WIDTH": width, "HEIGHT": height,
        "LENGTH": length, "FPS": float(frame_rate), "SEED": seed, "STEPS": steps,
        "CFG": cfg, "SAMPLER": sampler, "SCHEDULER": scheduler,
        "POSITIVE_PROMPT": prompt, "NEGATIVE_PROMPT": negative,
        "START_IMAGE": staged_image,
        "FILENAME_PREFIX": f"wan22_{Path(output_path).stem}",
    }
    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    workflow = _fill_template(template, values)

    log(f"[wan22] {length} quadros ({length/frame_rate:.1f}s @ {frame_rate}fps), "
        f"{width}x{height}, {steps} passos cfg={cfg}, video MUDO (sem audio_conditioning).")

    from script_pipeline.generate_storyboards import submit_and_wait
    entry = submit_and_wait(COMFY_SERVER, workflow, log=log, timeout=timeout)
    if entry is None:
        raise RuntimeError("ComfyUI (Wan 2.2) rejeitou ou nao completou o workflow -- ver log.")

    produced = None
    for _node_id, out in (entry.get("outputs") or {}).items():
        for key in ("videos", "images", "gifs"):
            for item in out.get(key, []) or []:
                if item.get("filename"):
                    produced = COMFYUI_OUTPUT_DIR / item.get("subfolder", "") / item["filename"]
                    break
            if produced:
                break
        if produced:
            break
    if produced is None or not Path(produced).exists():
        raise RuntimeError("Wan 2.2 concluiu mas nenhum arquivo de video foi encontrado na saida.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, output_path)
    log(f"[wan22] ok -> {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("output")
    ap.add_argument("--image", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=5.0)
    args = ap.parse_args()

    out = generate(args.prompt, args.output, image_path=args.image, width=args.width,
                   height=args.height, num_frames=args.num_frames, frame_rate=args.fps,
                   seed=args.seed, steps=args.steps, cfg=args.cfg)
    print("OK:", out)
