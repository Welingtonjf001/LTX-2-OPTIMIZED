"""Storyboard image generation: one txt2img render per scene, driven headlessly
through the project's already-existing ComfyUI instance (see comfy_run.py -- this
module inlines the same submit/poll pattern rather than shelling out to it, so it can
capture the actual output file path instead of just printing it).

Supports two checkpoint architectures, auto-selected by checkpoint filename:
  - SDXL (single-file checkpoint, e.g. sd_xl_base_1.0.safetensors): CheckpointLoaderSimple
    bundles model+clip+vae, workflow comfyui_workflows/storyboard_sdxl_txt2img.json.
  - FLUX (DiT-only checkpoint, e.g. flux-2-klein-9b-fp8.safetensors): needs a separate
    CLIP loader (Qwen3-8B text encoder for FLUX.2) and VAE loader, workflow
    comfyui_workflows/storyboard_flux_txt2img.json. CFG is fixed at 1.0 (flow-matching
    guidance-distilled model) with a FluxGuidance node controlling prompt adherence
    instead of classifier-free guidance.

Prompt per scene = scene["visual_prompt"] (from parse_screenplay's LLM enrichment,
when present) or a deterministic fallback assembled from location/time/action, PLUS
each present character's cast.json "descriptor" appended -- the same trick the music
pipeline already uses for cross-scene identity (repeat the character's fixed
description in every scene prompt; no IP-Adapter/face-lock in v1, see plan).

CLI: python -m script_pipeline.generate_storyboards --run-dir DIR
     [--checkpoint flux-2-klein-9b-fp8.safetensors] [--width 320] [--height 128]
     [--steps 8] [--cfg 7.0] [--clip Qwen3-8B-FP8-native-bf16.safetensors]
     [--vae flux2-vae.safetensors] [--guidance 3.5]
     [--comfy-server http://127.0.0.1:8188] [--no-auto-start]
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_TEMPLATES = {
    "sdxl": ROOT / "comfyui_workflows" / "storyboard_sdxl_txt2img.json",
    "flux": ROOT / "comfyui_workflows" / "storyboard_flux_txt2img.json",
    "sd35": ROOT / "comfyui_workflows" / "storyboard_sd35_txt2img.json",
    "flux1": ROOT / "comfyui_workflows" / "storyboard_flux1_txt2img.json",
}

# Padroes por MOTOR DE IMAGEM, num lugar so. Antes disto `render_shots` fixava
# steps=8 no codigo -- numero de modelo DESTILADO, que num modelo com CFG real
# devolve imagem crua. Cada motor traz o que ele precisa; quem escolhe o motor
# nao precisa saber de amostrador nem de nome de encoder.
#
# MEDIDO 2026-08-27 na 3090, still 1920x1088:
#   flux  9,4 GB de pesos + Qwen3-8B  -> pico ~24 GB, ~1 min/still quente,
#                                        ~4 min de carga fria, e ENCALHA a cada
#                                        ~6 stills (MEMORIAL 3.29)
#   sd35  5,1 GB + clip_g/clip_l/t5   -> pico ~12 GB, carga fria ~1 min
IMAGE_ENGINES = {
    "flux": {
        "checkpoint": "flux-2-klein-9b-fp8.safetensors",
        "clip": "Qwen3-8B-FP8-native-bf16.safetensors",
        "vae": "flux2-vae.safetensors",
        # Klein e destilado por guidance: CFG fica em 1.0 dentro do grafo e quem
        # controla adesao ao prompt e o FluxGuidance.
        "steps": 8, "cfg": 1.0, "guidance": 3.5,
        "descricao": "FLUX.2 Klein 9B fp8 -- melhor adesao a prompt longo, aceita imagem de referencia",
    },
    "sd35": {
        "checkpoint": "sd3.5_medium.safetensors",
        # Tres encoders, na ordem que o TripleCLIPLoader espera.
        "clip": "clip_g.safetensors,clip_l.safetensors,t5xxl_fp8_e4m3fn.safetensors",
        "vae": "",                      # vem do proprio checkpoint
        # SD 3.5 Medium NAO e destilado: precisa de CFG real e de bem mais passos.
        # E o preco de amostragem que se paga para ganhar na carga.
        "steps": 28, "cfg": 4.5, "guidance": 0.0,
        "descricao": "SD 3.5 Medium -- carrega em ~1 min e cabe em 12 GB; adesao a prompt longo e pior",
    },
    "sdxl": {
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "clip": "", "vae": "",
        "steps": 30, "cfg": 7.0, "guidance": 0.0,
        "descricao": "SDXL base -- legado, mantido porque o workflow ja existia",
    },
    # flux-krea e flux-kontext sao FLUX.1 (nao FLUX.2 Klein como "flux" acima):
    # precisam de DualCLIPLoader com clip_l+t5xxl, nao do CLIPLoader Qwen3 unico.
    # CONFIRMADO em 2026-09-01 via /object_info do ComfyUI ao vivo (nomes de
    # arquivo e nao adivinhados -- MEMORIAL 3.16 ja mostrou que isso morde).
    "flux-krea": {
        "checkpoint": "flux1-krea-dev_fp8_scaled.safetensors",
        "clip": "clip_l.safetensors,t5xxl_fp8_e4m3fn.safetensors",
        "vae": "vae\\ae.safetensors",
        # Ja vem fp8-scaled (11,9 GB); "default" carrega como esta salvo --
        # recastar um checkpoint ja escalado e onde perder qualidade a toa.
        "weight_dtype": "default",
        "steps": 28, "cfg": 1.0, "guidance": 4.5,
        "descricao": "FLUX.1 Krea dev -- estetica menos \"cara de IA\" que o Klein; ja fp8-scaled",
    },
    "flux-kontext": {
        "checkpoint": "flux1-kontext-dev.safetensors",
        "clip": "clip_l.safetensors,t5xxl_fp8_e4m3fn.safetensors",
        "vae": "vae\\ae.safetensors",
        # Peso bf16 de 23,8 GB -- sem cast fp8 nao sobra VRAM pro resto do grafo.
        "weight_dtype": "fp8_e4m3fn",
        "steps": 28, "cfg": 1.0, "guidance": 3.5,
        "descricao": "FLUX.1 Kontext dev -- editor por instrucao; aqui so txt2img "
                     "(sem imagem de entrada, ver generate_scene_storyboard), bf16 grande",
    },
}

# Nomes dos encoders do SD 3.5 quando o chamador nao passa a tripla explicita
# (por exemplo quem so trocou --image-engine e manteve o --clip do FLUX).
SD35_CLIPS_PADRAO = ("clip_g.safetensors", "clip_l.safetensors",
                     "t5xxl_fp8_e4m3fn.safetensors")

# LORA opcional nos STILLS -- pedido do usuario 2026-09-09, mesma pasta que o
# Storyboard-Director (projeto irmao, standalone) ja usa: os `.safetensors` de
# `models/loras/` sao para o TRANSFORMER DE VIDEO do LTX e sao INCOMPATIVEIS
# com estes checkpoints de imagem (FLUX/FLUX.1/SD3.5/SDXL). `loras_images/` e
# uma pasta PROPRIA, separada, e hoje pode estar vazia ate alguem colocar um
# LoRA treinado para um desses checkpoints -- "iniciar a opcao", nao entregar
# um LoRA curado. Cada arquitetura tem o node id do LoraLoaderModelOnly e o
# input que ele precisa rewirar (o que hoje recebe o MODEL do loader direto).
LORA_IMAGES_DIR = ROOT / "models" / "loras_images"

ARCH_MODEL_CONSUMER = {
    # arquitetura: (node id do consumidor, chave do input, [node id, slot] original)
    "flux": ("8", "model", ["1", 0]),
    "flux-ref": ("11", "model", ["1", 0]),  # character_flux_reference.json
    "flux1": ("8", "model", ["1", 0]),
    "sd35": ("8", "model", ["1", 0]),
    "sdxl": ("3", "model", ["4", 0]),
}


def available_loras_images() -> list[str]:
    """Lista os `.safetensors` disponiveis em `loras_images/`. Devolve []
    (sem erro) se a pasta nao existir -- e o estado padrao ate alguem baixar
    ou treinar um LoRA compativel com estes checkpoints de imagem."""
    if not LORA_IMAGES_DIR.exists():
        return []
    return sorted(p.name for p in LORA_IMAGES_DIR.glob("*.safetensors"))


def _apply_lora(workflow: dict, arch_key: str, lora_name: str, lora_strength: float) -> None:
    """Insere um LoraLoaderModelOnly entre o loader e quem consome MODEL,
    mutando `workflow` no lugar -- so afeta o transformer, nao o CLIP (LoRA de
    identidade/estilo nao precisa reescrever o encoder de texto). Node id "90"
    e livre em todo template hoje (nenhum passa de "13")."""
    consumer_id, input_key, model_ref = ARCH_MODEL_CONSUMER[arch_key]
    workflow["90"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": model_ref, "lora_name": lora_name, "strength_model": float(lora_strength),
    }}
    workflow[consumer_id]["inputs"][input_key] = ["90", 0]


def engine_defaults(engine: str) -> dict:
    """Padroes do motor de imagem. Levanta KeyError com a lista se o nome nao existe."""
    if engine not in IMAGE_ENGINES:
        raise KeyError(f"motor de imagem desconhecido {engine!r}; use {sorted(IMAGE_ENGINES)}")
    return dict(IMAGE_ENGINES[engine])


def _sd35_clip_names(clip: str) -> tuple[str, str, str]:
    """Aceita a tripla separada por virgula, ou cai no padrao.

    Quem chama com o `--clip` do FLUX (um encoder so, Qwen3) nao pode ter esse
    nome enfiado no TripleCLIPLoader: seria erro do ComfyUI num lugar onde a
    causa fica ilegivel. Uma entrada que nao seja tripla vira o padrao."""
    partes = [p.strip() for p in (clip or "").split(",") if p.strip()]
    return tuple(partes) if len(partes) == 3 else SD35_CLIPS_PADRAO


FLUX1_CLIPS_PADRAO = ("clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors")


def _flux1_clip_names(clip: str) -> tuple[str, str]:
    """Mesma logica de _sd35_clip_names, mas para o par clip_l+t5xxl do FLUX.1
    (DualCLIPLoader) -- diferente do CLIPLoader de encoder unico (Qwen3) do
    motor "flux" (FLUX.2 Klein)."""
    partes = [p.strip() for p in (clip or "").split(",") if p.strip()]
    return tuple(partes) if len(partes) == 2 else FLUX1_CLIPS_PADRAO
COMFYUI_DIR = ROOT / "ComfyUI"
COMFYUI_OUTPUT_DIR = COMFYUI_DIR / "output"

# O que nunca se quer num storyboard, seja qual for o meio da obra.
NEGATIVE_PROMPT_BASE = (
    "text, watermark, logo, signature, deformed, extra limbs, extra fingers, "
    "blurry, low quality, worst quality, jpeg artifacts, "
    # MEASURED (2026-08-10): FLUX drew SPEECH BALLOONS containing the dialogue into
    # storyboards, because the character descriptor still carried the character's
    # quoted line (fixed at the source in cast_characters._clean_descriptor -- this is
    # the second line of defence). Any lettering also survives into the video clip
    # conditioned on the image, as a smeared unreadable blob.
    "speech bubble, speech balloon, comic panel, caption, subtitles, lettering"
)

# Reforco de FOTORREALISMO -- so quando o roteiro NAO declara um meio proprio.
#
# MEASURED (2026-08-10): sem isto o storyboard saia como manga/ilustracao chapada
# em vez de material filmado, o que le como um meio diferente do video LTX que ele
# semeia. Mas em 2026-08-27 a cadeia ganhou direcao de arte (MEMORIAL 3.29), e ai
# esta lista passa a BRIGAR com o proprio pedido: um roteiro em "polished
# hand-drawn cel animation" recebia "cartoon, anime, manga, illustration, drawing"
# no negativo. O positivo venceu no teste, mas manter as duas metades do prompt se
# contradizendo e pedir resultado instavel de graca.
NEGATIVE_PROMPT_LIVE_ACTION = (
    "cartoon, anime, manga, illustration, drawing, painting, 3d render"
)

NEGATIVE_PROMPT_DEFAULT = NEGATIVE_PROMPT_BASE + ", " + NEGATIVE_PROMPT_LIVE_ACTION


COMMON_NEGATIVE_FILE = ROOT / "negative_prompt_common.txt"


def load_common_negative() -> str:
    """Termos negativos comuns ao video e aos stills, lidos de um arquivo.

    Existe como ARQUIVO e nao como constante para o usuario poder ajustar sem
    editar codigo -- e porque os dois consumidores (este modulo e o
    `ltx25_backend`) precisam do mesmo texto, e duas constantes divergiriam do
    mesmo jeito que os dois lancadores do ComfyUI divergiram (MEMORIAL 3.30).

    Linhas em branco e comecadas por '#' sao ignoradas. Arquivo ausente devolve
    string vazia, e ai cada lado fica so com o negativo proprio dele."""
    try:
        bruto = COMMON_NEGATIVE_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""
    partes = [l.strip().rstrip(",") for l in bruto.splitlines()
              if l.strip() and not l.lstrip().startswith("#")]
    return ", ".join(partes)


def negative_prompt_for(*, art_directed: bool) -> str:
    """O negativo que combina com o meio pedido, mais os termos comuns."""
    base = NEGATIVE_PROMPT_BASE if art_directed else NEGATIVE_PROMPT_DEFAULT
    comum = load_common_negative()
    return f"{base}, {comum}" if comum else base


def detect_architecture(checkpoint: str) -> str:
    """Arquitetura pelo NOME do arquivo de checkpoint.

    - "flux" -> DiT sozinho, precisa de CLIPLoader e VAELoader separados.
    - "sd3"  -> checkpoint unico (MMDiT + VAE) mais TripleCLIPLoader com
      clip_g/clip_l/t5xxl. O VAE vem de dentro do checkpoint, entao NAO ha
      VAELoader neste grafo.
    - resto  -> SDXL, checkpoint unico que ja traz model+clip+vae.

    Detectar pelo nome e fragil, mas e o que os tres formatos tem em comum aqui:
    o chamador passa um nome de arquivo, nao um tipo."""
    nome = checkpoint.lower()
    # "flux1-..." (Krea/Kontext/dev/schnell) precisa vir ANTES do "flux" generico:
    # os dois nomes contem "flux", mas so o FLUX.2 Klein usa o CLIPLoader Qwen3
    # unico que o branch "flux" abaixo monta.
    if "flux1" in nome:
        return "flux1"
    if "flux" in nome:
        return "flux"
    if "sd3" in nome:
        return "sd35"
    return "sdxl"


def _load_scenes(run_dir: Path) -> list[dict]:
    parse_dir = run_dir / "parse"
    enriched = parse_dir / "scenes_enriched.json"
    structural = parse_dir / "scenes.json"
    path = enriched if enriched.exists() else structural
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cast(run_dir: Path) -> dict:
    cast_path = run_dir / "characters" / "cast.json"
    if not cast_path.exists():
        return {}
    return json.loads(cast_path.read_text(encoding="utf-8"))


def build_prompt(scene: dict, cast: dict) -> str:
    if scene.get("visual_prompt"):
        base = scene["visual_prompt"]
    else:
        heading = ", ".join(p for p in (scene.get("location"), scene.get("time_of_day")) if p)
        base = f"{heading or 'scene'}. {scene.get('action_text', '')}".strip()
    descriptors = []
    for name in scene.get("characters", []):
        entry = cast.get(name)
        if entry and entry.get("descriptor"):
            descriptors.append(entry["descriptor"])
    if descriptors:
        base = base.rstrip(". ") + ". " + " ".join(d.rstrip(". ") + "." for d in descriptors)
    # The storyboard is a seed frame for LTX video, not artwork -- say so positively as
    # well as in the negative prompt, since the scene text alone left FLUX free to
    # answer in illustration style, which then set the look of the whole clip.
    return base.rstrip(". ") + ". Photorealistic cinematic film still, live action footage, natural lighting."


def _fill_template(template: dict, values: dict) -> dict:
    """Recursively substitute "{{TOKEN}}"-valued leaves with values[TOKEN], preserving
    the target's native type (int/float/str) -- a naive text-replace on the raw JSON
    would leave numeric fields quoted as strings, which ComfyUI's node schema rejects."""
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str) and node.startswith("{{") and node.endswith("}}"):
            token = node[2:-2]
            if token not in values:
                raise KeyError(f"Workflow template references unknown placeholder {{{{{token}}}}}")
            return values[token]
        return node
    return walk(copy.deepcopy(template))


def _http_json(url: str, payload: Optional[dict] = None, timeout: int = 30) -> dict:
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def comfy_is_up(server: str) -> bool:
    try:
        _http_json(f"{server}/system_stats", timeout=5)
        return True
    except Exception:
        return False


def stop_comfyui(port: int = 8188, *, log=print) -> bool:
    """Derruba quem estiver escutando na porta do ComfyUI. Devolve se derrubou.

    POR QUE ISTO PRECISA EXISTIR

    Um processo do ComfyUI decide a estrategia de alocacao quando CARREGA o
    modelo, nao continuamente (MEMORIAL 3.24). Entao um servidor que ja rodou o
    FLUX -- 9 GB de pesos mais o encoder Qwen3-8B -- fica com o alocador moldado
    para esse cenario, e quando o estagio seguinte pede o LTX 2.5, que sao 40 GB
    de transformer mais 25 GB de encoder num cartao de 24 GB, ele nao se
    recupera: vai a ~24,0 GB de 24,5, fica a 100% de uso e para de progredir.

    MEDIDO 2026-08-27, o MESMO clipe de 73 frames:

        servidor herdado do estagio de stills .... 55 min sem terminar
        servidor reiniciado limpo ................ 511 s (8,5 min)

    A passada dupla da secao 3.24 resolveu o recarregamento POR PLANO, mas nao a
    troca de cenario ENTRE as duas passadas -- e essa troca e exatamente o caso
    que a propria 3.24 registra como fatal. Um boot custa ~1 min; nao dar o boot
    custou quase uma hora.

    Nao usa psutil de proposito: `netstat`+`taskkill` e o que o
    start_comfyui_ltx.bat ja fazia, e nao acrescenta dependencia.

    Mecanismo delegado a `gpu_watchdog.free_port` desde 2026-09-04 -- eram
    duas copias do mesmo netstat+taskkill (esta e `decupagem_ui._free_port`)
    que podiam divergir; agora ha uma so implementacao."""
    from script_pipeline import gpu_watchdog
    derrubou = gpu_watchdog.free_port(port, log=log)
    if derrubou:
        log("ComfyUI encerrado para o proximo estagio subir limpo.")
        time.sleep(4)  # gpu_watchdog.free_port ja espera 2s; folga extra pro SO liberar a porta
    return derrubou


def comfy_launch_args(python_exe: str, *, port: int = 8188, cache_none: bool | None = None,
                      extra: list | None = None) -> list:
    """O comando do servidor ComfyUI, num lugar so.

    POR QUE ISTO EXISTE

    Havia DOIS lancadores com flags diferentes -- este e o de `ltx25_backend` --
    e quem subia o servidor primeiro decidia a politica para todo mundo, porque
    os dois voltam cedo quando a porta ja responde. Na cadeia de decupagem quem
    sobe primeiro e SEMPRE o estagio dos stills, entao o `--cache-none` que o
    `ltx25_backend` acha que esta passando nunca chegava a valer ali. Descoberto
    em 2026-08-27; ate entao o MEMORIAL 7 item 1 supunha o contrario, e media a
    lentidao da geracao de video contra uma flag que nao estava ligada.

    `cache_none` None = le `LTX_COMFY_CACHE_NONE` do ambiente. O padrao de cada
    chamador esta preservado: o backend 2.5 pede True (foi assim que a rota dele
    foi validada), o storyboard pede False. O ponto aqui NAO e unificar o valor
    -- e tornar a divergencia visivel e escolhida, em vez de sorteada pela ordem
    de execucao."""
    import os as _os
    if cache_none is None:
        cache_none = _os.environ.get("LTX_COMFY_CACHE_NONE", "0") not in ("", "0", "false", "False")
    args = [python_exe, "-u", "main.py", "--windows-standalone-build",
            "--extra-model-paths-config", "extra_model_paths.yaml",
            "--listen", "127.0.0.1", "--port", str(port)]
    if cache_none:
        args.append("--cache-none")
    # Flags so-de-bandeira do chamador entram no FIM: no meio elas separariam um
    # par --opcao valor e o argparse do ComfyUI leria o valor errado.
    args += list(extra or [])
    # Escotilha para experimentar flags de memoria do ComfyUI sem editar codigo:
    #   LTX_COMFY_EXTRA_ARGS="--disable-dynamic-vram --reserve-vram 1"
    # Existe porque as flags que decidem se um clipe cabe na placa
    # (--disable-dynamic-vram, --reserve-vram, --vram-headroom, --cuda-device)
    # sao justamente as que a gente precisa medir uma a uma.
    #
    # MEDIDO 2026-09-04: o DEFAULT (sem a variavel setada) costumava ser "" --
    # dynamic VRAM ligado -- e cada chamador que queria desliga-la tinha que
    # lembrar de exportar a variavel antes (so `start_decupagem.bat` e
    # `decupagem_ui.py` faziam isso). Qualquer invocacao direta (CLI, outro
    # script, ou eu mesmo testando manualmente) ficava com dynamic VRAM ligado
    # por omissao -- e travou de verdade: um still com imagem de referencia
    # (LoadImage+VAEEncode+ReferenceLatent, FLUX) ficou "rodando" por 7+ HORAS
    # sem terminar (GPU 100%, 24,1/24,5 GB, fila do ComfyUI com 43 planos
    # empacados atras dele), o mesmo sintoma ja documentado pro estagio de
    # VIDEO no MEMORIAL 3.33. Ate hoje --disable-dynamic-vram so tinha sido
    # medido necessario pro LTX 2.5; este caso prova que o FLUX dos stills,
    # com imagem de referencia, tem o mesmo problema. Sem nenhum caso medido
    # em que dynamic VRAM ligado seja melhor, o padrao agora e DESLIGADO --
    # setar `LTX_COMFY_EXTRA_ARGS=""` (string vazia, nao ausente) liga de
    # volta pra quem realmente quiser comparar. Ver MEMORIAL 3.58.
    if "LTX_COMFY_EXTRA_ARGS" in _os.environ:
        extra_env = _os.environ["LTX_COMFY_EXTRA_ARGS"].split()
    else:
        # EXCECAO GGUF -- MEDIDO 2026-09-07: com --disable-dynamic-vram, um
        # plano de video em gguf-q6k (decupagem, stills SD3.5 com imagem de
        # referencia + audio_conditioning) ficou "gerando" por 3 tentativas
        # seguidas, cada uma morta em exatamente 1802s pelo watchdog, GPU a
        # 100%/24,1-24,2 de 24,5 GB, historico do ComfyUI SEMPRE vazio --
        # nunca terminou nem uma vez, independente do numero de frames do
        # plano (153, ?, 121). O proprio log do ComfyUI-GGUF avisa isso na
        # subida do servidor ("if you use gguf we recommend keeping dynamic
        # vram enabled"), so que como texto de log, ninguem olha antes de
        # travar. O default de 2026-09-04 (`--disable-dynamic-vram` sempre
        # que a env var nao esta setada) foi medido so com bf16 LTX 2.5 e
        # FLUX dos stills -- GGUF nunca foi testado nessa combinacao. variant
        # distilled/dev continuam com a flag (nao mudou nada pra eles).
        variante = _os.environ.get("LTX25_VARIANT", "") or _os.environ.get("MINIMAX_H3_VARIANT", "")
        extra_env = [] if variante.startswith("gguf") else ["--disable-dynamic-vram"]
    args += extra_env
    return args


def ensure_comfyui_running(server: str, *, log=print, wait_seconds: int = 180,
                           watch_stalls: bool = True) -> bool:
    """`watch_stalls=True` (padrao) poe um `gpu_watchdog.StallWatch` de olho na
    fila deste servidor assim que ele sobe -- MEDIDO 2026-09-04: sem isto, um
    job preso na fila so era descoberto depois que 43 planos, um atras do
    outro, esperassem o PROPRIO timeout inteiro (7 horas ao todo). O watch
    detecta em minutos e reinicia sozinho ate 2x; na 3a trava seguida, para de
    tentar e so avisa (ver gpu_watchdog.StallWatch). `auto_recover=True` aqui
    porque matar+religar o ComfyUI e uma recuperacao ja usada em varios
    lugares deste projeto quando o servidor engasga -- nao e uma acao nova ou
    mais arriscada que o que ja se fazia na mao."""
    if comfy_is_up(server):
        if watch_stalls:
            _start_stall_watch_once(server, log=log)
        return True
    log("ComfyUI nao esta respondendo; iniciando (mesmo comando de start_comfyui_ltx.bat)...")
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3090 -- see start_comfyui_ltx.bat
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    # cache_none=None -> le LTX_COMFY_CACHE_NONE, que por padrao e 0. Ou seja: o
    # comportamento historico do storyboard (SEM a flag) continua sendo o padrao,
    # mas agora existe um interruptor, porque a flag muda VRAM disponivel e isso
    # decide se o estagio de video seguinte cabe na placa.
    cmd = comfy_launch_args(sys.executable, cache_none=None)
    log("ComfyUI: " + " ".join(cmd[3:]))
    # A saida do servidor ia para DEVNULL. Quando ESTE lancador e quem sobe o
    # ComfyUI -- que na cadeia de decupagem e sempre o caso, porque os stills vem
    # antes do video -- isso apagava tambem o log do ESTAGIO DE VIDEO, que roda
    # no mesmo servidor. MEDIDO 2026-08-27: fui ler `logs/comfyui_ltx25.log` para
    # saber quanto do tempo por clipe era carga de modelo e quanto era
    # amostragem, e o arquivo estava parado numa sessao antiga -- o servidor em
    # uso nao escrevia em lugar nenhum. Sem esse log nao da para medir carga.
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    server_log = open(logs_dir / "comfyui_storyboard.log", "a", encoding="utf-8", buffering=1)
    log(f"ComfyUI: log do servidor em {server_log.name}")
    subprocess.Popen(
        cmd, cwd=str(COMFYUI_DIR), env=env, creationflags=creationflags,
        stdout=server_log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if comfy_is_up(server):
            log("ComfyUI pronto.")
            if watch_stalls:
                _start_stall_watch_once(server, log=log)
            return True
        time.sleep(3)
    log(f"ComfyUI nao respondeu em {wait_seconds}s.")
    return False


_STALL_WATCHES: dict = {}


def _start_stall_watch_once(server: str, *, log=print) -> None:
    """Um StallWatch por servidor (server URL), nao um por chamada -- varios
    estagios chamam `ensure_comfyui_running` pro MESMO servidor dentro da
    mesma corrida, e nao faz sentido empilhar threads vigiando a mesma fila.

    `stall_seconds=1800` (era 300): o MESMO servidor atende tanto os stills
    (FLUX, ~1min cada, onde 300s ja seria folgado) quanto o estagio de VIDEO
    do plano de decupagem, que passa longe disso -- MEDIDO no MEMORIAL 3.35
    ate 1087s pra um unico plano em bf16 lowvram. Um numero curto o
    suficiente pros stills mata um plano de video legitimo antes de
    terminar -- exatamente o bug achado e corrigido no MiniMax H3
    (`minimax_h3_backend._start_stall_watch_once`, 2026-09-06): o
    detector so olha "mesmo job rodando ha N segundos", nao GPU ociosa, e
    o watchdog matava o servidor (`taskkill /F`) achando trava onde so
    havia um plano demorado de verdade. 1800s cobre o pior caso ja medido
    com folga."""
    if server in _STALL_WATCHES:
        return
    import re as _re
    import os as _os
    from script_pipeline import gpu_watchdog
    m = _re.search(r":(\d+)", server)
    port = int(m.group(1)) if m else 8188
    # EXCECAO GGUF -- MEDIDO 2026-09-07: o travamento do GGUF (race condition
    # no "async weight offloading" do ComfyUI + ComfyUI-GGUF sob VRAM
    # apertada, ver MEMORIAL 3.66) e categoricamente diferente do caso bf16
    # que justificou os 1800s -- no bf16 lowvram o job estava genuinamente
    # rodando devagar (1087s medido, chegou ao fim). No GGUF travado, o log
    # do proprio ComfyUI mostra "got prompt" e NUNCA sai a barra de progresso
    # do sampler -- ou seja, quando trava, trava ANTES de comecar a gerar de
    # verdade, nao durante. Os GGUF que terminam (varios medidos na mesma
    # corrida) fecham entre 115s e 600s, incluindo ate ~106s de inicializacao
    # de modelo no pior caso -- 900s cobre isso com folga generosa sem
    # herdar os 1800s inteiros de espera morta por travamento. bf16
    # (distilled/dev) continua em 1800s, sem mudanca.
    variante = _os.environ.get("LTX25_VARIANT", "")
    is_gguf = variante.startswith("gguf")
    stall_seconds = 900 if is_gguf else 1800
    # Mais tentativas no GGUF: cada uma custa metade do tempo (900s vs
    # 1800s) e a race e intermitente -- MEDIDO na mesma corrida, planos que
    # travaram na 1a tentativa terminaram normalmente na 2a ou 3a apos o
    # servidor reiniciar. 4 tentativas a 900s = mesmo teto de tempo que 2
    # tentativas a 1800s, com mais chances de a race nao repetir.
    max_recoveries = 4 if is_gguf else 2
    _STALL_WATCHES[server] = gpu_watchdog.start_stall_watch(
        server, port, log=log, stall_seconds=stall_seconds, auto_recover=True,
        max_auto_recoveries=max_recoveries)


def submit_and_wait(server: str, workflow: dict, *, timeout: int = 600, log=print) -> Optional[dict]:
    client_id = str(uuid.uuid4())
    try:
        res = _http_json(f"{server}/prompt", {"prompt": workflow, "client_id": client_id})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        log(f"ComfyUI rejeitou o workflow (HTTP {exc.code}): {body[:2000]}")
        return None
    prompt_id = res.get("prompt_id")
    if not prompt_id:
        log(f"Resposta inesperada de /prompt: {res}")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            history = _http_json(f"{server}/history/{prompt_id}")
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            # BUGFIX 2026-09-01: _http_json usa timeout de socket fixo (30s), mas o
            # log do ComfyUI mostra prompts levando ate 320s (troca de modelo em VRAM
            # -- MEMORIAL.md 3.30/3.35). Nesse trecho sincrono o HTTP do ComfyUI pode
            # nao responder por mais de 30s sem que o job tenha falhado de verdade.
            # Sem este except, um UNICO poll lento derrubava o estagio inteiro mesmo
            # dentro do orcamento de `timeout` (600s por padrao).
            log(f"/history nao respondeu a tempo ({type(exc).__name__}); tentando de novo.")
            time.sleep(2)
            continue
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return entry
            if status.get("status_str") == "error":
                log(f"ComfyUI falhou: {json.dumps(status, ensure_ascii=False)[:2000]}")
                return None
        time.sleep(2)
    log(f"Timeout esperando ComfyUI ({timeout}s).")
    return None


def _first_output_image(history_entry: dict) -> Optional[Path]:
    for _node_id, out in (history_entry.get("outputs") or {}).items():
        for item in out.get("images", []) or []:
            filename = item.get("filename")
            subfolder = item.get("subfolder", "")
            if filename:
                return COMFYUI_OUTPUT_DIR / subfolder / filename
    return None


def _stage_reference(image_path: str, tag: str) -> str:
    """Copy a reference photo into ComfyUI/input (LoadImage resolves by filename)."""
    comfy_input = COMFYUI_DIR / "input"
    comfy_input.mkdir(parents=True, exist_ok=True)
    staged = f"_charref_{tag}.png"
    from PIL import Image
    Image.open(image_path).convert("RGB").save(comfy_input / staged)
    return staged


def generate_scene_storyboard(
    scene: dict, cast: dict, *, server: str, checkpoint: str, width: int, height: int,
    steps: int, cfg: float, seed: int, out_path: Path, log=print,
    clip: str = "", vae: str = "", guidance: float = 3.5, prompt_override: str | None = None,
    reference_image: str | None = None, art_directed: bool = False,
    weight_dtype: str = "default", lora_name: str = "", lora_strength: float = 0.8,
) -> bool:
    architecture = detect_architecture(checkpoint)
    # A character reference photo routes through the ReferenceLatent graph, which a
    # controlled test in this project showed FLUX.2 Klein genuinely honours: same
    # prompt + same seed produced an unrelated person WITHOUT the reference and the
    # referenced person WITH it. Only FLUX has this path; SDXL falls back to text.
    used_reference_template = bool(reference_image and architecture == "flux")
    if used_reference_template:
        template = json.loads((ROOT / "comfyui_workflows" / "character_flux_reference.json").read_text(encoding="utf-8"))
    else:
        template = json.loads(WORKFLOW_TEMPLATES[architecture].read_text(encoding="utf-8"))
        reference_image = None
    prompt = prompt_override if prompt_override else build_prompt(scene, cast)
    values = {
        "SEED": seed, "STEPS": steps,
        "CHECKPOINT": checkpoint, "WIDTH": width, "HEIGHT": height,
        "POSITIVE_PROMPT": prompt,
        "NEGATIVE_PROMPT": negative_prompt_for(art_directed=art_directed),
        "FILENAME_PREFIX": f"storyboard_scene_{scene['index']:02d}",
    }
    if architecture == "flux":
        values["CLIP_NAME"] = clip
        values["VAE_NAME"] = vae
        values["GUIDANCE"] = guidance
    elif architecture == "flux1":
        clip_l, clip_t5 = _flux1_clip_names(clip)
        values["CLIP_L"], values["CLIP_T5"] = clip_l, clip_t5
        values["VAE_NAME"] = vae
        values["GUIDANCE"] = guidance
        values["WEIGHT_DTYPE"] = weight_dtype
    elif architecture == "sd35":
        g, l, t = _sd35_clip_names(clip)
        values["CLIP_G"], values["CLIP_L"], values["CLIP_T5"] = g, l, t
        values["CFG"] = cfg
    else:
        values["CFG"] = cfg
    if reference_image:
        values["REFERENCE_IMAGE"] = _stage_reference(
            reference_image, f"{scene['index']:02d}_{out_path.stem}"
        )
    workflow = _fill_template(template, values)
    if lora_name:
        arch_key = "flux-ref" if used_reference_template else architecture
        _apply_lora(workflow, arch_key, lora_name, lora_strength)
        log(f"Cena {scene['index']}: LoRA {lora_name} (forca {lora_strength}).")
    log(f"Cena {scene['index']}: {prompt[:120]}...")
    entry = submit_and_wait(server, workflow, log=log)
    if entry is None:
        return False
    image_path = _first_output_image(entry)
    if image_path is None or not image_path.exists():
        log(f"Cena {scene['index']}: ComfyUI concluiu mas nenhuma imagem foi encontrada.")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_path)
    log(f"Cena {scene['index']}: storyboard salvo em {out_path}")
    return True


def generate_scene_storyboard_with_consistency(
    scene: dict, cast: dict, *, out_path: Path, reference_image: str | None,
    seed: int, consistency_threshold: float | None, consistency_max_retries: int = 2,
    log=print, **kwargs,
) -> bool:
    """Mesmo `generate_scene_storyboard`, com auditoria de consistencia facial
    opcional -- porte de `render_shots.py::_still_for_shot` (decupagem, MEMORIAL
    3.53) pro caminho compartilhado do screenplay (MEMORIAL 3.54, "o que ja
    migra sozinho"). Sem `reference_image` ou sem `consistency_threshold`, e
    exatamente uma chamada normal -- custo extra so pra quem pediu."""
    if not reference_image or consistency_threshold is None:
        return generate_scene_storyboard(scene, cast, out_path=out_path,
                                         reference_image=reference_image, seed=seed, log=log, **kwargs)

    from script_pipeline.consistency_audit import check_consistency

    melhor_path, melhor_score = None, None
    for tentativa in range(consistency_max_retries + 1):
        seed_tentativa = seed + tentativa * 7919
        candidato = out_path if tentativa == 0 else out_path.with_suffix(f".tentativa{tentativa}.png")
        if not generate_scene_storyboard(scene, cast, out_path=candidato,
                                         reference_image=reference_image, seed=seed_tentativa, log=log, **kwargs):
            continue
        ok_cons, score = check_consistency(str(candidato), reference_image, threshold=consistency_threshold)
        log(f"  consistencia (tentativa {tentativa}, seed {seed_tentativa}): "
            f"{'sem rosto detectavel' if score is None else f'{score:.3f}'} "
            f"{'(dentro do limiar)' if ok_cons else '(ABAIXO do limiar)'}")
        if melhor_score is None or (score is not None and score > (melhor_score or -1)):
            melhor_path, melhor_score = candidato, score
        if ok_cons:
            break

    if melhor_path is None:
        return False
    if melhor_path != out_path:
        melhor_path.replace(out_path)
    for tentativa in range(1, consistency_max_retries + 1):
        sobra = out_path.with_suffix(f".tentativa{tentativa}.png")
        if sobra.exists():
            sobra.unlink(missing_ok=True)
    return out_path.exists()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-engine", default=None, choices=sorted(IMAGE_ENGINES),
                        help="motor de imagem: preenche checkpoint, encoders, passos, "
                             "CFG e guidance de uma vez (ver IMAGE_ENGINES). Sem isto "
                             "valem os valores explicitos abaixo, que e o comportamento "
                             "historico. Um valor explicito ganha do motor.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--width", type=int, default=1792, help="2x the render stage's --width (896), same aspect ratio, sharper source to downscale into the video.")
    parser.add_argument("--height", type=int, default=1024, help="2x the render stage's --height (512).")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=7.0, help="SDXL only (ignored for FLUX, which uses --guidance).")
    parser.add_argument("--clip", default="Qwen3-8B-FP8-native-bf16.safetensors",
                         help="FLUX/FLUX.1 (Krea/Kontext): text encoder filename(s) "
                              "-- Qwen3 unico para --image-engine flux, ou "
                              "\"clip_l.safetensors,t5xxl_fp8_e4m3fn.safetensors\" para flux-krea/flux-kontext.")
    parser.add_argument("--vae", default="flux2-vae.safetensors", help="FLUX/FLUX.1 only: VAE filename in ComfyUI/models/vae/.")
    parser.add_argument("--guidance", type=float, default=3.5, help="FLUX/FLUX.1 only: FluxGuidance scale.")
    parser.add_argument("--weight-dtype", default=None, dest="weight_dtype",
                         choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                         help="FLUX.1 only (flux-krea/flux-kontext): UNETLoader weight_dtype. "
                              "Sem isto, o motor escolhe (default para flux-krea, ja fp8-scaled; "
                              "fp8_e4m3fn para flux-kontext, bf16 de 23,8 GB).")
    parser.add_argument("--consistency-threshold", type=float, default=None,
                         help="auditoria automatica de consistencia facial (insightface) contra "
                              "a reference_image do plano/personagem -- MEMORIAL 3.53/3.54. Sem "
                              "isto, desligado (comportamento de sempre). So se aplica a planos "
                              "que JA tem reference_image (dialogo com foto/sheet em cast.json); "
                              "plano sem referencia nao muda.")
    parser.add_argument("--consistency-max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--comfy-server", default="http://127.0.0.1:8188")
    parser.add_argument("--no-auto-start", action="store_true")
    parser.add_argument("--only-scene", type=int, default=None,
                         help="Regenerate a single scene's storyboard (1-based scene index) instead of every scene.")
    parser.add_argument("--prompt-override", default=None,
                         help="Use this exact text as the prompt instead of build_prompt() -- only valid with --only-scene.")
    parser.add_argument("--per-shot", dest="per_shot", action="store_true", default=True,
                         help="One storyboard per SHOT of the scene's decoupage (default) -- without this every clip "
                              "in a scene is conditioned on the same image and the shots come out identical.")
    parser.add_argument("--per-scene", dest="per_shot", action="store_false",
                         help="Legacy: a single storyboard image for the whole scene.")
    args = parser.parse_args(argv)

    # O motor preenche o que o chamador nao disse. Ordem importa: quem passou o
    # valor explicitamente manda, para o caminho antigo continuar identico.
    _padrao = engine_defaults(args.image_engine) if args.image_engine else IMAGE_ENGINES["flux"]
    if not args.checkpoint:
        args.checkpoint = _padrao["checkpoint"]
    if args.image_engine:
        if "--clip" not in (argv or sys.argv):
            args.clip = _padrao["clip"]
        if "--vae" not in (argv or sys.argv):
            args.vae = _padrao["vae"]
        if "--steps" not in (argv or sys.argv):
            args.steps = _padrao["steps"]
        if "--cfg" not in (argv or sys.argv):
            args.cfg = _padrao["cfg"]
        if "--guidance" not in (argv or sys.argv):
            args.guidance = _padrao["guidance"]
        if args.weight_dtype is None:
            args.weight_dtype = _padrao.get("weight_dtype", "default")
    if args.weight_dtype is None:
        args.weight_dtype = "default"

    from script_pipeline import run_folder

    run_dir = Path(args.run_dir).resolve()
    scenes = _load_scenes(run_dir)
    cast = _load_cast(run_dir)
    storyboard_dir = run_folder.subdir(run_dir, "storyboard")

    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731

    if args.only_scene is not None:
        scenes = [s for s in scenes if s["index"] == args.only_scene]
        if not scenes:
            log(f"generate_storyboards: cena {args.only_scene} nao encontrada.")
            return 1
    elif args.prompt_override:
        parser.error("--prompt-override so e valido junto com --only-scene.")

    if not args.no_auto_start:
        if not ensure_comfyui_running(args.comfy_server, log=log):
            log("Nao foi possivel iniciar/alcancar o ComfyUI. Rode start_comfyui_ltx.bat manualmente e tente de novo.")
            return 1
    elif not comfy_is_up(args.comfy_server):
        log(f"ComfyUI nao esta respondendo em {args.comfy_server} (--no-auto-start impediu o start automatico).")
        return 1

    failures = 0
    seed_counter = 0
    for i, scene in enumerate(scenes):
        # One storyboard PER SHOT, not per scene.
        # MEASURED (2026-08-10): with a single image per scene, every clip in that
        # scene was conditioned on the SAME picture, so shots eight beats apart came
        # out visually identical -- same framing, same blocking, same background --
        # no matter how different their action descriptions were. Each shot in the
        # decoupage already carries its own visual description; giving each one its
        # own anchor image is what actually makes the shots differ. Falls back to the
        # old one-image-per-scene behaviour when there's no shot list.
        shot_list = scene.get("shot_list") or []
        if args.per_shot and shot_list and args.only_scene is None:
            scene_failures = 0
            for shot_i, shot in enumerate(shot_list, start=1):
                shot_reference = None
                if shot.get("type") == "action":
                    shot_prompt = build_prompt(
                        {**scene, "visual_prompt": shot.get("visual")}, cast
                    )
                else:
                    line = (scene.get("dialogue") or [])[shot.get("line_index", 0)]
                    beat = line.get("beat_visual") or scene.get("visual_prompt")
                    shot_prompt = build_prompt({**scene, "visual_prompt": beat}, cast)
                    shot_prompt += f" Close-up on {line.get('character', '')}."
                    # A dialogue shot is framed on ONE character, so that character's
                    # reference photo (when the user supplied one) is the right anchor.
                    # Action shots can hold several people, with no single face to lock.
                    shot_reference = (cast.get(line.get("character", "")) or {}).get("reference_image")
                shot_out = storyboard_dir / f"scene_{scene['index']:02d}_s{shot_i:03d}.png"
                seed_counter += 1
                if not generate_scene_storyboard_with_consistency(
                    scene, cast, server=args.comfy_server, checkpoint=args.checkpoint,
                    width=args.width, height=args.height, steps=args.steps, cfg=args.cfg,
                    seed=args.seed + seed_counter, out_path=shot_out, log=log,
                    clip=args.clip, vae=args.vae, guidance=args.guidance,
                    weight_dtype=args.weight_dtype,
                    prompt_override=shot_prompt, reference_image=shot_reference,
                    consistency_threshold=args.consistency_threshold,
                    consistency_max_retries=args.consistency_max_retries,
                ):
                    scene_failures += 1
            failures += scene_failures
            log(f"generate_storyboards: cena {scene['index']}: "
                f"{len(shot_list) - scene_failures}/{len(shot_list)} storyboard(s) por plano.")
            continue

        out_path = storyboard_dir / f"scene_{scene['index']:02d}.png"
        ok = generate_scene_storyboard(
            scene, cast, server=args.comfy_server, checkpoint=args.checkpoint,
            width=args.width, height=args.height, steps=args.steps, cfg=args.cfg,
            seed=args.seed + i, out_path=out_path, log=log,
            clip=args.clip, vae=args.vae, guidance=args.guidance,
            weight_dtype=args.weight_dtype,
            prompt_override=args.prompt_override if args.only_scene is not None else None,
        )
        if not ok:
            failures += 1

    log(f"generate_storyboards: {len(scenes) - failures}/{len(scenes)} storyboard(s) gerado(s).")
    if failures == 0:
        if args.only_scene is None:
            run_folder.mark_stage_complete(run_dir, "storyboard")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
