# -*- coding: utf-8 -*-
"""MiniMax H3 (Reference-to-Video) via a ComfyUI SEPARADO do LTX 2.5.

Mesmo padrao do `ltx25_backend.py` -- converter workflow oficial para API via
`comfy_workflow_tool`, escrever nos nos conhecidos por ID, submeter e esperar
-- mas para uma instalacao de ComfyUI INDEPENDENTE: outro venv, outra porta
(8189), outra GPU convention (`--cuda-device 1`, que aqui mapeia direto para
CUDA_VISIBLE_DEVICES, nao pelo `main.py` do LTX). As duas nunca disputam a
mesma porta nem o mesmo processo -- mas disputam a MESMA 3090 fisica, entao
nunca rode as duas ao mesmo tempo (ver `MEMORIAL.md` 3.40 e `ensure_stopped()`
abaixo).

O modelo espera referencias CITADAS NO TEXTO: "<Picture 1>", "<Picture 2>",
"<Audio 1>" dentro do prompt, alem de conectadas como imagem/audio de
referencia -- e o proprio exemplo oficial do workflow que ensina essa
convencao. `build_prompt_with_refs()` injeta isso automaticamente.

`length` (frames) so aceita `5 + 17*k` (24 fps; 124 = ~5s). Em vez de
reimplementar essa aritmetica, o proprio grafo ja faz a conta (no
ComfyMathExpression) -- so alimentamos a duracao em segundos.

CLI:
    .venv/Scripts/python.exe -m minimax_h3_backend \\
        --prompt "..." --ref-image a.png --ref-image b.png \\
        --ref-audio voz.wav --output-path out.mp4 --duration-seconds 5 --turbo

`--ref-audio` (2026-09-08, MEMORIAL 3.74): o workflow oficial deste checkout
NUNCA teve isso ligado (so ref_images.* vinha conectado, apesar do node
MiniMaxH3ReferenceToVideo aceitar `ref_audios` e ja trazer `audio_vae`
pronto) -- os nos LoadAudio sao criados por job em `build_workflow()`. Use
combinado com a fala exata no `--prompt`, nomeando o personagem que fala,
pra o modelo tentar gerar a fala real em vez de inventar uma voz propria.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
MINIMAX_ROOT = os.environ.get("MINIMAX_H3_ROOT",
                              r"E:\Users\home\Documents\MiniMax-H3")
COMFY_ROOT = os.path.join(MINIMAX_ROOT, "ComfyUI")
COMFY_PORT = int(os.environ.get("MINIMAX_H3_PORT", "8189"))
COMFY_SERVER = f"http://127.0.0.1:{COMFY_PORT}"
COMFY_OUTPUT = os.path.join(COMFY_ROOT, "output")
WORKFLOW_PATH = os.path.join(MINIMAX_ROOT, "workflows", "video_minimax_h3_r2v.json")

# IDs do workflow oficial -- ver a leitura feita contra /object_info em
# 2026-08-29 (MEMORIAL 3.40). Se o workflow mudar de estrutura, `base_api()`
# falha alto e cedo (checagem de nos esperados), nao em silencio.
N_PROMPT = "138"
N_RESOLUTION = "115"       # ResolutionSelector: aspect_ratio, megapixels, multiple
N_DURATION_S = "132"       # PrimitiveFloat: segundos, vira `length` via node 131
N_REF_IMAGE_0 = "137"
N_REF_IMAGE_1 = "139"
# Nao existiam no workflow oficial (so ref_images.* vinha ligado) -- IDs
# novos, fora da faixa 92-146 que o grafo original usa, criados por job em
# build_workflow(). Ver MEMORIAL 3.74.
N_REF_AUDIO_BASE = "150_ref_audio_"
N_SEED = "129"             # RandomNoise
N_TURBO_SWITCH = "146"     # PrimitiveBoolean: liga a LoRA turbo (4 passos) e o node 144
N_R2V = "136"              # MiniMaxH3ReferenceToVideo -- dono dos slots ref_images.*
N_CLIP = "128"             # CLIPLoader -- Qwen3-VL 32B, 15,7 GB so ele
N_UNET = "127"             # UNETLoader -- diffusion_models/*.safetensors ou *.gguf
N_VIDEO_VAE = "119"        # VAELoader (video) -- tambem usado por N_R2V pra encode() das refs
N_SAVE = "92"              # SaveVideo

# TensorRT pra VAE de video (2026-09-06): compilado via ComfyUI-H3VAE_TRT
# (github.com/lihaoyun6/ComfyUI-H3VAE_TRT) a partir dos ONNX oficiais
# (huggingface.co/lihaoyun6/MiniMax-H3-VAE-ONNX). So a VAE de VIDEO -- a de
# audio (node 120) nao tem engine TRT disponivel, fica como esta. O node
# MiniMaxH3TRTVAELoader troca o VAELoader inteiro por um so que faz
# encode+decode via TensorRT (a mesma VAE alimenta tanto o VAEDecode quanto
# o encode de imagem de referencia em N_R2V).
TRT_VAE = os.environ.get("MINIMAX_H3_TRT_VAE", "0") == "1"
TRT_DECODER_ENGINE = os.environ.get(
    "MINIMAX_H3_TRT_DECODER", "minimax_h3_vae_decoder.engine")
TRT_ENCODER_ENGINE = os.environ.get(
    "MINIMAX_H3_TRT_ENCODER", "minimax_h3_vae_encoder.engine")

# Override de checkpoint (2026-09-06): o workflow oficial
# (`video_minimax_h3_r2v.json`) traz o quantizado mais agressivo (w4a8
# pruned, 13 GB) fixo nos nos 127/128. Essas duas env vars trocam o arquivo
# sem tocar no JSON do workflow.
#
# PADRAO MUDOU pra FP8 (unet) + INT8 (text encoder) apos teste comparativo:
# MEDIDO 2026-09-06, mesma cena/seed, turbo 4 passos --
#   w4a8 (o antigo padrao) ......... 554-704s dependendo da cena
#   FP8+INT8 (novo padrao) ......... 397-666s -- SEMPRE mais rapido que o
#                                     w4a8 nas mesmas condicoes, alem de vir
#                                     de checkpoints menos comprimidos
#                                     (qualidade nominal maior)
#   GGUF Q4_K_M ..................... 707-986s -- mais lento que os dois
# Ver memoria de projeto "MiniMax H3 watchdog bug and quality tests" pro
# detalhe completo. Pra voltar ao w4a8, exporte MINIMAX_H3_UNET=
# minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors (e o CLIP
# correspondente).
# MINIMAX_VARIANTS (2026-09-07): nomeia os tres pares (unet, clip) medidos
# acima, pra escolher um so interruptor (`MINIMAX_H3_VARIANT=w4a8`) em vez de
# decorar/repetir os dois nomes de arquivo toda vez. `MINIMAX_H3_UNET`/
# `MINIMAX_H3_CLIP` continuam funcionando por cima -- setar qualquer um dos
# dois direto ainda vence, igual sempre foi; a variante so decide o PADRAO.
# CLIP do gguf-q4km: o GGUF so quantiza o UNET, o par usado no teste
# comparativo foi o mesmo int8 do padrao (nao existe um CLIP gguf pra isso).
MINIMAX_VARIANTS = {
    "fp8int8": ("minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
               "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "w4a8": ("minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors",
            "qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors"),
    "gguf-q4km": ("minimax_h3_ref2va-Q4_K_M.gguf",
                 "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
}
DEFAULT_MINIMAX_VARIANT = os.environ.get("MINIMAX_H3_VARIANT", "fp8int8").strip().lower()
if DEFAULT_MINIMAX_VARIANT not in MINIMAX_VARIANTS:
    raise ValueError(
        f"MINIMAX_H3_VARIANT={DEFAULT_MINIMAX_VARIANT!r} desconhecida; "
        f"use um de {sorted(MINIMAX_VARIANTS)}")
_variant_unet, _variant_clip = MINIMAX_VARIANTS[DEFAULT_MINIMAX_VARIANT]
UNET_FILENAME = os.environ.get("MINIMAX_H3_UNET", _variant_unet)
CLIP_FILENAME = os.environ.get("MINIMAX_H3_CLIP", _variant_clip)

DEFAULT_ASPECT = "16:9 (Widescreen)"
DEFAULT_MEGAPIXELS = 0.4

_server_proc = None
_server_log_handle = None
_base_api_cache: dict | None = None
_BOOT_LOCK_PATH = os.path.join(MINIMAX_ROOT, "logs", "minimax_h3.boot.lock")
_BOOT_LOCK_STALE_S = 600


def _log(msg: str, log_cb=None) -> None:
    print(msg, flush=True)
    if log_cb:
        try:
            log_cb(msg)
        except Exception:
            pass


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def server_is_up() -> bool:
    try:
        urllib.request.urlopen(f"{COMFY_SERVER}/", timeout=3)
        return True
    except Exception:
        return False


@contextlib.contextmanager
def _boot_lock(wait_s: int = 300):
    """Mesmo mutex entre processos do `ltx25_backend.py` -- ver o docstring
    la para o porque. Path proprio: as duas instancias de ComfyUI sao
    processos independentes, mas ainda cabe duas UIs desta rota tentarem
    subir o MiniMax H3 ao mesmo tempo."""
    os.makedirs(os.path.dirname(_BOOT_LOCK_PATH), exist_ok=True)
    inicio = time.time()
    while True:
        try:
            fd = os.open(_BOOT_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode("ascii"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                idade = time.time() - os.path.getmtime(_BOOT_LOCK_PATH)
            except OSError:
                idade = 0
            if idade > _BOOT_LOCK_STALE_S:
                try:
                    os.remove(_BOOT_LOCK_PATH)
                except OSError:
                    pass
                continue
            if time.time() - inicio > wait_s:
                break
            time.sleep(1)
    try:
        yield
    finally:
        try:
            os.remove(_BOOT_LOCK_PATH)
        except OSError:
            pass


def shutdown_server() -> None:
    global _server_proc, _server_log_handle
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None
    if _server_log_handle:
        _server_log_handle.close()
        _server_log_handle = None


def ensure_server(log_cb=None, boot_timeout: int = 300) -> None:
    """Sobe o ComfyUI do MiniMax H3 se nao estiver no ar. Idempotente."""
    global _server_proc, _server_log_handle
    if server_is_up():
        _start_stall_watch_once(log=lambda m: _log(m, log_cb))
        return
    with _boot_lock():
        if server_is_up():
            _start_stall_watch_once(log=lambda m: _log(m, log_cb))
            return
        _ensure_server_locked(log_cb=log_cb, boot_timeout=boot_timeout)


_STALL_WATCH = None


def _start_stall_watch_once(*, log=print) -> None:
    """Mesma logica de `generate_storyboards._start_stall_watch_once`, um
    watch so pro servidor do MiniMax H3 (porta propria, 8189) -- ver
    gpu_watchdog.StallWatch e MEMORIAL 3.58/3.60.

    `stall_seconds=300` (5 min) ERA CURTO DEMAIS pra esta configuracao --
    MEDIDO 2026-09-06: o watchdog matava o servidor (`taskkill /F`, sem
    traceback nenhum, parecia crash) sempre entre 330-340s, ANTES do job
    (clip 5s turbo, encoder de texto na CPU) terminar sozinho -- reproduzido
    3x seguidas, com e sem `--disable-dynamic-vram`. O detector so olha
    "mesmo prompt_id rodando ha N segundos", nao GPU ociosa -- e um clip
    com o encoder de 15 GB rodando na CPU passa boa parte do tempo com a GPU
    parada por design (ver `clip_on_cpu` em build_workflow), o que e normal,
    nao trava.

    900s (tentativa 1) ainda nao bastou: MEDIDO 2026-09-06 de novo, com um
    prompt bem mais longo (cena de dialogo, ~200 palavras) e duracao maior
    (15s em vez de 5s) -- matou de novo aos 943s. O tempo do job varia com
    tamanho do prompt (mais tokens pro encoder) E duracao pedida, entao um
    numero fixo baixo sempre vai ter uma combinacao que estoura. 1800s (30
    min) da folga de sobra pros casos pesados sem parar de pegar trava real
    (o pior caso ja medido, com prompt longo + 15s, fechou por volta de
    12-15 min).
    """
    global _STALL_WATCH
    if _STALL_WATCH is not None:
        return
    from script_pipeline import gpu_watchdog
    _STALL_WATCH = gpu_watchdog.start_stall_watch(
        COMFY_SERVER, COMFY_PORT, log=log, stall_seconds=1800,
        auto_recover=True, max_auto_recoveries=2)


def _ensure_server_locked(log_cb=None, boot_timeout: int = 300) -> None:
    global _server_proc, _server_log_handle

    def log(msg):
        _log(msg, log_cb)

    python_exe = os.path.join(MINIMAX_ROOT, "venv", "Scripts", "python.exe")
    if not os.path.isfile(python_exe):
        raise RuntimeError(f"venv do MiniMax H3 nao encontrado em {python_exe}")

    log("[minimax_h3] ComfyUI não está no ar; iniciando servidor (~1min)...")
    env = os.environ.copy()
    # Mesma GPU fisica que o LTX 2.5 usa (a 3090). `--cuda-device` do proprio
    # ComfyUI aplica isso antes do torch importar -- nao precisa do env aqui
    # tambem, mas setar os dois e redundante e inofensivo, e documenta a
    # intencao pra quem ler o processo na lista de tarefas.
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["HF_HOME"] = os.path.join(MINIMAX_ROOT, "hf-cache")
    env["PYTHONUTF8"] = "1"

    logs_dir = os.path.join(MINIMAX_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    _server_log_handle = open(os.path.join(logs_dir, "comfyui_minimax_h3.log"),
                              "a", encoding="utf-8", buffering=1)
    # SEM --lowvram: com o encoder de texto mandado para a CPU por workflow
    # (build_workflow's clip_on_cpu), o UNET + LoRA sozinhos (~14,5 GB) cabem
    # folgados nos 24,5 GB da placa -- --lowvram so fazia sentido quando os
    # dois disputavam VRAM juntos. MEDIDO 2026-08-29: com os dois na GPU e
    # --lowvram, 30 min sem terminar um clipe de 10s turbo. Ver MEMORIAL 3.41.
    #
    # SEM --cpu-vae tambem, desde que o torch deste venv virou cu130+ (3.42):
    # com o kernel rapido ligado, a amostragem passou a TERMINAR (617s, os 4
    # passos completos) e so falhou no VAEDecode -- "expected m1 and m2 to
    # have the same dtype, but got: float != struct c10::Half". O VAE forcado
    # a CPU roda em float32; o latente que chega da GPU esta em meia-precisao.
    # Com o encoder ja fora da GPU ha VRAM de sobra para o VAE tambem (~9,9 GB
    # carregado), entao a correcao mais simples e deixar o VAE na GPU, no
    # mesmo dtype do resto do grafo.
    # `--disable-dynamic-vram`: MEDIDO 2026-09-06 -- sem essa flag o servidor
    # morre em silencio (sem traceback, log para logo apos "CLIP/text encoder
    # model load device: cpu"), reproduzido 2x seguidas no mesmo ponto (~335s).
    # Mesmo bug ja documentado e corrigido pro LTX 2.5 no CLAUDE.md: o
    # DynamicVRAM do ComfyUI (ligado por padrao) engasga ao encenar um encoder
    # grande. Este servidor nunca tinha essa flag.
    cmd = [python_exe, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
           "--cuda-device", "1", "--disable-auto-launch", "--disable-dynamic-vram"]
    log("[minimax_h3] ComfyUI: " + " ".join(cmd[2:]))
    try:
        _server_proc = subprocess.Popen(
            cmd, cwd=COMFY_ROOT, env=env,
            stdout=_server_log_handle, stderr=subprocess.STDOUT,
        )
    except Exception:
        _server_log_handle.close()
        _server_log_handle = None
        raise

    import atexit
    atexit.register(shutdown_server)

    t0 = time.time()
    while time.time() - t0 < boot_timeout:
        if server_is_up():
            log(f"[minimax_h3] ComfyUI pronto em {time.time()-t0:.1f}s.")
            _start_stall_watch_once(log=log)
            return
        if _server_proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI do MiniMax H3 morreu ao subir (codigo {_server_proc.returncode}). "
                f"Log: {os.path.join(logs_dir, 'comfyui_minimax_h3.log')}")
        time.sleep(2)
    raise RuntimeError(f"ComfyUI do MiniMax H3 não respondeu em {boot_timeout}s.")


def _stage_input(path: str) -> str:
    dest_dir = os.path.join(COMFY_ROOT, "input")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{uuid.uuid4().hex[:8]}_{os.path.basename(path)}")
    shutil.copy2(path, dest)
    return dest


def base_api() -> dict:
    """Converte o workflow oficial uma vez, com o servidor no ar; cacheia."""
    global _base_api_cache
    if _base_api_cache is None:
        ensure_server()
        import comfy_workflow_tool as wf_tool
        ui = json.load(open(WORKFLOW_PATH, encoding="utf-8"))
        oi = wf_tool.fetch_object_info(COMFY_SERVER)
        api = wf_tool.convert(ui, oi)
        api = wf_tool.prune_to_output(api, N_SAVE)
        esperados = [N_PROMPT, N_RESOLUTION, N_DURATION_S, N_REF_IMAGE_0,
                    N_REF_IMAGE_1, N_SEED, N_TURBO_SWITCH, N_R2V, N_CLIP,
                    N_UNET, N_VIDEO_VAE, N_SAVE]
        faltando = [n for n in esperados if n not in api]
        if faltando:
            raise RuntimeError(
                "O workflow do MiniMax H3 mudou de estrutura: nós esperados "
                f"ausentes {faltando}. Reexporte `workflows/video_minimax_h3_r2v.json` "
                "do editor do ComfyUI e reveja os IDs no topo deste módulo.")
        if UNET_FILENAME.lower().endswith(".gguf"):
            # MEDIDO 2026-09-06: o UNETLoader oficial rejeita .gguf na
            # validacao ("value_not_in_list") -- so aceita o que esta na
            # pasta diffusion_models, e ele proprio nao sabe carregar GGUF.
            # Precisa trocar o NODE inteiro pro UnetLoaderGGUF (mesma saida
            # MODEL), igual feito em ltx25_backend.GGUF_VARIANTS. O arquivo
            # tem que estar em models/unet/ (categoria "unet", nao
            # "diffusion_models" -- e onde o node/GGUF le por padrao).
            api[N_UNET] = {"class_type": "UnetLoaderGGUF",
                           "inputs": {"unet_name": UNET_FILENAME}}
        else:
            api[N_UNET]["inputs"]["unet_name"] = UNET_FILENAME
        api[N_CLIP]["inputs"]["clip_name"] = CLIP_FILENAME
        if TRT_VAE:
            # Troca o node inteiro: VAELoader -> MiniMaxH3TRTVAELoader (saida
            # VAE identica). Os 4 links que apontavam pro 119 (VAEDecode E
            # o encode de referencia em N_R2V) continuam funcionando sem
            # mudar nada mais no grafo.
            api[N_VIDEO_VAE] = {
                "class_type": "MiniMaxH3TRTVAELoader",
                "inputs": {"decoder": TRT_DECODER_ENGINE, "encoder": TRT_ENCODER_ENGINE},
            }
        _base_api_cache = api
    return json.loads(json.dumps(_base_api_cache))  # copia rasa por job


def build_prompt_with_refs(prompt: str, n_refs: int, *, n_audio_refs: int = 0,
                           tag_style: str = "picture") -> str:
    """Prefixa o prompt com a citacao de referencia que o proprio MODELO
    espera -- e o exemplo oficial do workflow que ensina essa convencao
    (`Use <Picture 2> and <Picture 1> as reference frames...`). Sem citar, as
    imagens ainda condicionam a geracao (elas entram por input dedicado, nao
    so por texto), mas a citacao explicita e o que o exemplo oficial faz, e
    prompts de referencia SEM ela nao foram validados aqui.

    `tag_style`: "picture" (padrao, `<Picture N>`/`<Audio N>`, extraido do
    proprio workflow oficial deste checkout) ou "image" (`ImageN`/`AudioN`,
    sem colchete nem espaco, documentado em paginas de terceiros sobre o
    produto/API hospedado da MiniMax -- pode ser uma convencao de uma camada
    de parsing diferente da que o node LOCAL usa; existe aqui como opcao
    de teste A/B explicito, nao como substituicao assumida do padrao
    validado. Ver MEMORIAL 3.74."""
    se_picture = tag_style == "picture"
    if (n_refs <= 0 and n_audio_refs <= 0) or "<Picture" in prompt or "Image1" in prompt:
        return prompt
    partes = []
    if n_refs > 0:
        if se_picture:
            partes.append(" and ".join(f"<Picture {i + 1}>" for i in range(n_refs)))
        else:
            partes.append(" and ".join(f"Image{i + 1}" for i in range(n_refs)))
    if n_audio_refs > 0:
        if se_picture:
            partes.append(" and ".join(f"<Audio {i + 1}>" for i in range(n_audio_refs)))
        else:
            partes.append(" and ".join(f"Audio{i + 1}" for i in range(n_audio_refs)))
    citacoes = " and ".join(partes)
    plural = "s" if (n_refs + n_audio_refs) > 1 else ""
    return f"Use {citacoes} as reference{plural}. {prompt}"


def build_workflow(prompt: str, *, ref_images: list[str] | None = None,
                   ref_audios: list[str] | None = None,
                   aspect_ratio: str = DEFAULT_ASPECT,
                   megapixels: float = DEFAULT_MEGAPIXELS,
                   duration_seconds: float = 5.0, seed: int = 42,
                   turbo: bool = True, clip_on_cpu: bool = True,
                   filename_prefix: str = "minimax_h3") -> dict:
    """`clip_on_cpu`: manda o encoder de texto (Qwen3-VL 32B, 15,7 GB) rodar
    na CPU em vez de disputar VRAM com o UNET.

    MEDIDO 2026-08-29: com os dois na GPU, `--lowvram` tinha que trocar
    camada por camada entre CPU e GPU o tempo todo -- 15,7 GB de encoder +
    12,5 GB de UNET somam 28,2 GB, acima dos 24,5 GB da placa antes mesmo do
    latente. Um clipe de 10s em modo turbo (4 passos) passou de 30 min sem
    terminar. O encoder roda UMA VEZ por geração (antes da amostragem
    propriamente dita); o UNET roda em CADA passo. Tirar o maior peso do
    caminho que se repete é a otimização óbvia -- ver MEMORIAL 3.41."""
    api = base_api()
    api[N_PROMPT]["inputs"]["value"] = prompt
    api[N_RESOLUTION]["inputs"]["aspect_ratio"] = aspect_ratio
    api[N_RESOLUTION]["inputs"]["megapixels"] = megapixels
    api[N_DURATION_S]["inputs"]["value"] = duration_seconds
    api[N_SEED]["inputs"]["noise_seed"] = seed
    api[N_TURBO_SWITCH]["inputs"]["value"] = bool(turbo)
    api[N_CLIP]["inputs"]["device"] = "cpu" if clip_on_cpu else "default"
    api[N_SAVE]["inputs"]["filename_prefix"] = filename_prefix
    # `format` e COMFY_DYNAMICCOMBO_V3 -- o mesmo tipo de widget que
    # `comfy_workflow_tool.convert()` ja falhava em preencher no workflow do
    # LTX 2.5 (CLAUDE.md 3.16). MEDIDO 2026-08-29: sem isto, SaveVideo.execute()
    # falhava faltando o argumento na ULTIMA etapa do grafo, depois de toda a
    # amostragem (4 passos) e o VAEDecode ja terem terminado.
    api[N_SAVE]["inputs"]["format"] = "auto"

    refs = (ref_images or [])[:2]  # o grafo oficial so tem 2 slots ligados
    ref_nodes = [N_REF_IMAGE_0, N_REF_IMAGE_1]
    for i, node_id in enumerate(ref_nodes):
        if i < len(refs):
            api[node_id]["inputs"]["image"] = os.path.basename(refs[i])
        else:
            # Sem referencia nesse slot: remove o no e a entrada que apontava
            # pra ele, ou o ComfyUI tenta carregar o "example.png" do template.
            api.pop(node_id, None)
            if N_R2V in api:
                api[N_R2V]["inputs"].pop(f"ref_images.ref_image_{i}", None)

    # AUDIO de referencia (2026-09-08, MEMORIAL 3.74) -- ao contrario das
    # imagens, o workflow oficial NAO trazia nenhum slot de audio conectado
    # (confirmado inspecionando o grafo convertido: so ref_images.*, nada de
    # ref_audios.*), mesmo o node MiniMaxH3ReferenceToVideo aceitando
    # `ref_audios` e ja vindo com `audio_vae` ligado (node 120, a MESMA VAE
    # que decodifica o audio de saida). Os nos LoadAudio sao criados aqui,
    # por job -- nao em base_api() -- porque base_api() e cacheada uma vez e
    # compartilhada (copiada) entre jobs; nao ha "no fixo" de audio pra so
    # trocar o nome do arquivo como acontece com as imagens.
    audios = (ref_audios or [])[:2]
    for i, path in enumerate(audios):
        node_id = f"{N_REF_AUDIO_BASE}{i}"
        api[node_id] = {"class_type": "LoadAudio",
                        "inputs": {"audio": os.path.basename(path)}}
        api[N_R2V]["inputs"][f"ref_audios.ref_audio_{i}"] = [node_id, 0]
    return api


def submit_and_wait(api: dict, log_cb=None, timeout: int = 3600,
                    expect_node: str = N_SAVE) -> list[str]:
    def log(msg):
        _log(msg, log_cb)

    ensure_server(log_cb=log_cb)
    try:
        res = _post_json(f"{COMFY_SERVER}/prompt",
                         {"prompt": api, "client_id": str(uuid.uuid4())})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"ComfyUI (MiniMax H3) rejeitou o workflow "
                           f"(HTTP {e.code}):\n{body[:3000]}") from None

    pid = res.get("prompt_id")
    log(f"[minimax_h3] enfileirado: {pid}")
    t0 = time.time()
    last_report = 0.0
    while time.time() - t0 < timeout:
        hist = _get_json(f"{COMFY_SERVER}/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                outputs = entry.get("outputs") or {}
                if expect_node in outputs:
                    files = [it["filename"] for items in outputs[expect_node].values()
                             for it in (items if isinstance(items, list) else [])
                             if isinstance(it, dict) and it.get("filename")]
                else:
                    files = [it["filename"]
                             for _nid, out in outputs.items()
                             for _k, items in out.items()
                             for it in (items if isinstance(items, list) else [])
                             if isinstance(it, dict) and it.get("filename")]
                log(f"[minimax_h3] concluído em {time.time()-t0:.1f}s -> {files}")
                return files
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"Erro no ComfyUI (MiniMax H3): {msgs[-3:]}")
        elapsed = time.time() - t0
        if elapsed - last_report >= 15:
            last_report = elapsed
            log(f"[minimax_h3] gerando... {elapsed:.0f}s")
        time.sleep(3)
    raise RuntimeError(f"Timeout após {timeout}s.")


def generate(prompt: str, output_path: str, *, ref_images: list[str] | None = None,
            ref_audios: list[str] | None = None,
            aspect_ratio: str = DEFAULT_ASPECT, megapixels: float = DEFAULT_MEGAPIXELS,
            duration_seconds: float = 5.0, seed: int = 42, turbo: bool = True,
            clip_on_cpu: bool = True, auto_cite_refs: bool = True,
            tag_style: str = "picture", log_cb=None,
            timeout: int = 3600) -> str:
    """Gera um clipe MiniMax H3 e copia para *output_path*.

    `ref_images`: 0 a 2 caminhos de imagem (identidade/estilo de referencia).
    `ref_audios`: 0 a 2 caminhos de audio (timbre/cadencia/emocao de voz --
    ver MEMORIAL 3.74. Combine com falas exatas no `prompt`, nomeando quem
    fala, pra o modelo gerar a fala real em vez de inventar uma).
    `auto_cite_refs`: injeta a citacao no prompt automaticamente -- ver
    build_prompt_with_refs (`tag_style` escolhe `<Picture N>`/`<Audio N>`,
    padrao validado neste checkout, ou `ImageN`/`AudioN`, formato documentado
    pra um produto/API hospedado que pode nao ser este node local -- ver
    MEMORIAL 3.74 antes de trocar o padrao). `turbo`: LoRA de 4 passos
    (rapido) contra 20 passos sem ela. `clip_on_cpu`: ver build_workflow --
    default ligado porque MEDIDO ser a diferenca entre "nao termina" e
    terminar (3.41). `duration_seconds`: o grafo arredonda para o multiplo
    valido mais proximo (5 + 17k frames a 24fps) sozinho."""
    refs = ref_images or []
    audio_refs = ref_audios or []
    staged = [_stage_input(p) for p in refs]
    staged_audio = [_stage_input(p) for p in audio_refs]
    prompt_final = (build_prompt_with_refs(prompt, len(staged), n_audio_refs=len(staged_audio),
                                           tag_style=tag_style)
                    if auto_cite_refs else prompt)
    api = build_workflow(
        prompt_final, ref_images=staged, ref_audios=staged_audio, aspect_ratio=aspect_ratio,
        megapixels=megapixels, duration_seconds=duration_seconds, seed=seed,
        turbo=turbo, clip_on_cpu=clip_on_cpu,
        filename_prefix=os.path.splitext(os.path.basename(output_path))[0]
        or "minimax_h3")
    _log(f"[minimax_h3] {'turbo (4 passos)' if turbo else '20 passos'}, "
        f"{duration_seconds}s pedidos, {len(staged)} referencia(s) de imagem, "
        f"{len(staged_audio)} de audio, clip {'na CPU' if clip_on_cpu else 'na GPU'}", log_cb)
    try:
        files = submit_and_wait(api, log_cb=log_cb, timeout=timeout)
        if not files:
            raise RuntimeError("A geração terminou sem produzir arquivo de saída.")
        src = os.path.join(COMFY_OUTPUT, files[0])
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        shutil.copy2(src, output_path)
        return output_path
    finally:
        for p in staged + staged_audio:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MiniMax H3 reference-to-video via ComfyUI")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--ref-image", action="append", default=[], dest="ref_images")
    ap.add_argument("--ref-audio", action="append", default=[], dest="ref_audios",
                    help="audio de referencia (timbre/cadencia/emocao de voz), 0-2. "
                         "Combine com a fala exata no --prompt, nomeando o personagem.")
    ap.add_argument("--tag-style", default="picture", choices=["picture", "image"],
                    help="convencao de citacao de referencia no prompt: 'picture' "
                         "(<Picture N>/<Audio N>, padrao validado neste checkout) ou "
                         "'image' (ImageN/AudioN, documentado pra produto/API hospedado "
                         "-- teste A/B, ver MEMORIAL 3.74).")
    ap.add_argument("--aspect-ratio", default=DEFAULT_ASPECT)
    ap.add_argument("--megapixels", type=float, default=DEFAULT_MEGAPIXELS)
    ap.add_argument("--duration-seconds", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--turbo", action="store_true", default=True)
    ap.add_argument("--no-turbo", dest="turbo", action="store_false")
    ap.add_argument("--clip-on-cpu", action="store_true", default=True)
    ap.add_argument("--clip-on-gpu", dest="clip_on_cpu", action="store_false",
                    help="mantem o encoder de 32B na GPU -- so testado sem isto, "
                         "que travou por 30 min. Ver MEMORIAL 3.41.")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    out = generate(
        args.prompt, args.output_path, ref_images=args.ref_images,
        ref_audios=args.ref_audios, tag_style=args.tag_style,
        aspect_ratio=args.aspect_ratio, megapixels=args.megapixels,
        duration_seconds=args.duration_seconds, seed=args.seed, turbo=args.turbo,
        clip_on_cpu=args.clip_on_cpu, timeout=args.timeout)
    print(f"OK -> {out}")
