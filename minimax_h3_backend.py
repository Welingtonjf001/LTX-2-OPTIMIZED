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
        --output-path out.mp4 --duration-seconds 5 --turbo
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
N_SEED = "129"             # RandomNoise
N_TURBO_SWITCH = "146"     # PrimitiveBoolean: liga a LoRA turbo (4 passos) e o node 144
N_R2V = "136"              # MiniMaxH3ReferenceToVideo -- dono dos slots ref_images.*
N_CLIP = "128"             # CLIPLoader -- Qwen3-VL 32B, 15,7 GB so ele
N_SAVE = "92"              # SaveVideo

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
        return
    with _boot_lock():
        if server_is_up():
            return
        _ensure_server_locked(log_cb=log_cb, boot_timeout=boot_timeout)


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
    cmd = [python_exe, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT),
           "--cuda-device", "1", "--disable-auto-launch"]
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
                    N_REF_IMAGE_1, N_SEED, N_TURBO_SWITCH, N_R2V, N_CLIP, N_SAVE]
        faltando = [n for n in esperados if n not in api]
        if faltando:
            raise RuntimeError(
                "O workflow do MiniMax H3 mudou de estrutura: nós esperados "
                f"ausentes {faltando}. Reexporte `workflows/video_minimax_h3_r2v.json` "
                "do editor do ComfyUI e reveja os IDs no topo deste módulo.")
        _base_api_cache = api
    return json.loads(json.dumps(_base_api_cache))  # copia rasa por job


def build_prompt_with_refs(prompt: str, n_refs: int) -> str:
    """Prefixa o prompt com a citacao de referencia que o proprio MODELO
    espera -- e o exemplo oficial do workflow que ensina essa convencao
    (`Use <Picture 2> and <Picture 1> as reference frames...`). Sem citar, as
    imagens ainda condicionam a geracao (elas entram por input dedicado, nao
    so por texto), mas a citacao explicita e o que o exemplo oficial faz, e
    prompts de referencia SEM ela nao foram validados aqui."""
    if n_refs <= 0 or "<Picture" in prompt:
        return prompt
    citacoes = " and ".join(f"<Picture {i + 1}>" for i in range(n_refs))
    return f"Use {citacoes} as reference frame{'s' if n_refs > 1 else ''}. {prompt}"


def build_workflow(prompt: str, *, ref_images: list[str] | None = None,
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
            aspect_ratio: str = DEFAULT_ASPECT, megapixels: float = DEFAULT_MEGAPIXELS,
            duration_seconds: float = 5.0, seed: int = 42, turbo: bool = True,
            clip_on_cpu: bool = True, auto_cite_refs: bool = True, log_cb=None,
            timeout: int = 3600) -> str:
    """Gera um clipe MiniMax H3 e copia para *output_path*.

    `ref_images`: 0 a 2 caminhos de imagem (identidade/estilo de referencia).
    `auto_cite_refs`: injeta a citacao `<Picture N>` no prompt automaticamente
    -- ver build_prompt_with_refs. `turbo`: LoRA de 4 passos (rapido) contra
    20 passos sem ela. `clip_on_cpu`: ver build_workflow -- default ligado
    porque MEDIDO ser a diferenca entre "nao termina" e terminar (3.41).
    `duration_seconds`: o grafo arredonda para o multiplo valido mais
    proximo (5 + 17k frames a 24fps) sozinho."""
    refs = ref_images or []
    staged = [_stage_input(p) for p in refs]
    prompt_final = build_prompt_with_refs(prompt, len(staged)) if auto_cite_refs else prompt
    api = build_workflow(
        prompt_final, ref_images=staged, aspect_ratio=aspect_ratio,
        megapixels=megapixels, duration_seconds=duration_seconds, seed=seed,
        turbo=turbo, clip_on_cpu=clip_on_cpu,
        filename_prefix=os.path.splitext(os.path.basename(output_path))[0]
        or "minimax_h3")
    _log(f"[minimax_h3] {'turbo (4 passos)' if turbo else '20 passos'}, "
        f"{duration_seconds}s pedidos, {len(staged)} referencia(s), "
        f"clip {'na CPU' if clip_on_cpu else 'na GPU'}", log_cb)
    try:
        files = submit_and_wait(api, log_cb=log_cb, timeout=timeout)
        if not files:
            raise RuntimeError("A geração terminou sem produzir arquivo de saída.")
        src = os.path.join(COMFY_OUTPUT, files[0])
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        shutil.copy2(src, output_path)
        return output_path
    finally:
        for p in staged:
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
        aspect_ratio=args.aspect_ratio, megapixels=args.megapixels,
        duration_seconds=args.duration_seconds, seed=args.seed, turbo=args.turbo,
        clip_on_cpu=args.clip_on_cpu, timeout=args.timeout)
    print(f"OK -> {out}")
