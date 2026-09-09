"""Gradio test bench for the installed MiniMax H3 ComfyUI (port 8189)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import uuid

import gradio as gr

ROOT = Path(__file__).resolve().parent
COMFY = Path(os.environ.get("MINIMAX_H3_ROOT", r"E:\Users\home\Documents\MiniMax-H3")) / "ComfyUI"
SERVER = "http://127.0.0.1:" + os.environ.get("MINIMAX_H3_PORT", "8189")
OUTPUT = ROOT / "outputs" / "minimax_h3_test"
MODEL = "minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors"
CLIP = "qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
DEFAULT_PROMPT = "A cinematic shot of a bamboo forest at night, leaves swaying gently in the wind, soft moonlight, natural ambient sounds."


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:6000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ComfyUI indisponível em {SERVER}. Execute MiniMax-H3/start_minimax_h3.bat.") from exc


def build_workflow(prompt, width=640, height=384, seconds=5, seed=42, turbo=True, refs=(), prefix="minimax_h3_test", clip_on_cpu=False):
    if not prompt or not prompt.strip():
        raise ValueError("Informe um prompt.")
    if any(int(v) != v or v < 32 or v > 1344 or v % 32 for v in (width, height)):
        raise ValueError("Largura e altura devem ser múltiplos de 32, entre 32 e 1344.")
    if not math.isfinite(seconds) or not 0 < seconds <= 15:
        raise ValueError("Use duração entre 0 e 15 segundos para este teste.")
    if int(seed) != seed or not 0 <= seed <= 2**53 - 1:
        raise ValueError("Seed deve ser um inteiro entre 0 e 2^53-1.")
    if len(refs) > 2:
        raise ValueError("Use no máximo duas referências.")
    frames = max(5, round(seconds * 24))
    frames += (5 - frames % 17) % 17
    tags = [f"<Picture {i + 1}>" for i in range(len(refs)) if f"<Picture {i + 1}>" not in prompt]
    if tags:
        prompt = "Use " + " and ".join(tags) + " as visual references.\n" + prompt

    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    graph = {
        "1": node("UNETLoader", unet_name=MODEL, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=CLIP, type="minimax", device="cpu" if clip_on_cpu else "default"),
        "3": node("VAELoader", vae_name=VIDEO_VAE),
        "4": node("VAELoader", vae_name=AUDIO_VAE),
        "5": node("MiniMaxH3ReferenceToVideo", clip=["2", 0], vae=["3", 0], audio_vae=["4", 0],
                  prompt=prompt, width=int(width), height=int(height), length=frames, ref_image_size="match"),
        "6": node("BasicGuider", model=["15" if turbo else "1", 0], conditioning=["5", 0]),
        "7": node("RandomNoise", noise_seed=int(seed)),
        "8": node("KSamplerSelect", sampler_name="res_multistep"),
        "9": node("BasicScheduler", model=["1", 0], scheduler="simple", steps=4 if turbo else 20, denoise=1.0),
        "10": node("SamplerCustomAdvanced", noise=["7", 0], guider=["6", 0], sampler=["8", 0],
                   sigmas=["9", 0], latent_image=["5", 1]),
        "11": node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": node("CreateVideo", images=["11", 0], audio=["12", 0], fps=24.0),
        "14": node("SaveVideo", video=["13", 0], filename_prefix=prefix, format="mp4", **{"format.codec": "h264"}),
    }
    if turbo:
        graph["15"] = node("LoraLoaderModelOnly", model=["1", 0], lora_name=LORA, strength_model=1.0)
    for index, name in enumerate(refs):
        key = str(20 + index)
        graph[key] = node("LoadImage", image=name)
        graph["5"]["inputs"][f"ref_images.ref_image_{index}"] = [key, 0]
    return graph


def validate_environment(graph):
    info = request("/object_info")
    for entry in graph.values():
        kind = entry["class_type"]
        if kind not in info:
            raise ValueError(f"Nó ausente no ComfyUI: {kind}")
        required = info[kind]["input"].get("required", {})
        for key, spec in required.items():
            if key not in entry["inputs"]:
                raise ValueError(f"Entrada ausente: {kind}.{key}")
            value = entry["inputs"][key]
            if isinstance(spec[0], list) and not isinstance(value, list) and value not in spec[0]:
                raise ValueError(f"Opção/modelo indisponível: {kind}.{key} = {value}")


def diagnose():
    try:
        validate_environment(build_workflow(DEFAULT_PROMPT))
        queue = request("/queue")
        return f"OK: nós e modelos disponíveis em {SERVER}. Fila: {len(queue.get('queue_running', []))} em execução, {len(queue.get('queue_pending', []))} pendentes."
    except Exception as exc:
        return str(exc)


def generate(prompt, image1, image2, width, height, seconds, seed, turbo, clip_on_cpu=False):
    run = OUTPUT / uuid.uuid4().hex[:12]
    run.mkdir(parents=True, exist_ok=True)
    staged = []
    lines = []
    graph_path = run / "workflow_api.json"
    try:
        # Validate parameters before copying files or submitting GPU work.
        graph = build_workflow(prompt, width, height, seconds, seed, turbo, clip_on_cpu=clip_on_cpu)
        validate_environment(graph)
        for path in (image1, image2):
            if path:
                source = Path(path)
                target = COMFY / "input" / f"gradio_h3_{uuid.uuid4().hex}{source.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                staged.append(target)
        graph = build_workflow(prompt, width, height, seconds, seed, turbo,
                               [p.name for p in staged], f"gradio_h3_{run.name}", clip_on_cpu=clip_on_cpu)
        graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        result = request("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
        pid = result["prompt_id"]
        lines.append(f"Prompt: {pid}\n{graph['5']['inputs']['length']} frames, 24 fps. Encoder: {'CPU' if clip_on_cpu else 'automático (ComfyUI)'}. Modelo mantém o sampler nativo de áudio/vídeo.")
        started = time.monotonic()
        while True:
            history = request(f"/history/{pid}").get(pid)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    for event, detail in status.get("messages", []):
                        if event == "execution_error":
                            raise RuntimeError(f"Nó {detail.get('node_id')} ({detail.get('node_type')}): "
                                               f"{detail.get('exception_message')}\n" + "".join(detail.get("traceback", [])))
                        if event == "execution_interrupted":
                            raise RuntimeError("Geração cancelada no ComfyUI.")
                    raise RuntimeError("ComfyUI informou falha. Consulte o log do servidor.")
                if status.get("completed") or status.get("status_str") == "success":
                    for items in history.get("outputs", {}).get("14", {}).values():
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            if not isinstance(item, dict) or not item.get("filename"):
                                continue
                            source = (COMFY / "output" / item.get("subfolder", "") / item["filename"]).resolve()
                            if not source.is_relative_to((COMFY / "output").resolve()):
                                raise RuntimeError("Saída fora da pasta output do ComfyUI.")
                            if source.suffix.lower() != ".mp4":
                                continue
                            video = run / source.name
                            shutil.copy2(source, video)
                            lines.append(f"Concluído em {time.monotonic() - started:.0f}s: {video}")
                            yield str(video), "\n".join(lines), str(graph_path)
                            return
                    raise RuntimeError("ComfyUI terminou sem retornar um MP4 no nó SaveVideo.")
            if time.monotonic() - started > 3600:
                raise TimeoutError(f"Espera excedeu 1 hora. Prompt {pid} pode continuar no ComfyUI; consulte a fila em {SERVER}.")
            yield None, "\n".join(lines) + f"\nAguardando ComfyUI: {time.monotonic() - started:.0f}s", str(graph_path)
            time.sleep(2)
    except Exception as exc:
        lines.append(f"ERRO: {exc}")
        yield None, "\n".join(lines), str(graph_path) if graph_path.exists() else None
    finally:
        (run / "log.txt").write_text("\n".join(lines), encoding="utf-8")
        # Keep staged inputs: queued work can outlive a browser session or timeout.


def build_ui():
    with gr.Blocks(title="MiniMax H3 — Teste") as demo:
        gr.Markdown("# MiniMax H3 · Teste de vídeo + áudio\nWorkflow corrigido com sampler AV nativo, encoder MiniMax e VAEs separados.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=6)
                with gr.Row():
                    image1 = gr.Image(label="Referência 1 (opcional)", type="filepath")
                    image2 = gr.Image(label="Referência 2 (opcional)", type="filepath")
                with gr.Row():
                    width = gr.Slider(256, 1344, value=640, step=32, label="Largura")
                    height = gr.Slider(256, 1344, value=384, step=32, label="Altura")
                seconds = gr.Slider(1, 15, value=5, step=0.5, label="Duração solicitada (s)", info="24 fps; arredondamento para 5 + 17k frames. Comece com 5 segundos.")
                seed = gr.Number(value=42, precision=0, label="Seed")
                turbo = gr.Checkbox(value=True, label="Turbo: LoRA de 4 passos (desligado: 20 passos)")
                clip_on_cpu = gr.Checkbox(value=False, label="Encoder na CPU (mais lento; reduz uso de VRAM)")
                button = gr.Button("Gerar vídeo com áudio", variant="primary")
            with gr.Column():
                video = gr.Video(label="Resultado")
                status = gr.Textbox(label="Progresso / erro", lines=12)
                workflow = gr.File(label="Workflow API utilizado")
                check = gr.Button("Verificar conexão, nós e modelos")
                environment = gr.Textbox(label="Diagnóstico", interactive=False)
        gr.Markdown(f"Backend: [{SERVER}]({SERVER}). Arquivos em `outputs/minimax_h3_test`. O teste usa a mesma GPU do ComfyUI.")
        button.click(generate, [prompt, image1, image2, width, height, seconds, seed, turbo, clip_on_cpu],
                     [video, status, workflow], concurrency_limit=1, api_name="generate")
        check.click(diagnose, outputs=environment, queue=False)
    return demo


if __name__ == "__main__":
    build_ui().queue(max_size=4).launch(server_name="127.0.0.1", server_port=int(os.environ.get("MINIMAX_H3_UI_PORT", "7916")),
                                      allowed_paths=[str(OUTPUT)])
