"""WebUI de teste para o LoRA `ltx_2.3_Multi-Ref Character Storyboard_V2` (e
qualquer outro LoRA em models/loras/) sobre o checkpoint LTX 2.3.

Por que uma UI dedicada em vez de encaixar isso nas UIs de produção
(music_maker_ui, screenplay_ui etc.): estas usam o pipeline `ltx_pipelines`
nativo (subprocesso .venv), não ComfyUI, para o vídeo 2.3 -- só o storyboard
(stills) delas passa por ComfyUI. Testar um LoRA baixado de fora (o post do
RunningHub que motivou isso) especificamente com controle fino de força do
LoRA, sem afetar as UIs validadas, pediu um caminho próprio.

Arquitetura final (híbrida, depois de 4 bugs investigados e corrigidos -- ver
memory/project_lora_storyboard_test_ui.md para a história completa):
  - Texto (Gemma3) é codificado pela rota NATIVA já validada em produção
    (`ltx_pipelines`/`model_ledger.py`, via lora_storyboard_encode.py), não
    pelo ComfyUI -- o carregamento do Gemma3 pelo `LTXVGemmaCLIPModelLoader`
    do ComfyUI trava nesta GPU (bug do subsistema aimdo/DynamicVRAM,
    documentado na memória).
  - Vídeo (checkpoint 2.3 + LoRA + sampler + VAE decode) roda no ComfyUI --
    essa parte nunca teve problema; só precisa do conditioning no formato
    certo (vídeo+áudio concatenados -- este checkpoint é um modelo conjunto
    áudio+vídeo).
Ver lora_storyboard_backend.py para o grafo ComfyUI montado à mão (evita o
bug documentado do conversor de subgrafos) e lora_storyboard_encode.py para
a codificação nativa do texto.

Porta 7915 -- ver CLAUDE.md para a lista de portas já usadas neste repo.
"""
import os
import threading
import traceback

import gradio as gr

import lora_storyboard_backend as backend

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "lora_storyboard_test")

_LOG_LOCK = threading.Lock()
_RUN_STATE = {"log": [], "running": False}


def _log_cb(msg: str) -> None:
    with _LOG_LOCK:
        _RUN_STATE["log"].append(msg)


def stop_generation() -> str:
    return backend.interrupt()


def _generate(prompt, negative, lora_name, lora_strength, width, height, frames,
             fps, steps, cfg, seed, image_path, image_strength):
    if not lora_name:
        yield "Nenhum LoRA encontrado em models/loras/.", None
        return
    if not prompt.strip():
        yield "Informe um prompt.", None
        return

    with _LOG_LOCK:
        _RUN_STATE["log"] = ["Enfileirando no ComfyUI (sobe o servidor compartilhado se ainda não estiver no ar)..."]
        _RUN_STATE["running"] = True

    result = {"path": None, "error": None}

    def worker():
        try:
            result["path"] = backend.generate(
                prompt=prompt, output_dir=OUTPUT_DIR,
                lora_name=lora_name, lora_strength=float(lora_strength),
                negative=negative, width=int(width), height=int(height),
                num_frames=int(frames), frame_rate=float(fps),
                steps=int(steps), cfg=float(cfg), seed=int(seed),
                image_path=image_path or None, image_strength=float(image_strength),
                log_cb=_log_cb,
            )
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            with _LOG_LOCK:
                _RUN_STATE["running"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        with _LOG_LOCK:
            log_text = "\n".join(_RUN_STATE["log"][-500:])
            running = _RUN_STATE["running"]
        yield log_text, None
        if not running:
            break
        t.join(timeout=1.0)

    if result["error"]:
        yield log_text + "\n\n[Falhou]\n" + result["error"][-3000:], None
    else:
        yield log_text + f"\n\nOK: {result['path']}", result["path"]


def build_ui() -> gr.Blocks:
    loras = backend.list_loras()
    default_lora = next((l for l in loras if "multi-ref" in l.lower() or "storyboard" in l.lower()), None) \
        or (loras[0] if loras else None)

    with gr.Blocks(title="Teste de LoRA — LTX 2.3 (ComfyUI)") as demo:
        gr.Markdown(
            "# Teste de LoRA — LTX 2.3\n"
            "Roda o checkpoint `ltx-2.3-22b-distilled-fp8` + um LoRA de "
            "`models/loras/`. Rota híbrida: o texto é codificado pela rota "
            "**nativa** (mesmo código já validado em produção — o carregamento "
            "de texto pelo ComfyUI trava nesta GPU), e o vídeo (checkpoint + "
            "LoRA + sampler) roda no **ComfyUI**, reaproveitando o servidor "
            "compartilhado das outras UIs deste projeto (stills, 2.5) se já "
            "estiver no ar."
        )
        with gr.Row():
            lora_name = gr.Dropdown(label="LoRA", choices=loras, value=default_lora,
                                    allow_custom_value=True)
            lora_strength = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="Força do LoRA")
        prompt = gr.Textbox(label="Prompt", lines=4)
        negative = gr.Textbox(label="Prompt negativo", lines=2,
                              value=backend.DEFAULT_NEGATIVE)
        with gr.Row():
            width = gr.Slider(256, 1280, value=768, step=32, label="Largura")
            height = gr.Slider(256, 768, value=512, step=32, label="Altura")
            frames = gr.Slider(9, 361, value=97, step=8, label="Frames",
                               info="9 a 361 (=1+8k) — a 24 fps até ~15s.")
        with gr.Row():
            fps = gr.Slider(12, 30, value=24, step=1, label="FPS")
            steps = gr.Slider(1, 30, value=8, step=1, label="Passos de denoise",
                              info="O checkpoint distilled foi ajustado para ~8.")
            cfg = gr.Slider(0.0, 15.0, value=3.0, step=0.1, label="CFG")
            seed = gr.Number(value=1234, precision=0, label="Seed")
        with gr.Accordion("Imagem → vídeo (opcional)", open=False):
            image_in = gr.Image(label="Imagem de condicionamento", type="filepath")
            image_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Força do condicionamento")
        with gr.Row():
            generate_btn = gr.Button("Gerar", variant="primary")
            stop_btn = gr.Button("Parar geração", variant="stop")
        status = gr.Textbox(label="Status / log ao vivo", lines=16, max_lines=30, autoscroll=True)
        video = gr.Video(label="Resultado", format="mp4")

        gen_event = generate_btn.click(
            _generate,
            inputs=[prompt, negative, lora_name, lora_strength, width, height, frames,
                    fps, steps, cfg, seed, image_in, image_strength],
            outputs=[status, video],
        )
        stop_btn.click(stop_generation, outputs=[status], cancels=[gen_event])
    return demo


def standalone(port: int = 7915):
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7915)
    standalone(ap.parse_args().port)
