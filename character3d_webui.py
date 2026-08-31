"""MVP local: fotos -> referências FLUX -> pipeline 3D.

The heavy model runners are intentionally optional. This UI can be used to
prepare a reproducible job and will call ComfyUI when it is running.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
G = Path(os.environ.get("CHAR3D_MODEL_ROOT", r"G:\models"))
COMFY = Path(os.environ.get("COMFYUI_ROOT", str(ROOT / "ComfyUI")))
FLUX_NAME = "flux-2-klein-9b-fp8.safetensors"
FLUX_CANDIDATES = [G / FLUX_NAME, ROOT / "models" / FLUX_NAME,
                   COMFY / "models" / "checkpoints" / FLUX_NAME,
                   COMFY / "models" / "diffusion_models" / FLUX_NAME]
JOBS = ROOT / "outputs" / "character3d"
JOBS.mkdir(parents=True, exist_ok=True)
SAM_MODELS = ROOT / "sam-3d-body" / "sam_3d_body" / "models"
HUNYUAN_CACHE = G / "huggingface-cache" / "hub"

def status():
    flux = next((p for p in FLUX_CANDIDATES if p.exists()), None)
    rows = [
        f"FLUX 2 Klein 9B FP8: {'OK' if flux else 'não encontrado'}",
        f"SAM 3D Body: {'OK' if (SAM_MODELS / 'model.ckpt').exists() and (SAM_MODELS / 'mhr_model.pt').exists() else 'modelos ausentes'}",
        f"Hunyuan3D: {'OK' if HUNYUAN_CACHE.exists() else 'cache/modelos ausentes'}",
        f"ComfyUI: {'OK' if COMFY.exists() else 'não encontrado'}",
    ]
    return "\n\n".join(rows)


def next_job():
    """Reserva character_0001, character_0002 e assim por diante."""
    indexes = []
    for entry in JOBS.iterdir():
        match = re.fullmatch(r"character_(\d{4})", entry.name)
        if entry.is_dir() and match:
            indexes.append(int(match.group(1)))
    sequence = max(indexes, default=0) + 1
    job = JOBS / f"character_{sequence:04d}"
    job.mkdir()
    return sequence, job


def copy_references(photos, job):
    references_dir = job / "references"
    references_dir.mkdir()
    saved = []
    for index, item in enumerate(photos, start=1):
        source = Path(getattr(item, "name", item))
        destination = references_dir / f"reference_{index:02d}{source.suffix.lower() or '.png'}"
        shutil.copy2(source, destination)
        saved.append(str(destination))
    return saved


def run_hunyuan_with_progress(command, job, progress):
    """Mantém um log do Hunyuan e traduz as etapas longas para a barra da UI."""
    log_path = job / "hunyuan3d.log"
    tail = []
    pbr_started = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in iter(process.stdout.readline, ""):
            log.write(line)
            tail.append(line)
            tail = tail[-100:]
            if "Diffusion Sampling" in line:
                match = re.search(r"(\d{1,3})%", line)
                if match:
                    progress(0.62 + int(match.group(1)) / 100 * 0.10, desc="Hunyuan3D: criando geometria")
            elif "Fetching 13 files" in line or "Loading pipeline components" in line:
                pbr_started = True
                progress(0.74, desc="Hunyuan3D: carregando modelos de material PBR")
            elif pbr_started and ("custom timestep schedule" in line or "torch.range" in line):
                progress(0.80, desc="Hunyuan3D: renderizando textura PBR — esta fase pode levar vários minutos")
        return_code = process.wait()
    return return_code, "".join(tail), log_path


def run_pipeline(photos, prompt, quality, use_flux, progress=gr.Progress()):
    if not photos or not 1 <= len(photos) <= 3:
        raise gr.Error("Envie de 1 a 3 fotos da mesma pessoa.")

    sequence, job = next_job()
    progress(0.05, desc="Criando job sequencial")
    references = copy_references(photos, job)
    flux = next((path for path in FLUX_CANDIDATES if path.exists()), None)
    manifest = {
        "sequence": sequence, "photos": references, "prompt": prompt, "quality": quality,
        "flux_enabled": use_flux, "model_root": str(G), "comfyui_root": str(COMFY),
        "flux_checkpoint": str(flux) if flux else FLUX_NAME,
        "pipeline": ["sam3d_body", "hunyuan3d", "blender"],
    }
    (job / "job.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sam_out = job / "sam_body"; sam_out.mkdir()
    progress(0.1, desc="Executando SAM 3D Body")
    cmd = ["python", str(ROOT / "sam_body_worker.py"), str(job / "references"), str(sam_out)]
    sam = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if sam.returncode: raise gr.Error("SAM falhou: " + sam.stderr[-1200:])

    body_files = []
    for index, body in enumerate(sorted(sam_out.glob("*.obj")), start=1):
        target = sam_out / f"character_{sequence:04d}_body_{index:02d}.obj"
        body.replace(target)
        body_files.append(str(target))
    if not body_files:
        raise gr.Error("SAM 3D Body não produziu um OBJ.")

    progress(0.6, desc="Hunyuan3D: geometria e textura PBR — pode levar vários minutos")
    glb = job / f"character_{sequence:04d}_textured.glb"
    hun_code, hun_log, _ = run_hunyuan_with_progress(
        ["python", str(ROOT / "hunyuan_worker.py"), references[0], str(glb)], job, progress
    )
    if hun_code or not glb.exists():
        raise gr.Error("Hunyuan3D falhou: " + (hun_log[-1500:] if hun_log else "Não foi criado o arquivo GLB."))
    progress(1.0, desc="Concluído")
    message = f"### Personagem {sequence:04d} concluído\nUse o visualizador para girar e aproximar o GLB."
    return body_files, str(glb), str(glb), str(job), message


def open_artifact_folder(job_path):
    if not job_path:
        raise gr.Error("Execute um personagem antes de abrir a pasta.")
    path = Path(job_path).resolve()
    try:
        path.relative_to(JOBS.resolve())
    except ValueError as error:
        raise gr.Error("Pasta de artefatos inválida.") from error
    if not path.is_dir():
        raise gr.Error("A pasta de artefatos não existe mais.")
    subprocess.Popen(["explorer.exe", str(path)])
    return f"Pasta aberta: `{path}`"

with gr.Blocks(title="Character 3D Lab") as demo:
    gr.Markdown(
        "# Character 3D Lab\n"
        "**1–3 fotos → corpo-base SAM → malha PBR Hunyuan → arquivos 3D**\n\n"
        "Envie frente e, se possível, perfil/costas. O SAM processa todas as fotos; "
        "neste MVP o Hunyuan usa a primeira como referência visual principal."
    )
    with gr.Row():
        with gr.Column(scale=2):
            photos = gr.File(label="Fotos de referência (1 a 3)", file_count="multiple", file_types=["image"], type="filepath")
            prompt = gr.Textbox(label="Direção visual", value="personagem 3D realista, roupa e cabelo consistentes, fundo neutro")
            quality = gr.Radio(["MVP rápido", "Alta qualidade"], value="MVP rápido", label="Qualidade")
            use_flux = gr.Checkbox(True, label="Registrar FLUX para referências consistentes")
            run = gr.Button("Gerar personagem 3D", variant="primary", size="lg")
        with gr.Column():
            state = gr.Markdown(status())
            result_message = gr.Markdown()
            artifact_folder = gr.Textbox(label="Pasta dos artefatos", interactive=False)
            open_folder = gr.Button("Abrir pasta dos artefatos")
            folder_message = gr.Markdown()

    gr.Markdown("## Visualizador 3D")
    viewer = gr.Model3D(label="Modelo Hunyuan3D texturizado", height=560, clear_color=(0.08, 0.09, 0.12, 1.0))
    with gr.Row():
        body_files = gr.File(label="Corpos SAM 3D Body (.obj)", file_count="multiple")
        glb_file = gr.File(label="Modelo Hunyuan3D (.glb)")

    run.click(
        run_pipeline, [photos, prompt, quality, use_flux],
        [body_files, glb_file, viewer, artifact_folder, result_message], api_name=False,
    )
    open_folder.click(open_artifact_folder, artifact_folder, folder_message, api_name=False)
    gr.Markdown(
        "### Próxima etapa no Blender\n"
        "Use o OBJ do SAM como corpo/rig de referência e o GLB texturizado como geometria visual. "
        "A transferência de pesos e a retopologia continuam necessárias para animação confiável."
    )

if __name__ == "__main__":
    # show_api removido no Gradio 5/6 (temos 6.20) -- ver video_doctor_ui.
    # theme movido do Blocks para o launch (Gradio 6).
    demo.launch(server_name=os.environ.get("LTX_UI_HOST", "127.0.0.1"), server_port=int(os.getenv("CHAR3D_PORT", "7865")),
                share=False, theme=gr.themes.Soft())
