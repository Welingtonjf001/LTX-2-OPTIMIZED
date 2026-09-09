"""UI Gradio para o skill img2threejs: pipeline imagem->3D, visualizador e
exportador de modelo.

Por que existe: o img2threejs (clonado em `E:\\Users\\home\\Documents\\img2threejs`,
junction em `~/.claude/skills/img2threejs`) roda como skill do Claude Code — a
geração em si (blockout -> structural -> form -> ... -> optimization) depende de
revisão visual feita pelo AGENTE a cada passe, então não dá para automatizar
ponta a ponta num botão Gradio: o julgamento de "isso bate com a referência" é
visual, não script. O que esta UI automatiza é tudo que É determinístico nesse
caminho:

  1. Rodar o intake (`forge/stage1_intake/probe_image.py`) e ESCALONAR o
     assessment/spec (`new_pre_spec_assessment.py`, `new_sculpt_spec.py`) —
     essas duas últimas escrevem um ESQUELETO com campos "unassessed"; o
     conteúdo real (proporções, materiais, inventário de detalhes) precisa
     vir de inspeção visual — cole aqui o spec.json que o `/img2threejs` no
     Claude Code já tiver terminado, ou edite o esqueleto à mão para um teste
     simples.
  2. Validar o spec (`validate_sculpt_spec.py --strict-quality`).
  3. Gerar a factory TypeScript (`generate_threejs_factory.py`).
  4. Congelar essa factory num `.glb` (via tools/img2threejs_export/, Node +
     three.js + esbuild — Gradio não roda Three.js/TS ao vivo) e exibir no
     viewer 3D orbitável ao lado da referência, com download.

A assinatura real gerada por `generate_threejs_factory.py` é
`create{Nome}Model(options: ProceduralModelOptions = {})` — um parâmetro só; o
ObjectSculptSpec fica embutido no código gerado, não é passado em runtime
(confirmado lendo generate_threejs_factory.py:2288,3444 — o README do skill
descreve `createXModel(spec, options)`, que não bate com o gerador real).

Três abas:
  1. Imagem -> 3D (pipeline) — o fluxo acima, com pausa manual no gate de
     revisão (não tem como pular).
  2. Converter factory TypeScript -> GLB — usa isso direto se você já tem uma
     factory pronta (gerada aqui ou pelo /img2threejs no Claude Code).
  3. Visualizar modelo pronto — só upload/órbita/download de um .glb já feito.

Porta 7914 — 7803/7804/7903/7904/7911/7810/7960/7961/7912/7913/7860 já usadas
pelas outras UIs deste repo (ver CLAUDE.md).
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback

import gradio as gr

_ROOT = os.path.dirname(os.path.abspath(__file__))
_EXPORT_DIR = os.path.join(_ROOT, "tools", "img2threejs_export")
_EXPORT_SCRIPT = os.path.join(_EXPORT_DIR, "export_to_glb.mjs")
_NODE_MODULES = os.path.join(_EXPORT_DIR, "node_modules")

_SKILL_ROOT = r"E:\Users\home\Documents\img2threejs"
_FORGE = os.path.join(_SKILL_ROOT, "forge")
_RUNS_DIR = os.path.join(_ROOT, "img2threejs_runs")
_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")


def _node_ready():
    if shutil.which("node") is None:
        return False, "Node.js não encontrado no PATH. Instale Node 18+ para converter factories TS."
    if not os.path.isdir(_NODE_MODULES):
        return False, (f"Dependências não instaladas. Rode uma vez:\n"
                        f"  cd \"{_EXPORT_DIR}\" && npm install")
    return True, ""


def convert_factory(factory_file, fn_name, options_file):
    if not factory_file:
        return "Envie o arquivo da factory (.ts).", None
    if not fn_name or not fn_name.strip():
        return "Informe o nome da função exportada (ex.: createObjectNameModel).", None

    ok, msg = _node_ready()
    if not ok:
        return msg, None

    out_dir = tempfile.mkdtemp(prefix="img2threejs_glb_")
    out_path = os.path.join(out_dir, "model.glb")

    cmd = ["node", _EXPORT_SCRIPT,
           "--factory", factory_file, "--fn", fn_name.strip(),
           "--out", out_path]
    if options_file:
        cmd += ["--options", options_file]

    try:
        proc = subprocess.run(cmd, cwd=_EXPORT_DIR, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=180)
    except subprocess.TimeoutExpired:
        return "Conversão excedeu 180s — a factory pode depender de algo pesado ou travar em Node.", None
    except Exception:
        return f"Falha ao chamar o conversor Node:\n{traceback.format_exc()[-1500:]}", None

    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 or not os.path.exists(out_path):
        return f"Falhou (código {proc.returncode}):\n{log[-2000:]}", None
    return f"OK.\n{log.strip()}", out_path


# --- Pipeline imagem -> 3D (etapas deterministicas do forge/) ---------------

def _slugify(name):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "object"


def _run_py(args, cwd, timeout=120):
    cmd = [_PY, *args]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=env)
        return proc.returncode, (proc.stdout or ""), (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "", f"excedeu {timeout}s"
    except Exception:
        return -1, "", traceback.format_exc()[-2000:]


def intake_and_scaffold(image_path, target_name, is_character):
    if not image_path or not os.path.exists(image_path):
        return "Envie uma imagem de referência.", "", None
    if not target_name or not target_name.strip():
        return "Dê um nome ao objeto/personagem (usado no nome da factory).", "", None
    if not os.path.isdir(_FORGE):
        return f"Skill não encontrado em {_SKILL_ROOT} — confira a instalação.", "", None

    slug = _slugify(target_name)
    workdir = os.path.join(_RUNS_DIR, f"{slug}_{int(time.time())}")
    os.makedirs(workdir, exist_ok=True)

    logs = [f"Diretório de trabalho: {workdir}"]

    rc, out, err = _run_py(
        [os.path.join(_FORGE, "stage1_intake", "probe_image.py"), image_path],
        cwd=workdir,
    )
    logs.append("--- probe_image ---\n" + (out or err))
    if rc == 0:
        with open(os.path.join(workdir, "probe.json"), "w", encoding="utf-8") as f:
            f.write(out)

    assessment_path = os.path.join(workdir, "assessment.json")
    args = [os.path.join(_FORGE, "stage2_spec", "new_pre_spec_assessment.py"),
            target_name.strip(), "--image", image_path, "--out", assessment_path]
    if is_character:
        args.append("--character")
    rc, out, err = _run_py(args, cwd=workdir)
    logs.append("--- new_pre_spec_assessment ---\n" + (out + err).strip())
    if rc != 0 or not os.path.exists(assessment_path):
        return "\n\n".join(logs), "", None

    spec_path = os.path.join(workdir, "spec.json")
    args = [os.path.join(_FORGE, "stage2_spec", "new_sculpt_spec.py"),
            target_name.strip(), "--image", image_path,
            "--assessment", assessment_path, "--out", spec_path]
    if is_character:
        args.append("--character")
    rc, out, err = _run_py(args, cwd=workdir)
    logs.append("--- new_sculpt_spec ---\n" + (out + err).strip())
    if rc != 0 or not os.path.exists(spec_path):
        return "\n\n".join(logs), "", None

    with open(spec_path, "r", encoding="utf-8") as f:
        spec_text = f.read()

    logs.append(
        "\nEsqueleto criado. Os campos vêm como \"unassessed\"/vazios — isso é "
        "esperado: o script só monta a estrutura, quem preenche proporções, "
        "materiais e o inventário de detalhes é inspeção visual (o "
        "/img2threejs no Claude Code, ou você editando o JSON abaixo à mão "
        "para um teste simples). Sem isso preenchido, a validação --strict-quality "
        "no passo 2 vai reprovar — por desenho, não bug."
    )
    return "\n\n".join(logs), spec_text, workdir


def validate_spec(spec_text, workdir):
    if not workdir:
        return "Rode o passo 1 primeiro (cria o diretório de trabalho)."
    try:
        json.loads(spec_text)
    except json.JSONDecodeError as e:
        return f"spec.json não é JSON válido: {e}"
    spec_path = os.path.join(workdir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_text)

    rc, out, err = _run_py(
        [os.path.join(_FORGE, "stage2_spec", "validate_sculpt_spec.py"),
         spec_path, "--json", "--strict-quality"],
        cwd=workdir,
    )
    log = (out + err).strip()
    try:
        result = json.loads(out)
        status = "PASS" if result.get("ok") else "FAIL"
        lines = [f"{status}"]
        for w in result.get("warnings", []):
            lines.append(f"  warning: {w}")
        for e in result.get("errors", []):
            lines.append(f"  error: {e}")
        return "\n".join(lines)
    except Exception:
        return log or f"código de saída {rc}"


def generate_factory(spec_text, workdir, target_name):
    if not workdir:
        return "Rode o passo 1 primeiro.", "", ""
    try:
        json.loads(spec_text)
    except json.JSONDecodeError as e:
        return f"spec.json não é JSON válido: {e}", "", ""
    spec_path = os.path.join(workdir, "spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_text)

    factory_path = os.path.join(workdir, "factory.ts")
    rc, out, err = _run_py(
        [os.path.join(_FORGE, "stage3_build", "generate_threejs_factory.py"),
         spec_path, "--out", factory_path, "--force"],
        cwd=workdir,
    )
    log = (out + err).strip()
    if rc != 0 or not os.path.exists(factory_path):
        return f"Bloqueado (código {rc}):\n{log}", "", ""

    with open(factory_path, "r", encoding="utf-8") as f:
        code = f.read()
    m = re.search(r"export function (create\w+Model)\(", code)
    fn_name = m.group(1) if m else ""
    return f"OK: {factory_path}\n{log}", factory_path, fn_name


def build_ui():
    with gr.Blocks(title="img2threejs — Imagem para 3D") as demo:
        gr.Markdown(
            "# img2threejs — Imagem para 3D\n"
            "Pipeline do skill `img2threejs`: imagem de referência → spec → "
            "factory Three.js → `.glb` visualizável. O único passo que esta UI "
            "**não automatiza** é o julgamento visual do spec (proporções, "
            "materiais, inventário de detalhes) — isso é o motivo do skill "
            "existir como agente e não como script; rode `/img2threejs` no "
            "Claude Code para preencher isso de verdade, ou edite o JSON à mão "
            "para um objeto simples."
        )

        with gr.Tab("1. Imagem → 3D (pipeline)"):
            with gr.Row():
                pipe_img = gr.Image(label="Imagem de referência", type="filepath")
                with gr.Column():
                    pipe_name = gr.Textbox(label="Nome do objeto/personagem",
                                           placeholder="ex.: War Hauler Truck")
                    pipe_character = gr.Checkbox(label="É um personagem (trilha de anatomia)?")
                    step1_btn = gr.Button("1. Criar assessment + spec (esqueleto)", variant="primary")
            log1 = gr.Textbox(label="Log — intake/scaffold", lines=8, interactive=False)

            spec_code = gr.Code(
                label="spec.json — edite aqui (ou cole um spec já completo, ex. produzido pelo /img2threejs)",
                language="json", lines=18,
            )
            workdir_state = gr.State(None)

            with gr.Row():
                step2_btn = gr.Button("2. Validar (--strict-quality)")
                step3_btn = gr.Button("3. Gerar factory .ts")
            log2 = gr.Textbox(label="Resultado da validação", lines=6, interactive=False)
            log3 = gr.Textbox(label="Log — geração da factory", lines=6, interactive=False)

            factory_state = gr.State("")
            fn_state = gr.State("")
            step4_btn = gr.Button("4. Converter para GLB e visualizar", variant="primary")
            with gr.Row():
                pipe_ref_echo = gr.Image(label="Imagem de referência", type="filepath", interactive=False)
                pipe_model_view = gr.Model3D(label="Resultado", interactive=False)
            log4 = gr.Textbox(label="Log — conversão GLB", lines=4, interactive=False)

            step1_btn.click(fn=intake_and_scaffold,
                            inputs=[pipe_img, pipe_name, pipe_character],
                            outputs=[log1, spec_code, workdir_state])
            step2_btn.click(fn=validate_spec, inputs=[spec_code, workdir_state], outputs=[log2])
            step3_btn.click(fn=generate_factory, inputs=[spec_code, workdir_state, pipe_name],
                            outputs=[log3, factory_state, fn_state])

            def _convert_and_echo(factory_path, fn_name, ref_img):
                log, glb = convert_factory(factory_path, fn_name, None)
                return log, glb, ref_img

            step4_btn.click(fn=_convert_and_echo, inputs=[factory_state, fn_state, pipe_img],
                            outputs=[log4, pipe_model_view, pipe_ref_echo])

        with gr.Tab("2. Converter factory TypeScript → GLB"):
            gr.Markdown(
                "Converte uma factory `create{Nome}Model(options)` (já gerada "
                "aqui ou pelo skill) num `.glb` estático. Requer Node.js e, uma "
                "vez, `npm install` em `tools/img2threejs_export/`."
            )
            with gr.Row():
                factory_in = gr.File(label="Factory .ts", file_types=[".ts"], type="filepath")
                options_in = gr.File(label="options.json (ProceduralModelOptions, opcional)",
                                     file_types=[".json"], type="filepath")
            fn_in = gr.Textbox(label="Nome da função exportada",
                               placeholder="createObjectNameModel")
            convert_btn = gr.Button("Converter para GLB", variant="primary")
            log_out = gr.Textbox(label="Log", lines=8, interactive=False)
            with gr.Row():
                ref_img_2 = gr.Image(label="Imagem de referência", type="filepath")
                model_view_2 = gr.Model3D(label="Resultado", interactive=False)

            convert_btn.click(fn=convert_factory,
                              inputs=[factory_in, fn_in, options_in],
                              outputs=[log_out, model_view_2])

        with gr.Tab("3. Visualizar modelo pronto"):
            gr.Markdown(
                "Arraste um `.glb`, `.gltf`, `.obj` ou `.stl` já exportado. O "
                "componente serve para upload, órbita e download ao mesmo tempo."
            )
            with gr.Row():
                ref_img_1 = gr.Image(label="Imagem de referência", type="filepath")
                model_view_1 = gr.Model3D(
                    label="Modelo 3D (clique para enviar, arraste para orbitar, baixe pelo ícone)",
                    interactive=True,
                )

    return demo


def standalone(port: int = 7914):
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7914)
    standalone(ap.parse_args().port)
