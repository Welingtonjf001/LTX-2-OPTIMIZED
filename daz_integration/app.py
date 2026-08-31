from pathlib import Path
import os
import logging
import traceback
import gradio as gr
from .character_schema import CharacterSpec
from .local_llm import interpret
from .daz_controller import create_script, run_daz, DEFAULT_FIGURE_PRESET, SCRIPT_LIBRARY_DIR

ROOT = Path(__file__).resolve().parent
PICTURES = Path.home() / "Pictures"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "daz_character_ui.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("daz_character_ui")


def build(description, reference, endpoint, model, daz_path, output_dir, launch):
    logger.info("INÍCIO | descrição=%r | referência=%r | endpoint=%s | modelo=%s | abrir_daz=%s",
                description, reference, endpoint, model, launch)
    try:
        if not description or not description.strip():
            raise ValueError("Digite uma descrição para o personagem.")
        logger.info("LLM | enviando descrição para %s (%s)", endpoint, model)
        spec = interpret(description, endpoint, model)
        logger.info("LLM OK | especificação=%s", spec.to_json().replace("\n", " "))
        script = create_script(spec, output_dir, reference)
        logger.info("DSA OK | arquivo=%s", script)
        status = f"OK — especificação criada.\nScript: {script}"
        if launch:
            logger.info("DAZ | iniciando %s", daz_path)
            status += "\n" + run_daz(script, daz_path)
        logger.info("FIM OK")
        return spec.to_json(), str(script), status
    except Exception as exc:
        logger.exception("FALHA | %s", exc)
        kind = type(exc).__name__
        hint = "Verifique se o Ollama está ativo e se o modelo foi baixado." if "urlopen" in traceback.format_exc() else "Consulte o log para detalhes."
        message = f"ERRO ({kind}): {exc}\n\n{hint}\nLog: {LOG_DIR / 'daz_character_ui.log'}"
        return "", "", message


def preview_from_options(base, gender, age, height, weight, muscularity, body_shape, hair, outfit, color, shoes, figure_preset, daz_path, output_dir):
    spec = CharacterSpec(base_figure=base, gender=gender, age=int(age), height=float(height),
                         weight=float(weight), muscularity=float(muscularity), body_shape=body_shape,
                         hair=hair, outfit=outfit, outfit_color=color, shoes=shoes)
    logger.info("PRÉVIA | opções=%s", spec.to_json().replace("\n", " "))
    try:
        script = create_script(spec, output_dir, figure_preset=figure_preset or None)
        status = "PRÉVIA GERADA\nPreset: " + (figure_preset or str(DEFAULT_FIGURE_PRESET)) + "\n" + run_daz(script, daz_path)
        return spec.to_json(), str(script), status
    except Exception as exc:
        logger.exception("PRÉVIA FALHOU | %s", exc)
        return "", "", f"ERRO NA PRÉVIA ({type(exc).__name__}): {exc}\nLog: {LOG_DIR / 'daz_character_ui.log'}"


def main():
    with gr.Blocks(title="DAZ Character Director") as demo:
        gr.Markdown("# DAZ Character Director\nDescreva a figura; o LLM local traduz a intenção em parâmetros Genesis.")
        with gr.Row():
            with gr.Column():
                description = gr.Textbox(label="Descrição", lines=6, value="Mulher adulta, atlética, alta, cabelo longo castanho, roupa de dança preta")
                reference = gr.Image(label="Referência opcional", type="filepath")
                endpoint = gr.Textbox(label="Endpoint LLM local", value=os.getenv("DAZ_LLM_ENDPOINT", "http://127.0.0.1:11434"))
                model = gr.Textbox(label="Modelo", value=os.getenv("DAZ_LLM_MODEL", "qwen3.6-35b-a3b:latest"))
                daz_path = gr.Textbox(label="DAZ Studio.exe", value=r"C:\Program Files\DAZ 3D\DAZStudio6\DAZStudio.exe")
                output = gr.Textbox(label="Pasta de scripts DAZ", value=str(SCRIPT_LIBRARY_DIR))
                launch = gr.Checkbox(label="Abrir DAZ e executar script", value=False)
                button = gr.Button("Interpretar e montar", variant="primary")
                picture_files = sorted(
                    [p for p in PICTURES.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
                    key=lambda p: str(p).lower(),
                )[:12]
                examples = [
                    ["Mulher adulta atlética, alta, cabelo longo castanho, roupa de dança preta", str(picture_files[0]) if picture_files else None],
                    ["Homem adulto musculoso, pele escura, cabelo curto, jaqueta vermelha e botas", str(picture_files[1]) if len(picture_files) > 1 else None],
                    ["Personagem andrógino jovem adulto, corpo esguio, cabelo prateado, roupa futurista branca", str(picture_files[2]) if len(picture_files) > 2 else None],
                    ["Mulher madura elegante, cabelo curto grisalho, vestido azul e sapatos sociais", str(picture_files[3]) if len(picture_files) > 3 else None],
                ]
                gr.Examples(examples=examples, inputs=[description, reference], label="Exemplos rápidos — texto + imagem de Pictures")
                gr.Markdown("### Prévia por opções do DAZ")
                base = gr.Dropdown(["Genesis 8 Female", "Genesis 8 Male", "Genesis 9"], value="Genesis 9", label="Figura base")
                gender = gr.Dropdown(["female", "male", "androgynous"], value="female", label="Gênero")
                age = gr.Slider(18, 90, value=28, step=1, label="Idade")
                height = gr.Slider(140, 220, value=170, step=1, label="Altura (cm)")
                weight = gr.Slider(40, 150, value=65, step=1, label="Peso (kg)")
                muscularity = gr.Slider(-1, 1, value=0, step=.05, label="Muscularidade")
                body_shape = gr.Dropdown(["natural", "athletic", "slim", "muscular", "curvy"], value="natural", label="Tipo corporal")
                hair = gr.Dropdown(["default", "short brown", "long brown", "long black", "blonde", "silver"], value="default", label="Cabelo / preset")
                outfit = gr.Dropdown(["casual", "dance outfit", "dress", "futuristic", "suit", "armor"], value="casual", label="Roupa / preset")
                color = gr.Dropdown(["black", "white", "red", "blue", "silver", "gold"], value="black", label="Cor")
                shoes = gr.Dropdown(["default", "sneakers", "dance shoes", "boots", "heels"], value="default", label="Calçado")
                figure_preset = gr.Textbox(label="Preset da figura (.duf)", value=str(DEFAULT_FIGURE_PRESET), placeholder=r"Ex.: C:\Users\...\Genesis 9 Female.duf")
                preview_button = gr.Button("Gerar prévia no DAZ", variant="secondary")
            with gr.Column():
                spec = gr.Code(label="Especificação", language="json")
                script = gr.Textbox(label="Script DSA")
                status = gr.Textbox(label="Status / logs", lines=8, show_copy_button=True)
        button.click(build, [description, reference, endpoint, model, daz_path, output, launch], [spec, script, status], api_name=False)
        preview_button.click(preview_from_options, [base, gender, age, height, weight, muscularity, body_shape, hair, outfit, color, shoes, figure_preset, daz_path, output], [spec, script, status], api_name=False)
    # 0.0.0.0 funciona também em ambientes Windows em que o Gradio não consegue
    # validar a acessibilidade de 127.0.0.1 durante o boot.
    # show_api removido no Gradio 5/6 (temos 6.20) -- ver video_doctor_ui.
    demo.launch(server_name=os.environ.get("LTX_UI_HOST", "127.0.0.1"), server_port=7865, inbrowser=True)


if __name__ == "__main__": main()
