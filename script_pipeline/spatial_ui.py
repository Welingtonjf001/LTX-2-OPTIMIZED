"""Local UI for the persistent spatial pipeline. python -m script_pipeline.spatial_ui"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(project, spec, demo, qwen, stage, denoise):
    project = Path(project).expanduser().resolve()
    command = [sys.executable,'-u','-m','script_pipeline.spatial_pipeline','--project',str(project),
               '--stage',stage,'--denoise',str(denoise)]
    if demo:
        command.append('--demo')
    elif spec.strip():
        command += ['--spec',str(Path(spec).expanduser().resolve())]
    if qwen:
        command.append('--qwen')
    log = ''
    with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,encoding='utf-8',errors='replace') as child:
        for line in child.stdout:
            log += line
            yield log[-20000:],None
        code = child.wait()
    log += f'\nProcesso encerrado: {code}\n'
    output = project/'entregas/spatial_20s.mp4'
    yield log[-20000:], str(output) if output.exists() else None


def main():
    import gradio as gr
    with gr.Blocks(title='Continuidade espacial') as app:
        gr.Markdown('# Continuidade espacial\nEstado persistente, câmeras, blocking, stills e vídeo com auditoria.')
        project = gr.Textbox(label='Pasta do projeto',value=str(ROOT/'outputs/spatial_pilot_20260919'))
        spec = gr.Textbox(label='Arquivo de cena e planos (JSON)')
        demo = gr.Checkbox(label='Usar piloto de 20 segundos',value=True)
        qwen = gr.Checkbox(label='Interpretar o evento do piloto com Qwen',value=True)
        stage = gr.Dropdown(['prepare','blocking','stills','audit-stills','video','audit-video','all'],value='all',label='Etapa')
        denoise = gr.Slider(.1,1,value=.65,step=.05,label='Força da transformação do blocking')
        button = gr.Button('Executar')
        log = gr.Textbox(label='Progresso',lines=16)
        video = gr.Video(label='Resultado / vídeo de diagnóstico')
        button.click(run_stage,[project,spec,demo,qwen,stage,denoise],[log,video],concurrency_limit=1)
    app.launch(server_name='127.0.0.1',server_port=7879,allowed_paths=[str(ROOT/'outputs')])


if __name__ == '__main__':
    main()
