"""Aba Gradio do video_doctor, para embutir nas UIs de geração.

Por que existe como módulo separado: as UIs de 2.3 e 2.5 são arquivos grandes e
independentes (music_maker v2/v3, web_ui, film_maker, storyplay). Duplicar a
interface de diagnóstico em cada uma significaria seis cópias divergindo. Aqui
é uma função que constrói o bloco, e cada UI a chama de dentro do seu
`gr.Blocks`.

DESENHO: revisão antes de reparo, não reparo automático.

Isso não é excesso de zelo -- é resultado de medição. MEDIDO 2026-08-24: o
detector marcou como defeito uma mão ENTRANDO EM QUADRO, e tanto o warp quanto
o RIFE apagaram a mão ao interpolar entre as âncoras. A métrica melhorou 38%
justamente por destruir o conteúdo que a fazia piorar.

Nenhuma medida temporal separa com segurança "matéria surgindo porque o modelo
errou" de "matéria surgindo porque entrou em cena". Quem separa é quem olha.
Por isso o fluxo é: analisar -> ver as tiras -> escolher o que reparar.
"""
import json
import os
import traceback

import gradio as gr

import video_doctor as vd

_STATE = {"plan": None, "video": None, "previews": []}


def _preview_dir(video: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(video)) or ".", "_doctor")
    os.makedirs(base, exist_ok=True)
    return base


def analyze(video, flow, z_res, z_local, z_luma):
    if not video or not os.path.exists(video):
        return "Informe o caminho de um vídeo existente.", [], gr.update(choices=[], value=[])
    try:
        frames = vd.read_frames(video)
        info = vd.probe(video)
        m = vd.measure(frames, flow)
        segs = vd.classify(m, z_res=float(z_res), z_luma=float(z_luma),
                           z_res_local=float(z_local))
        plan = {"video": os.path.abspath(video), "frames": len(frames),
                "fps": info["fps"], "width": info["width"], "height": info["height"],
                "global_flicker": vd.global_flicker(m), "segments": segs}
        outdir = _preview_dir(video)
        previews = vd.export_previews(video, plan, outdir) if segs else []
        _STATE.update(plan=plan, video=video, previews=previews)
        json.dump(plan, open(os.path.join(outdir, "plano.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        lines = [f"{len(frames)} frames · {info['width']}x{info['height']} · "
                 f"{info['fps']:.2f} fps",
                 f"resíduo de fluxo: mediana {sum(m['residual'])/len(m['residual']):.2f}"]
        if plan["global_flicker"]:
            lines.append("FLICKER GLOBAL detectado no clipe inteiro.")
        if not segs:
            lines.append("\nNenhum segmento anômalo. Nada a corrigir.")
        else:
            lines.append(f"\n{len(segs)} segmento(s). REVISE AS TIRAS ABAIXO antes de "
                         "reparar: movimento rápido legítimo (uma mão entrando em "
                         "quadro, por exemplo) aparece igual a defeito, e reparar "
                         "isso APAGA o movimento.")
            for i, s in enumerate(segs):
                lines.append(
                    f"  [{i}] frames {s['start']}-{s['end']} ({s['length']}f) → "
                    f"{s['action'].upper()} · {s['reason']}")
        choices = [f"[{i}] {s['start']}-{s['end']} {s['action']}"
                   for i, s in enumerate(segs)]
        return "\n".join(lines), previews, gr.update(choices=choices, value=[])
    except Exception:
        return f"Falhou:\n{traceback.format_exc()[-1500:]}", [], gr.update(choices=[], value=[])


def repair(selected, interp, prompt, seed, suffix):
    plan, video = _STATE.get("plan"), _STATE.get("video")
    if not plan or not video:
        return "Rode a análise primeiro.", None
    if not selected:
        return "Nenhum segmento selecionado. Marque o que quer reparar.", None
    idx = {int(s.split("]")[0].lstrip("[")) for s in selected}
    sub = dict(plan, segments=[s for i, s in enumerate(plan["segments"]) if i in idx])
    out = os.path.splitext(video)[0] + (suffix or "_corrigido") + ".mp4"

    class A:  # do_repair espera um namespace de argparse
        pass
    a = A()
    a.video, a.output, a.plan = video, out, None
    a.flow, a.interp = "dis", interp
    a.prompt, a.seed = (prompt or None), int(seed or 42)
    a.no_global_deflicker = True   # o global não é escolhível aqui; evita surpresa
    a.only = None
    try:
        vd.do_repair(a, sub)
        return (f"Pronto: {out}\nSegmentos aplicados: {sorted(idx)}\n"
                f"Compare com o original antes de aceitar."), out
    except Exception:
        return f"Falhou:\n{traceback.format_exc()[-1500:]}", None


def build_doctor_tab(get_default_video=None, label="Diagnóstico e correção",
                     container="tab"):
    """Constrói o bloco. `get_default_video` é um callable sem argumentos que
    devolve o caminho do último vídeo gerado pela UI hospedeira -- assim o
    botão "usar o último vídeo" funciona sem a UI precisar expor estado.

    `container`: "tab" ou "accordion".

    Existe porque metade das UIs deste repo NÃO usa abas -- web_ui_*,
    film_maker_* e a decupagem são layout plano. Um `gr.Tab` solto ali cria um
    container de aba única embaixo de todo o resto, que fica ilegível. O
    conteúdo é o mesmo; muda só a casca."""
    if container == "accordion":
        caixa = gr.Accordion(label, open=False)
    else:
        caixa = gr.Tab(label)
    with caixa:
        gr.Markdown(
            "### Diagnóstico temporal pós-geração\n"
            "Acha frames em que o conteúdo muda de um jeito que o movimento não "
            "explica — rosto ou mão derretendo, textura pulsando, brilho oscilando.\n\n"
            "**Revise as tiras antes de reparar.** O detector encontra *mudança*, "
            "e nem toda mudança é defeito: movimento rápido legítimo aparece igual, "
            "e repará-lo apaga o movimento."
        )
        with gr.Row():
            video_in = gr.Textbox(label="Vídeo", scale=4,
                                  placeholder="caminho do .mp4 (final, já concatenado/upscalado)")
            if get_default_video is not None:
                use_last = gr.Button("Usar o último gerado", scale=1)
                use_last.click(fn=lambda: (get_default_video() or ""), outputs=[video_in])
        with gr.Accordion("Limiares (só mexa se houver falso positivo/negativo)", open=False):
            with gr.Row():
                flow = gr.Dropdown(["dis", "raft"], value="dis", label="Fluxo óptico",
                                   info="dis = rápido. raft = melhor, mais lento.")
                z_res = gr.Number(value=vd.Z_RESIDUAL, label="z resíduo global")
                z_local = gr.Number(value=vd.Z_RESIDUAL_LOCAL, label="z resíduo local")
                z_luma = gr.Number(value=vd.Z_LUMA, label="z luminância")
        analyze_btn = gr.Button("Analisar", variant="primary")
        report = gr.Textbox(label="Relatório", lines=10, interactive=False)
        gallery = gr.Gallery(label="Tiras por segmento (vermelho = acusado, verde = referência)",
                             columns=1, height=260)
        picks = gr.CheckboxGroup(choices=[], label="Reparar quais segmentos")
        with gr.Row():
            interp = gr.Dropdown(
                ["rife", "warp"], value="rife" if vd.rife_available() else "warp",
                label="Interpolação",
                info=("RIFE instalado" if vd.rife_available() else "RIFE ausente; só warp"))
            seed = gr.Number(value=42, label="Seed (regeneração)")
            suffix = gr.Textbox(value="_corrigido", label="Sufixo da saída")
        prompt = gr.Textbox(
            label="Prompt (só para regeneração parcial via LTX)", lines=2,
            info="Segmentos classificados como REGENERATE precisam dele; sem prompt caem para interpolação.")
        repair_btn = gr.Button("Reparar selecionados")
        result = gr.Textbox(label="Resultado", lines=4, interactive=False)
        preview = gr.Video(label="Vídeo corrigido")

        analyze_btn.click(fn=analyze, inputs=[video_in, flow, z_res, z_local, z_luma],
                          outputs=[report, gallery, picks])
        repair_btn.click(fn=repair, inputs=[picks, interp, prompt, seed, suffix],
                         outputs=[result, preview])
    return video_in


def standalone(port: int = 7912):
    with gr.Blocks(title="Video Doctor — LTX") as demo:
        gr.Markdown("# Video Doctor\nDiagnóstico e correção temporal para clipes do LTX (2.3 e 2.5).")
        build_doctor_tab()
    # show_api foi removido no Gradio 5/6 (temos 6.20): passar levanta
    # TypeError e a UI nem chega a subir. Corrigido 2026-08-27.
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7912)
    standalone(ap.parse_args().port)
