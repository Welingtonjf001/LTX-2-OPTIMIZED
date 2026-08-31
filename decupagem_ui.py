"""UI da cadeia de DECUPAGEM (roteiro -> filme), porta 7913.

POR QUE ESTA UI EXISTE

O `start_decupagem.bat` roda a cadeia inteira e termina -- e por vinte minutos a
unica coisa que a pessoa ve e console. Os artefatos intermediarios existem todos
em disco (roteiro reconstruido, cast, decupagem, stills, animatic) e nao havia
como olhar para eles enquanto a coisa anda.

Duas coisas que esta UI conserta, alem de mostrar imagem:

1. RETOMAR. O `.bat` cria um diretorio novo com timestamp a cada execucao, entao
   uma corrida interrompida no meio dos stills recomeca do zero -- mesmo o
   `run_decupagem` sabendo retomar (ele pula etapa cujo artefato ja existe).
   MEDIDO 2026-08-27: uma corrida morreu em 3 de 6 stills e nao havia como
   continuar de onde parou sem montar a linha de comando na mao. Aqui a lista de
   corridas existentes e o primeiro controle da tela.

2. EDITAR ANTES DO CARO. O proprio log do estagio de cast diz "edite este
   arquivo antes de continuar". O `cast.json` decide aparencia e voz de cada
   personagem, e um descritor errado se propaga para TODOS os stills e clipes.
   Editar aqui e rodar a etapa seguinte custa menos que descobrir no filme.

O QUE ELA NAO FAZ

Nao reimplementa nenhum estagio. Ela monta a mesma argv que o `.bat` monta e
transmite o stdout -- se a cadeia mudar, a UI acompanha sozinha. O unico
conhecimento proprio dela e onde cada artefato cai, que e o contrato do
`run_folder`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

import video_doctor_ui  # pos-producao: diagnostico e correcao temporal

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS_DIR = ROOT / "outputs" / "decupagem"

# Mesmas paradas do orquestrador. Importadas, nao copiadas: duas listas
# divergiriam do mesmo jeito que os dois lancadores do ComfyUI divergiram.
sys.path.insert(0, str(ROOT))
from script_pipeline.run_decupagem import PARADAS  # noqa: E402
from script_pipeline.shot_plan import STYLES  # noqa: E402

# As duas variaveis que decidem se o estagio de VIDEO fecha nesta placa.
# Sem `--disable-dynamic-vram` o mesmo plano de 81 frames nao fechava em 13 min
# e passa a fechar em 151 s -- ver MEMORIAL.md secao 3.33. Ficam aqui pelo mesmo
# motivo que estao no start_decupagem.bat: nao sao ajuste fino.
# `distilled` e NAO `distilled-int8`: o int8 cabe inteiro na placa, o ComfyUI o
# carrega todo ("full load: True") e nao sobra VRAM para o latente -- um plano de
# 129 frames travou duas vezes. O bf16 nao cabe, o gerenciador descarrega 19 GB
# e o mesmo plano fecha. Ver MEMORIAL 3.35.
ENV_VIDEO = {
    "LTX25_VARIANT": "distilled",
    "LTX_COMFY_EXTRA_ARGS": "--disable-dynamic-vram",
}

_PROC: dict = {"p": None}


# --------------------------------------------------------------------------
# leitura de artefatos
# --------------------------------------------------------------------------
def listar_runs() -> list:
    if not RUNS_DIR.is_dir():
        return []
    runs = [d for d in RUNS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_")]
    return [d.name for d in sorted(runs, key=lambda d: d.stat().st_mtime, reverse=True)]


def stills_de(run: Path) -> list:
    d = run / "shots" / "stills"
    return [str(p) for p in sorted(d.glob("shot*.png"))] if d.is_dir() else []


def _plano(run: Path) -> dict:
    try:
        return json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _indice_do_still(caminho: str) -> int:
    """`shot007_medium.png` -> 7. -1 quando o nome nao segue o padrao."""
    nome = Path(caminho).stem
    try:
        return int(nome[4:].split("_", 1)[0])
    except (ValueError, IndexError):
        return -1


def stills_com_legenda(run: Path) -> list:
    """(caminho, legenda) para a galeria. A legenda diz QUAL plano e -- sem
    isso, doze imagens parecidas nao dizem qual delas refazer."""
    plan = _plano(run)
    shots = plan.get("shots") or []
    saida = []
    for c in stills_de(run):
        i = _indice_do_still(c)
        s = shots[i] if 0 <= i < len(shots) else {}
        legenda = f"#{i} {s.get('framing', '?')}"
        if s.get("subject"):
            legenda += f" · {s['subject']}"
        if s.get("style"):
            legenda += f" · {s['style']}"
        saida.append((c, legenda))
    return saida


def prompts_do_still(nome_run: str, evt: gr.SelectData) -> str:
    """Os DOIS prompts do plano clicado. Ver shot_plan: o still carrega
    enquadramento e o video carrega movimento -- sao textos diferentes e a
    duvida "por que esta imagem saiu assim" quase sempre se responde aqui."""
    if not nome_run:
        return ""
    run = RUNS_DIR / nome_run
    itens = stills_com_legenda(run)
    if not (0 <= evt.index < len(itens)):
        return ""
    i = _indice_do_still(itens[evt.index][0])
    shots = _plano(run).get("shots") or []
    if not (0 <= i < len(shots)):
        return f"(plano {i} nao esta no shot_plan atual)"
    sh = shots[i]
    return (f"TOMADA {i} — {sh.get('framing')} / {sh.get('angle')} / "
            f"{sh.get('movement')} — {sh.get('seconds')}s ({sh.get('frames')}f) — "
            f"estilo {sh.get('style')}\n"
            f"lado de tela: {sh.get('screen_side') or '-'}   sujeito: {sh.get('subject') or '-'}\n"
            f"\n--- PROMPT DA IMAGEM (still) ---\n{sh.get('storyboard_prompt', '')}\n"
            f"\n--- PROMPT DO VIDEO ---\n{sh.get('video_prompt', '')}")


def video_de(run: Path):
    """O melhor video disponivel: filme montado, senao o rascunho."""
    for c in (run / "final" / "movie.mp4", run / "shots" / "animatic.mp4"):
        if c.exists():
            return str(c)
    return None


def texto_de(caminho: Path, limite: int = 20000) -> str:
    try:
        return caminho.read_text(encoding="utf-8")[:limite]
    except OSError:
        return ""


def resumo_do_plano(run: Path) -> str:
    p = run / "parse" / "shot_plan.json"
    if not p.exists():
        return "(a decupagem ainda nao foi feita)"
    try:
        plan = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"(shot_plan.json ilegivel: {e})"
    linhas = [f"{plan['total_shots']} planos, {plan['total_seconds']}s -- estilo {plan.get('style')}",
              "",
              f"{'#':>3} {'tipo':<12} {'enquadre':<14} {'mov':<9} {'seg':>6} {'frames':>7} {'lado':<6} sujeito"]
    for s in plan.get("shots", []):
        linhas.append(f"{s['position']:>3} {str(s.get('type')):<12} {s['framing']:<14} "
                      f"{s['movement']:<9} {s['seconds']:>6} {s['frames']:>7} "
                      f"{str(s.get('screen_side') or '-'):<6} {s.get('subject') or '-'}")
    return "\n".join(linhas)


def carregar_run(nome: str):
    """Tudo o que a tela mostra de uma corrida, para o seletor e para o refresh."""
    if not nome:
        return "", "", "(nenhuma corrida selecionada)", [], None
    run = RUNS_DIR / nome
    return (texto_de(run / "parse" / "screenplay_auto.txt") or "(roteiro ja vinha formatado)",
            texto_de(run / "characters" / "cast.json"),
            resumo_do_plano(run),
            stills_com_legenda(run),
            video_de(run))


def salvar_cast(nome: str, conteudo: str) -> str:
    if not nome:
        return "Selecione uma corrida primeiro."
    try:
        json.loads(conteudo)          # nao gravar JSON quebrado por cima do bom
    except json.JSONDecodeError as e:
        return f"JSON invalido, nada gravado: {e}"
    alvo = RUNS_DIR / nome / "characters" / "cast.json"
    alvo.parent.mkdir(parents=True, exist_ok=True)
    alvo.write_text(conteudo, encoding="utf-8")
    return (f"cast.json gravado. Os stills usam o descritor na chave de cache, "
            f"entao rodar de novo refaz so os planos afetados.")


def apagar_stills(nome_run: str, quais: str) -> str:
    """Marca stills para REFAZER, apagando o arquivo e a entrada do manifesto.

    Nao regenera aqui: quem regenera e o proprio estagio de stills, que refaz o
    que estiver faltando. Apagar tambem a entrada do manifesto e obrigatorio --
    `render_shots` guarda ali a chave (prompt + enquadramento + referencia +
    motor) e, com o arquivo ausente mas a entrada viva, o glob de retomada pode
    achar um PNG antigo de outra versao do prompt e reusa-lo em silencio.

    `quais`: vazio ou "todos" = tudo; senao lista/intervalos "0,3,5-7"."""
    if not nome_run:
        return "Selecione uma corrida primeiro."
    run = RUNS_DIR / nome_run
    d = run / "shots" / "stills"
    if not d.is_dir():
        return "Esta corrida ainda nao tem stills."

    alvo = (quais or "").strip().lower()
    if alvo in ("", "todos", "all", "*"):
        indices = None
    else:
        indices = set()
        for parte in alvo.replace(";", ",").split(","):
            parte = parte.strip()
            if not parte:
                continue
            try:
                if "-" in parte:
                    a, b = parte.split("-", 1)
                    indices.update(range(int(a), int(b) + 1))
                else:
                    indices.add(int(parte))
            except ValueError:
                return f"Nao entendi {parte!r}. Use numeros, intervalos (3-7) ou 'todos'."

    man_path = d / "stills.json"
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        man = {}

    apagados = []
    for png in sorted(d.glob("shot*.png")):
        i = _indice_do_still(str(png))
        if indices is not None and i not in indices:
            continue
        try:
            png.unlink()
            apagados.append(i)
            man.pop(str(i), None)
        except OSError as e:
            return f"Nao consegui apagar {png.name}: {e}"

    try:
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass

    if not apagados:
        return "Nenhum still correspondeu — confira os numeros."
    return (f"{len(apagados)} still(s) marcados para refazer: {sorted(apagados)}. "
            f"Rode com 'Ir até' = stills (ou animatic) para regerar so esses.")


# --------------------------------------------------------------------------
# execucao
# --------------------------------------------------------------------------
def _argv(run: Path, script: str, estilo: str, trocas: str, largura: int,
          altura: int, ate: str, motor: str, recast: bool,
          motor_img: str = "flux") -> list:
    cmd = [PY, "-u", "-m", "script_pipeline.run_decupagem",
           "--run-dir", str(run), "--style", estilo, "--ate", ate,
           "--width", str(int(largura)), "--height", str(int(altura)),
           "--engine", motor, "--image-engine", motor_img]
    # --script so na primeira vez: com o parse ja feito, passar de novo nao
    # muda nada, mas deixa a UI dependendo de um caminho que pode ter sumido.
    if script and not (run / "parse" / "scenes.json").exists():
        cmd += ["--script", script]
    if trocas.strip():
        cmd += ["--style-changes", trocas.strip()]
    if recast:
        cmd.append("--recast")
    return cmd


def rodar(nome_run, script, novo_nome, estilo, trocas, largura, altura, ate, motor,
          recast, motor_img="flux"):
    """Executa a cadeia transmitindo o stdout. Gerador: a UI recebe cada linha.

    O subprocesso e o MESMO que o .bat dispara. A UI nao reimplementa etapa
    nenhuma -- se o orquestrador mudar, isto acompanha."""
    if _PROC["p"] is not None and _PROC["p"].poll() is None:
        yield "Ja existe uma corrida em andamento. Pare antes de comecar outra.", [], None, ""
        return

    if nome_run:
        run = RUNS_DIR / nome_run
    else:
        base = (novo_nome or "").strip() or (Path(script).stem if script else "")
        if not base:
            yield "Informe um roteiro (para uma corrida nova) ou selecione uma existente.", [], None, ""
            return
        run = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M')}_{base}"

    if not (run / "parse" / "scenes.json").exists() and not script:
        yield (f"A corrida {run.name} nao tem parse feito e nenhum roteiro foi informado.",
               [], None, "")
        return

    run.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(ENV_VIDEO)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
    # NAO definir CUDA_VISIBLE_DEVICES: o gemma4_worker espera as duas placas
    # visiveis e faz set_device(1) sozinho. Mesma nota do start_decupagem.bat.

    cmd = _argv(run, script, estilo, trocas, largura, altura, ate, motor, recast,
                motor_img)
    linhas = [f"$ {' '.join(cmd[3:])}", f"(corrida: {run})", ""]
    yield "\n".join(linhas), stills_com_legenda(run), video_de(run), resumo_do_plano(run)

    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    _PROC["p"] = p
    ultimo = 0.0
    try:
        for linha in p.stdout:
            linhas.append(linha.rstrip("\n"))
            # A galeria e o plano so sao relidos a cada ~1,5s: um still leva
            # dezenas de segundos, e varrer o disco a cada linha de log so
            # gastaria E/S para mostrar a mesma coisa.
            agora = time.time()
            if agora - ultimo > 1.5:
                ultimo = agora
                yield ("\n".join(linhas[-400:]), stills_com_legenda(run), video_de(run),
                       resumo_do_plano(run))
    finally:
        p.wait()
        _PROC["p"] = None
    linhas.append("")
    linhas.append(f"[fim] codigo de saida {p.returncode}")
    yield "\n".join(linhas[-400:]), stills_com_legenda(run), video_de(run), resumo_do_plano(run)


def parar() -> str:
    p = _PROC.get("p")
    if p is None or p.poll() is not None:
        return "Nada rodando."
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
    return ("Corrida encerrada. Os artefatos prontos ficam -- selecione a mesma "
            "corrida e rode de novo para continuar de onde parou.")


# --------------------------------------------------------------------------
def build() -> None:
    # A tela abre JA mostrando a corrida mais recente. Alem de ser o que a
    # pessoa quase sempre quer ver, isso evita uma tela vazia que nao diz se a
    # UI esta funcionando ou se nao ha nada para mostrar.
    _runs = listar_runs()
    _inicial = _runs[0] if _runs else None
    _auto0, _cast0, _plano0, _stills0, _video0 = carregar_run(_inicial)

    with gr.Blocks(title="Decupagem — LTX") as demo:
        gr.Markdown(
            "# Decupagem\n"
            "Roteiro → cast → voz → decupagem → stills → animatic → filme. "
            "Mesma cadeia do `start_decupagem.bat`, com o log à vista e retomada."
        )

        with gr.Row():
            with gr.Column(scale=2):
                runs = gr.Dropdown(choices=_runs, label="Corrida existente (para RETOMAR)",
                                   value=_inicial, allow_custom_value=False)
            with gr.Column(scale=1):
                atualizar = gr.Button("Atualizar lista")
                # Botao explicito alem do evento do dropdown: o `change` do
                # Dropdown depende de uma selecao de verdade e nao dispara
                # quando o valor chega por outro caminho (restaurar sessao,
                # teclado, automacao). Um botao sempre dispara.
                carregar = gr.Button("Carregar", variant="secondary")

        with gr.Row():
            script = gr.Textbox(label="Roteiro .txt (só para corrida NOVA)", scale=3,
                                placeholder=r"C:\Users\user\Desktop\roteiro.txt")
            novo_nome = gr.Textbox(label="Nome da corrida nova (opcional)", scale=1)

        with gr.Row():
            estilo = gr.Dropdown(choices=sorted(STYLES), value="classico", label="Estilo")
            ate = gr.Dropdown(choices=PARADAS, value="animatic", label="Ir até")
            largura = gr.Number(value=960, label="Largura", precision=0)
            altura = gr.Number(value=544, label="Altura", precision=0)
        with gr.Row():
            trocas = gr.Textbox(label="Trocas de estilo — por cena (3:tenso) ou por tomada (1.10:intimista); vigora até a próxima marca", scale=2,
                                placeholder="1:classico,1.10:intimista,2:tenso")
            motor = gr.Textbox(value="qwen3.6-35b-a3b:latest", label="Motor LLM", scale=2)
            recast = gr.Checkbox(value=False, label="Reescolher vozes")
        with gr.Row():
            motor_img = gr.Dropdown(
                choices=["flux", "sd35", "sdxl"], value="flux", scale=1,
                label="Motor das imagens",
                info="flux: obedece melhor enquadramento e lado de tela, e o unico "
                     "com imagem de referencia por personagem. sd35: carrega em ~1 min "
                     "contra ~4 e cabe em ~12 GB contra ~24, mas erra o enquadramento. "
                     "Trocar o motor REFAZ os stills (ele entra na chave de cache).")

        with gr.Row():
            btn = gr.Button("Rodar", variant="primary", scale=2)
            btn_parar = gr.Button("Parar", scale=1)

        log = gr.Textbox(label="Log", lines=22, interactive=False, autoscroll=True,
                         max_lines=22)

        with gr.Row():
            galeria = gr.Gallery(value=_stills0, columns=4, height=320, show_label=True,
                                 label="Stills — clique num para ver o prompt que o gerou")
            video = gr.Video(value=_video0, label="Rascunho / filme", height=320)

        prompt_sel = gr.Textbox(label="Prompts da tomada selecionada", lines=9,
                                interactive=False,
                                placeholder="Clique numa imagem acima.")

        with gr.Accordion("Refazer imagens", open=False):
            gr.Markdown(
                "Apaga os stills escolhidos (e a entrada deles no manifesto) para que "
                "a próxima execução os gere de novo. **Não regenera sozinho**: depois "
                "de marcar, rode com *Ir até* = `stills` ou `animatic`.\n\n"
                "Use quando editar o `cast.json`, trocar o estilo de um trecho ou o "
                "motor de imagem — nesses casos o prompt muda e o still velho fica "
                "vencido."
            )
            with gr.Row():
                quais = gr.Textbox(label="Quais tomadas", scale=3,
                                   placeholder="todos   |   0,3,5-7")
                btn_refazer = gr.Button("Marcar para refazer", scale=1)

        with gr.Accordion("Decupagem (shot_plan)", open=False):
            plano = gr.Textbox(value=_plano0, label="", lines=14, interactive=False)

        with gr.Accordion("Elenco — edite ANTES dos stills", open=False):
            gr.Markdown(
                "O descritor decide a aparência em **todos** os planos e a voz em "
                "todas as falas. Corrigir aqui é mais barato que descobrir no filme."
            )
            cast = gr.Textbox(value=_cast0, label="cast.json", lines=16)
            salvar = gr.Button("Salvar cast.json")
            aviso = gr.Textbox(label="", interactive=False)

        with gr.Accordion("Roteiro reconstruído (quando a entrada era prosa)", open=False):
            auto = gr.Textbox(value=_auto0, label="screenplay_auto.txt", lines=16, interactive=False)

        # Pos-producao, FORA do fluxo de geracao de proposito: e ferramenta
        # aplicada ao video ja pronto, e so quando a pessoa quiser. Em accordion
        # porque esta tela e de layout plano -- uma aba solta criaria um
        # container de aba unica embaixo de tudo.
        video_doctor_ui.build_doctor_tab(
            get_default_video=lambda: video_de(RUNS_DIR / (_inicial or "")) or "",
            label="🩺 Diagnóstico e correção (video doctor)",
            container="accordion")

        # ---- ligacoes ----
        atualizar.click(fn=lambda: gr.update(choices=listar_runs()), outputs=runs)
        runs.change(fn=carregar_run, inputs=runs,
                    outputs=[auto, cast, plano, galeria, video])
        carregar.click(fn=carregar_run, inputs=runs,
                       outputs=[auto, cast, plano, galeria, video])
        btn.click(fn=rodar,
                  inputs=[runs, script, novo_nome, estilo, trocas, largura, altura,
                          ate, motor, recast, motor_img],
                  outputs=[log, galeria, video, plano])
        galeria.select(fn=prompts_do_still, inputs=runs, outputs=prompt_sel)
        btn_refazer.click(fn=apagar_stills, inputs=[runs, quais], outputs=aviso)
        btn_parar.click(fn=parar, outputs=aviso)
        salvar.click(fn=salvar_cast, inputs=[runs, cast], outputs=aviso)

    # show_api foi removido no Gradio 6 (temos 6.20) -- passar levanta TypeError.
    # As UIs antigas deste repo ainda passam; ver video_doctor_ui.standalone().
    # allowed_paths: sem isto o Gradio recusa servir os PNG/MP4 de outputs/ e a
    # galeria fica vazia sem dizer por que.
    demo.launch(server_name="127.0.0.1", server_port=7913, inbrowser=False,
                allowed_paths=[str(RUNS_DIR)])


if __name__ == "__main__":
    build()
