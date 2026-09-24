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

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

import video_doctor_ui  # pos-producao: diagnostico e correcao temporal

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS_DIR = ROOT / "outputs" / "decupagem"
PORT = 7913


def _free_port(port: int, *, log=print) -> None:
    """Derruba quem estiver escutando na porta ANTES do demo.launch().

    MEDIDO 2026-09-04: fechar a janela do console (ou o processo travar) as
    vezes nao mata o processo do Gradio -- ele fica orfao, ainda escutando
    7913, e a proxima tentativa de subir a UI morre com `OSError: Cannot
    find empty port in range: 7913-7913` em vez de simplesmente reusar ou
    liberar a porta. `demo.launch()` nao tem uma opcao "mate quem estiver no
    caminho"; o Gradio so tenta outra porta se `server_port` for None, e esta
    UI fixa a porta de proposito (bookmark, .bat, etc. apontam pra 7913).

    Delegado a `gpu_watchdog.free_port` desde 2026-09-04 (era uma copia do
    mesmo netstat+taskkill de `generate_storyboards.stop_comfyui`)."""
    sys.path.insert(0, str(ROOT))
    from script_pipeline import gpu_watchdog
    if gpu_watchdog.free_port(port, log=log):
        log(f"[decupagem_ui] porta {port}: instancia orfa encerrada antes de subir.")

# Mesmas paradas do orquestrador. Importadas, nao copiadas: duas listas
# divergiriam do mesmo jeito que os dois lancadores do ComfyUI divergiram.
sys.path.insert(0, str(ROOT))
from script_pipeline.run_decupagem import PARADAS  # noqa: E402
from script_pipeline.shot_plan import STYLES  # noqa: E402

# As duas variaveis que decidem se o estagio de VIDEO fecha nesta placa.
# Sem `--disable-dynamic-vram` o mesmo plano de 81 frames nao fechava em 13 min
# e passa a fechar em 151 s -- ver MEMORIAL.md secao 3.33. Ficam aqui pelo mesmo
# motivo que estao no start_decupagem.bat: nao sao ajuste fino.
# `w4a8-v10` e o padrao desde 2026-09-12 (MEMORIAL 3.77): destilado em 4 bits,
# ~1,6x mais rapido que o bf16 com o modelo carregado, qualidade julgada maior em
# 3 de 3 comparacoes, validado nesta cadeia com I2V + fala. NAO troque por
# `distilled-int8`: ele cabe inteiro e nao sobra VRAM para o latente -- um plano
# de 129 frames travou duas vezes (MEMORIAL 3.35). O que decide o travamento e a
# folga DEPOIS da carga, e o w4a8-v10 (14,9 GB) deixa mais folga (MEMORIAL 3.51).
ENV_VIDEO = {
    "LTX25_VARIANT": "w4a8-v10",
    "LTX_COMFY_EXTRA_ARGS": "--disable-dynamic-vram",
}

_PROC: dict = {"p": None}


def _descarregar_ollama() -> list[str]:
    """Libera RAM/VRAM dos modelos; nao encerra o servidor Ollama."""
    from script_pipeline.ollama_runtime import unload_all
    return unload_all(log=lambda message: print(f"[decupagem_ui] {message}", flush=True))


# Ctrl+C e encerramento normal da janela/servidor passam por atexit. A corrida
# tambem chama a mesma limpeza no proprio finally, pois a WebUI pode continuar
# aberta depois que o filme termina.
atexit.register(_descarregar_ollama)


# --------------------------------------------------------------------------
# motor LLM: qual instancia do Ollama esta no ar importa (ver CLAUDE.md) --
# o catalogo muda de maquina pra maquina e de sessao pra sessao (quem sobe
# `ollama serve` e com qual OLLAMA_MODELS). Um default FIXO no codigo (como
# era ate 2026-09-09) trava com HTTP 404 assim que a instancia no ar nao
# serve aquela tag especifica -- MEDIDO pelo usuario: "qwen2.5:32b-instruct-
# q4_K_M" (o default corrigido depois do crash do qwen3.6-35b-a3b, ver
# MEMORIAL 3.65/3.76) nem sempre esta entre os modelos baixados.
# --------------------------------------------------------------------------
_MOTOR_LLM_FALLBACK = ["qwen2.5:32b-instruct-q4_K_M", "gemma4",
                       "mistral-nemo:12b-instruct-2407-q4_K_M", "qwen3.6-35b-a3b:latest"]
# Preferencia quando o modelo documentado como seguro nao esta na lista:
# modelos DENSOS (nao MoE) de INSTRUCAO GERAL primeiro -- o crash medido
# (§3.65) e especifico de roteamento MoE em prompt longo (nao ha caso
# medido de modelo denso falhando do mesmo jeito), e a tarefa aqui e
# enriquecimento NARRATIVO (cena, emocao, prompt visual), nao codigo -- um
# modelo "coder" fica DEPOIS dos de instrucao geral de proposito, mesmo
# sendo denso e do tamanho certo. "qwen3.6-35b-a3b" fica por ultimo.
_MOTOR_LLM_PREFERENCIA = ["qwen2.5:32b-instruct-q4_K_M", "gemma4:31b",
                          "gemma4-32k:latest", "gemma4-64k:latest", "phi4:14b",
                          "phi4-64k:latest", "mistral-nemo:12b-instruct-2407-q4_K_M",
                          "gemma4", "qwen2.5-coder:32b", "qwen2.5-coder-64k:latest",
                          "qwen3.6-35b-a3b:latest"]


def listar_modelos_ollama() -> list:
    """Tags realmente servidas pela instancia do Ollama no ar AGORA (timeout
    curto -- nao pode travar o boot da UI se o Ollama estiver desligado).
    [] se inacessivel; quem chama cai no fallback hardcoded."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            dados = json.load(r)
        return sorted(m["name"] for m in dados.get("models", []))
    except Exception:
        return []


def motor_llm_padrao() -> tuple:
    """(choices, default) para o dropdown "Motor LLM" -- se o Ollama estiver
    no ar, usa a lista REAL (evita escolher por padrao uma tag que devolve
    HTTP 404); senao cai no fallback hardcoded, do jeito que sempre foi."""
    disponiveis = listar_modelos_ollama()
    if not disponiveis:
        return _MOTOR_LLM_FALLBACK, _MOTOR_LLM_FALLBACK[0]
    padrao = next((m for m in _MOTOR_LLM_PREFERENCIA if m in disponiveis), disponiveis[0])
    # A tag preferida (mesmo se nao estiver instalada agora) continua na
    # lista -- digitavel/selecionavel achando por texto -- pra nao esconder
    # a decisao documentada so porque esta instancia nao a tem baixada hoje.
    choices = sorted(set(disponiveis) | {_MOTOR_LLM_FALLBACK[0]})
    return choices, padrao


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


def indice_do_still_selecionado(nome_run: str, evt: gr.SelectData):
    """So o numero da tomada clicada -- preenche `regen_indice` sozinho, pra
    nao obrigar a pessoa a contar posicao na galeria pra saber o que digitar."""
    if not nome_run:
        return None
    itens = stills_com_legenda(RUNS_DIR / nome_run)
    if not (0 <= evt.index < len(itens)):
        return None
    return _indice_do_still(itens[evt.index][0])


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


def resumo_enriquecimento(run: Path) -> str:
    """O resultado do parse + enriquecimento por LLM (scenes_enriched.json --
    cai para scenes.json, sem LLM, se aquele ainda nao existe): por cena,
    local/periodo/direcao de arte e, por fala, a emocao e o beat visual que o
    LLM inferiu. E o unico lugar da UI que mostra ISTO -- antes so dava pra
    ver abrindo o JSON na mao. Pedido do usuario 2026-09-04: caixa com
    rolagem, pra nao empurrar o resto da tela pra baixo com 5+ cenas."""
    p = run / "parse" / "scenes_enriched.json"
    enriquecido = p.exists()
    if not enriquecido:
        p = run / "parse" / "scenes.json"
    if not p.exists():
        return "(parse ainda nao foi feito)"
    try:
        cenas = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return f"({p.name} ilegivel: {e})"

    linhas = [f"{len(cenas)} cena(s) -- {'com enriquecimento LLM' if enriquecido else 'SEM enriquecimento (LLM nao rodou/falhou)'}", ""]
    for c in cenas:
        linhas.append(f"=== CENA {c.get('index')} -- {c.get('heading_raw') or '(sem cabecalho)'} ===")
        linhas.append(f"local: {c.get('location') or '-'}   periodo: {c.get('time_of_day') or '-'}")
        if c.get("art_direction"):
            linhas.append(f"direcao de arte: {c['art_direction']}")
        personagens = c.get("characters") or []
        if personagens:
            linhas.append(f"personagens: {', '.join(personagens)}")
        for d in c.get("dialogue") or []:
            trecho = (d.get("text") or "")[:80]
            extra = f" [{d['emotion']}]" if d.get("emotion") else ""
            linhas.append(f"  {d.get('character', '?'):<10}{extra}: {trecho}")
            if d.get("beat_visual"):
                linhas.append(f"             beat: {d['beat_visual'][:100]}")
        linhas.append("")
    return "\n".join(linhas)


def carregar_run(nome: str):
    """Tudo o que a tela mostra de uma corrida, para o seletor e para o refresh."""
    if not nome:
        return "", "", "(nenhuma corrida selecionada)", "(nenhuma corrida selecionada)", [], None
    run = RUNS_DIR / nome
    return (texto_de(run / "parse" / "screenplay_auto.txt") or "(roteiro ja vinha formatado)",
            texto_de(run / "characters" / "cast.json"),
            resumo_do_plano(run),
            resumo_enriquecimento(run),
            stills_com_legenda(run),
            video_de(run))


def carregar_run_completo(nome: str, roteiro_path_atual: str = "") -> tuple:
    """Mesmo que `carregar_run`, mais o que a TELA (nao so os dados) precisa
    mostrar ao trocar de corrida: mensagem de status, trilha de estagio
    zerada (nao ha corrida rodando neste momento) -- a corrida carregada ja
    tem `parse/scenes.json` proprio, entao trazer OUTRO roteiro so faz
    sentido se a pessoa pedir explicitamente.

    BUG CORRIGIDO 2026-09-09 (achado pelo usuario testando de verdade): esta
    funcao tambem dispara via `runs.change` quando "Salvar corrida" seleciona
    a corrida recem-criada -- e SEMPRE zerava `roteiro_path`, mesmo quando a
    pessoa ja tinha clicado "Usar texto colado" ANTES de salvar o nome da
    corrida. Resultado: "Rodar" via nome_run preenchido mas script vazio, e
    a mensagem "corrida sem parse e nenhum roteiro informado" -- roteiro que
    a pessoa acabou de trazer, perdido em silencio. Agora recebe o
    `roteiro_path` ATUAL como input e so o zera se essa corrida especifica
    JA tem parse proprio (nesse caso, sim, faz sentido limpar -- trazer outro
    roteiro so por pedido explicito); caso contrario preserva o que estava
    preparado."""
    auto, cast, plano, enriq, stills, video = carregar_run(nome)
    if not nome:
        return (auto, cast, plano, enriq, stills, video,
                "Nenhuma corrida selecionada. Escolha uma existente e clique \"Carregar corrida "
                "selecionada\" -- ou, se acabou de clicar \"① Nova corrida\", dê um nome acima e "
                "clique \"Salvar corrida\".",
                "", "Passo 2: arraste um .txt OU cole o texto e clique \"Usar texto colado\".",
                _stage_html(None, -1), "")
    ja_tem_parse = (RUNS_DIR / nome / "parse" / "scenes.json").exists()
    # So preserva se a corrida foi criada AGORA (ultimos 2 min) -- cobre o
    # caso real do bug ("Usar texto colado" antes de "Salvar corrida", que
    # dispara este mesmo evento por tabela) sem arriscar grudar um roteiro
    # de OUTRA corrida numa corrida antiga e abandonada que a pessoa
    # resolveu selecionar de novo pelo dropdown.
    criada_agora = (time.time() - (RUNS_DIR / nome).stat().st_mtime) < 120
    roteiro_preparado = (bool(roteiro_path_atual) and Path(roteiro_path_atual).exists()
                         and criada_agora)
    # Projeto mestre ja guarda a fonte versionada em production.json. Depois
    # de reiniciar a WebUI, esse caminho precisa voltar para a tela mesmo que
    # o parse ainda nao exista; depender apenas do estado efemero do textbox
    # tornava uma corrida interrompida impossivel de retomar.
    roteiro_projeto = None
    try:
        from script_pipeline.production_project import project_for
        raiz_projeto = project_for(RUNS_DIR / nome)
        candidato = raiz_projeto / "roteiro" / "roteiro.txt" if raiz_projeto else None
        if candidato and candidato.exists():
            roteiro_projeto = candidato
    except (OSError, ValueError, TypeError):
        roteiro_projeto = None
    if ja_tem_parse:
        roteiro_path_novo = ""
        aviso_roteiro_txt = (
            "Roteiro desta corrida já processado (veja \"Parse + enriquecimento\" abaixo). "
            "Só traga um roteiro aqui se quiser SUBSTITUIR o desta corrida."
        )
    elif roteiro_preparado:
        roteiro_path_novo = roteiro_path_atual
        aviso_roteiro_txt = (
            f"✅ Roteiro já preparado ({Path(roteiro_path_atual).name}) continua em uso "
            "nesta corrida. Passo 3: clique Rodar (aba Stills).")
    elif roteiro_projeto is not None:
        roteiro_path_novo = str(roteiro_projeto)
        aviso_roteiro_txt = (
            f"✅ Roteiro restaurado do projeto mestre ({roteiro_projeto.name}). "
            "Passo 3: clique Rodar (aba Stills).")
    else:
        roteiro_path_novo = ""
        aviso_roteiro_txt = ("Corrida nova, sem roteiro ainda. Passo 2: arraste um .txt OU cole "
                            "o texto e clique \"Usar texto colado\".")
    return (auto, cast, plano, enriq, stills, video,
            _corrida_status_pronta(nome),
            roteiro_path_novo, aviso_roteiro_txt,
            _stage_html(None, -1), resumo_descriptor_gaps(cast))


def resumo_descriptor_gaps(cast_texto: str) -> str:
    """Aviso destacado quando algum personagem ficou com descritor
    incompleto (`descriptor_gaps` -- ver `cast_characters.py::
    _audit_and_fix_descriptors`). Antes so dava pra ver lendo o JSON bruto
    linha por linha ou rolando o log da corrida -- pedido do usuario
    2026-09-09."""
    try:
        cast = json.loads(cast_texto) if (cast_texto or "").strip() else {}
    except json.JSONDecodeError:
        return ""
    if not cast:
        return ""
    incompletos = {nome: info.get("descriptor_gaps") for nome, info in cast.items()
                  if isinstance(info, dict) and info.get("descriptor_gaps")}
    if not incompletos:
        return "✅ Todos os descritores completos (cabelo, rosto/porte, roupa, item único)."
    linhas = [f"- **{nome}**: faltando {', '.join(gaps)}" for nome, gaps in incompletos.items()]
    return ("⚠️ **Descritor incompleto** (a aparência pode sair genérica nos stills desse "
            "personagem):\n" + "\n".join(linhas))


def salvar_cast(nome: str, conteudo: str) -> tuple:
    if not nome:
        return "Selecione uma corrida primeiro.", gr.update()
    try:
        json.loads(conteudo)          # nao gravar JSON quebrado por cima do bom
    except json.JSONDecodeError as e:
        return f"JSON invalido, nada gravado: {e}", gr.update()
    alvo = RUNS_DIR / nome / "characters" / "cast.json"
    alvo.parent.mkdir(parents=True, exist_ok=True)
    alvo.write_text(conteudo, encoding="utf-8")
    return (f"cast.json gravado. Os stills usam o descritor na chave de cache, "
            f"entao rodar de novo refaz so os planos afetados.",
            resumo_descriptor_gaps(conteudo))


# --------------------------------------------------------------------------
# roteiro: PONTO UNICO de entrada, por arquivo (arrastar/clicar) OU por texto
# colado -- pedido do usuario 2026-09-09: os dois caminhos existiam como
# campos separados e confusos ("Roteiro .txt" de um lado, "Texto do roteiro"
# do outro), sem deixar claro que os dois terminavam no MESMO lugar. Os dois
# agora convergem em `roteiro_path` (visivel na tela, e o unico valor que
# `rodar()` de fato usa como `--script`).
# --------------------------------------------------------------------------
ROTEIROS_DIR = RUNS_DIR / "_roteiros"


def _slug(texto: str) -> str:
    base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (texto or "").strip())
    return base or time.strftime("roteiro_%Y%m%d_%H%M%S")


def usar_arquivo_roteiro(arquivo, nome_run: str) -> tuple:
    """Dispara sozinho quando um .txt e arrastado ou escolhido -- sem
    precisar de um botao extra, porque so ha UMA coisa sensata a fazer com
    um arquivo que acabou de chegar.

    Sem ARQUIVO, NAO MEXE em nada (gr.update() nos dois campos) em vez de
    mostrar aviso -- este evento tambem dispara quando `usar_texto_colado`
    limpa a caixa de arquivo de proposito (par com o texto colado), e um
    aviso aqui apagaria a mensagem de sucesso que aquele acabou de escrever."""
    if not arquivo:
        return gr.update(), gr.update()
    caminho = Path(arquivo)
    try:
        linhas = len(caminho.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError as e:
        return "", f"Nao consegui ler o arquivo: {e}"
    return str(caminho), f"✅ Roteiro pronto (arquivo): {caminho.name} -- {linhas} linha(s). Agora clique Rodar (aba abaixo)."


def usar_texto_colado(texto: str, nome_run: str) -> tuple:
    """Grava o texto colado como .txt em `_roteiros/` e devolve o mesmo
    formato que `usar_arquivo_roteiro` -- os dois caminhos de entrada
    terminam no MESMO lugar. Tambem limpa o campo de arquivo (`gr.File`),
    para o estado na tela nunca sugerir os dois ao mesmo tempo."""
    texto = (texto or "").strip()
    if not texto:
        return gr.update(), "Cole o texto do roteiro antes de clicar aqui.", gr.update()
    ROTEIROS_DIR.mkdir(parents=True, exist_ok=True)
    base = _slug(nome_run)
    caminho = ROTEIROS_DIR / f"{base}.txt"
    n = 2
    while caminho.exists():
        caminho = ROTEIROS_DIR / f"{base}_{n}.txt"
        n += 1
    caminho.write_text(texto, encoding="utf-8")
    linhas = len(texto.splitlines())
    return (str(caminho),
            f"✅ Roteiro pronto (colado): salvo em {caminho.name} -- {linhas} linha(s). Agora clique Rodar (aba abaixo).",
            gr.update(value=None))


# --------------------------------------------------------------------------
# gestao de corrida: nova / criar (salvar nome) / apagar
# --------------------------------------------------------------------------
def nova_corrida() -> tuple:
    """"① Nova corrida": so limpa a TELA (roteiro, cast, plano, stills,
    log, status) e desmarca a corrida selecionada -- nao cria nem apaga nada
    em disco. Prepara a interface para "Salvar" (que ai sim cria a pasta) ou
    para carregar outra corrida existente. Pedido do usuario: clicar em Nova
    tem que deixar a tela pronta pra um roteiro/prompt diferente, sem
    resquicio da corrida anterior."""
    auto, cast, plano, enriq, stills, video = carregar_run(None)
    return (
        gr.update(value=None),                      # runs: desmarcado
        "",                                          # novo_nome
        None,                                        # arquivo_roteiro
        "",                                          # roteiro_colado
        "",                                          # roteiro_path
        "Aguardando: traga o roteiro depois de salvar a corrida (passo 2).",  # aviso_roteiro
        "Tela limpa. Passo 1: dê um nome acima e clique \"Salvar corrida\".",  # aviso_corrida
        auto, cast, plano, enriq, stills, video,
        "",                                          # log
        _stage_html(None, -1),                       # estagio_html
        "",                                          # cast_gaps_aviso
    )


def criar_corrida(nome: str) -> tuple:
    """"Salvar corrida": cria a PASTA da corrida nova (com timestamp, mesmo
    esquema que `rodar()` já usava para corrida sem nome) e a seleciona --
    sem isso não havia como ter uma corrida "de verdade" (com nome escolhido
    pela pessoa) antes de rodar a cadeia inteira."""
    base = _slug(nome) if (nome or "").strip() else None
    if base is None:
        return gr.update(), "Informe um nome antes de clicar \"Salvar corrida\"."
    nome_final = f"{time.strftime('%Y%m%d_%H%M')}_{base}"
    run = RUNS_DIR / nome_final
    if run.exists():
        return gr.update(), f"Já existe uma corrida chamada '{nome_final}' -- tente outro nome."
    run.mkdir(parents=True)
    novas = listar_runs()
    return (gr.update(choices=novas, value=nome_final),
            f"✅ Corrida '{nome_final}' criada e selecionada. Passo 3: traga o roteiro abaixo.")


def apagar_corrida(nome_run: str, confirmacao: str) -> tuple:
    """"Apagar corrida": remove a pasta inteira da corrida selecionada. Exige
    digitar o NOME EXATO da corrida num campo ao lado -- a unica confirmacao
    disponivel sem um dialogo de confirmar nativo do Gradio, e o suficiente
    para evitar apagar por clique acidental."""
    if not nome_run:
        return "Selecione uma corrida para apagar.", gr.update()
    if (confirmacao or "").strip() != nome_run:
        return f"Para confirmar, digite exatamente '{nome_run}' no campo de confirmação e clique de novo.", gr.update()
    run = RUNS_DIR / nome_run
    if run.exists():
        shutil.rmtree(run)
    novas = listar_runs()
    return f"🗑️ Corrida '{nome_run}' apagada.", gr.update(choices=novas, value=(novas[0] if novas else None))


# --------------------------------------------------------------------------
# trilha de estagios: "estamos aqui" durante uma corrida (rodar())
# --------------------------------------------------------------------------
# As mesmas tags que `run_decupagem.py::passo()` imprime entre colchetes --
# ver ali (`passo("1 parse", ...)`, etc.). Nomes amigaveis so pra exibicao;
# a TAG (chave do dict) precisa bater exatamente com o que passo() imprime.
ESTAGIOS = [
    ("1 parse", "Ler roteiro"),
    ("2 cast", "Elenco"),
    ("E emocao", "Emoção"),
    ("4 tts", "Voz (TTS)"),
    ("S estrutura", "Estrutura"),
    ("P decupagem", "Decupagem"),
    ("C character-sheet", "Ref. personagem"),
    ("5-D stills", "Stills"),
    ("R rascunho", "Animatic"),
    ("5-D video", "Vídeo"),
    ("6 lipsync", "Lip-sync"),
    ("7 mix", "Mixagem"),
    ("8 montagem", "Montagem"),
    ("9 verificacao", "Verificação"),
]
_ESTAGIO_INDICE = {tag: i for i, (tag, _) in enumerate(ESTAGIOS)}
_ESTAGIO_RE = re.compile(r"^\[([^\]]+)\]")


def _stage_html(atual: int | None, feito_ate: int, falhou: bool = False) -> str:
    """Uma fileira de "chips": cinza = ainda não chegou, azul = ESTAMOS AQUI,
    verde = concluído, vermelho = falhou aqui. É a "linha de status" que o
    usuário pediu -- em vez de só texto, marca visualmente onde a corrida
    está dentro das 14 etapas possíveis (algumas opcionais/paradas antes)."""
    chips = []
    for i, (_, label) in enumerate(ESTAGIOS):
        if i == atual:
            estilo = "background:#dc2626;color:#fff;font-weight:600;" if falhou else \
                     "background:#2563eb;color:#fff;font-weight:600;"
            marcador = "✗ " if falhou else "▶ "
        elif i <= feito_ate:
            estilo, marcador = "background:#15803d;color:#fff;", "✓ "
        else:
            estilo, marcador = "background:#e5e7eb;color:#6b7280;", ""
        chips.append(
            f'<span style="{estilo}padding:5px 11px;margin:2px;border-radius:6px;'
            f'display:inline-block;font-size:12px;">{marcador}{label}</span>'
        )
    return "<div>" + "".join(chips) + "</div>"


# --------------------------------------------------------------------------
# aba de character sheet (2026-09-03, pedido do usuario -- ver MEMORIAL 3.53)
# --------------------------------------------------------------------------
# Referencia barata pra identidade, sem treinar LoRA nenhum: um still SO, com
# tres vistas do MESMO personagem lado a lado (frente/3-4/perfil), fundo
# neutro. Vira a `reference_image` do personagem em cast.json -- o mesmo
# campo que generate_scene_storyboard ja sabe usar (FLUX Klein/FLUX.1), so
# que agora alimentado por uma sheet desenhada pra isso em vez do primeiro
# still que calhou de sair perto o bastante.
SHEET_PROMPT_TEMPLATE = (
    "character reference turnaround sheet, three full-body views of the exact "
    "SAME person side by side on one image -- front view, three-quarter view, "
    "side profile view -- identical costume, identical hairstyle, identical "
    "proportions and face across all three views, neutral flat grey studio "
    "background, even soft studio lighting, no shadow gradient between panels. "
    "{descritor} Photorealistic, sharp focus, consistent character design "
    "sheet, no text, no watermark, no logo."
)


# --------------------------------------------------------------------------
# aba de pre-visualizacao de stills: estilo, etnia/aparencia, regeneracao
# --------------------------------------------------------------------------
# Modificadores de TEXTO, nao checkpoints diferentes -- o motor de imagem
# continua sendo o escolhido em `motor_img` (FLUX/SD3.5/etc). Combinaveis
# entre si (Estilo x Aparencia) porque os dois so se concatenam ao prompt do
# still que ja existe no shot_plan; nenhum dos dois muda estrutura, so texto.
STYLE_PRESETS: dict[str, str] = {
    "(nenhum)": "",
    "cinematografico realista": "cinematic photography, realistic film look, "
        "35mm lens, shallow depth of field, natural film grain, color-graded",
    "vintage / filme antigo": "vintage film photography, aged film stock look, "
        "faded colors, soft grain, light halation, 1970s cinematography",
    "anime japones": "japanese anime style, cel-shaded, clean line art, "
        "vibrant flat colors, anime key visual composition",
    "animacao 3D (Pixar/DreamWorks)": "3D animated film style, Pixar/DreamWorks "
        "look, stylized proportions, soft global illumination, subsurface "
        "scattering skin shading",
    "estilo Disney": "classic Disney animation style, expressive character "
        "design, warm painterly lighting, storybook illustration quality",
    "stop-motion": "stop-motion animation style, handcrafted felt/clay puppet "
        "look, visible fingerprint texture, miniature set photography, "
        "Laika/Aardman aesthetic",
}

ETHNICITY_PRESETS: dict[str, str] = {
    "(nenhuma)": "",
    "japonesa": "Japanese ethnicity, East Asian features",
    "coreana": "Korean ethnicity, East Asian features",
    "chinesa": "Chinese ethnicity, East Asian features",
    "asiatica (generica)": "Asian ethnicity",
    "americana": "American, diverse Western features",
    "europeia": "European ethnicity, Western features",
}


def regenerar_still(nome_run: str, indice, estilos: list, etnia: str,
                     motor_img: str, lora: str = "(nenhum)", lora_strength: float = 0.8) -> tuple:
    """Regenera UM still na hora (nao so marca para refazer -- gera de
    verdade e devolve a galeria ja atualizada), com Estilo(s) + Aparencia
    combinados ao prompt original do plano.

    Diferenca de proposito para "Marcar para refazer": aquele apaga e deixa o
    PROXIMO `render_shots_stage --stills-only` regerar com o prompt CANONICO
    do shot_plan (sem estilo extra). Este aqui e o caminho de EXPERIMENTAR uma
    variante -- estilo/etnia NAO sao gravados no shot_plan, entao uma corrida
    completa futura volta ao prompt original (a chave de cache em
    `_still_key` e computada a partir do shot_plan, nao do que esta em disco,
    entao o still com estilo fica automaticamente "desatualizado" pra ela --
    comportamento correto, nao residual)."""
    if not nome_run:
        return None, "Selecione uma corrida primeiro.", None
    try:
        idx = int(indice)
    except (TypeError, ValueError):
        return None, "Informe o numero da tomada (clique numa imagem da galeria, ou digite).", None

    run = RUNS_DIR / nome_run
    plan = _plano(run)
    shots = plan.get("shots") or []
    if not (0 <= idx < len(shots)):
        return None, f"Tomada {idx} nao existe neste shot_plan (0..{len(shots)-1}).", None
    shot = shots[idx]

    sufixo = ", ".join(
        [STYLE_PRESETS.get(e, "") for e in (estilos or []) if STYLE_PRESETS.get(e)]
        + ([ETHNICITY_PRESETS.get(etnia, "")] if ETHNICITY_PRESETS.get(etnia) else [])
    )
    prompt = shot.get("storyboard_prompt", "")
    if sufixo:
        prompt = f"{prompt}, {sufixo}"

    stills_dir = run / "shots" / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    out_path = stills_dir / f"shot{idx:03d}_{shot.get('framing', 'still')}.png"

    # Referencia: cast.json do personagem (character sheet ou still anterior
    # gravado a mao), senao a que o manifesto ja registrou pra esta tomada --
    # simplificado de proposito. O laco completo em render_shots.render()
    # tambem encadeia o PRIMEIRO still de cada personagem/locacao como
    # referencia dos seguintes; aqui, um preview avulso, isso reintroduziria
    # o mesmo custo (subir ComfyUI, decidir ordem) por uma unica imagem.
    ref = None
    sujeito = shot.get("subject") or ""
    if sujeito:
        try:
            cast = json.loads((run / "characters" / "cast.json").read_text(encoding="utf-8"))
            ref = (cast.get(sujeito) or {}).get("reference_image")
        except (OSError, json.JSONDecodeError):
            pass
    if not ref:
        man = json.loads((stills_dir / "stills.json").read_text(encoding="utf-8")) \
            if (stills_dir / "stills.json").exists() else {}
        ref = (man.get(str(idx)) or {}).get("reference")

    sys.path.insert(0, str(ROOT))
    from script_pipeline.generate_storyboards import (
        engine_defaults, ensure_comfyui_running, generate_scene_storyboard,
    )
    motor = engine_defaults(motor_img)

    def log(msg):
        print(f"[regen still {idx}] {msg}", flush=True)

    if not ensure_comfyui_running("http://127.0.0.1:8188", log=log):
        return None, "ComfyUI nao subiu -- confira o log do terminal.", None

    ok = generate_scene_storyboard(
        {"index": idx}, {}, server="http://127.0.0.1:8188",
        checkpoint=motor["checkpoint"], width=960, height=544,
        steps=motor["steps"], cfg=motor["cfg"], guidance=motor["guidance"],
        weight_dtype=motor.get("weight_dtype", "default"),
        seed=int(time.time()) % 100000, out_path=out_path,
        clip=motor["clip"], vae=motor["vae"],
        prompt_override=prompt, reference_image=ref,
        art_directed=bool(shot.get("art_direction")), log=log,
        lora_name="" if lora in (None, "(nenhum)") else lora, lora_strength=lora_strength,
    )
    if not (ok and out_path.exists()):
        return stills_com_legenda(run), "Geracao falhou -- confira o log do terminal.", None
    status = f"Tomada {idx} regenerada" + (f" com: {sufixo}" if sufixo else " (sem estilo extra)") + "."
    return stills_com_legenda(run), status, str(out_path)


def gerar_sheet_personagem(nome_run: str, personagem: str, descritor: str, motor_img: str,
                           lora: str = "(nenhum)", lora_strength: float = 0.8):
    if not nome_run:
        return None, "Selecione uma corrida primeiro."
    if not personagem.strip():
        return None, "Informe o nome do personagem."
    if not descritor.strip():
        return None, "Informe o descritor visual (cole o de cast.json, ou escreva um novo)."

    run = RUNS_DIR / nome_run
    out_dir = run / "characters" / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in personagem.strip())
    out_path = out_dir / f"{slug}.png"

    sys.path.insert(0, str(ROOT))
    from script_pipeline.generate_storyboards import (
        engine_defaults, ensure_comfyui_running, generate_scene_storyboard,
    )

    motor = engine_defaults(motor_img)
    prompt = SHEET_PROMPT_TEMPLATE.format(descritor=descritor.strip())

    def log(msg):
        print(f"[sheet] {msg}", flush=True)

    if not ensure_comfyui_running("http://127.0.0.1:8188", log=log):
        return None, "ComfyUI nao subiu -- confira o log do terminal."

    ok = generate_scene_storyboard(
        {"index": 0}, {}, server="http://127.0.0.1:8188",
        checkpoint=motor["checkpoint"], width=1024, height=1024,
        steps=motor["steps"], cfg=motor["cfg"], guidance=motor["guidance"],
        weight_dtype=motor.get("weight_dtype", "default"),
        seed=hash(personagem) % 100000, out_path=out_path,
        clip=motor["clip"], vae=motor["vae"],
        prompt_override=prompt, log=log,
        lora_name="" if lora in (None, "(nenhum)") else lora, lora_strength=lora_strength,
    )
    if not (ok and out_path.exists()):
        return None, "Geracao falhou -- confira o log do terminal."
    return str(out_path), (f"Sheet gerada -> {out_path}. Use \"Usar como referencia\" "
                           f"pra gravar em cast.json, ou copie o caminho a mao.")


def usar_sheet_como_referencia(nome_run: str, personagem: str, caminho_sheet: str, conteudo_cast: str):
    """Grava `caminho_sheet` como reference_image de `personagem` no JSON que
    esta na caixa de texto do cast -- MESMA caixa que `salvar_cast` grava, pra
    nao ter dois caminhos de escrita em cast.json divergindo."""
    if not caminho_sheet:
        return conteudo_cast, "Gere a sheet primeiro."
    if not personagem.strip():
        return conteudo_cast, "Informe o nome do personagem."
    try:
        dados = json.loads(conteudo_cast) if conteudo_cast.strip() else {}
    except json.JSONDecodeError as e:
        return conteudo_cast, f"cast.json na caixa esta invalido, nada mudado: {e}"
    entrada = dados.setdefault(personagem.strip(), {})
    entrada["reference_image"] = caminho_sheet
    novo_texto = json.dumps(dados, ensure_ascii=False, indent=2)
    return novo_texto, (f"reference_image de {personagem} atualizado na caixa abaixo -- "
                        f"clique \"Salvar cast.json\" pra gravar em disco.")


def carregar_sheet_report(nome_run: str) -> tuple:
    """Le `characters/sheet_report.json` (Fase A automatica, `--character-
    sheet`, ver `script_pipeline/character_sheet.py`) e devolve uma galeria
    com TODOS os candidatos de TODOS os personagens, legendados com o
    personagem, a posicao, a similaridade media e um marcador no escolhido
    (medoid) -- pedido do usuario 2026-09-09: a etapa ja existe no backend
    (run_decupagem --character-sheet) desde 2026-09-08 mas nunca teve UI para
    revisar OU trocar a escolha automatica. Clicar num candidato aqui
    preenche os campos do "Character sheet" manual logo acima, que ja sabe
    gravar `reference_image` em cast.json -- reusa o caminho de escrita
    existente em vez de duplicar logica."""
    if not nome_run:
        return [], "Selecione uma corrida primeiro.", []
    report_path = RUNS_DIR / nome_run / "characters" / "sheet_report.json"
    if not report_path.exists():
        return [], ("Nenhum sheet_report.json nesta corrida -- rode com \"Gerar character "
                    "sheet automatica\" marcado (aba Motores) para gerar um."), []
    try:
        relatorio = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [], f"sheet_report.json invalido: {e}", []

    itens = []
    for nome, info in relatorio.items():
        candidatos = info.get("candidates") or []
        escolhido = info.get("reference_image")
        score = info.get("score")
        for i, caminho in enumerate(candidatos):
            marca = " · ESCOLHIDO (medoid)" if caminho == escolhido else ""
            legenda = f"{nome} · candidato {i + 1}/{len(candidatos)}{marca}"
            if score is not None and caminho == escolhido:
                legenda += f" · sim {score:.3f}"
            itens.append((caminho, legenda))
    if not itens:
        return [], "sheet_report.json existe mas nao tem candidatos registrados.", []
    status = f"{len(relatorio)} personagem(ns), {len(itens)} candidato(s) no total."
    return itens, status, itens


def selecionar_candidato_sheet(itens_legendas: list, evt: gr.SelectData) -> tuple:
    """Galeria.select devolve o indice clicado -- extrai personagem+caminho da
    legenda "NOME · candidato N/M[...]" que `carregar_sheet_report` monta (via
    o `gr.State` que guarda a mesma lista), e preenche os campos do sheet
    manual (personagem/imagem) para o botao "Usar como referencia deste
    personagem", ja existente, gravar."""
    if not itens_legendas or not (0 <= evt.index < len(itens_legendas)):
        return gr.update(), gr.update()
    caminho, legenda = itens_legendas[evt.index]
    nome = legenda.split(" · ", 1)[0]
    return nome, caminho


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


def regenerar_selecao(nome_run: str, quais: str, motor_img: str, largura, altura,
                      lora: str = "(nenhum)", lora_strength: float = 0.8) -> tuple:
    """Apaga E regenera JA os stills da selecao (mesmo formato de tomadas que
    'Marcar para refazer' aceita -- numeros/intervalos/'todos'), sem esperar
    uma corrida completa. Pedido do usuario 2026-09-04: "Após marcar para
    refazer coloque um botão para regenerar seleção do textbox" -- ou seja,
    o botao ao lado nao so marca, ele espera terminar e devolve a galeria
    pronta. Reusa o mesmo `render_shots_stage --only-shots` que a corrida
    completa usa (nao reimplementa geracao), so que bloqueando ate acabar."""
    if not nome_run:
        return "Selecione uma corrida primeiro.", stills_com_legenda(RUNS_DIR / nome_run) if nome_run else []
    run = RUNS_DIR / nome_run
    aviso = apagar_stills(nome_run, quais)
    # "Nenhum still correspondeu" NAO bloqueia: significa so que nao havia
    # nada pra apagar (ja apagado antes, ou a tomada ainda nunca foi gerada)
    # -- a geracao abaixo roda do mesmo jeito e preenche o que faltar. So os
    # erros de INSUMO (corrida sem stills/, numero invalido) impedem seguir.
    if aviso.startswith(("Selecione", "Esta corrida", "Nao entendi")):
        return aviso, stills_com_legenda(run)

    alvo = (quais or "").strip().lower()
    only_shots_arg = None if alvo in ("", "todos", "all", "*") else quais.strip()

    env = os.environ.copy()
    env.update(ENV_VIDEO)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")

    cmd = [PY, "-u", "-m", "script_pipeline.render_shots_stage",
           "--run-dir", str(run), "--width", str(int(largura)), "--height", str(int(altura)),
           "--stills-only", "--image-engine", motor_img]
    if only_shots_arg:
        cmd += ["--only-shots", only_shots_arg]
    if lora and lora != "(nenhum)":
        cmd += ["--lora", lora, "--lora-strength", str(lora_strength)]

    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    saida = ((r.stdout or "") + (r.stderr or ""))[-1500:]
    status = f"{aviso}\n\nRegeneração terminada (código {r.returncode}).\n{saida}"
    return status, stills_com_legenda(run)


def limpar_cache_stills(nome_run: str, quais: str, derivados: bool, liberar_comfy: bool) -> str:
    """Limpador de cache para uma NOVA geracao de stills (pedido do usuario 2026-09-13).

    "Marcar para refazer" apaga so o PNG e a entrada do manifesto. O que ficava para tras
    e reaproveitado em silencio depois que o still muda:
      - clipe (`shots/clips/shotNNN.key`): VISTO 2026-09-13, closes novos com os clipes de
        plano medio antigos reusados;
      - lip-sync (`lipsync/shotNNN.key` e `_synced.mp4`), animatic e auditoria de storyboard;
      - o cache de nos/modelos do ComfyUI (`POST /free`), que segura o FLUX carregado.
    Nao apaga clipes .mp4: sem a .key eles sao refeitos na proxima passada de video."""
    if not nome_run:
        return "Selecione uma corrida primeiro."
    run = RUNS_DIR / nome_run
    linhas = [apagar_stills(nome_run, quais)]
    if linhas[0].startswith(("Selecione", "Nao entendi")):
        return linhas[0]

    alvo = (quais or "").strip().lower()
    todos = alvo in ("", "todos", "all", "*")
    indices = set()
    if not todos:
        for parte in alvo.replace(";", ",").split(","):
            parte = parte.strip()
            if "-" in parte:
                a, b = parte.split("-", 1)
                indices.update(range(int(a), int(b) + 1))
            elif parte:
                indices.add(int(parte))

    def _casa(p: Path) -> bool:
        m = re.match(r"shot(\d+)", p.name)
        return bool(m) and (todos or int(m.group(1)) in indices)

    if derivados:
        removidos = []
        for pasta, padroes in ((run / "shots" / "clips", ("shot*.key",)),
                               (run / "lipsync", ("shot*.key", "shot*_synced.mp4"))):
            if not pasta.is_dir():
                continue
            for padrao in padroes:
                for p in pasta.glob(padrao):
                    if _casa(p):
                        p.unlink(missing_ok=True)
                        removidos.append(f"{pasta.name}/{p.name}")
        for p in (run / "shots" / "animatic.mp4", run / "shots" / "storyboard_audit.json"):
            if p.exists():
                p.unlink(missing_ok=True)
                removidos.append(f"shots/{p.name}")
        linhas.append(f"Derivados invalidados ({len(removidos)}): "
                      + (", ".join(removidos[:12]) + (" ..." if len(removidos) > 12 else "")
                         if removidos else "nenhum"))

    if liberar_comfy:
        import urllib.request
        try:
            req = urllib.request.Request("http://127.0.0.1:8188/free", method="POST",
                                         data=json.dumps({"unload_models": True,
                                                          "free_memory": True}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            linhas.append("ComfyUI (8188): modelos descarregados e cache de nós liberado.")
        except OSError:
            linhas.append("ComfyUI (8188) fora do ar — nada a liberar.")
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# execucao
# --------------------------------------------------------------------------
def _argv(run: Path, script: str, estilo: str, trocas: str, largura: int,
          altura: int, ate: str, motor: str, recast: bool,
          motor_img: str = "flux", motor_video: str = "ltx",
          motor_voz: str = "auto", consistencia: float | None = None,
          camera_llm: bool = False, ltx_variant: str = "w4a8-v10",
          minimax_variant: str = "fp8int8", lora: str = "(nenhum)",
          lora_strength: float = 0.8, character_sheet_on: bool = False,
          character_sheet_n: int = 4, minimax_ref_audio: bool = False,
          minimax_no_still: bool = False, minimax_chain_max_seconds: float | None = None,
          ltx_no_still: bool = False, ltx_chain_max_seconds: float | None = None,
          spatial_on: bool = False, spatial_spec: str = "", spatial_denoise: float = .65,
          spatial_diagnostic: bool = False, motion_conditioning: bool = False,
          visual_unresolved: str = "block",
          video_loras: list | None = None, ic_reference: str = "off",
          ic_strength: float = 1.0, extras: list | None = None) -> list:
    cmd = [PY, "-u", "-m", "script_pipeline.run_decupagem",
           "--run-dir", str(run), "--style", estilo, "--ate", ate,
           "--width", str(int(largura)), "--height", str(int(altura)),
           "--engine", motor, "--image-engine", motor_img,
           "--video-engine", motor_video, "--tts-engine", motor_voz,
           "--ltx-variant", ltx_variant, "--minimax-variant", minimax_variant]
    # --script so na primeira vez: com o parse ja feito, passar de novo nao
    # muda nada, mas deixa a UI dependendo de um caminho que pode ter sumido.
    if script and not (run / "parse" / "scenes.json").exists():
        cmd += ["--script", script]
    if trocas.strip():
        cmd += ["--style-changes", trocas.strip()]
    if recast:
        cmd.append("--recast")
    # Auditoria de consistencia facial (insightface, MEMORIAL 3.53) -- opt-in
    # de proposito (custo extra por still). None/0 desliga; qualquer valor >0
    # liga com esse limiar. Pedido do usuario 2026-09-04: religar antes de
    # retomar os planos restantes do Beatriz/Lucas.
    if consistencia:
        cmd += ["--consistency-threshold", str(consistencia)]
    # LLM decide movimento de camera/luz por plano, dentro do vocabulario do
    # Estilo escolhido (shot_plan.CAMERA_STYLE_VOCAB) -- pedido do usuario
    # 2026-09-04. Custa 1 chamada de Ollama por cena; opt-in.
    if camera_llm:
        cmd.append("--camera-llm")
    if motion_conditioning:
        cmd.append("--motion-conditioning")
    if visual_unresolved == "continue":
        cmd += ["--visual-unresolved", "continue"]
    # LoRA opcional nos STILLS (nunca no video) -- pedido do usuario
    # 2026-09-09: models/loras_images/, separado dos LoRAs de video do LTX em
    # models/loras/, que sao incompativeis com estes checkpoints de imagem.
    if lora and lora != "(nenhum)":
        cmd += ["--lora", lora, "--lora-strength", str(lora_strength)]
    if character_sheet_on:
        cmd += ["--character-sheet", "--character-sheet-candidates", str(int(character_sheet_n))]
    if minimax_ref_audio and motor_video == "minimax":
        cmd.append("--minimax-ref-audio")
    # Pedido do usuario 2026-09-16: "e se usarmos so os descritivos?" (sem
    # still) e a opcao de dividir plano longo em sub-planos encadeados por
    # ultimo-frame (MEDIDO em _test_minimax_duration_cap.py: ~16s+ trava
    # 30min+ numa chamada so). Os dois so tem efeito com motor_video minimax.
    if minimax_no_still and motor_video == "minimax":
        cmd.append("--minimax-no-still")
    if minimax_chain_max_seconds and motor_video == "minimax":
        cmd += ["--minimax-chain-max-seconds", str(minimax_chain_max_seconds)]
    if ltx_no_still and motor_video == "ltx":
        cmd.append("--ltx-no-still")
    if ltx_chain_max_seconds and motor_video == "ltx":
        cmd += ["--ltx-chain-max-seconds", str(ltx_chain_max_seconds)]
    if spatial_on:
        if motor_img != "flux":
            raise ValueError("Modo espacial exige Motor das imagens = flux")
        if not spatial_spec or not Path(str(spatial_spec)).exists():
            raise ValueError("Modo espacial exige um arquivo JSON de blocking existente")
        cmd += ["--spatial-spec", str(spatial_spec), "--spatial-denoise", str(float(spatial_denoise)),
                "--reuse-plan"]
        if spatial_diagnostic:
            cmd.append("--visual-diagnostic")  # roda o gate, nao bloqueia, para no animatic
    # LoRAs de VIDEO do LTX (pedido do usuario 2026-09-12) -- so com motor ltx. O
    # dropdown mostra "chave (forca)" do catalogo ltx_loras.py; aqui vira chave:forca.
    if motor_video == "ltx":
        for rotulo in video_loras or []:
            chave, _, resto = str(rotulo).partition(" (")
            forca = resto.rstrip(")")
            cmd += ["--video-lora", f"{chave}:{forca}" if forca else chave]
        if ic_reference and ic_reference != "off":
            cmd += ["--ic-reference", ic_reference, "--ic-strength", str(ic_strength)]
    # Lip-sync, guia do IC e pos-producao: ja chegam como argumentos prontos de _rodar_ui.
    cmd += list(extras or [])
    return cmd


def rodar(nome_run, script, novo_nome, estilo, trocas, largura, altura, ate, motor,
          recast, motor_img="flux", motor_video="ltx", motor_voz="auto",
          consistencia=None, camera_llm=False, ltx_variant="w4a8-v10",
          minimax_variant="fp8int8", lora="(nenhum)", lora_strength=0.8,
          character_sheet_on=False, character_sheet_n=4, minimax_ref_audio=False,
          minimax_no_still=False, minimax_chain_max_seconds=None,
          ltx_no_still=False, ltx_chain_max_seconds=None,
          spatial_on=False, spatial_spec="", spatial_denoise=.65,
          spatial_diagnostic=False, motion_conditioning=False, visual_unresolved="block",
          video_loras=None, ic_reference="off", ic_strength=1.0, extras=None):
    """Executa a cadeia transmitindo o stdout. Gerador: a UI recebe cada linha.

    O subprocesso e o MESMO que o .bat dispara. A UI nao reimplementa etapa
    nenhuma -- se o orquestrador mudar, isto acompanha."""
    if _PROC["p"] is not None and _PROC["p"].poll() is None:
        yield "Ja existe uma corrida em andamento. Pare antes de comecar outra.", [], None, "", _stage_html(None, -1)
        return

    if nome_run:
        run = RUNS_DIR / nome_run
    else:
        base = (novo_nome or "").strip() or (Path(script).stem if script else "")
        if not base:
            yield ("Informe um roteiro (para uma corrida nova) ou selecione uma existente.",
                   [], None, "", _stage_html(None, -1))
            return
        run = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M')}_{base}"

    if not (run / "parse" / "scenes.json").exists() and not script:
        yield (f"A corrida {run.name} nao tem parse feito e nenhum roteiro foi informado.",
               [], None, "", _stage_html(None, -1))
        return

    # Falhar antes de abrir o subprocesso. Antes desta verificacao, a UI
    # mostrava a lista fallback quando o Ollama estava desligado e aceitava a
    # corrida; o parser entao gastava 3 tentativas por cena e seguia sem o
    # enriquecimento pedido. Para stills, o Qwen3-VL tambem e requisito do
    # gate visual padrao.
    disponiveis = set(listar_modelos_ollama())
    exigidos = set()
    if motor and not str(motor).startswith("nvidia/"):
        exigidos.add(str(motor))
    if PARADAS.index(ate) >= PARADAS.index("stills"):
        exigidos.add("qwen3-vl:30b")
    ausentes = sorted(exigidos - disponiveis)
    if ausentes:
        estado = ("Ollama indisponivel." if not disponiveis else
                  "Modelos ausentes na instancia Ollama ativa: " + ", ".join(ausentes))
        yield (estado + " Inicie a instancia correta e tente novamente.", [], None, "",
               _stage_html(None, -1))
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
                motor_img, motor_video, motor_voz, consistencia, camera_llm,
                ltx_variant, minimax_variant, lora, lora_strength,
                character_sheet_on, character_sheet_n, minimax_ref_audio,
                minimax_no_still, minimax_chain_max_seconds,
                ltx_no_still, ltx_chain_max_seconds,
                spatial_on, spatial_spec, spatial_denoise,
                spatial_diagnostic, motion_conditioning, visual_unresolved,
                video_loras, ic_reference, ic_strength, extras)
    linhas = [f"$ {' '.join(cmd[3:])}", f"(corrida: {run})", ""]
    # Rastreio da "trilha de estagios" (a linha de status "estamos aqui" que o
    # usuario pediu): cada linha `[TAG] ...` que `run_decupagem.py::passo()`
    # imprime avanca o estagio ATUAL; tudo antes dele vira "concluido".
    atual, feito_ate = None, -1
    yield "\n".join(linhas), stills_com_legenda(run), video_de(run), resumo_do_plano(run), _stage_html(atual, feito_ate)

    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    _PROC["p"] = p
    ultimo = 0.0
    try:
        for linha in p.stdout:
            linhas.append(linha.rstrip("\n"))
            m = _ESTAGIO_RE.match(linha.strip())
            if m and m.group(1) in _ESTAGIO_INDICE:
                if atual is not None:
                    feito_ate = max(feito_ate, atual)
                atual = _ESTAGIO_INDICE[m.group(1)]
            # A galeria e o plano so sao relidos a cada ~1,5s: um still leva
            # dezenas de segundos, e varrer o disco a cada linha de log so
            # gastaria E/S para mostrar a mesma coisa.
            agora = time.time()
            if agora - ultimo > 1.5:
                ultimo = agora
                yield ("\n".join(linhas[-400:]), stills_com_legenda(run), video_de(run),
                       resumo_do_plano(run), _stage_html(atual, feito_ate))
    finally:
        p.wait()
        _PROC["p"] = None
        _descarregar_ollama()
    linhas.append("")
    linhas.append(f"[fim] codigo de saida {p.returncode}")
    sucesso = p.returncode == 0
    if sucesso and atual is not None:
        feito_ate = atual
        atual = None
    yield ("\n".join(linhas[-400:]), stills_com_legenda(run), video_de(run), resumo_do_plano(run),
           _stage_html(atual, feito_ate, falhou=not sucesso and atual is not None))


def parar() -> str:
    p = _PROC.get("p")
    if p is None or p.poll() is not None:
        descarregados = _descarregar_ollama()
        return ("Nada rodando. " +
                (f"Ollama liberado: {', '.join(descarregados)}."
                 if descarregados else "Nenhum modelo Ollama estava carregado."))
    try:
        if sys.platform == "win32":
            # O orquestrador abre um subprocesso por etapa. Terminar apenas o
            # pai deixa parse/TTS/ComfyUI orfaos; /T encerra a arvore inteira.
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           capture_output=True, text=True, timeout=15)
        else:
            p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
    finally:
        _PROC["p"] = None
        _descarregar_ollama()
    return ("Corrida encerrada. Os artefatos prontos ficam -- selecione a mesma "
            "corrida e rode de novo para continuar de onde parou.")


HELP_TEXT = """
**A tela funciona em 3 passos, sempre nesta ordem:**

**① Corrida.** Escolha uma corrida existente no menu e clique
**"Carregar corrida selecionada"** para retomar de onde parou -- OU clique
**"① Nova corrida"**, digite um nome no campo ao lado e clique **"Salvar
corrida"** para começar uma do zero. **"Atualizar lista de corridas"** só
relê a pasta em disco (útil se você criou uma corrida por fora, no
`start_decupagem.bat`); ele não carrega nada na tela sozinho.

**② Roteiro.** Traga o texto de UM jeito: arraste um `.txt` na caixa (ou
clique para escolher) -- entra sozinho -- OU cole o texto na caixa de baixo e
clique **"Usar texto colado"**. Os dois terminam no mesmo lugar: um roteiro
pronto, mostrado na linha de status logo abaixo da caixa.

**③ Rodar.** Ajuste Estilo / "Ir até" / Motores se quiser (abas acima) e
clique **"Rodar"**, na aba Stills. A cadeia passa sozinha por: ler/enriquecer
o roteiro → elenco → emoção das falas → voz (TTS) → estrutura narrativa →
decupagem (plano por plano) → [opcional] referência de personagem → stills →
animatic → vídeo → lip-sync → mixagem → montagem final. A **trilha de
estágios** abaixo do cabeçalho acende em azul o estágio atual e marca em
verde os já concluídos -- essa é a linha "estamos aqui".

**Antes de deixar rodar até o vídeo**, revise o Elenco (aba Decupagem) e os
Stills (aba Stills) -- corrigir ali é mais barato que descobrir no filme
pronto. **"Ir até" = animatic** é o ponto de revisão barato (nada de vídeo
ainda); rode de novo sem essa trava para seguir até o final.
"""


def _corrida_status_pronta(nome: str) -> str:
    return f"Corrida '{nome}' carregada. Passo 2: confira o roteiro abaixo (ou traga um novo) e siga para Rodar."


# --------------------------------------------------------------------------
def criar_projeto_mestre(titulo, pasta, roteiro, duracao):
    from script_pipeline.production_project import create_project
    nome = time.strftime('%Y%m%d_%H%M%S') + '_' + _slug(titulo)
    run = RUNS_DIR / nome
    path = create_project(pasta, run, Path(roteiro).read_text(encoding='utf-8'), titulo, float(duracao))
    return gr.update(choices=listar_runs(), value=nome), path, resumo_projeto_mestre(nome)


def resumo_projeto_mestre(nome):
    from script_pipeline.production_project import project_for, read, sync_media
    root = project_for(RUNS_DIR / nome) if nome else None
    if root is None:
        return 'Corrida sem projeto mestre.'
    sync_media(RUNS_DIR / nome)
    run = RUNS_DIR / nome
    continuity = read(run / 'shots/continuity_audit.json', {})
    return json.dumps({'pasta': str(root), 'projeto': read(root / 'projeto.json'),
                       'cobertura': read(root / 'editorial/cobertura.json'),
                       'planos': len(read(root / 'editorial/timeline.json', {}).get('shots', [])),
                       'auditoria_continuidade': continuity.get('status', 'pendente'),
                       'bloqueios_continuidade': len(continuity.get('blocking', [])),
                       'avisos_objetos': len(continuity.get('warnings', []))},
                      ensure_ascii=False, indent=2)


PROJECT_DOCUMENTS = ['biblia/continuidade.json', 'biblia/locacoes.json', 'biblia/personagens.json',
                     'biblia/voz.json', 'audio/cues.json', 'editorial/timeline.json', 'vfx/jobs.json', 'projeto.json']


def documento_projeto(nome, documento, conteudo=None):
    from script_pipeline.production_project import project_for, read, write
    if documento not in PROJECT_DOCUMENTS:
        raise gr.Error('Documento não permitido.')
    if _PROC['p'] is not None and _PROC['p'].poll() is None and conteudo is not None:
        raise gr.Error('Aguarde a etapa atual terminar antes de editar o projeto.')
    root = project_for(RUNS_DIR / nome)
    if root is None:
        raise gr.Error('Selecione uma corrida com projeto mestre.')
    if conteudo is not None:
        data = json.loads(conteudo)
        if documento == 'editorial/timeline.json':
            from script_pipeline.production_project import validate_timeline
            validate_timeline(data)
        previous = read(root / documento)
        from script_pipeline.production_project import digest
        write(root / 'roteiro/versoes' / (Path(documento).stem + '_' + digest(previous)[:16] + '.json'), previous)
        write(root / documento, data)
    return json.dumps(read(root / documento, {}), ensure_ascii=False, indent=2)


def importar_tomada(nome, plano, caminho, aprovar, entrada):
    from script_pipeline.production_project import project_for
    from script_pipeline.production_post import set_take
    root = project_for(RUNS_DIR / nome)
    if not root:
        raise gr.Error('Corrida sem projeto mestre.')
    set_take(root, plano.strip(), caminho.strip(), bool(aprovar), int(entrada))
    return resumo_projeto_mestre(nome)


def exportar_projeto(nome, aprovado):
    from script_pipeline.production_post import conform_movie
    if _PROC['p'] is not None and _PROC['p'].poll() is None:
        raise gr.Error('Aguarde a geração terminar antes de conformar.')
    return conform_movie(RUNS_DIR / nome, approved_only=bool(aprovado))


def build() -> None:
    # A tela abre JA mostrando a corrida mais recente. Alem de ser o que a
    # pessoa quase sempre quer ver, isso evita uma tela vazia que nao diz se a
    # UI esta funcionando ou se nao ha nada para mostrar.
    _runs = listar_runs()
    _inicial = _runs[0] if _runs else None
    _auto0, _cast0, _plano0, _enriq0, _stills0, _video0 = carregar_run(_inicial)

    with gr.Blocks(title="Decupagem — LTX") as demo:
        gr.Markdown(
            "# Decupagem\n"
            "Roteiro → cast → voz → decupagem → stills → animatic → filme. "
            "Mesma cadeia do `start_decupagem.bat`, com o log à vista e retomada."
        )
        with gr.Accordion("❓ Como funciona (leia antes da primeira vez)", open=False):
            gr.Markdown(HELP_TEXT)

        # Trilha de estagios: "linha de status" pedida pelo usuario -- acende
        # o estagio ATUAL em azul e marca os concluidos em verde. Visivel o
        # tempo todo (nao so na aba Stills), porque e a resposta a "em que pe
        # esta a corrida" nas 3 abas. Fora de qualquer corrida, fica cinza.
        estagio_html = gr.HTML(_stage_html(None, -1))

        # Selecao de corrida: FORA das abas, de proposito -- e o unico estado
        # que as tres abas (Decupagem/Motores/Stills) compartilham.
        with gr.Row():
            with gr.Column(scale=2):
                runs = gr.Dropdown(choices=_runs, label="Corrida existente (para RETOMAR)",
                                   value=_inicial, allow_custom_value=False)
            with gr.Column(scale=1):
                atualizar = gr.Button(
                    "Atualizar lista de corridas",
                    elem_id="btn-atualizar",
                )
                # Botao explicito alem do evento do dropdown: o `change` do
                # Dropdown depende de uma selecao de verdade e nao dispara
                # quando o valor chega por outro caminho (restaurar sessao,
                # teclado, automacao). Um botao sempre dispara.
                carregar = gr.Button("Carregar corrida selecionada", variant="secondary")

        # PASSO 1 -- nova corrida ou apagar uma existente. Fica logo abaixo do
        # seletor de corrida, de proposito: e a mesma area de decisao ("qual
        # corrida eu uso"), so que para os dois casos que o seletor sozinho
        # nao cobre (comecar do zero, remover).
        with gr.Row():
            nova_btn = gr.Button("① Nova corrida", scale=1)
            novo_nome = gr.Textbox(label="Nome da corrida (nova ou para renomear a confirmação de exclusão)", scale=2)
            salvar_nome_btn = gr.Button("Salvar corrida", variant="secondary", scale=1)
        with gr.Row():
            apagar_confirma = gr.Textbox(
                label="Para apagar, digite aqui o NOME EXATO da corrida selecionada acima", scale=2)
            apagar_btn = gr.Button("🗑 Apagar corrida", variant="stop", scale=1)
        aviso_corrida = gr.Textbox(
            label="", interactive=False,
            value=(_corrida_status_pronta(_inicial) if _inicial else
                   "Passo 1: carregue uma corrida existente, ou clique \"① Nova corrida\"."))

        # Rodar/Parar ficavam so dentro da aba "Stills" -- clicavel so depois de navegar
        # ate la, sem indicio nas outras abas de que e ali que se inicia a corrida (achado
        # de operabilidade rodando de verdade, 2026-09-22). Visivel em QUALQUER aba agora.
        with gr.Row():
            btn = gr.Button("▶ Rodar", variant="primary", scale=2)
            btn_parar = gr.Button("■ Parar", scale=1)

        with gr.Tabs():
            with gr.Tab("Projeto mestre"):
                gr.Markdown("Projeto persistente fora do motor: roteiro versionado, continuidade, cobertura e timeline. As tomadas geradas ficam pendentes de revisão visual.")
                project_title = gr.Textbox(label="Título do filme", value="VOO 702 - CÉU TURBULENTO")
                project_root = gr.Textbox(label="Pasta do projeto", value=str(ROOT.parent / "Filmes" / "Voo702"))
                project_source = gr.Textbox(label="Arquivo do roteiro para importar")
                project_duration = gr.Number(label="Duração alvo (segundos)", value=300)
                project_create = gr.Button("Criar projeto e vincular corrida", variant="primary")
                project_refresh = gr.Button("Atualizar projeto / exportar timeline")
                project_status = gr.Textbox(label="Estado do projeto", lines=14, interactive=False)
                with gr.Accordion('Continuidade, som e edição', open=False):
                    project_doc = gr.Dropdown(PROJECT_DOCUMENTS, value=PROJECT_DOCUMENTS[0], label='Documento do projeto')
                    project_load = gr.Button('Ler documento')
                    project_json = gr.Textbox(label='Documento JSON editável', lines=18)
                    project_save = gr.Button('Salvar nova versão')
                    gr.Markdown('Timeline: selecione uma tomada existente pelo selected_take; ajuste trim_in em frames. Aprovação visual é explícita. Eventos sonoros: media, start, source_in, duration, gain, fade_in, fade_out (tempos em segundos).')
                with gr.Accordion('Tomada de movimento / VFX / alternativa', open=False):
                    take_shot = gr.Textbox(label='ID permanente do plano (SH_...)')
                    take_path = gr.Textbox(label='Arquivo de vídeo da tomada')
                    take_in = gr.Number(label='Entrada na tomada (frames)', value=0, precision=0)
                    take_approval = gr.Checkbox(label='Aprovar esta tomada após revisão visual', value=False)
                    take_import = gr.Button('Importar e selecionar tomada')
                delivery_approved = gr.Checkbox(label='Exigir todas as tomadas aprovadas para entrega', value=True)
                delivery_button = gr.Button('Conformar filme e exportar cinco stems')
                delivery_path = gr.Textbox(label='Entrega editorial', interactive=False)
            # ---------------------------------------------------------------
            # ABA 1: DECUPAGEM -- roteiro, cast, e o resultado do parse/LLM
            # ---------------------------------------------------------------
            with gr.Tab("Decupagem"):
                gr.Markdown("### Passo 2 — Traga o roteiro (um jeito só: arquivo OU texto colado)")
                with gr.Row():
                    arquivo_roteiro = gr.File(
                        label="Arraste um .txt aqui, ou clique para escolher", scale=1,
                        file_types=[".txt"], file_count="single", type="filepath")
                    with gr.Column(scale=1):
                        roteiro_colado = gr.Textbox(
                            label="...ou cole o texto do roteiro / prompt audiovisual aqui", lines=6,
                            placeholder="Cole aqui o roteiro em prosa, formato de cena, ou prompt "
                                        "audiovisual -- prose_to_screenplay.py cuida da conversão "
                                        "na primeira etapa, igual a um arquivo carregado.")
                        usar_texto_btn = gr.Button("Usar texto colado", variant="secondary")
                roteiro_path = gr.Textbox(label="Roteiro em uso (preenchido automaticamente)",
                                          interactive=False)
                aviso_roteiro = gr.Textbox(
                    label="", interactive=False,
                    value=("Roteiro desta corrida já processado (veja \"Parse + enriquecimento\" abaixo). "
                           "Só traga um roteiro aqui se quiser SUBSTITUIR o desta corrida."
                           if _inicial and (RUNS_DIR / _inicial / "parse" / "scenes.json").exists() else
                           "Passo 2: arraste um .txt OU cole o texto e clique \"Usar texto colado\"."))

                with gr.Row():
                    estilo = gr.Dropdown(choices=sorted(STYLES), value="classico", label="Estilo")
                    ate = gr.Dropdown(choices=PARADAS, value="animatic", label="Ir até")
                    recast = gr.Checkbox(value=False, label="Reescolher vozes")
                trocas = gr.Textbox(
                    label="Trocas de estilo — por cena (3:tenso) ou por tomada (1.10:intimista); vigora até a próxima marca",
                    placeholder="1:classico,1.10:intimista,2:tenso")

                with gr.Accordion("Parse + enriquecimento (o que o LLM entendeu do roteiro)", open=True):
                    gr.Markdown(
                        "Por cena: local, período, personagens e — por fala — a emoção e o "
                        "\"beat\" visual que o LLM inferiu quando não havia parentético "
                        "explícito no roteiro."
                    )
                    enriquecimento = gr.Textbox(value=_enriq0, label="", lines=16, max_lines=16,
                                                interactive=False)

                with gr.Accordion("Decupagem (shot_plan — tomada por tomada)", open=False):
                    plano = gr.Textbox(value=_plano0, label="", lines=14, max_lines=14,
                                       interactive=False)

                with gr.Accordion("Elenco — edite ANTES dos stills", open=False):
                    gr.Markdown(
                        "O descritor decide a aparência em **todos** os planos e a voz em "
                        "todas as falas. Corrigir aqui é mais barato que descobrir no filme."
                    )
                    cast_gaps_aviso = gr.Markdown(value=resumo_descriptor_gaps(_cast0))
                    cast = gr.Textbox(value=_cast0, label="cast.json", lines=16)
                    salvar = gr.Button("Salvar cast.json")
                    aviso = gr.Textbox(label="", interactive=False)

                with gr.Accordion("Character sheet (referência de identidade, antes de gerar stills)", open=False):
                    gr.Markdown(
                        "Gera UM still com três vistas do mesmo personagem lado a lado "
                        "(frente/3-4/perfil) — âncora mais forte de identidade que um still "
                        "avulso, sem precisar treinar LoRA nenhum. Vira `reference_image` do "
                        "personagem no `cast.json` (o mesmo campo que o FLUX já sabe usar)."
                    )
                    with gr.Row():
                        sheet_personagem = gr.Textbox(label="Personagem (nome EXATO em cast.json)", scale=2)
                        sheet_motor = gr.Dropdown(
                            choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext", "zimage", "qwen-image-2.1", "hidream", "qwen-image"],
                            value="flux", label="Motor", scale=1)
                    sheet_descritor = gr.Textbox(
                        label="Descritor visual (cole o de cast.json, ou escreva)", lines=2)
                    sheet_btn = gr.Button("Gerar sheet")
                    sheet_img = gr.Image(label="Sheet gerada", interactive=False, type="filepath")
                    sheet_status = gr.Textbox(label="", interactive=False)
                    sheet_usar_btn = gr.Button("Usar como referência deste personagem")

                    gr.Markdown(
                        "---\n**Character sheet automática (Fase A, medoid)** — gera N "
                        "candidatos por personagem e escolhe sozinha o mais central "
                        "(maior similaridade facial média aos outros); ligue em "
                        "\"Gerar character sheet automática\" na aba Motores antes de "
                        "rodar. Revise ou troque a escolha aqui depois."
                    )
                    sheet_report_btn = gr.Button("Revisar candidatos da última corrida")
                    sheet_report_status = gr.Textbox(label="", interactive=False)
                    sheet_report_gallery = gr.Gallery(
                        columns=4, height=360, show_label=False,
                        label="Clique num candidato para preenchê-lo acima e usar como referência")
                    sheet_report_itens = gr.State([])

                with gr.Accordion("Roteiro reconstruído (quando a entrada era prosa)", open=False):
                    auto = gr.Textbox(value=_auto0, label="screenplay_auto.txt", lines=16, interactive=False)

            # ---------------------------------------------------------------
            # ABA 2: MOTORES -- qual modelo/engine faz cada parte
            # ---------------------------------------------------------------
            with gr.Tab("Motores"):
                with gr.Row():
                    _motor_choices, _motor_padrao = motor_llm_padrao()
                    _motor_ao_vivo = _motor_choices != _MOTOR_LLM_FALLBACK
                    motor = gr.Dropdown(
                        choices=_motor_choices, value=_motor_padrao, scale=2,
                        label=f"Motor LLM (parse/cast/emoção/estrutura) -- "
                              f"{'detectado no Ollama ao vivo' if _motor_ao_vivo else 'lista padrão (Ollama não respondeu ao abrir a UI)'}",
                        allow_custom_value=True,
                        info="Tag do Ollama (`ollama list` mostra o que já está baixado). Aceita "
                             "qualquer tag digitada, mesmo fora da lista -- a escolhida aqui é "
                             "conferida contra a instância no ar quando você clicar Rodar, não "
                             "antes. ⚠️ qwen3.6-35b-a3b:latest (MoE) crasha o backend CUDA do "
                             "Ollama em prompt longo -- o próprio enriquecimento do parse -- "
                             "MEDIDO e documentado no CLAUDE.md/MEMORIAL §3.65; só escolha se "
                             "souber o que está fazendo.")
                    consistencia = gr.Number(
                        value=0.35, label="Auditoria de consistência facial (limiar)", scale=1,
                        info="InsightFace/ArcFace compara cada still novo ao still de "
                             "referência do personagem e tenta de novo (outra seed) se a "
                             "similaridade ficar abaixo disto -- 0 desliga. Só tem efeito "
                             "em planos COM referência (ver MEMORIAL 3.53).")
                with gr.Row():
                    camera_llm = gr.Checkbox(
                        value=False, label="LLM refina câmera/luz por plano", scale=1,
                        info="Dentro do Estilo escolhido (aba Decupagem), o LLM pode trocar "
                             "movimento de câmera e adicionar uma tag de luz plano a plano "
                             "-- só dentro do vocabulário permitido para aquele Estilo (nunca "
                             "livre). Custa 1 chamada de Ollama por cena, na etapa de "
                             "decupagem. Sem isto, tudo determinístico como sempre.")
                    motion_conditioning = gr.Checkbox(
                        value=False, label="Condicionamento de movimento (MotionBricks)", scale=1,
                        info="Estágio [M], antes só existia por CLI (--motion-conditioning). Produz "
                             "o condicionamento textual de trajetória/pose que os motores de vídeo "
                             "aceitam hoje, a partir de sujeito/co-sujeito/lado de tela já resolvidos "
                             "pela decupagem. Não é o diretor de movimento por InterGen/pose real "
                             "(motion_director.py) -- esse continua só por CLI, fora desta corrida.")
                with gr.Row():
                    visual_unresolved = gr.Radio(
                        choices=["block", "continue"], value="block", scale=1,
                        label="Se o gate visual não aprovar um plano após as tentativas",
                        info="block (padrão): a corrida para e mostra shots/visual_gate_*_retries.json. "
                             "continue: segue com o plano reprovado (fica registrado no relatório) -- "
                             "use quando souber que é falso positivo (ex.: close legítimo sem ambiente "
                             "visível) e não quiser gastar mais uma rodada de GPU.")
                    modo_all = gr.Checkbox(
                        value=False, label="🚀 Modo ALL (câmera+movimento+ritmo)", scale=1,
                        info="Liga de uma vez: refino de câmera por LLM, condicionamento de movimento "
                             "e segue mesmo com gate visual pendente (block vira continue) -- os "
                             "avanços que só existiam por CLI. A auditoria de ritmo (planos longos/"
                             "curtos demais) roda sempre, com ou sem o Modo ALL. NÃO liga o estado "
                             "espacial 3D: aquele exige projeto mestre + Motor das imagens = flux "
                             "+ um spec já pronto (aba Motores, acordeão próprio) -- ligar aqui sem "
                             "isso pronto faria a corrida falhar, então fica de fora.")
                with gr.Row():
                    motor_img = gr.Dropdown(
                        choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext", "zimage", "qwen-image-2.1", "hidream", "qwen-image"], value="flux", scale=1,
                        label="Motor das imagens",
                        info="flux: obedece melhor enquadramento e lado de tela, e o unico "
                             "com imagem de referencia por personagem. sd35: carrega em ~1 min "
                             "contra ~4 e cabe em ~12 GB contra ~24, mas erra o enquadramento. "
                             "flux-krea/flux-kontext: FLUX.1 (ver MEMORIAL 3.48), nao testados "
                             "nesta cadeia ate agora. qwen-image-2.1: servidor local separado, "
                             "texto/edicao e ate 10 referencias. Trocar o motor REFAZ os stills (ele entra "
                             "na chave de cache).")
                with gr.Accordion("Continuidade espacial 3D (opcional)", open=False):
                    spatial_on = gr.Checkbox(
                        value=False, label="Ativar estado espacial persistente", scale=1,
                        info="Executa blocking Blender, depth/máscaras e condicionamento RGB antes dos stills. "
                             "Exige projeto mestre vinculado, Motor das imagens = flux e um spec JSON que cubra "
                             "todos os IDs estáveis do shot_plan.")
                    spatial_spec = gr.Textbox(
                        value="", label="Spec JSON de locação/planos espaciais", scale=2,
                        placeholder="C:/.../spatial_spec.json")
                    spatial_denoise = gr.Slider(
                        minimum=.1, maximum=1, value=.65, step=.05,
                        label="Força de transformação do blocking",
                        info="Menor preserva mais o render técnico; maior permite aparência mais cinematográfica.")
                    spatial_diagnostic = gr.Checkbox(
                        value=False, label="Animatic diagnóstico: continuar se o gate visual bloquear",
                        info="Roda o gate e grava o relatório, mas não bloqueia nem regenera, e a "
                             "corrida para no animatic (exige 'Ir até' = animatic ou antes; "
                             "vídeo, lipsync e montagem são recusados). Não aprova stills.")
                    gr.Markdown("Modo opt-in. A aprovação visual continua obrigatória antes do vídeo.")
                    motor_video = gr.Dropdown(
                        choices=["ltx", "minimax", "longcat", "wan"], value="ltx", scale=1,
                        label="Motor de video",
                        info="ltx: LTX 2.5, fala vem do TTS (estagios lipsync/mix rodam normal). "
                             "minimax: MiniMax H3 -- fala e lip-sync NATIVOS a partir do texto "
                             "do plano (o video_prompt precisa ja carregar o dialogo, entre "
                             "aspas, como nos exemplos de teste); lipsync/mix viram passthrough. "
                             "wan: Wan 2.2 TI2V 5B, mesmo ComfyUI do ltx -- video SEMPRE MUDO, "
                             "fala vem do TTS/lipsync normal (como o ltx). Validado com GPU "
                             "real 2026-09-18; sem LoRA/IC-LoRA/encadeamento ainda.")
                    motor_voz = gr.Dropdown(
                        choices=["auto", "xtts", "qwen", "fish"], value="auto", scale=1,
                        label="Motor de voz (TTS)",
                        info="auto: XTTS/Qwen, sem setup extra (padrao). fish: qualidade/emocao "
                             "melhores, mas precisa do servidor do fish-speech JA NO AR "
                             "(fish-speech/START_API.ps1, ~1 min, ~22 GB de VRAM) -- sem ele, cada "
                             "fala falha com mensagem clara. Recente (2026-09-03), nao amadurecido "
                             "ainda -- ver MEMORIAL 3.53/3.55. So afeta o motor ltx (minimax fala "
                             "nativamente, nao usa TTS nenhum).")
                with gr.Row():
                    ltx_variant = gr.Dropdown(
                        choices=["w4a8-v10", "distilled", "dev", "gguf-q6k"], value="w4a8-v10", scale=1,
                        label="Variante LTX 2.5 (só com Motor de vídeo = ltx)",
                        info="w4a8-v10 (padrão desde 2026-09-12): 4 bits, ~1,6x mais rápido que o "
                             "distilled (bf16) com o modelo carregado, qualidade julgada maior em 3 de 3 "
                             "comparações (MEMORIAL 3.77). distilled = bf16, padrão anterior. dev = CFG "
                             "real, mais lento, negative prompt funciona de verdade.")
                    minimax_variant = gr.Dropdown(
                        choices=["fp8int8", "w4a8", "gguf-q4km"], value="fp8int8", scale=1,
                        label="Variante MiniMax H3 (só com Motor de vídeo = minimax)",
                        info="MEDIDO 2026-09-06, mesma cena/seed, turbo 4 passos: fp8int8 "
                             "(padrão) 11min6s -- sempre mais rápido que w4a8 (11min44s) e com "
                             "checkpoints menos comprimidos. gguf-q4km 16min26s, o mais lento "
                             "dos três -- só vale se VRAM for o limite.")
                with gr.Row():
                    ltx_no_still = gr.Checkbox(
                        value=False, scale=1,
                        label="LTX 2.5: sem still (T2V puro, só descritivo textual)",
                        info="Não manda o still do plano como imagem inicial (I2V) -- gera "
                             "puro texto-para-vídeo. ltx25_backend já aceita isso "
                             "(image_path=None); sem esta opção o caminho da decupagem sempre "
                             "ancorava no still. Pedido do usuário 2026-09-16, para comparar "
                             "continuidade puramente textual. Só com Motor de vídeo = ltx.")
                    ltx_chain_max_seconds = gr.Number(
                        value=None, scale=1, precision=1,
                        label="LTX 2.5: dividir plano acima de N segundos (encadeado)",
                        info="Mesmo princípio do continuous_chain.py, aplicado dentro do plano "
                             "da decupagem: sub-planos curtos encadeados por último frame "
                             "decodificado. Vazio = desligado (comportamento de sempre). Só "
                             "com Motor de vídeo = ltx.")
                with gr.Row():
                    minimax_ref_audio = gr.Checkbox(
                        value=False, scale=1,
                        label="MiniMax H3: usar áudio de referência real (voz do TTS)",
                        info="Manda o WAV já sintetizado para cada fala (timbre/cadência reais) "
                             "como ref_audios do MiniMax H3, além das imagens de referência -- "
                             "MEMORIAL 3.74. Só tem efeito com Motor de vídeo = minimax E "
                             "dialogue/lines.json já gerado (estágio 4 TTS). Opt-in: validado só "
                             "com uma fala isolada até agora, não com a cadeia de produção "
                             "inteira -- acompanhe o log do estágio '5-D video' na primeira vez.")
                with gr.Row():
                    minimax_no_still = gr.Checkbox(
                        value=False, scale=1,
                        label="MiniMax H3: sem still (só descritivo textual guia a identidade)",
                        info="Não manda o still do plano como referência de imagem -- só o "
                             "TEXTO (descritor do personagem no prompt) guia a identidade. A "
                             "sheet do personagem, quando existe, continua indo (é ela que "
                             "ancora a identidade ENTRE planos; sem ela o drift entre planos "
                             "tende a piorar). Pedido do usuário 2026-09-16, para comparar "
                             "continuidade puramente textual contra a rota com still. Só tem "
                             "efeito com Motor de vídeo = minimax.")
                    minimax_chain_max_seconds = gr.Number(
                        value=None, scale=1, precision=1,
                        label="MiniMax H3: dividir plano acima de N segundos (encadeado)",
                        info="Planos mais longos que N viram sub-planos curtos encadeados "
                             "(último frame decodificado + sheet do personagem alimentam o "
                             "sub-plano seguinte) em vez de uma chamada só -- MEDIDO em "
                             "_test_minimax_duration_cap.py que planos de ~16s+ ficam "
                             "instáveis (30min+ sem terminar) numa chamada única. 6.0 é o "
                             "ponto de partida conservador já validado. Vazio = desligado "
                             "(comportamento de sempre, cada plano é uma chamada só). A fala "
                             "inteira do plano vai para CADA sub-plano -- revise a sincronia "
                             "visualmente antes de confiar em produção contínua.")
                with gr.Row():
                    largura = gr.Number(value=960, label="Largura", precision=0)
                    altura = gr.Number(value=544, label="Altura", precision=0)
                with gr.Row():
                    character_sheet_on = gr.Checkbox(
                        value=False, scale=1, label="Forçar character sheet (Fase A, medoid) com 1 só personagem",
                        info="Já liga sozinho com 2+ personagens no cast (evita identidade roubada "
                             "entre coadjuvantes). Gera N retratos candidatos por personagem ANTES "
                             "dos stills e escolhe o MEDOID -- maior similaridade facial média aos "
                             "outros candidatos -- como reference_image. Marque aqui só para forçar "
                             "com 1 personagem só. Revise/troque a escolha na aba Decupagem > "
                             "Character sheet > \"Revisar candidatos da última corrida\".")
                    character_sheet_n = gr.Number(
                        value=4, precision=0, scale=1, label="Candidatos por personagem")
                with gr.Row():
                    from script_pipeline.generate_storyboards import available_loras_images
                    lora = gr.Dropdown(
                        choices=["(nenhum)"] + available_loras_images(), value="(nenhum)", scale=2,
                        label="LoRA dos stills (opcional)",
                        info="Aplicado ao motor de IMAGEM (flux/sd35/sdxl/flux-krea/flux-kontext), "
                             "nunca ao vídeo. Arquivos em models/loras_images/ -- pasta PRÓPRIA, "
                             "separada de models/loras/ (LoRAs de vídeo do LTX, incompatíveis "
                             "aqui). Vazio até alguém colocar um LoRA treinado para um desses "
                             "checkpoints; \"iniciar a opção\", não um LoRA curado.")
                    lora_strength = gr.Slider(
                        minimum=0, maximum=2, value=0.8, step=0.05, scale=1,
                        label="Força do LoRA")
                # LoRAs de VIDEO do LTX 2.5, lip-sync e pos-producao (pedidos do usuario
                # 2026-09-12/13): cada LoRA com liga/desliga e forca exposta, padrao =
                # a do catalogo (ltx_loras.py). Nada liga sozinho.
                import ltx_loras
                with gr.Accordion("LoRAs do LTX 2.5: geração, lip-sync e pós-produção", open=False):
                    gr.Markdown(
                        "**LoRAs comuns na geração** (só Motor de vídeo = ltx). Treinados no 2.3, "
                        "carregam no 2.5 (verificado); efeito a validar vendo. ATENÇÃO: "
                        "motion-enhancer-n4w é afinado para NSFW; talking-head-av é de UM "
                        "personagem do autor; cdrama-char puxa rostos dos atores da série. No "
                        "w4a8 o LoRA é requantizado — se não fizer diferença, teste no gguf-q6k.")
                    lora_chaves, lora_liga, lora_forca = [], [], []
                    for _spec in ltx_loras.video_loras_available():
                        with gr.Row():
                            lora_chaves.append(_spec.key)
                            lora_liga.append(gr.Checkbox(value=False, scale=1,
                                                         label=f"{_spec.key}" + (f" (gatilho {_spec.trigger})" if _spec.trigger else "")))
                            lora_forca.append(gr.Slider(minimum=0.0, maximum=1.5, value=_spec.strength,
                                                        step=0.05, scale=2, label=f"força ({_spec.strength:g} padrão)"))
                    with gr.Row():
                        ic_reference = gr.Dropdown(
                            choices=["off", "ingredients", "msr"], value="off", scale=1,
                            label="IC-LoRA de referência de personagem",
                            info="msr = sujeitos + cenário (melhor no teste de 2026-09-12); ingredients = "
                                 "folha da character sheet. O still continua sendo o 1º quadro.")
                        ic_strength = gr.Slider(minimum=0.0, maximum=1.5, value=1.0, step=0.05, scale=1,
                                                label="força do IC-LoRA (1.0 padrão)")
                        ic_guide_strength = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, scale=1,
                                                      label="força da guia (1.0 padrão)")
                    with gr.Row():
                        lipsync_engine = gr.Dropdown(
                            choices=["auto", "latentsync", "wav2lip", "dubit", "none"], value="auto", scale=1,
                            label="Lip-sync",
                            info="auto = LatentSync/Wav2Lip (como sempre); dubit = IC-LoRA DubIt refaz a "
                                 "boca no LTX; none = sem lip-sync, fica a boca que o LTX gerou.")
                        dubit_audio = gr.Dropdown(choices=["congelar", "gerar"], value="congelar", scale=1,
                                                  label="DubIt: áudio",
                                                  info="congelar = voz do TTS; gerar = o modelo gera a fala do texto.")
                        dubit_strength = gr.Slider(minimum=0.0, maximum=1.5, value=1.0, step=0.05, scale=1,
                                                   label="DubIt: força (1.0 padrão)")
                        dubit_guide = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, scale=1,
                                                label="DubIt: força da guia (1.0 padrão)")
                    with gr.Row():
                        post_deblur = gr.Checkbox(value=False, scale=1, label="Pós: Deblur 2.5")
                        post_deblur_s = gr.Slider(minimum=0.0, maximum=1.5, value=1.0, step=0.05, scale=1,
                                                  label="Deblur: força (1.0 padrão)")
                        post_upscale = gr.Checkbox(value=False, scale=1, label="Pós: Upscaler x2 2.5 (lento)")
                        post_upscale_s = gr.Slider(minimum=0.0, maximum=1.5, value=1.0, step=0.05, scale=1,
                                                   label="Upscaler: força (1.0 padrão)")

            # ---------------------------------------------------------------
            # ABA 3: STILLS -- rodar/parar, rascunho, previas, regenerar
            # ---------------------------------------------------------------
            with gr.Tab("Stills"):
                log = gr.Textbox(label="Log", lines=22, interactive=False, autoscroll=True,
                                 max_lines=22)
                video = gr.Video(value=_video0, label="Rascunho / filme", height=320)

                galeria = gr.Gallery(value=_stills0, columns=4, height=420, show_label=True,
                                     label="Stills — clique num para ver o prompt e preencher "
                                           "a tomada abaixo")

                prompt_sel = gr.Textbox(label="Prompts da tomada selecionada", lines=9,
                                        interactive=False,
                                        placeholder="Clique numa imagem acima.")

                with gr.Accordion("Regenerar UMA tomada agora, com Estilo/Aparência", open=True):
                    gr.Markdown(
                        "Gera de novo NA HORA (não precisa rodar o estágio inteiro) e a "
                        "galeria acima é atualizada sozinha ao terminar. Estilo/Aparência "
                        "aqui são só para **experimentar** esta tomada — não ficam "
                        "gravados no shot_plan, então uma corrida completa depois volta "
                        "ao prompt original (é o comportamento certo, não um resíduo)."
                    )
                    with gr.Row():
                        regen_indice = gr.Number(label="Tomada # (clique numa imagem acima, ou digite)",
                                                 precision=0, scale=1)
                        regen_motor_img = gr.Dropdown(
                            choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext", "zimage", "qwen-image-2.1", "hidream", "qwen-image"],
                            value="flux", label="Motor da imagem", scale=1)
                    with gr.Row():
                        regen_estilo = gr.Dropdown(
                            choices=list(STYLE_PRESETS), value=["(nenhum)"], multiselect=True,
                            label="Estilo (combináveis)", scale=2)
                        regen_etnia = gr.Dropdown(
                            choices=list(ETHNICITY_PRESETS), value="(nenhuma)",
                            label="Aparência / etnia do personagem", scale=1)
                    regen_btn = gr.Button("Regenerar esta tomada", variant="primary")
                    regen_status = gr.Textbox(label="", interactive=False)
                    regen_preview = gr.Image(label="Resultado", interactive=False, type="filepath")

                with gr.Accordion("Refazer/Regenerar em lote", open=False):
                    gr.Markdown(
                        "**Marcar para refazer**: apaga os stills escolhidos (e a entrada deles "
                        "no manifesto) para a PRÓXIMA execução (*Ir até* = `stills`/`animatic`) "
                        "gerar de novo — não regenera sozinho.\n\n"
                        "**Regenerar seleção agora**: faz as duas coisas — apaga e já gera de "
                        "novo, na hora, com o motor de imagem escolhido na aba Motores — sem "
                        "precisar rodar o estágio inteiro de novo.\n\n"
                        "Use quando editar o `cast.json`, trocar o estilo de um trecho ou o "
                        "motor de imagem — nesses casos o prompt muda e o still velho fica "
                        "vencido."
                    )
                    with gr.Row():
                        quais = gr.Textbox(label="Quais tomadas", scale=3,
                                           placeholder="todos   |   0,3,5-7")
                        btn_refazer = gr.Button("Marcar para refazer", scale=1)
                        btn_regenerar_selecao = gr.Button("Regenerar seleção agora", scale=1,
                                                          variant="primary")
                    gr.Markdown(
                        "**🧹 Limpar cache**: além dos stills da seleção, invalida o que foi "
                        "feito A PARTIR deles — chave dos clipes, lip-sync, animatic e auditoria "
                        "de storyboard — e pode liberar os modelos/cache do ComfyUI. Sem isso, um "
                        "still novo pode seguir com o clipe velho. Clipes `.mp4` não são "
                        "apagados: sem a chave, a próxima passada de vídeo refaz."
                    )
                    with gr.Row():
                        limpar_derivados = gr.Checkbox(value=True, scale=2,
                                                       label="Invalidar clipes, lip-sync e animatic da seleção")
                        limpar_comfy = gr.Checkbox(value=False, scale=2,
                                                   label="Liberar cache do ComfyUI (8188) — não use com geração em andamento")
                        btn_limpar_cache = gr.Button("🧹 Limpar cache", scale=1, variant="stop")
                    aviso_refazer = gr.Textbox(label="", interactive=False, lines=6)

                # Pos-producao, FORA do fluxo de geracao de proposito: e ferramenta
                # aplicada ao video ja pronto, e so quando a pessoa quiser.
                video_doctor_ui.build_doctor_tab(
                    get_default_video=lambda: video_de(RUNS_DIR / (_inicial or "")) or "",
                    label="🩺 Diagnóstico e correção (video doctor)",
                    container="accordion")

        # ---- ligacoes ----
        # Atualizar SO rele a pasta em disco (nao carrega nada na tela) --
        # Carregar/selecionar a corrida E' o que traz os dados pra tela.
        project_create.click(fn=criar_projeto_mestre,
                             inputs=[project_title, project_root, project_source, project_duration],
                             outputs=[runs, roteiro_path, project_status])
        project_refresh.click(fn=resumo_projeto_mestre, inputs=runs, outputs=project_status)
        project_load.click(fn=documento_projeto, inputs=[runs, project_doc], outputs=project_json)
        project_save.click(fn=documento_projeto, inputs=[runs, project_doc, project_json], outputs=project_json)
        take_import.click(fn=importar_tomada, inputs=[runs, take_shot, take_path, take_approval, take_in], outputs=project_status)
        delivery_button.click(fn=exportar_projeto, inputs=[runs, delivery_approved], outputs=delivery_path)
        atualizar.click(fn=lambda: (gr.update(choices=listar_runs()),
                                    "Lista de corridas atualizada -- selecione uma e clique "
                                    "\"Carregar corrida selecionada\"."),
                        outputs=[runs, aviso_corrida])
        runs.change(fn=carregar_run_completo, inputs=[runs, roteiro_path],
                    outputs=[auto, cast, plano, enriquecimento, galeria, video,
                             aviso_corrida, roteiro_path, aviso_roteiro, estagio_html, cast_gaps_aviso])
        carregar.click(fn=carregar_run_completo, inputs=[runs, roteiro_path],
                       outputs=[auto, cast, plano, enriquecimento, galeria, video,
                                aviso_corrida, roteiro_path, aviso_roteiro, estagio_html, cast_gaps_aviso])
        nova_btn.click(fn=nova_corrida,
                       outputs=[runs, novo_nome, arquivo_roteiro, roteiro_colado, roteiro_path,
                                aviso_roteiro, aviso_corrida, auto, cast, plano, enriquecimento,
                                galeria, video, log, estagio_html, cast_gaps_aviso])
        salvar_nome_btn.click(fn=criar_corrida, inputs=novo_nome, outputs=[runs, aviso_corrida])
        apagar_btn.click(fn=apagar_corrida, inputs=[runs, apagar_confirma],
                         outputs=[aviso_corrida, runs])
        arquivo_roteiro.change(fn=usar_arquivo_roteiro, inputs=[arquivo_roteiro, runs],
                               outputs=[roteiro_path, aviso_roteiro])
        usar_texto_btn.click(fn=usar_texto_colado, inputs=[roteiro_colado, runs],
                             outputs=[roteiro_path, aviso_roteiro, arquivo_roteiro])
        _entradas_base = [runs, roteiro_path, novo_nome, estilo, trocas, largura, altura,
                          ate, motor, recast, motor_img, motor_video, motor_voz,
                          consistencia, camera_llm, ltx_variant, minimax_variant,
                          lora, lora_strength, character_sheet_on, character_sheet_n,
                          minimax_ref_audio, minimax_no_still, minimax_chain_max_seconds,
                          ltx_no_still, ltx_chain_max_seconds,
                          spatial_on, spatial_spec, spatial_denoise, spatial_diagnostic,
                          motion_conditioning, visual_unresolved, modo_all]
        _entradas_lora = [ic_reference, ic_strength, ic_guide_strength, lipsync_engine,
                          dubit_strength, dubit_guide, dubit_audio,
                          post_deblur, post_deblur_s, post_upscale, post_upscale_s]

        def _rodar_ui(*valores):
            """Traduz os controles de LoRA/lip-sync/pos-producao em argumentos da cadeia.
            So o que foi LIGADO vira argumento: tudo desligado = corrida de sempre."""
            nb, nl = len(_entradas_base), len(_entradas_lora)
            base = list(valores[:nb])
            modo_all_ligado = base.pop()   # ultimo de _entradas_base; nao e argumento de rodar()
            if modo_all_ligado:
                # indices dentro de base, na MESMA ordem de _entradas_base (sem o modo_all
                # que acabou de sair): camera_llm=14, motion_conditioning=-2, visual_unresolved=-1
                base[14] = True
                base[-2] = True
                base[-1] = "continue"
            (ic_ref, ic_s, ic_g, lip, dub_s, dub_g, dub_a,
             deb, deb_s, up, up_s) = valores[nb:nb + nl]
            ligados = valores[nb + nl:nb + nl + len(lora_chaves)]
            forcas = valores[nb + nl + len(lora_chaves):]
            escolhidos = [f"{k} ({float(f):g})" for k, on, f in zip(lora_chaves, ligados, forcas) if on]
            extras = []
            if ic_ref != "off":
                extras += ["--ic-guide-strength", str(ic_g)]
            if lip != "auto":
                extras += ["--lipsync-engine", lip]
            if lip == "dubit":
                extras += ["--dubit-strength", str(dub_s), "--dubit-guide-strength", str(dub_g),
                           "--dubit-audio", dub_a]
            if deb:
                extras += ["--post-deblur", "--post-deblur-strength", str(deb_s)]
            if up:
                extras += ["--post-upscale", "--post-upscale-strength", str(up_s)]
            yield from rodar(*base, video_loras=escolhidos, ic_reference=ic_ref,
                             ic_strength=ic_s, extras=extras)

        evento_rodar = btn.click(
            fn=_rodar_ui,
            inputs=_entradas_base + _entradas_lora + lora_liga + lora_forca,
            outputs=[log, galeria, video, plano, estagio_html])
        # Clicar numa imagem preenche o texto do prompt E o numero da tomada no
        # regenerador -- sem isso a pessoa teria que contar posicao na galeria
        # a mao pra saber que numero digitar.
        galeria.select(fn=prompts_do_still, inputs=runs, outputs=prompt_sel)
        galeria.select(fn=indice_do_still_selecionado, inputs=runs, outputs=regen_indice)
        btn_refazer.click(fn=apagar_stills, inputs=[runs, quais], outputs=aviso_refazer)
        btn_regenerar_selecao.click(fn=regenerar_selecao,
                                    inputs=[runs, quais, motor_img, largura, altura,
                                            lora, lora_strength],
                                    outputs=[aviso_refazer, galeria])
        btn_limpar_cache.click(fn=limpar_cache_stills,
                               inputs=[runs, quais, limpar_derivados, limpar_comfy],
                               outputs=aviso_refazer
                               ).then(fn=lambda n: stills_com_legenda(RUNS_DIR / n) if n else [],
                                      inputs=runs, outputs=galeria)
        # queue=False faz o clique chegar enquanto o gerador da corrida ocupa
        # a fila; cancels interrompe o streaming da funcao e `parar` mata a
        # arvore de subprocessos no Windows.
        btn_parar.click(fn=parar, outputs=aviso, queue=False,
                        cancels=[evento_rodar])
        salvar.click(fn=salvar_cast, inputs=[runs, cast], outputs=[aviso, cast_gaps_aviso])
        sheet_btn.click(fn=gerar_sheet_personagem,
                        inputs=[runs, sheet_personagem, sheet_descritor, sheet_motor,
                                lora, lora_strength],
                        outputs=[sheet_img, sheet_status])
        sheet_usar_btn.click(fn=usar_sheet_como_referencia,
                             inputs=[runs, sheet_personagem, sheet_img, cast],
                             outputs=[cast, sheet_status])
        sheet_report_btn.click(fn=carregar_sheet_report, inputs=runs,
                               outputs=[sheet_report_gallery, sheet_report_status, sheet_report_itens])
        sheet_report_gallery.select(fn=selecionar_candidato_sheet, inputs=sheet_report_itens,
                                    outputs=[sheet_personagem, sheet_img])
        regen_btn.click(fn=regenerar_still,
                        inputs=[runs, regen_indice, regen_estilo, regen_etnia, regen_motor_img,
                                lora, lora_strength],
                        outputs=[galeria, regen_status, regen_preview])

    # show_api foi removido no Gradio 6 (temos 6.20) -- passar levanta TypeError.
    # As UIs antigas deste repo ainda passam; ver video_doctor_ui.standalone().
    # allowed_paths: sem isto o Gradio recusa servir os PNG/MP4 de outputs/ e a
    # galeria fica vazia sem dizer por que.
    _free_port(PORT)
    demo.launch(server_name="127.0.0.1", server_port=PORT, inbrowser=True,
                allowed_paths=[str(RUNS_DIR)])


if __name__ == "__main__":
    build()
