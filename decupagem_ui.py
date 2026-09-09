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


# --------------------------------------------------------------------------
# roteiro colado direto na UI, sem precisar de um .txt em disco antes
# --------------------------------------------------------------------------
ROTEIROS_DIR = RUNS_DIR / "_roteiros"


def salvar_roteiro_colado(texto: str, novo_nome: str) -> tuple:
    """Grava o texto colado como .txt em `_roteiros/` e devolve o CAMINHO --
    preenche o campo "Roteiro .txt" existente em vez de abrir um segundo
    caminho de execucao. `rodar()` continua so aceitando arquivo (e e o que
    `run_decupagem --script` espera); isto so poupa a pessoa de salvar o
    arquivo a mao antes de colar o caminho aqui.

    `_roteiros/` (nome com underscore) ja e ignorado por `listar_runs()` --
    mesma convencao usada nesta sessao pro roteiro Beatriz/Lucas."""
    texto = (texto or "").strip()
    if not texto:
        return gr.update(), "Cole o texto do roteiro antes de salvar."
    ROTEIROS_DIR.mkdir(parents=True, exist_ok=True)
    base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (novo_nome or "").strip())
    if not base:
        base = time.strftime("roteiro_%Y%m%d_%H%M%S")
    caminho = ROTEIROS_DIR / f"{base}.txt"
    n = 2
    while caminho.exists():
        caminho = ROTEIROS_DIR / f"{base}_{n}.txt"
        n += 1
    caminho.write_text(texto, encoding="utf-8")
    return str(caminho), f"Roteiro salvo em {caminho} -- preenchido no campo acima. Clique Rodar."


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


# --------------------------------------------------------------------------
# execucao
# --------------------------------------------------------------------------
def _argv(run: Path, script: str, estilo: str, trocas: str, largura: int,
          altura: int, ate: str, motor: str, recast: bool,
          motor_img: str = "flux", motor_video: str = "ltx",
          motor_voz: str = "auto", consistencia: float | None = None,
          camera_llm: bool = False, ltx_variant: str = "distilled",
          minimax_variant: str = "fp8int8", lora: str = "(nenhum)",
          lora_strength: float = 0.8, character_sheet_on: bool = False,
          character_sheet_n: int = 4) -> list:
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
    # LoRA opcional nos STILLS (nunca no video) -- pedido do usuario
    # 2026-09-09: models/loras_images/, separado dos LoRAs de video do LTX em
    # models/loras/, que sao incompativeis com estes checkpoints de imagem.
    if lora and lora != "(nenhum)":
        cmd += ["--lora", lora, "--lora-strength", str(lora_strength)]
    if character_sheet_on:
        cmd += ["--character-sheet", "--character-sheet-candidates", str(int(character_sheet_n))]
    return cmd


def rodar(nome_run, script, novo_nome, estilo, trocas, largura, altura, ate, motor,
          recast, motor_img="flux", motor_video="ltx", motor_voz="auto",
          consistencia=None, camera_llm=False, ltx_variant="distilled",
          minimax_variant="fp8int8", lora="(nenhum)", lora_strength=0.8,
          character_sheet_on=False, character_sheet_n=4):
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
                motor_img, motor_video, motor_voz, consistencia, camera_llm,
                ltx_variant, minimax_variant, lora, lora_strength,
                character_sheet_on, character_sheet_n)
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
    _auto0, _cast0, _plano0, _enriq0, _stills0, _video0 = carregar_run(_inicial)

    with gr.Blocks(title="Decupagem — LTX") as demo:
        gr.Markdown(
            "# Decupagem\n"
            "Roteiro → cast → voz → decupagem → stills → animatic → filme. "
            "Mesma cadeia do `start_decupagem.bat`, com o log à vista e retomada."
        )

        # Selecao de corrida: FORA das abas, de proposito -- e o unico estado
        # que as tres abas (Decupagem/Motores/Stills) compartilham.
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

        with gr.Tabs():
            # ---------------------------------------------------------------
            # ABA 1: DECUPAGEM -- roteiro, cast, e o resultado do parse/LLM
            # ---------------------------------------------------------------
            with gr.Tab("Decupagem"):
                with gr.Row():
                    script = gr.Textbox(label="Roteiro .txt (só para corrida NOVA)", scale=3,
                                        placeholder=r"C:\Users\user\Desktop\roteiro.txt")
                    novo_nome = gr.Textbox(label="Nome da corrida nova (opcional)", scale=1)

                with gr.Accordion("Ou cole o texto do roteiro aqui (em vez de apontar um arquivo)", open=False):
                    roteiro_colado = gr.Textbox(
                        label="Texto do roteiro / prompt audiovisual", lines=10,
                        placeholder="Cole aqui o roteiro em prosa, formato de cena, ou prompt "
                                    "audiovisual -- prose_to_screenplay.py cuida da conversão "
                                    "na primeira etapa, igual a um arquivo carregado.")
                    salvar_roteiro_btn = gr.Button("Salvar e usar como roteiro desta corrida")
                    aviso_roteiro = gr.Textbox(label="", interactive=False)

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
                            choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext"],
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
                    motor = gr.Dropdown(
                        choices=["qwen3.6-35b-a3b:latest", "gemma4", "qwen2.5:32b-instruct-q4_K_M",
                                 "mistral-nemo:12b-instruct-2407-q4_K_M"],
                        value="qwen3.6-35b-a3b:latest", label="Motor LLM (parse/cast/emoção/estrutura)", scale=2,
                        allow_custom_value=True,
                        info="Tag do Ollama (`ollama list` mostra o que já está baixado). "
                             "Aceita qualquer tag digitada, mesmo fora da lista.")
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
                with gr.Row():
                    motor_img = gr.Dropdown(
                        choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext"], value="flux", scale=1,
                        label="Motor das imagens",
                        info="flux: obedece melhor enquadramento e lado de tela, e o unico "
                             "com imagem de referencia por personagem. sd35: carrega em ~1 min "
                             "contra ~4 e cabe em ~12 GB contra ~24, mas erra o enquadramento. "
                             "flux-krea/flux-kontext: FLUX.1 (ver MEMORIAL 3.48), nao testados "
                             "nesta cadeia ate agora. Trocar o motor REFAZ os stills (ele entra "
                             "na chave de cache).")
                    motor_video = gr.Dropdown(
                        choices=["ltx", "minimax"], value="ltx", scale=1,
                        label="Motor de video",
                        info="ltx: LTX 2.5, fala vem do TTS (estagios lipsync/mix rodam normal). "
                             "minimax: MiniMax H3 -- fala e lip-sync NATIVOS a partir do texto "
                             "do plano (o video_prompt precisa ja carregar o dialogo, entre "
                             "aspas, como nos exemplos de teste); lipsync/mix viram passthrough.")
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
                        choices=["distilled", "dev", "gguf-q6k"], value="distilled", scale=1,
                        label="Variante LTX 2.5 (só com Motor de vídeo = ltx)",
                        info="MEDIDO 2026-09-06, mesma cena/seed: distilled (padrão) 13min17s -- "
                             "gguf-q6k 4min25s, ~3x mais rápido. dev = CFG real, mais lento, "
                             "negative prompt funciona de verdade (não comparado neste teste).")
                    minimax_variant = gr.Dropdown(
                        choices=["fp8int8", "w4a8", "gguf-q4km"], value="fp8int8", scale=1,
                        label="Variante MiniMax H3 (só com Motor de vídeo = minimax)",
                        info="MEDIDO 2026-09-06, mesma cena/seed, turbo 4 passos: fp8int8 "
                             "(padrão) 11min6s -- sempre mais rápido que w4a8 (11min44s) e com "
                             "checkpoints menos comprimidos. gguf-q4km 16min26s, o mais lento "
                             "dos três -- só vale se VRAM for o limite.")
                with gr.Row():
                    largura = gr.Number(value=960, label="Largura", precision=0)
                    altura = gr.Number(value=544, label="Altura", precision=0)
                with gr.Row():
                    character_sheet_on = gr.Checkbox(
                        value=False, scale=1, label="Gerar character sheet automática (Fase A, medoid)",
                        info="Gera N retratos candidatos por personagem ANTES dos stills e "
                             "escolhe o MEDOID -- maior similaridade facial média aos outros "
                             "candidatos -- como reference_image, em vez do primeiro still que "
                             "acontecer de sair. Revise/troque a escolha na aba Decupagem > "
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

            # ---------------------------------------------------------------
            # ABA 3: STILLS -- rodar/parar, rascunho, previas, regenerar
            # ---------------------------------------------------------------
            with gr.Tab("Stills"):
                with gr.Row():
                    btn = gr.Button("Rodar", variant="primary", scale=2)
                    btn_parar = gr.Button("Parar", scale=1)

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
                            choices=["flux", "sd35", "sdxl", "flux-krea", "flux-kontext"],
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
                    aviso_refazer = gr.Textbox(label="", interactive=False, lines=6)

                # Pos-producao, FORA do fluxo de geracao de proposito: e ferramenta
                # aplicada ao video ja pronto, e so quando a pessoa quiser.
                video_doctor_ui.build_doctor_tab(
                    get_default_video=lambda: video_de(RUNS_DIR / (_inicial or "")) or "",
                    label="🩺 Diagnóstico e correção (video doctor)",
                    container="accordion")

        # ---- ligacoes ----
        atualizar.click(fn=lambda: gr.update(choices=listar_runs()), outputs=runs)
        runs.change(fn=carregar_run, inputs=runs,
                    outputs=[auto, cast, plano, enriquecimento, galeria, video])
        carregar.click(fn=carregar_run, inputs=runs,
                       outputs=[auto, cast, plano, enriquecimento, galeria, video])
        btn.click(fn=rodar,
                  inputs=[runs, script, novo_nome, estilo, trocas, largura, altura,
                          ate, motor, recast, motor_img, motor_video, motor_voz,
                          consistencia, camera_llm, ltx_variant, minimax_variant,
                          lora, lora_strength, character_sheet_on, character_sheet_n],
                  outputs=[log, galeria, video, plano])
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
        btn_parar.click(fn=parar, outputs=aviso)
        salvar.click(fn=salvar_cast, inputs=[runs, cast], outputs=aviso)
        salvar_roteiro_btn.click(fn=salvar_roteiro_colado, inputs=[roteiro_colado, novo_nome],
                                 outputs=[script, aviso_roteiro])
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
