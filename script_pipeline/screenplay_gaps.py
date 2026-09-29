"""Auditoria de LACUNAS do roteiro, antes da decupagem: o que o texto nao diz e o
video vai precisar inventar.

POR QUE EXISTE

MEDIDO 2026-09-27 (CERCO EM SEUL, MEMORIAL 3.132): o filme tinha cara de thriller,
mas nao dava para ler quem protege quem, quem foge e como a ameaca evolui. A maior
parte dos erros nascia ANTES do video: acao sem agente cadastrado ("the President"),
acao fisica sem alvo, fala cujo beat repete a acao vizinha (o evento recomeca), fala
colada numa acao fisica que um close nao mostra. O motor desenha o que o texto
deixa em aberto -- e desenha diferente a cada plano.

O QUE FAZ

1. Checagens DETERMINISTICAS (sempre, sem modelo): cenario, aparencia do elenco,
   voz adivinhada, agente e alvo de cada acao, pessoa agindo sem cadastro, fala que
   repete a acao vizinha, fala sobre acao fisica, e a lista de MUDANCAS DE ESTADO
   (explode, cai, fecha, algema...) que a continuidade precisa carregar depois.
2. Com `--engine` (Ollama), o LLM propoe por acao: agente, alvo, objetos, posicao,
   estado inicial e final, e o que falta -- sempre marcado como SUGESTAO INFERIDA.
   O codigo valida nomes contra o elenco. Nada disso reescreve o roteiro.

Saida: `parse/lacunas.json` e `parse/lacunas.md` (o que acrescentar ao roteiro, por
plano). Nunca bloqueia por padrao; `--bloquear` devolve 2 se houver lacuna critica.

CLI:
  python -m script_pipeline.screenplay_gaps --run-dir DIR [--engine qwen3.6-35b-a3b:latest] [--bloquear]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline.cast_characters import _PAPEL_RE, personagens_citados  # noqa: E402
from script_pipeline.motion_conditioner import infer_primitive  # noqa: E402

# Primitivas que exigem um SEGUNDO participante (alvo) para serem filmaveis.
PRIMITIVAS_COM_ALVO = {"takedown", "restrain", "escort", "protect", "contact"}
# Primitivas de acao fisica: nao cabem num close de fala.
PRIMITIVAS_FISICAS = PRIMITIVAS_COM_ALVO | {"pursue", "fire", "locomote", "reach"}

_PESSOA_SEM_NOME = re.compile(
    r"\b(he|she|they|him|her|someone|somebody|a man|a woman|the man|the woman|a person|"
    r"ele|ela|eles|elas|algu[eé]m|um homem|uma mulher|o homem|a mulher)\b", re.IGNORECASE)
_ALVO_DE_TIRO = re.compile(r"\b(at|toward|towards|into|contra|em dire[cç][aã]o|no|na|nos|nas)\b",
                           re.IGNORECASE)
# Verbo que MUDA o estado do mundo -- precisa persistir nos planos seguintes.
_MUDANCA_DE_ESTADO = re.compile(
    r"\b(explod\w*|shatter\w*|collaps\w*|falls?|falling|crash\w*|catches fire|burn\w*|"
    r"clos(?:e|es|ing)|open(?:s|ing)?|lock\w*|breaks?|broken|removes?|takes? off|tears? off|"
    r"drops?|handcuff\w*|dies|killed|injur\w*|wounded|"
    r"explode|estilha[cç]\w*|desab\w*|despenc\w*|cai|caem|pega fogo|fecha\w*|abre\w*|"
    r"quebra\w*|arranca\w*|solta\w*|algema\w*|morre\w*|ferid\w*)\b", re.IGNORECASE)
_PALAVRA = re.compile(r"[a-zà-ú0-9]+", re.IGNORECASE)
_VAZIAS = {"the", "a", "an", "and", "of", "to", "in", "on", "at", "his", "her", "their", "with",
           "as", "o", "a", "os", "as", "e", "de", "do", "da", "em", "no", "na", "um", "uma"}


def _tokens(texto: str) -> set[str]:
    return {t.lower() for t in _PALAVRA.findall(texto or "") if t.lower() not in _VAZIAS}


def _semelhanca(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _lacuna(tipo: str, gravidade: str, cena: int, item: int | None, texto: str,
            o_que_falta: str) -> dict:
    return {"tipo": tipo, "gravidade": gravidade, "cena": cena, "item": item,
            "texto": texto, "o_que_falta": o_que_falta}


def audit_deterministic(scenes: list[dict], cast: dict) -> tuple[list[dict], list[dict]]:
    """(lacunas, mudancas_de_estado). `item` e o indice no shot_list da cena."""
    lacunas: list[dict] = []
    estados: list[dict] = []
    papeis_cadastrados = {a.lower() for info in cast.values() for a in (info.get("aliases") or [])}
    papeis_cadastrados |= {n.lower() for n in cast}

    for nome, info in cast.items():
        if info.get("on_screen") is False:
            continue
        desc = info.get("descriptor") or ""
        if not desc or "please edit" in desc or info.get("descriptor_gaps"):
            lacunas.append(_lacuna("aparencia", "atencao", 0, None, nome,
                                   f"descrever {nome}: cabelo, idade/porte, roupa com cores e um item "
                                   f"distintivo (faltando: {', '.join(info.get('descriptor_gaps') or ['descricao'])})"))
        voz = info.get("voice") or {}
        if voz.get("gender_guessed") and info.get("line_count"):
            lacunas.append(_lacuna("voz", "atencao", 0, None, nome,
                                   f"indicar genero/idade de {nome} (a voz foi escolhida por palpite)"))

    for sc in scenes:
        cena = sc.get("index", 0)
        if not (sc.get("location") or "").strip():
            lacunas.append(_lacuna("cenario", "atencao", cena, None, sc.get("heading_raw", ""),
                                   "cabecalho com o LOCAL (INT./EXT. + lugar nomeavel)"))
        if not (sc.get("time_of_day") or "").strip():
            lacunas.append(_lacuna("cenario", "info", cena, None, sc.get("heading_raw", ""),
                                   "periodo do dia/clima (DIA, NOITE, chuva...)"))
        dialogo = sc.get("dialogue") or []
        itens = sc.get("shot_list") or []
        vizinhos_de_acao = {}
        for i, item in enumerate(itens):
            if item.get("type") == "action":
                vizinhos_de_acao[i] = str(item.get("visual") or "")

        for i, item in enumerate(itens):
            if item.get("type") == "action":
                texto = str(item.get("visual") or "")
                citados = personagens_citados(texto, cast)
                ator = item.get("actor") or (citados[0] if citados else "")
                if not ator and _PESSOA_SEM_NOME.search(texto):
                    lacunas.append(_lacuna("agente", "critica", cena, i, texto,
                                           "QUEM executa a acao (nome do personagem ou papel cadastrado)"))
                for m in _PAPEL_RE.finditer(texto):
                    papel = m.group(1).lower()
                    if papel not in papeis_cadastrados:
                        lacunas.append(_lacuna("pessoa_sem_cadastro", "critica", cena, i, texto,
                                               f"descrever \"{m.group(0)}\" (idade, roupa, traco "
                                               f"distintivo) -- age em cena e nao esta no elenco"))
                primitiva, _ = infer_primitive(texto, ator or "X", "", framing="medium", speaking=False)
                outros = [c for c in citados if c != ator]
                if primitiva in PRIMITIVAS_COM_ALVO and not outros:
                    lacunas.append(_lacuna("alvo", "critica", cena, i, texto,
                                           "SOBRE QUEM a acao acontece (alvo com nome/papel) e onde "
                                           "cada um esta"))
                if primitiva == "fire" and not _ALVO_DE_TIRO.search(texto):
                    lacunas.append(_lacuna("alvo", "critica", cena, i, texto,
                                           "CONTRA O QUE/QUEM o tiro e dado (janela, pessoa, direcao)"))
                if _MUDANCA_DE_ESTADO.search(texto):
                    estados.append({"cena": cena, "item": i, "texto": texto,
                                    "manter_depois": "o novo estado (destruido, fechado, caido, "
                                                     "algemado...) precisa aparecer nos planos seguintes"})
                continue

            li = item.get("line_index")
            if li is None or not (0 <= li < len(dialogo)):
                continue
            linha = dialogo[li]
            beat = str(linha.get("beat_visual") or "")
            falante = linha.get("character") or ""
            if not beat:
                lacunas.append(_lacuna("fala_sem_visual", "info", cena, i, linha.get("text", ""),
                                       f"o que {falante} FAZ enquanto fala (gesto, olhar, posicao)"))
                continue
            for j in (i - 1, i + 1):
                if j in vizinhos_de_acao and _semelhanca(beat, vizinhos_de_acao[j]) >= 0.45:
                    lacunas.append(_lacuna("repeticao", "atencao", cena, i, beat,
                                           f"a fala repete a acao do item {j} -- o evento recomeca na "
                                           f"tela. Escrever a REACAO ou a CONSEQUENCIA, nao a mesma acao"))
                    break
            primitiva, _ = infer_primitive(beat, falante, "", framing="medium", speaking=False)
            if primitiva in PRIMITIVAS_FISICAS:
                lacunas.append(_lacuna("fala_sobre_acao", "atencao", cena, i, beat,
                                       "fala durante acao fisica: um close nao mostra a acao. Separar "
                                       "ACAO (plano proprio) e FALA (reacao), ou marcar a fala como "
                                       "voz sobre a acao"))
    return lacunas, estados


GAPS_SYSTEM_PROMPT = (
    "Voce e um assistente de continuidade de cinema. Responda APENAS com JSON valido no "
    'formato {"eventos": [{"item": 0, "agente": "...", "alvo": "...", "objetos": ["..."], '
    '"posicao": "...", "estado_inicial": "...", "estado_final": "...", "falta": ["..."], '
    '"sugestao": "..."}]}. Para cada ACAO numerada, diga quem age (agente), sobre quem/o que '
    "(alvo), objetos envolvidos, onde cada um esta (posicao), o estado antes e depois. Use os "
    "NOMES do elenco listado; para quem nao esta no elenco, use o papel (ex.: 'the President'). "
    "'falta': SOMENTE o que o TEXTO nao diz E muda o que a camera mostra (escolha entre: "
    "agente, alvo, posicao, estado_inicial, estado_final, objetos, tempo); lista VAZIA se a "
    "acao ja esta clara -- nao marque tudo por precaucao. 'sugestao': UMA frase curta, no "
    "idioma do roteiro, que o roteirista poderia acrescentar para fechar a lacuna (vazia se "
    "nada falta). Nao invente fatos que contradigam o texto; quando inferir, deixe claro."
)


def audit_llm(scenes: list[dict], cast: dict, engine: str, log=print) -> list[dict]:
    """Sugestoes por acao, INFERIDAS pelo LLM. Uma chamada por cena, prompt curto
    (o qwen3.6 MoE cai com prompt longo -- CLAUDE.md, ~8k caracteres)."""
    from script_pipeline.story_structure import _call_ollama

    nomes = sorted(cast)
    sugestoes = []
    for sc in scenes:
        itens = [(i, str(it.get("visual") or "")) for i, it in enumerate(sc.get("shot_list") or [])
                 if it.get("type") == "action"]
        if not itens:
            continue
        user = (f"Elenco: {', '.join(nomes) or '(vazio)'}\n"
                f"Texto da cena (trecho): {(sc.get('action_text') or '')[:3000]}\n\nAcoes:\n"
                + "\n".join(f"{i}: {t}" for i, t in itens))
        payload = _call_ollama(GAPS_SYSTEM_PROMPT, user, engine, log=log) or {}
        validos = {i for i, _ in itens}
        for ev in payload.get("eventos") or []:
            try:
                i = int(ev.get("item"))
            except (TypeError, ValueError):
                continue
            if i not in validos:
                continue
            falta = [str(f) for f in (ev.get("falta") or []) if str(f).strip()]
            # MEDIDO 2026-09-27: com o prompt antigo o modelo marcava as 7 categorias em
            # TODA acao -- lista que nao discrimina nada. Descarta como diagnostico.
            if len(falta) >= 6:
                falta = []
            if not falta and not ev.get("sugestao"):
                continue
            sugestoes.append({"cena": sc.get("index", 0), "item": i, "texto": dict(itens)[i],
                              "agente": ev.get("agente"), "alvo": ev.get("alvo"),
                              "objetos": ev.get("objetos") or [], "posicao": ev.get("posicao"),
                              "estado_inicial": ev.get("estado_inicial"),
                              "estado_final": ev.get("estado_final"),
                              "falta": falta, "sugestao": ev.get("sugestao"), "inferido": True})
    log(f"[lacunas] LLM ({engine}): {len(sugestoes)} sugestao(oes) por acao")
    return sugestoes


_TITULOS = {
    "pessoa_sem_cadastro": "Pessoas que agem sem estar no elenco",
    "agente": "Acoes sem agente definido",
    "alvo": "Acoes sem alvo definido",
    "repeticao": "Falas que repetem a acao vizinha",
    "fala_sobre_acao": "Falas durante acao fisica",
    "fala_sem_visual": "Falas sem indicacao do que o personagem faz",
    "aparencia": "Aparencia incompleta",
    "voz": "Voz escolhida por palpite",
    "cenario": "Cenario",
}


def render_markdown(relatorio: dict) -> str:
    l = ["# Lacunas do roteiro", "",
         f"{relatorio['resumo']['critica']} critica(s), {relatorio['resumo']['atencao']} de atencao, "
         f"{relatorio['resumo']['info']} informativa(s). Guia do que indicar: "
         "`script_pipeline/GUIA_ROTEIRO.md`.", ""]
    por_tipo: dict[str, list] = {}
    for lac in relatorio["lacunas"]:
        por_tipo.setdefault(lac["tipo"], []).append(lac)
    for tipo in _TITULOS:
        if tipo not in por_tipo:
            continue
        l += [f"## {_TITULOS[tipo]}", ""]
        for lac in por_tipo[tipo]:
            onde = f"cena {lac['cena']}" + (f", item {lac['item']}" if lac["item"] is not None else "")
            l.append(f"- **{lac['gravidade']}** ({onde}): _{lac['texto'][:140]}_ — falta: {lac['o_que_falta']}")
        l.append("")
    if relatorio["mudancas_de_estado"]:
        l += ["## Continuidade a manter", ""]
        for e in relatorio["mudancas_de_estado"]:
            l.append(f"- cena {e['cena']}, item {e['item']}: _{e['texto'][:140]}_ — {e['manter_depois']}")
        l.append("")
    if relatorio.get("sugestoes_llm"):
        l += ["## Sugestoes do LLM (inferidas — confirme antes de usar)", ""]
        for s in relatorio["sugestoes_llm"]:
            l.append(f"- cena {s['cena']}, item {s['item']}: _{s['texto'][:120]}_ — falta: "
                     f"{', '.join(s['falta']) or '-'}; sugestao: {s.get('sugestao') or '-'}")
        l.append("")
    return "\n".join(l)


def audit(run_dir: Path, *, engine: str | None = None, log=print) -> dict:
    parse = run_dir / "parse"
    cenas_path = parse / "scenes_enriched.json"
    if not cenas_path.exists():
        cenas_path = parse / "scenes.json"
    scenes = json.loads(cenas_path.read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes", [])
    cast_path = run_dir / "characters" / "cast.json"
    cast = json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.exists() else {}
    lacunas, estados = audit_deterministic(scenes, cast)
    sugestoes = []
    if engine:
        try:
            sugestoes = audit_llm(scenes, cast, engine, log=log)
        except Exception as exc:  # o LLM nunca derruba a auditoria deterministica
            log(f"[lacunas] LLM indisponivel ({type(exc).__name__}: {exc}); so a parte deterministica")
    resumo = {g: sum(1 for x in lacunas if x["gravidade"] == g) for g in ("critica", "atencao", "info")}
    relatorio = {"resumo": resumo, "lacunas": lacunas, "mudancas_de_estado": estados,
                 "sugestoes_llm": sugestoes, "fonte": cenas_path.name}
    (parse / "lacunas.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    (parse / "lacunas.md").write_text(render_markdown(relatorio), encoding="utf-8")
    log(f"[lacunas] {resumo['critica']} critica(s), {resumo['atencao']} de atencao, {resumo['info']} "
        f"info; {len(estados)} mudanca(s) de estado a carregar -> {parse / 'lacunas.md'}")
    for lac in [x for x in lacunas if x["gravidade"] == "critica"][:8]:
        log(f"  CRITICA cena {lac['cena']} item {lac['item']}: {lac['o_que_falta']}")
    return relatorio


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", default=None, help="modelo do Ollama para as sugestoes por acao")
    ap.add_argument("--bloquear", action="store_true", help="devolve 2 se houver lacuna critica")
    args = ap.parse_args(argv)
    relatorio = audit(Path(args.run_dir).resolve(), engine=args.engine)
    return 2 if (args.bloquear and relatorio["resumo"]["critica"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
