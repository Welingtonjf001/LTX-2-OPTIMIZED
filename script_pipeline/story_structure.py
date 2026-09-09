"""Leitura do roteiro INTEIRO, de uma vez, para produzir a camada que falta:
estrutura dramática e continuidade entre cenas.

POR QUE ISTO É UMA ETAPA SEPARADA, E NÃO UM PARÂMETRO DO PARSE

O enriquecimento de `parse_screenplay` roda **uma chamada independente por
cena**, e `_build_scene_user_prompt(scene)` monta o prompt com dados só daquela
cena — cabeçalho, local, período, ação, personagens, falas. Nenhuma referência
à anterior ou à seguinte. Então ele **não pode** produzir arco, progressão nem
continuidade: não é limitação do modelo, é do contexto que ele recebe. Trocar
por um modelo maior não mudaria nada.

Aqui é o contrário: o roteiro entra inteiro, numa chamada só, e o que sai é
justamente o que depende de ver o conjunto.

O QUE ESTA ETAPA NÃO FAZ, DE PROPÓSITO

Ela **não reescreve nada** — nem diálogo, nem ação, nem cabeçalho. A saída é
puramente aditiva: um JSON de metadados por cena, ao lado do roteiro.

Isso não é timidez. `prose_to_screenplay` existe com verificação verbatim
porque diálogo aqui vai para TTS e depois para lip-sync: uma fala parafraseada
é falada errada em silêncio, e só se descobre assistindo. Já aconteceu nesta
base (rubrica escrita sob um cue, que o TTS falaria como diálogo). Se esta
etapa pudesse reescrever, ela desfaria essa garantia justamente onde ela é
mais cara. Se alucinar, perde-se anotação — nunca uma fala.

DETERMINÍSTICO PRIMEIRO

Boa parte do que parece exigir LLM é conta. Ordem das cenas, quem entra e sai,
quando um local se repete, volume de diálogo, salto de período do dia — tudo
sai dos objetos já parseados, sem modelo, sem erro. A LLM recebe esse esqueleto
pronto e só preenche o que é interpretação: função dramática, tensão, propósito
e o que mudou no estado da história.

PARA QUE SERVE RIO ABAIXO

`function` e `tension` são o que permite a decupagem escolher cobertura: uma
virada não se filma como um respiro. Sem esta camada, variar ângulo seria
variar por variar — melhor que o plano fixo de hoje, mas ainda não é direção.
`continuity` alimenta o problema de identidade entre cenas: figurino e local
precisam ser repetidos verbatim nos prompts, e é aqui que se sabe quando
mudam legitimamente.

CLI:
    python -m script_pipeline.story_structure --run RUN_DIR [--engine qwen3.6-35b-a3b:latest]
    python -m script_pipeline.story_structure --scenes scenes.json --out estrutura.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

# Vocabulário fechado. Enum em vez de texto livre porque estes valores viram
# CHAVE de política na decupagem -- "virada" tem de casar exatamente, e um
# modelo deixado à solta devolve "ponto de virada", "turning point", "clímax
# parcial" para a mesma coisa.
FUNCTIONS = ["estabelecimento", "desenvolvimento", "virada", "climax", "respiro", "desfecho"]
LINKS = ["continuo", "corte", "salto_temporal", "retorno"]

DEFAULT_ENGINE = os.environ.get("LTX_STRUCTURE_ENGINE", "qwen3.6-35b-a3b:latest")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


# --------------------------------------------------------------------------
# passe determinístico
# --------------------------------------------------------------------------
def _slug(text: str) -> str:
    """Id estável de local. Normaliza acento e caixa para que 'Varanda do
    Castelo' e 'VARANDA DO CASTELO' sejam o MESMO lugar -- sem isso, um
    retorno ao mesmo cenário passa despercebido e a continuidade se perde
    exatamente onde ela importa."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-zA-Z0-9]+", "_", t).strip("_").lower()
    return t or "local_sem_nome"


def derive_deterministic(scenes: list[dict]) -> dict:
    """Tudo o que é conta, não interpretação. Nenhum modelo envolvido."""
    out_scenes = []
    locations: dict[str, dict] = {}
    characters: dict[str, dict] = {}
    prev_chars: set[str] = set()
    prev_loc: str | None = None
    prev_time: str | None = None

    for s in scenes:
        idx = int(s.get("index", len(out_scenes) + 1))
        loc_id = _slug(s.get("location") or s.get("heading_raw") or "")
        chars = list(s.get("characters") or [])
        cur = set(chars)
        dialogue = s.get("dialogue") or []

        loc = locations.setdefault(loc_id, {"heading": s.get("heading_raw", ""),
                                            "location": s.get("location", ""), "scenes": []})
        revisits = bool(loc["scenes"])          # já apareceu antes neste roteiro
        loc["scenes"].append(idx)

        for name in chars:
            characters.setdefault(name, {"scenes": [], "first_scene": idx})["scenes"].append(idx)

        out_scenes.append({
            "index": idx,
            "location_id": loc_id,
            "location": s.get("location", ""),
            "time_of_day": s.get("time_of_day", ""),
            "characters": chars,
            # Diferença com a cena anterior: quem chegou, quem saiu. É o que
            # permite dizer "ela entra" em vez de "ela está lá" no prompt.
            "entering": sorted(cur - prev_chars),
            "exiting": sorted(prev_chars - cur),
            "revisits_location": revisits,
            "same_location_as_previous": loc_id == prev_loc,
            "time_changed": bool(prev_time) and s.get("time_of_day") != prev_time,
            "dialogue_lines": len(dialogue),
            "action_chars": len(s.get("action_text") or ""),
        })
        prev_chars, prev_loc, prev_time = cur, loc_id, s.get("time_of_day")

    # link_to_previous é DERIVADO, não perguntado. MEDIDO 2026-08-26: pedindo à
    # LLM, ela devolveu "continuo" para as cinco cenas do roteiro de teste --
    # inclusive na que vai de MEIA-NOITE para AMANHECER, salto óbvio. A evidência
    # (mudou o período? mudou o local? o local já tinha aparecido?) está toda na
    # camada determinística, então perguntar era criar fonte de erro de graça.
    for i, d in enumerate(out_scenes):
        if i == 0:
            d["link_to_previous"] = None
        elif d["time_changed"]:
            d["link_to_previous"] = "salto_temporal"
        elif d["same_location_as_previous"]:
            d["link_to_previous"] = "continuo"
        elif d["revisits_location"]:
            d["link_to_previous"] = "retorno"
        else:
            d["link_to_previous"] = "corte"

    return {"scenes": out_scenes, "locations": locations, "characters": characters}


# --------------------------------------------------------------------------
# passe de interpretação (LLM, roteiro inteiro numa chamada)
# --------------------------------------------------------------------------
def _build_prompt(scenes: list[dict], det: dict, language: str) -> tuple[str, str]:
    system = (
        "Voce e um analista de roteiro. Recebe um roteiro INTEIRO e devolve APENAS JSON valido, "
        "sem markdown, sem comentario.\n"
        "NAO reescreva nada. NAO copie falas. NAO invente cenas. Voce apenas ANOTA.\n"
        '{"logline": "<uma frase>", '
        '"acts": [{"name": "<nome>", "scenes": [<indices>]}], '
        '"scenes": [{"index": <n>, "function": "<' + "|".join(FUNCTIONS) + '>", '
        '"tension": <0.0 a 1.0>, "purpose": "<uma linha: para que esta cena existe>", '
        '"state_change": ["<o que mudou na historia depois desta cena>"]}]}\n'
        "REGRAS:\n"
        "- Uma entrada em scenes para CADA indice recebido, na mesma ordem.\n"
        "- function DEVE ser um dos valores listados, exatamente.\n"
        "- tension e um numero, nao texto.\n"
        "- purpose e state_change descrevem FUNCAO NARRATIVA, nao o que se ve.\n"
        f"- Escreva purpose e state_change em {language}.\n"
    )

    blocos = []
    for s, d in zip(scenes, det["scenes"]):
        dial = s.get("dialogue") or []
        # As falas entram RESUMIDAS (quem fala e quantos caracteres), nunca o
        # texto: o modelo nao precisa delas para julgar funcao dramatica, e
        # nao mandar o texto elimina a chance de ele devolver fala reescrita.
        quem = ", ".join(sorted({(l.get("character") or "?") for l in dial})) or "(sem falas)"
        blocos.append(
            f"CENA {d['index']} | {s.get('heading_raw','')}\n"
            f"  local={d['location_id']} periodo={d['time_of_day']} "
            f"{'(RETORNA a local ja usado)' if d['revisits_location'] else ''}\n"
            f"  personagens={', '.join(d['characters']) or '(nenhum)'}"
            f"{'  entram=' + ', '.join(d['entering']) if d['entering'] else ''}"
            f"{'  saem=' + ', '.join(d['exiting']) if d['exiting'] else ''}\n"
            f"  acao: {(s.get('action_text') or '(nenhuma)')[:400]}\n"
            f"  falas: {len(dial)} ({quem})\n"
        )
    user = ("Roteiro completo, cena a cena:\n\n" + "\n".join(blocos) +
            f"\nDevolva o JSON para as {len(blocos)} cenas.")
    return system, user


def _call_ollama(system: str, user: str, model: str, log=print) -> dict | None:
    payload = {
        "model": model, "stream": False, "format": "json",
        # think=False: modelo de raciocinio gasta o orcamento no campo `thinking`
        # e devolve content VAZIO -- falha silenciosa que parece recusa.
        # Medido com qwen3.6 (ver MEMORIAL.md secao 3.13).
        "think": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.2, "num_ctx": 32768},
    }
    req_bytes = json.dumps(payload).encode("utf-8")
    # RETRY (2026-09-09, auditoria pos-Storyboard-Director): mesmo fix de
    # `parse_screenplay.py` (2026-09-07, MEMORIAL 3.65) portado pra ca -- esta
    # e a implementacao COMPARTILHADA que `cast_characters.py` e
    # `prompt_polish.py` chamam, e ate agora so o `_call_ollama` PROPRIO do
    # parse_screenplay tinha retry. Uma falha HTTP transiente na primeira
    # chamada nao pode custar o descritor de um personagem inteiro se a
    # segunda tentativa teria funcionado. 3x o timeout de conexao curto, nao
    # 3x os 600s de geracao.
    body = None
    for tentativa in range(3):
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=req_bytes,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body = json.load(r)
            if tentativa:
                log(f"[story_structure] Ollama ok na tentativa {tentativa + 1}/3.")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log(f"[story_structure] modelo '{model}' nao existe nesta instancia do Ollama.")
                try:
                    tags = json.load(urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10))
                    log("  servidos: " + ", ".join(m["name"] for m in tags.get("models", [])))
                except Exception:
                    pass
                return None
            log(f"[story_structure] Ollama HTTP {e.code} (tentativa {tentativa + 1}/3).")
        except Exception as e:
            log(f"[story_structure] Ollama inacessivel ({type(e).__name__}): {e} (tentativa {tentativa + 1}/3).")
        if tentativa < 2:
            time.sleep(2)
    if body is None:
        return None

    content = (body.get("message") or {}).get("content") or ""
    if not content.strip():
        log("[story_structure] Ollama devolveu content vazio "
            f"(done_reason={body.get('done_reason')}). Modelo de raciocinio sem think=False?")
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        log("[story_structure] resposta nao era JSON valido.")
        return None


# --------------------------------------------------------------------------
# verificação
# --------------------------------------------------------------------------
def verify(structure: dict, scenes: list[dict], log=print) -> list[str]:
    """Confere o que a LLM devolveu. Devolve a lista de problemas -- vazia
    quando está tudo certo.

    O item que mais importa é o vazamento de diálogo: se uma fala aparecer nos
    campos anotados, algo saiu do papel de anotador e virou reescrita, que é
    exatamente o que esta etapa promete não fazer."""
    problems = []
    got = {int(s.get("index", -1)) for s in structure.get("scenes", [])}
    want = {int(s.get("index", i + 1)) for i, s in enumerate(scenes)}
    if got != want:
        problems.append(f"cenas anotadas {sorted(got)} != cenas do roteiro {sorted(want)}")

    for s in structure.get("scenes", []):
        i = s.get("index")
        if s.get("function") not in FUNCTIONS:
            problems.append(f"cena {i}: function invalida {s.get('function')!r}")
        try:
            t = float(s.get("tension"))
            if not 0.0 <= t <= 1.0:
                problems.append(f"cena {i}: tension fora de 0..1 ({t})")
        except (TypeError, ValueError):
            problems.append(f"cena {i}: tension nao numerica ({s.get('tension')!r})")

    # vazamento de diálogo
    falas = []
    for sc in scenes:
        for line in (sc.get("dialogue") or []):
            txt = (line.get("text") or "").strip()
            if len(txt) >= 25:      # trechos curtos coincidem por acaso
                falas.append(txt)
    anotado = json.dumps(structure, ensure_ascii=False).lower()
    for txt in falas:
        if txt.lower()[:40] in anotado:
            problems.append(f"VAZAMENTO: fala aparece na anotacao -> {txt[:50]!r}")

    for p in problems:
        log(f"[story_structure] {p}")
    return problems


# --------------------------------------------------------------------------
# montagem
# --------------------------------------------------------------------------
def build(scenes: list[dict], *, engine: str = DEFAULT_ENGINE, language: str = "pt",
          log=print) -> dict:
    det = derive_deterministic(scenes)
    log(f"[story_structure] {len(scenes)} cena(s), {len(det['locations'])} local(is), "
        f"{len(det['characters'])} personagem(ns) -- passe deterministico pronto.")

    system, user = _build_prompt(scenes, det, language)
    llm = _call_ollama(system, user, engine, log=log)

    merged = {
        "logline": (llm or {}).get("logline", ""),
        "acts": (llm or {}).get("acts", []),
        "locations": det["locations"],
        "characters": det["characters"],
        "scenes": [],
        "llm_engine": engine if llm else None,
    }
    by_index = {int(s.get("index", -1)): s for s in (llm or {}).get("scenes", [])}
    for d in det["scenes"]:
        extra = by_index.get(d["index"], {})
        merged["scenes"].append({**d,
                                 "function": extra.get("function"),
                                 "tension": extra.get("tension"),
                                 "purpose": extra.get("purpose"),
                                 "state_change": extra.get("state_change") or []})
    if llm is None:
        log("[story_structure] sem LLM: saindo so com a camada deterministica "
            "(local, presenca, retornos, entradas/saidas). function/tension ficam nulos.")
    else:
        problems = verify(merged, scenes, log=log)
        merged["problems"] = problems
        if problems:
            log(f"[story_structure] {len(problems)} problema(s) -- a camada deterministica "
                "continua valida; revise o JSON antes de usar as anotacoes.")
    return merged


def summary(structure: dict) -> str:
    linhas = []
    if structure.get("logline"):
        linhas.append(f"LOGLINE: {structure['logline']}")
    for a in structure.get("acts") or []:
        linhas.append(f"ATO {a.get('name','?')}: cenas {a.get('scenes')}")
    linhas.append("")
    linhas.append(f"{'cena':>5} {'funcao':<16} {'tens':>5} {'local':<22} {'elo':<14} proposito")
    for s in structure.get("scenes", []):
        t = s.get("tension")
        linhas.append(
            f"{s['index']:>5} {str(s.get('function') or '-'):<16} "
            f"{(f'{float(t):.2f}' if isinstance(t, (int, float)) else '-'):>5} "
            f"{s['location_id'][:22]:<22} {str(s.get('link_to_previous') or '-'):<14} "
            f"{(s.get('purpose') or '')[:60]}")
    return "\n".join(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Estrutura dramatica e continuidade entre cenas")
    ap.add_argument("--run", help="pasta de run com scenes.json")
    ap.add_argument("--scenes", help="caminho direto de um scenes.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--language", default="pt")
    args = ap.parse_args()

    if args.scenes:
        path = Path(args.scenes)
    elif args.run:
        path = Path(args.run) / "scenes.json"
    else:
        ap.error("informe --run ou --scenes")
    if not path.exists():
        print(f"[story_structure] nao encontrei {path}", file=sys.stderr)
        return 1

    data = json.load(open(path, encoding="utf-8"))
    scenes = data["scenes"] if isinstance(data, dict) and "scenes" in data else data

    structure = build(scenes, engine=args.engine, language=args.language)
    out = Path(args.out) if args.out else path.parent / "story_structure.json"
    json.dump(structure, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print()
    print(summary(structure))
    print(f"\n[story_structure] salvo em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
