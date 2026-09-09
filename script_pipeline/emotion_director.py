"""Direção de voz: escolhe a EMOÇÃO de cada fala e a VOZ de cada personagem.

POR QUE ISTO EXISTE

A biblioteca de vozes já traz 12 vozes emotivas com 17 takes cada -- raiva,
surpresa, angústia, alegria, medo, desdém, grito -- e `synthesize_dialogue` já
escolhe o take POR FALA. O mecanismo estava certo; faltava o casamento.

MEDIDO 2026-08-26, numa cena com cinco direções distintas de atuação:

    (em tom de alerta sussurrado)      -> nenhuma  -> 16_neutra.wav
    (sobressaltada, arregalando os olhos) -> nenhuma  -> 16_neutra.wav
    (com urgência cômica)              -> nenhuma  -> 16_neutra.wav
    (com reverência e um sorriso doce) -> nenhuma  -> 16_neutra.wav
    (empolgada, dando uma piscadinha)  -> nenhuma  -> 16_neutra.wav

Cinco de cinco em neutro. `voice_library.match_emotion` casa por PALAVRA-CHAVE,
e rubrica de roteiro não usa as palavras da lista: usa "sobressaltada",
"empolgada", "com reverência". Ampliar a lista de sinônimos seria enxugar gelo
-- a variedade da linguagem é o problema, não a falta de três palavras.

Aqui o casamento é SEMÂNTICO: um LLM lê a rubrica e escolhe um dos 17 slugs.
O casamento por palavra-chave continua como fallback, e o neutro continua como
último recurso -- porque emoção errada é pior que emoção ausente.

O QUE É VERIFICADO

O slug devolvido tem de estar no conjunto REAL de takes daquela voz. Modelo que
inventa "empolgacao" não derruba a síntese: cai no fallback. Mesma disciplina de
`story_structure` -- enum fechado, validação depois.

CLI:
    python -m script_pipeline.emotion_director --run RUN_DIR [--engine qwen3.6-35b-a3b:latest]
    python -m script_pipeline.emotion_director --run RUN_DIR --recast   # rever as vozes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_ENGINE = os.environ.get("LTX_STRUCTURE_ENGINE", "qwen3.6-35b-a3b:latest")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# Como cada slug soa, em uma linha. Vai no prompt: o nome do arquivo sozinho
# ("06_desapontamento_APROX") não diz ao modelo o que o distingue de "desanimo".
SLUG_GLOSSARIO = {
    "raiva": "irritado, furioso, voz dura",
    "surpresa": "pego de surpresa, sobressaltado, arregalado",
    "desanimo": "sem energia, resignado, para baixo",
    "angustia": "aflito, apertado, sofrendo",
    "tristeza": "triste, choroso, pesado",
    "desapontamento": "decepcionado, frustrado com algo que falhou",
    "espontanea_entusiasmada": "empolgado e solto, falando rápido de animação",
    "com_medo": "assustado, tenso, alerta, sussurro de aviso",
    "alegre": "feliz, sorrindo, leve",
    "apaixonada": "terno, afetuoso, admirado",
    "sensual": "baixo, insinuante",
    "desdem": "desprezo, ironia, superioridade",
    "confusa": "sem entender, hesitante",
    "excitada": "agitado, urgente, elétrico",
    "calma": "sereno, controlado, gentil, reverente",
    "neutra": "sem carga emocional particular",
    "grito": "gritando, no volume máximo",
}


def _ollama(system: str, user: str, model: str, log=print) -> dict | None:
    payload = {"model": model, "stream": False, "format": "json", "think": False,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "options": {"temperature": 0.1, "num_ctx": 16384}}
    req_bytes = json.dumps(payload).encode("utf-8")
    # RETRY (2026-09-09): mesmo fix de parse_screenplay.py (MEMORIAL 3.65) --
    # sem isto uma falha HTTP transiente derrubava a direcao emocional do
    # ROTEIRO INTEIRO pro fallback de palavra-chave (o proprio docstring da
    # funcao promete "uma chamada para o roteiro inteiro" -- e exatamente por
    # cobrir tanta coisa numa chamada so que vale insistir antes de desistir).
    body = None
    for tentativa in range(3):
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=req_bytes,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = json.load(r)
            if tentativa:
                log(f"[emotion_director] Ollama ok na tentativa {tentativa + 1}/3.")
            break
        except Exception as e:
            log(f"[emotion_director] Ollama indisponivel ({type(e).__name__}) "
                f"(tentativa {tentativa + 1}/3).")
        if tentativa < 2:
            time.sleep(2)
    if body is None:
        log("[emotion_director] Ollama indisponivel apos 3 tentativas; usando palavra-chave.")
        return None
    content = (body.get("message") or {}).get("content") or ""
    if not content.strip():
        log("[emotion_director] resposta vazia (modelo de raciocinio sem think=False?).")
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else None


def direct_emotions(scenes: list, *, engine: str = DEFAULT_ENGINE, log=print) -> dict:
    """(cena, fala) -> slug de emoção. Uma chamada para o roteiro inteiro."""
    from script_pipeline.voice_library import match_emotion

    falas = []
    for sc in scenes:
        for i, l in enumerate(sc.get("dialogue") or []):
            falas.append({"scene": sc["index"], "line": i,
                          "character": l.get("character", ""),
                          "rubrica": l.get("parenthetical") or l.get("emotion") or "",
                          "texto": (l.get("text") or "")[:160]})
    if not falas:
        return {}

    slugs = list(SLUG_GLOSSARIO)
    system = (
        "Voce dirige atuacao de voz. Para cada fala, escolha UMA emocao da lista. "
        "Responda APENAS JSON valido.\n"
        "EMOCOES (slug: como soa):\n"
        + "\n".join(f"  {k}: {v}" for k, v in SLUG_GLOSSARIO.items()) + "\n"
        '{"emocoes": [{"scene": <n>, "line": <n>, "emocao": "<slug>"}]}\n'
        "REGRAS:\n"
        "- Uma entrada para CADA fala recebida.\n"
        "- emocao DEVE ser exatamente um dos slugs listados.\n"
        "- Use a RUBRICA como fonte principal; o texto da fala desempata.\n"
        "- Na duvida entre duas, prefira a menos extrema.\n"
        "- 'neutra' so quando realmente nao houver carga emocional.\n"
    )
    user = "Falas:\n" + "\n".join(
        f"cena {f['scene']} fala {f['line']} | {f['character']} | "
        f"rubrica: {f['rubrica'] or '(nenhuma)'} | texto: {f['texto']}"
        for f in falas)

    resposta = _ollama(system, user, engine, log=log)
    escolhido: dict = {}
    if resposta:
        for e in resposta.get("emocoes", []):
            slug = str(e.get("emocao", "")).strip().lower()
            if slug in SLUG_GLOSSARIO:
                escolhido[(int(e["scene"]), int(e["line"]))] = slug
            else:
                log(f"[emotion_director] slug invalido descartado: {slug!r}")

    # Fallback por palavra-chave onde o LLM não decidiu.
    disponiveis = set(slugs)
    faltando = 0
    for f in falas:
        k = (f["scene"], f["line"])
        if k not in escolhido:
            escolhido[k] = match_emotion(f["rubrica"], disponiveis) or "neutra"
            faltando += 1
    if faltando:
        log(f"[emotion_director] {faltando} fala(s) resolvida(s) por palavra-chave/neutro.")
    return escolhido


# --------------------------------------------------------------------------
def cast_voices(cast: dict, *, engine: str = DEFAULT_ENGINE, log=print) -> dict:
    """personagem -> id de voz emotiva, escolhido pelo DESCRITOR.

    O nome de cada voz codifica faixa e timbre (`F05_madura_nobre_ruth`,
    `M02_jovem_energico_ryan`), então a escolha é uma leitura do descritor --
    idade, temperamento, posição social -- contra esses rótulos. Feito à mão
    seria melhor; feito por LLM é melhor que a ordem alfabética, que é o que
    acontece hoje quando o cast atribui a primeira voz livre do gênero."""
    from script_pipeline.voice_library import list_emotive_voices, voice_gender

    vozes = sorted(list_emotive_voices())
    if not cast:
        return {}
    system = (
        "Voce faz casting de voz. Escolha a voz que melhor corresponde a cada personagem. "
        "Responda APENAS JSON valido.\n"
        "VOZES (o nome indica faixa etaria e timbre):\n"
        + "\n".join(f"  {v}" for v in vozes) + "\n"
        '{"vozes": {"NOME_DO_PERSONAGEM": "<id_da_voz>"}}\n'
        "REGRAS:\n"
        "- O prefixo F e voz feminina, M masculina. RESPEITE o genero informado.\n"
        "- Use idade e temperamento do descritor.\n"
        "- Personagens diferentes devem receber vozes DIFERENTES quando possivel.\n"
    )
    user = "Personagens:\n" + "\n".join(
        f"{n} | genero: {v.get('voice', {}).get('gender', '?')} | {v.get('descriptor', '')[:200]}"
        for n, v in cast.items())

    resposta = _ollama(system, user, engine, log=log)
    out = {}
    if resposta:
        for nome, vid in (resposta.get("vozes") or {}).items():
            if nome not in cast:
                continue
            if vid not in vozes:
                log(f"[emotion_director] voz inexistente descartada: {vid!r}")
                continue
            # Gênero é regra dura: uma voz masculina num personagem feminino é
            # erro audível imediato, e o modelo às vezes escorrega nisso.
            g = (cast[nome].get("voice") or {}).get("gender")
            if g and voice_gender(vid) and voice_gender(vid) != g:
                log(f"[emotion_director] {nome}: voz {vid} nao bate com genero {g}; mantida a atual.")
                continue
            out[nome] = vid
    return out


# --------------------------------------------------------------------------
def apply_to_run(run_dir: Path, *, engine: str, recast: bool, log=print) -> dict:
    """Grava a emoção escolhida em cada fala do scenes.json e, com --recast,
    a voz de cada personagem no cast.json.

    A emoção vai para o campo `emotion`, que `synthesize_dialogue.build_jobs` já
    lê (`line.get("parenthetical") or line.get("emotion")`). Como a rubrica tem
    precedência, escreve-se o SLUG na rubrica quando ela existe -- é o que faz
    `pick_take` casar, já que ele também usa palavra-chave e o slug É a
    palavra-chave exata."""
    parse_dir = run_dir / "parse"
    scenes_path = parse_dir / "scenes_enriched.json"
    if not scenes_path.exists():
        scenes_path = parse_dir / "scenes.json"
    dados = json.load(open(scenes_path, encoding="utf-8"))
    scenes = dados["scenes"] if isinstance(dados, dict) and "scenes" in dados else dados

    emocoes = direct_emotions(scenes, engine=engine, log=log)
    n = 0
    for sc in scenes:
        for i, l in enumerate(sc.get("dialogue") or []):
            slug = emocoes.get((sc["index"], i))
            if not slug:
                continue
            l["emotion"] = slug
            # A rubrica original fica preservada em `parenthetical_original`:
            # ela ainda vale para o prompt de vídeo e para leitura humana.
            if l.get("parenthetical") and "parenthetical_original" not in l:
                l["parenthetical_original"] = l["parenthetical"]
            l["parenthetical"] = slug
            n += 1
    if isinstance(dados, dict) and "scenes" in dados:
        dados["scenes"] = scenes
    else:
        dados = scenes
    json.dump(dados, open(scenes_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"[emotion_director] {n} fala(s) com emocao dirigida -> {scenes_path.name}")

    resultado = {"emocoes": {f"{k[0]}:{k[1]}": v for k, v in emocoes.items()}}

    if recast:
        cast_path = run_dir / "characters" / "cast.json"
        if cast_path.exists():
            cast = json.load(open(cast_path, encoding="utf-8"))
            novas = cast_voices(cast, engine=engine, log=log)
            for nome, vid in novas.items():
                antiga = cast[nome].setdefault("voice", {}).get("emotive_voice")
                cast[nome]["voice"]["emotive_voice"] = vid
                log(f"[emotion_director] {nome}: {antiga} -> {vid}")
            json.dump(cast, open(cast_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            resultado["vozes"] = novas
        else:
            log("[emotion_director] cast.json nao encontrado; --recast ignorado.")
    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description="Direcao de voz: emocao por fala e voz por personagem")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--recast", action="store_true",
                    help="tambem reescolhe a VOZ de cada personagem pelo descritor")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[emotion_director] run nao encontrado: {run_dir}", file=sys.stderr)
        return 1
    r = apply_to_run(run_dir, engine=args.engine, recast=args.recast)
    print(json.dumps(r, ensure_ascii=False, indent=2)[:1200])
    print("\nRode synthesize_dialogue de novo para a voz sair com a emocao dirigida.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
