# -*- coding: utf-8 -*-
"""Estagio opcional [P2]: mede e reescreve os prompts de video de um shot_plan.

O `shot_plan` produz prompts CORRETOS mas em lista de fragmentos: acao, emocao,
descritor, movimento e look colados por pontos. O prompt bom para o LTX -- ver
MEMORIAL 3.37 -- e um paragrafo com CADEIA DE ACAO (toca -> vira -> fala),
ancoras de boca, beat de fechamento e desenho de som, na ordem que o modelo
pesa: acao, personagem, ambiente, camera, audio por ultimo.

Divisao de trabalho deliberada:

  - O SCRIPT mede (score_prompt): 9 perguntas viram 9 testes de presenca,
    deterministicos, em milissegundos. E o arbitro.
  - A LLM reescreve (polish_prompt): conectores, ancoras, beat final -- o que
    heuristica nao faz. Ela NAO decide fato novo: recebe uma ficha com os
    campos que o shot_plan ja fixou (quem, onde, enquadramento, emocao) e so
    tem permissao de reorganizar.
  - O script valida a LLM: se o prompt polido perde o sujeito, perde a fala
    citada verbatim, ou pontua MENOS que o original, fica o original. A LLM
    nunca pode piorar o plano em silencio.

CLI:
  python -m script_pipeline.prompt_polish --run-dir DIR            # dry-run
  python -m script_pipeline.prompt_polish --run-dir DIR --apply    # grava
  python -m script_pipeline.prompt_polish --text "..."             # so mede
Flags: --quotes inclui a fala verbatim com ancoras de boca (ver o aviso na
ficha: em clipe CURTO a fala citada ja queimou legenda -- MEMORIAL 3.22).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline.story_structure import _call_ollama, DEFAULT_ENGINE  # noqa: E402

# ---------------------------------------------------------------------------
# medicao -- as 9 perguntas como testes de presenca
# ---------------------------------------------------------------------------

_CONECTORES = re.compile(
    r"\b(then|and then|as |while |before |after |turns?|reaches|steps?|"
    r"leans?|rises?|lowers?|crosses)\b", re.I)
_VERBOS_ACAO = re.compile(
    r"\b(touch\w*|grip\w*|rais\w*|turn\w*|step\w*|gestur\w*|look\w*|walk\w*|"
    r"lean\w*|point\w*|draw\w*|lift\w*|watch\w*|reach\w*|clos\w*|open\w*|"
    r"speak\w*|says?|answer\w*|kneel\w*|run\w*|stand\w*)\b", re.I)
_CAMERA = re.compile(
    r"\b(camera|push(es)? in|pull(s)? (back|out)|pan(s)?|orbit\w*|tracking|"
    r"dolly|handheld|static shot|zoom\w*|circling)\b", re.I)
_AUDIO = re.compile(
    r"\b(audio:|sound of|wind|hum(s|ming)?|music|orchestral|rumbl\w*|"
    r"vibrat\w*|footsteps|ambien\w*|silence|echo\w*)\b", re.I)
_AMBIENTE = re.compile(
    r"\b(chamber|room|forest|street|hall|cave|castle|interior|exterior|"
    r"moonlight|sunlight|lamplight|torchlight|archway|wall|ceiling|floor|"
    r"lit by|illuminated)\b", re.I)
_MEIO = re.compile(
    r"\b(3d animation|cel animation|live action|photoreal\w*|anime|stop.motion|"
    r"storybook|illustration|cinematic)\b", re.I)
_ENTREGA = re.compile(
    r"\b(urgently|calmly|softly|firmly|steady voice|deep voice|whisper\w*|"
    r"shout\w*|trembling|excited\w*|fearful\w*)\b", re.I)
_BOCA = re.compile(r"\bclos(es|ing) (her|his|their) mouth\b", re.I)
_QUOTE = re.compile(r'["“”]')


def score_prompt(text: str, *, has_dialogue: bool = False) -> dict:
    """As 9 perguntas do MEMORIAL 3.37 como testes booleanos.

    Devolve item -> bool, mais `total`, `max` e `faltando`. Itens de fala so
    contam quando o plano TEM fala -- senao a pontuacao puniria plano de acao
    por nao citar dialogo, que e o certo para ele."""
    t = text or ""
    frases = [f for f in re.split(r"(?<=[.!?])\s+", t.strip()) if f]
    itens = {
        # 1. quem esta em quadro (nome proprio capitalizado fora de inicio de frase
        #    e uma heuristica fraca; presenca de descritor fisico e mais estavel)
        "sujeito_descrito": bool(re.search(
            r"\b(woman|man|mage|warrior|girl|boy|figure|hair|cloak|armor)\b", t, re.I)),
        # 2. cadeia de acao: mais de um verbo E um conector entre eles
        "cadeia_de_acao": len(_VERBOS_ACAO.findall(t)) >= 2 and bool(_CONECTORES.search(t)),
        # 5. beat de fechamento: a ULTIMA frase sem aspas e com verbo -- o clipe
        #    sabe onde terminar (mesmo principio do keyframe final)
        "beat_final": bool(frases) and not _QUOTE.search(frases[-1])
                      and bool(_VERBOS_ACAO.search(frases[-1]) or
                               re.search(r"\b(fad\w*|dim\w*|settl\w*|ends?)\b",
                                         frases[-1], re.I)),
        # 6. onde estamos
        "ambiente": bool(_AMBIENTE.search(t)),
        # 7. meio da obra -- sem ele o mesmo roteiro sai metade fotoreal,
        #    metade ilustrado (MEMORIAL 3.24)
        "meio_da_obra": bool(_MEIO.search(t)),
        # 8. o que a camera faz
        "camera": bool(_CAMERA.search(t)),
        # 9. o que se ouve alem das falas -- sem isso o modelo inventa (3.36.2)
        "audio": bool(_AUDIO.search(t)),
    }
    if has_dialogue:
        itens["fala_citada"] = bool(_QUOTE.search(t))
        itens["entrega_da_fala"] = bool(_ENTREGA.search(t))
        itens["ancora_de_boca"] = bool(_BOCA.search(t))
    # ordem: acao antes de ambiente, ambiente antes de camera/audio.
    # Comparacao por primeira ocorrencia; ausencia nao pune (ja punida acima).
    pos = {k: (m.start() if (m := rx.search(t)) else None) for k, rx in
           (("acao", _VERBOS_ACAO), ("amb", _AMBIENTE), ("cam", _CAMERA))}
    itens["ordem_ltx"] = not (
        (pos["acao"] is not None and pos["amb"] is not None and pos["acao"] > pos["amb"]) or
        (pos["amb"] is not None and pos["cam"] is not None and pos["amb"] > pos["cam"]))
    total = sum(itens.values())
    return {"itens": itens, "total": total, "max": len(itens),
            "faltando": [k for k, v in itens.items() if not v]}


# ---------------------------------------------------------------------------
# reescrita -- a LLM recebe a ficha, nao o prompt
# ---------------------------------------------------------------------------

POLISH_SYSTEM = """You rewrite video-generation prompts for the LTX-2 model.
You receive a FACT SHEET. Write ONE English paragraph using ONLY those facts.
Never invent characters, objects, places or events not in the sheet.

Structure, in this exact order:
1. Action chain: what the subject does, as a connected sequence (use connectors:
   touches X, turns toward Y and..., then...). If the sheet gives
   dialogue_verbatim as the token <<FALA>>, write the speech moment like this
   example -- and says urgently, “<<FALA>>”. Lyra closes her mouth. -- picking
   a real verb and adverb that match the delivery field. The token <<FALA>> is
   the ONLY thing kept literally; the real dialogue is substituted later by the
   caller. Never write placeholder text in angle brackets yourself.
2. Character(s): physical description; screen side if given.
3. Environment and lighting.
4. Art direction / medium (exactly as given).
5. Camera: starting framing and movement.
6. Audio LAST: ambient sound design; if dialogue exists add 'spoken dialogue
   exclusively in natural Brazilian Portuguese, accurate lip sync, no
   overlapping speech'.
End the action chain with the closing beat if one is given.
Respond as JSON: {"prompt": "<the paragraph>"}"""


def _ficha(shot: dict, scene: dict, fala: str | None) -> str:
    campos = {
        "subject": shot.get("subject") or "",
        "action": shot.get("_acao") or "",
        "emotion(visible)": shot.get("_emocao_visivel") or "",
        "descriptor": shot.get("_descritor") or "",
        "screen_side": shot.get("screen_side") or "",
        "framing": shot.get("framing") or "",
        "camera_movement": shot.get("movement") or "",
        "location": scene.get("location") or "",
        "time_of_day": scene.get("time_of_day") or "",
        "environment": (scene.get("visual_prompt") or "")[:400],
        "art_direction": shot.get("art_direction") or "",
        "closing_beat": shot.get("_beat_final") or "",
        # A LLM NUNCA carrega o texto da fala -- so posiciona o token, e o
        # chamador substitui. MEDIDO 2026-08-28: pedir a citacao verbatim
        # funcionou para fala curta e falhou 4/4 vezes para a fala longa de 4
        # frases; o qwen3.6 simplesmente a OMITIA ("...speaks to Thoren, then
        # closes her mouth"), mesmo com instrucao estrita na retentativa.
        # Fidelidade e trabalho de script, nao de modelo.
        "dialogue_verbatim": "<<FALA>>" if fala else "(none -- do not quote anything)",
        "delivery": shot.get("_entrega") or "",
    }
    return "\n".join(f"{k}: {v}" for k, v in campos.items() if v)


def polish_prompt(shot: dict, scene: dict, *, fala: str | None = None,
                  model: str = DEFAULT_ENGINE, log=print) -> tuple[str, dict, dict]:
    """Devolve (prompt_final, score_antes, score_depois).

    O prompt final e o polido SO se ele nao perder nada: sujeito presente,
    fala verbatim intacta (quando pedida), score >= original. Qualquer
    violacao devolve o original -- a LLM nunca piora em silencio."""
    original = shot.get("video_prompt") or ""
    tem_fala = bool(fala)
    antes = score_prompt(original, has_dialogue=tem_fala)

    # verbatim: compara sem aspas tipograficas e com espacos normalizados
    limpo = lambda s: re.sub(r"\s+", " ", _QUOTE.sub("", s)).strip()  # noqa: E731

    def _fala_intacta(texto: str) -> bool:
        """Cada FRASE da fala presente verbatim. Por frase, nao pelo bloco:
        o proprio sistema pede ancora de boca 'apos cada linha citada', entao o
        modelo tem licenca para segmentar a citacao -- o que ele nao tem e
        licenca para trocar uma palavra. Exigir o bloco contiguo reprovava
        segmentacao legitima (foi o que derrubou a fala de 4 frases da Lyra
        duas vezes seguidas)."""
        if not fala:
            return True
        frases = [f.strip() for f in re.split(r"(?<=[.!?…])\s+", fala)
                  if f.strip()]
        alvo = limpo(texto)
        return all(limpo(f) in alvo for f in frases)

    ficha = _ficha(shot, scene, fala)
    polido = ""
    # DUAS tentativas. A fala viaja como token <<FALA>> e e substituida AQUI,
    # em codigo -- entao a unica falha possivel do modelo e nao emitir o token,
    # e e exatamente isso que a retentativa cobra. A checagem verbatim continua
    # como ultima linha de defesa, mas agora so falha se o proprio texto da
    # fala contiver o token (impossivel na pratica).
    for aviso in ("", "\nCRITICAL: the literal token <<FALA>> must appear "
                      "inside the quotes. Emit it unchanged."):
        resp = _call_ollama(POLISH_SYSTEM + aviso, ficha, model, log=log)
        polido = ((resp or {}).get("prompt") or "").strip()
        if not polido:
            continue
        if fala:
            if "<<FALA>>" not in polido:
                log("  [polish] token da fala ausente; retentando")
                polido = ""
                continue
            polido = polido.replace("<<FALA>>", fala.strip())
        # Sobra de template e pior que fragmento: "<delivery verb + adverb>"
        # IMPRESSO no prompt vazou no primeiro teste real (planos 3 e 9 da
        # Mei-Li). Qualquer colchete angular remanescente reprova.
        if re.search(r"<[^>]{1,60}>", polido):
            log("  [polish] sobra de template no texto; retentando")
            polido = ""
            continue
        break
    if not polido:
        return original, antes, antes

    depois = score_prompt(polido, has_dialogue=tem_fala)
    sujeito = (shot.get("subject") or "").strip()
    if sujeito and sujeito.lower() not in polido.lower():
        log(f"  [polish] descartado: perdeu o sujeito {sujeito}")
        return original, antes, antes
    if not _fala_intacta(polido):
        log("  [polish] descartado: fala alterada mesmo apos retentativa")
        return original, antes, antes
    if depois["total"] < antes["total"]:
        log(f"  [polish] descartado: score caiu {antes['total']}->{depois['total']}")
        return original, antes, antes
    return polido, antes, depois


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--text", default=None, help="so mede este prompt e sai")
    ap.add_argument("--apply", action="store_true",
                    help="grava os prompts polidos no shot_plan.json "
                         "(o original vai para video_prompt_original)")
    ap.add_argument("--quotes", action="store_true",
                    help="inclui a fala verbatim com ancoras de boca. Em clipe "
                         "CURTO a fala citada ja queimou legenda (MEMORIAL 3.22); "
                         "em clipe longo e o modo de dialogo nativo do 2.5.")
    ap.add_argument("--model", default=DEFAULT_ENGINE)
    args = ap.parse_args()

    if args.text:
        r = score_prompt(args.text, has_dialogue=bool(_QUOTE.search(args.text)))
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    if not args.run_dir:
        ap.error("--run-dir ou --text")

    run = Path(args.run_dir)
    plan_path = run / "parse" / "shot_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cenas = json.loads((run / "parse" / "scenes_enriched.json").read_text(encoding="utf-8"))
    por_indice = {c.get("index"): c for c in cenas}

    # Descritores fisicos do cast.json. Sem eles o modelo escreve "is a
    # handmaid" no lugar de "young elegant woman in mint-green silk robes" --
    # aconteceu no primeiro teste real (Mei-Li): a ficha tinha o campo, a CLI
    # nunca o preenchia, e a reescrita saia MAIS pobre que o original.
    descritores = {}
    cast_path = run / "characters" / "cast.json"
    if cast_path.exists():
        cast = json.loads(cast_path.read_text(encoding="utf-8"))
        descritores = {k.upper(): (v.get("descriptor") or "")
                       for k, v in cast.items()}

    mudados = 0
    for i, shot in enumerate(plan["shots"]):
        scene = por_indice.get(shot.get("scene")) or {}
        fala = None
        if args.quotes and shot.get("type") == "dialogue":
            li = shot.get("line_index")
            dlg = scene.get("dialogue") or []
            if li is not None and 0 <= li < len(dlg):
                fala = dlg[li].get("text")
                shot["_entrega"] = dlg[li].get("emotion") or ""
        # ficha: expor os fragmentos que o video_prompt ja embute
        shot["_acao"] = (shot.get("video_prompt") or "").split(".")[0]
        shot["_descritor"] = descritores.get((shot.get("subject") or "").upper(), "")
        # a emocao ja calculada, traduzida para o que se VE (3.36.4)
        try:
            from script_pipeline.shot_plan import emocao_visivel
            if shot.get("type") == "dialogue":
                dlg = scene.get("dialogue") or []
                li = shot.get("line_index")
                if li is not None and 0 <= li < len(dlg):
                    shot["_emocao_visivel"] = emocao_visivel(dlg[li].get("emotion"))
        except ImportError:
            pass
        novo, antes, depois = polish_prompt(shot, scene, fala=fala,
                                            model=args.model)
        marca = "=" if novo == shot.get("video_prompt") else "*"
        print(f"[{i}] {marca} score {antes['total']}/{antes['max']} -> "
              f"{depois['total']}/{depois['max']}"
              + (f"  faltava: {', '.join(antes['faltando'])}" if antes['faltando'] else ""))
        if novo != shot.get("video_prompt"):
            mudados += 1
            print(f"    ANTES : {shot['video_prompt'][:160]}")
            print(f"    DEPOIS: {novo[:160]}")
            if args.apply:
                shot["video_prompt_original"] = shot["video_prompt"]
                shot["video_prompt"] = novo
        for k in ("_acao", "_entrega", "_descritor", "_emocao_visivel"):
            shot.pop(k, None)

    if args.apply and mudados:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n{mudados} prompt(s) atualizados em {plan_path}")
    elif mudados:
        print(f"\n{mudados} prompt(s) melhorariam; rode com --apply para gravar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
