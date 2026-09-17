"""Editor de MONTAGEM (v1) -- pedido do usuario 2026-09-16, depois de ver o
teste "sem still" do palacio repetir uma acao (a princesa se levantando duas
vezes) e variar de ritmo sem justificativa narrativa.

O QUE JA EXISTIA E O QUE ISTO E DIFERENTE

`video_doctor.py`, `storyboard_audit.py` e `clip_identity_audit.py` medem
defeito de PIXEL (artefato entre quadros) ou de IDENTIDADE (rosto bate com a
referencia). Nenhum dos tres entende CONTEUDO -- nenhum sabe que "a princesa
se levanta" aconteceu duas vezes em planos diferentes, ou que uma danca de 2s
esta curta demais pro que a acao pede. `video_doctor` e explicito no proprio
MEMORIAL: "nada e reescrito" -- so mede.

Este script MEDE (sempre) e, opt-in (--apply), AGE: marca planos duplicados
para regenerar com seed diferente + uma dica textual pra nao repetir a acao
do primeiro, e alarga a duracao de planos que a IA julgou curtos demais pro
que a acao descreve. NAO reordena planos sozinho (mudar a ordem afeta lado de
tela, encadeamento de referencia e a narrativa em si -- fica so como
SUGESTAO no relatorio, pra revisao humana) e NAO encolhe plano longo
sozinho (encolher fala corta audio -- risco demais pra v1).

METODO

1. Extrai um frame do MEIO de cada clipe (nao o primeiro/ultimo -- e onde a
   acao geralmente esta em andamento, nao em transicao).
2. Descreve a acao de cada plano com um VLM local (qwen2.5vl:7b via Ollama,
   ja instalado neste checkout -- ver MEMORIAL, investigacoes 2026-09-12).
3. Manda a lista inteira (todas as descricoes + beat + segundos planejados)
   pro LLM de texto de sempre (qwen3.6, `_call_ollama` compartilhado) num
   UNICO passe, pra ele julgar duplicacao e ritmo com o contexto da cena
   inteira, nao plano a plano isolado.
4. Escreve `<run>/final/editorial_report.json` sempre. Com --apply, muta
   `shot_plan.json` (seed + duracao) e invalida o cache dos planos afetados
   (apaga o `.key`) -- a regeneracao em si (GPU) fica pro usuario rodar
   depois com `render_shots_stage.py --videos-only --only-shots N,M,...`,
   exatamente como o relatorio final imprime pra copiar e colar.

CLI:
    python -m script_pipeline.story_editor --run-dir DIR [--apply]
        [--vision-model qwen2.5vl:7b] [--engine qwen3.6-35b-a3b:latest]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434"


def _extrair_frame_meio(video_path: str, out_path: Path) -> bool:
    """Frame do MEIO do clipe -- porte de `render_shots._extrair_ultimo_frame`,
    trocando o indice (n-1) por (n//2)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame))


def _describe_frame(frame_path: Path, model: str, log=print) -> str:
    """Uma frase curta descrevendo a ACAO no frame, via VLM local (Ollama).
    Falha vira string vazia -- o chamador decide se aceita follow sem essa
    descricao (nao derruba a auditoria inteira por um still ilegivel)."""
    try:
        img_b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    payload = {
        "model": model, "stream": False,
        "messages": [{
            "role": "user",
            "content": ("Describe in ONE short sentence (max 20 words) what action "
                        "is happening in this frame -- who is doing what, physically. "
                        "No scene description, no adjectives about mood -- just the "
                        "concrete physical action."),
            "images": [img_b64],
        }],
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
        return (body.get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        log(f"[story_editor] VLM falhou em {frame_path.name}: {e}")
        return ""


_JUDGE_SYSTEM = (
    "Voce e um editor de montagem revisando a decupagem de UMA cena. Recebe, "
    "na ordem dos planos, a acao PLANEJADA (beat) e a acao REALMENTE GERADA "
    "(descrita por um modelo de visao a partir do video), com a duracao em "
    "segundos de cada plano. Responda em JSON com duas chaves:\n"
    '"duplicados": lista de grupos -- cada grupo e {"planos": [indices], '
    '"motivo": "..."} -- planos cuja acao GERADA e a MESMA acao fisica '
    "acontecendo de novo (nao just personagens iguais fazendo coisas "
    "diferentes -- a MESMA acao, tipo 'levanta da cadeira' aparecendo duas "
    "vezes). So aponte duplicacao clara, nao pareca duas acoes parecidas por "
    "coincidencia (dois planos de dialogo estatico NAO sao duplicados so "
    "por serem os dois estaticos).\n"
    '"ritmo": lista de {"plano": indice, "veredito": "curto"|"longo"|"ok", '
    '"motivo": "..."} -- so inclua planos "curto" ou "longo" (omita os "ok" '
    "da lista, pra nao poluir); julgue a duracao EM SEGUNDOS contra o quanto "
    "a acao DESCRITA parece precisar (uma danca inteira em 1.5s e curta; um "
    "plano estatico de dialogo em 8s nao e longo so por ser comprido, se a "
    "fala preenche o tempo).\n"
    "Responda SO o JSON, nenhum texto fora dele."
)


def _reescrever_beat(shot: dict, motivo: str, outros_beats: list[str],
                     model: str, log=print) -> str | None:
    """Fix #6 (auditoria externa 2026-09-16, achado #6): o v1 so anexava uma
    dica generica ("don't repeat this action") ao video_prompt sem tocar no
    BEAT planejado -- o motor recebia instrucao contraditoria (beat pedindo a
    MESMA acao, dica pedindo outra) e nada garantia coerencia com o resto da
    cena. Agora pede ao LLM um beat NOVO, fisicamente diferente da acao
    duplicada, mas grounded nos outros beats da mesma cena (mesmo personagem/
    tom/continuidade -- sem inventar objeto ou evento que contradiga a cena).
    None se a chamada falhar -- o chamador cai de volta pro hint antigo."""
    from script_pipeline.story_structure import _call_ollama

    system = (
        "Voce reescreve UM beat de acao de um plano de filme que estava "
        "duplicando a acao fisica de outro plano da MESMA cena. Troque a "
        "acao concreta por outra plausivel e DIFERENTE, mantendo o mesmo "
        "personagem, tom emocional e continuidade -- nao invente um evento, "
        "objeto ou personagem que nao esteja implicito nos outros beats. "
        'Responda em JSON: {"beat": "novo beat, uma frase curta, mesmo '
        'estilo do beat original"}. So o JSON, nenhum texto fora dele.'
    )
    user = (
        f"Beat original (duplicava outro plano -- motivo: {motivo}):\n"
        f"\"{shot.get('beat', '')}\"\n\n"
        "Outros beats da mesma cena, para contexto (a acao nova nao pode "
        "repetir nenhum deles):\n"
        + "\n".join(f"- {b}" for b in outros_beats if b)
    )
    payload = _call_ollama(system, user, model, log=log) or {}
    novo = (payload.get("beat") or "").strip()
    return novo or None


def _julgar(planos: list[dict], model: str, log=print) -> dict:
    from script_pipeline.story_structure import _call_ollama

    linhas = []
    for p in planos:
        linhas.append(
            f"plano {p['indice']} (cena {p['scene']}, {p['seconds']}s, "
            f"sujeito={p.get('subject') or '-'}): "
            f"planejado=\"{p['beat'][:200]}\" | gerado=\"{p['descricao'] or '(sem descricao)'}\""
        )
    user = "Planos da cena, em ordem:\n\n" + "\n".join(linhas)
    payload = _call_ollama(_JUDGE_SYSTEM, user, model, log=log) or {}
    return {
        "duplicados": payload.get("duplicados") or [],
        "ritmo": payload.get("ritmo") or [],
    }


def _carregar_planos(run: Path) -> tuple[dict, list[dict]]:
    plan_path = run / "parse" / "shot_plan.json"
    clips_path = run / "scenes" / "clips.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    clips = json.loads(clips_path.read_text(encoding="utf-8")) if clips_path.exists() else []
    clip_por_indice = {}
    for c in clips:
        # id e "sceneNN_shotNNN" -- o indice do shot_plan e o numero depois de "shot"
        try:
            idx = int(c["id"].rsplit("shot", 1)[1])
        except (KeyError, ValueError, IndexError):
            continue
        clip_por_indice[idx] = c
    return plan, [{"indice": i, **s, "clip": clip_por_indice.get(i)}
                  for i, s in enumerate(plan["shots"])]


def audit(run_dir: str, *, vision_model: str = "qwen2.5vl:7b",
         engine: str = "qwen3.6-35b-a3b:latest", apply: bool = False,
         log=print) -> dict:
    run = Path(run_dir).resolve()
    plan, shots = _carregar_planos(run)

    frames_dir = run / "final" / "_editorial_frames"
    planos_info = []
    for s in shots:
        clip = s.get("clip")
        video_path = clip.get("video_path") if clip else None
        descricao = ""
        if video_path and Path(video_path).exists():
            frame_path = frames_dir / f"shot{s['indice']:03d}.png"
            if _extrair_frame_meio(video_path, frame_path):
                descricao = _describe_frame(frame_path, vision_model, log=log)
                log(f"[story_editor] plano {s['indice']}: {descricao or '(vazio)'}")
            else:
                log(f"[story_editor] plano {s['indice']}: nao consegui extrair frame de {video_path}")
        else:
            log(f"[story_editor] plano {s['indice']}: sem clipe gerado, pulando descricao")
        planos_info.append({
            "indice": s["indice"], "scene": s.get("scene"),
            "subject": s.get("subject"), "beat": s.get("beat", ""),
            "seconds": s.get("seconds"), "descricao": descricao,
        })

    log("[story_editor] julgando duplicacao e ritmo (LLM de texto, cena inteira de uma vez)...")
    veredito = _julgar(planos_info, engine, log=log)

    relatorio = {
        "run_dir": str(run), "planos": planos_info,
        "duplicados": veredito["duplicados"], "ritmo": veredito["ritmo"],
        "aplicado": False,
    }

    if apply and (veredito["duplicados"] or veredito["ritmo"]):
        relatorio["aplicado"] = True
        relatorio["mudancas"] = _aplicar(run, plan, veredito, engine, log=log)
        out_plan = run / "parse" / "shot_plan.json"
        out_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[story_editor] shot_plan.json atualizado -> {out_plan}")

    out_path = run / "final" / "editorial_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[story_editor] relatorio -> {out_path}")
    return relatorio


def _invalidar_cache(run: Path, indice: int, log=print) -> None:
    """Apaga a marca de cache (.key) do plano -- proxima passada de video
    regenera em vez de reaproveitar, mesmo mecanismo que os bugfixes A02-A04
    da auditoria (2026-09-16) ja usam pra cache por conteudo."""
    marca = run / "shots" / "clips" / f"shot{indice:03d}.key"
    if marca.exists():
        marca.unlink()
        log(f"[story_editor]   cache invalidado: {marca.name}")


def _aplicar(run: Path, plan: dict, veredito: dict, engine: str, log=print) -> dict:
    from script_pipeline.shot_plan import _video_prompt

    mudancas = {"regenerar_seed": [], "duracao_ajustada": []}
    SEED_OFFSET = 90001  # deslocamento grande, fora da faixa normal de seed+i

    # Duplicados: mantem a PRIMEIRA ocorrencia do grupo, marca as OUTRAS pra
    # regenerar com seed diferente E com o BEAT reescrito (Fix #6) -- antes so
    # anexava uma dica generica ao video_prompt, deixando o "beat" planejado
    # (que ainda descrevia a MESMA acao) contradizendo a dica.
    for grupo in veredito["duplicados"]:
        indices = sorted(grupo.get("planos") or [])
        if len(indices) < 2:
            continue
        mantido, repetidos = indices[0], indices[1:]
        outros_beats = [plan["shots"][i].get("beat", "") for i in indices
                        if 0 <= i < len(plan["shots"])]
        for idx in repetidos:
            if not (0 <= idx < len(plan["shots"])):
                continue
            shot = plan["shots"][idx]
            shot["editorial_v1_seed_bump"] = SEED_OFFSET + idx
            motivo = grupo.get("motivo", "")
            novo_beat = _reescrever_beat(shot, motivo, outros_beats, engine, log=log)
            if novo_beat:
                shot["beat"] = novo_beat
                shot["editorial_v1_beat_rewritten"] = True
                shot["video_prompt"] = _video_prompt(
                    action=novo_beat, movement=shot.get("movement", "static"),
                    descriptor=shot.get("descriptor", ""), look=shot.get("look_base", ""),
                    quote=shot.get("quote"), subject=shot.get("subject", ""),
                    fallback=shot.get("fallback", ""), emotion=shot.get("emotion"),
                    framing=shot.get("framing", ""), speaking=shot.get("quote") is not None,
                    co_subject=shot.get("co_subject", ""))
                # Reconstruir o video_prompt do zero apagaria a clausula de
                # movimento que motion_conditioner.apply_motion_score anexou
                # (--motion-conditioning); recoloca-la mantem o condicionamento.
                if shot.get("motion_prompt"):
                    shot["video_prompt"] = shot["video_prompt"].rstrip(".") + f". {shot['motion_prompt']}."
                log(f"[story_editor] plano {idx}: beat reescrito -> \"{novo_beat}\"")
            else:
                # Fallback (LLM indisponivel/falhou): comportamento antigo,
                # so a dica textual anexada.
                dica = (" IMPORTANT: this must show a DIFFERENT physical action than "
                        "any previous shot in this scene -- do not repeat an action "
                        "already shown (e.g. standing up, sitting down) unless the "
                        "text explicitly describes it happening again.")
                if dica not in (shot.get("video_prompt") or ""):
                    shot["video_prompt"] = (shot.get("video_prompt") or "") + dica
                log(f"[story_editor] plano {idx}: reescrita de beat falhou, usando dica generica")
            _invalidar_cache(run, idx, log=log)
            mudancas["regenerar_seed"].append(
                {"plano": idx, "motivo": motivo, "mantido": mantido,
                 "beat_reescrito": bool(novo_beat)})
            log(f"[story_editor] plano {idx}: marcado pra regenerar (duplicava o plano {mantido})")

    # Ritmo "curto": alarga a duracao proporcionalmente (nao mexe em "longo" --
    # encolher e mais arriscado, fica so como sugestao no relatorio).
    fps = plan.get("fps", 24.0)
    for item in veredito["ritmo"]:
        if item.get("veredito") != "curto":
            continue
        idx = item.get("plano")
        if idx is None or not (0 <= idx < len(plan["shots"])):
            continue
        shot = plan["shots"][idx]
        antigo = shot.get("seconds", 0.0)
        novo = round(antigo * 1.5, 2)
        shot["seconds"] = novo
        shot["frames"] = max(9, ((int(novo * fps) - 1) // 8) * 8 + 1)
        shot["editorial_v1_duration_extended"] = True
        _invalidar_cache(run, idx, log=log)
        mudancas["duracao_ajustada"].append(
            {"plano": idx, "de": antigo, "para": novo, "motivo": item.get("motivo", "")})
        log(f"[story_editor] plano {idx}: {antigo}s -> {novo}s ({item.get('motivo', '')})")

    return mudancas


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--vision-model", default="qwen2.5vl:7b")
    ap.add_argument("--engine", default="qwen3.6-35b-a3b:latest")
    ap.add_argument("--apply", action="store_true",
                    help="muta shot_plan.json (seed+dica pra duplicados, duracao pra planos "
                         "curtos) e invalida o cache dos planos afetados. Sem isto, so mede "
                         "e escreve o relatorio -- nada e alterado.")
    args = ap.parse_args(argv)

    relatorio = audit(args.run_dir, vision_model=args.vision_model,
                      engine=args.engine, apply=args.apply)

    print(f"\n{len(relatorio['duplicados'])} grupo(s) de duplicacao, "
          f"{len(relatorio['ritmo'])} plano(s) com ritmo sinalizado.")
    if relatorio["duplicados"]:
        print("\nDUPLICADOS:")
        for g in relatorio["duplicados"]:
            print(f"  planos {g.get('planos')}: {g.get('motivo', '')}")
    if relatorio["ritmo"]:
        print("\nRITMO:")
        for r in relatorio["ritmo"]:
            print(f"  plano {r.get('plano')} [{r.get('veredito')}]: {r.get('motivo', '')}")

    if relatorio["aplicado"]:
        afetados = sorted(
            {m["plano"] for m in relatorio["mudancas"]["regenerar_seed"]} |
            {m["plano"] for m in relatorio["mudancas"]["duracao_ajustada"]})
        if afetados:
            run = relatorio["run_dir"]
            lista = ",".join(str(i) for i in afetados)
            print(f"\nPara regenerar so os planos afetados:\n"
                  f'  .venv/Scripts/python.exe -m script_pipeline.render_shots_stage '
                  f'--run-dir "{run}" --width 960 --height 544 --stills-only --only-shots {lista}\n'
                  f'  .venv/Scripts/python.exe -m script_pipeline.render_shots_stage '
                  f'--run-dir "{run}" --width 960 --height 544 --videos-only --engine minimax '
                  f'--only-shots {lista}\n'
                  f'(depois rode lipsync_scenes/mix_audio/assemble_final normalmente pra remontar)')
    elif relatorio["duplicados"] or relatorio["ritmo"]:
        print("\nRode de novo com --apply para marcar os planos afetados para regeneracao.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
