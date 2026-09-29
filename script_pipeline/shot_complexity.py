"""Mede a COMPLEXIDADE de encenacao de cada plano e recomenda quanto pre-visualizar
antes de gastar GPU -- e em que motor.

POR QUE EXISTE

Texto + still segura bem um plano de uma pessoa falando ou de um lugar. Nao segura
dois corpos em contato (derrubar, algemar, escoltar), causa e efeito em poucos
segundos (explode -> cai -> reage) ou objeto trocando de mao: MEDIDO no CERCO EM SEUL
(2026-09-27, MEMORIAL 3.132), foram exatamente esses planos que sairam ilegiveis nos
DOIS motores de video. Pre-visualizacao custa pouco nos planos simples e e o unico
jeito de dirigir os complexos -- mas so vale a pena onde o risco e real. Este modulo
mede, nao renderiza: diz ONDE o blocking 2D basta e ONDE um previs 3D de baixa
resolucao (manequins, `spatial_pipeline`/`motion_director`) paga o custo.

PONTUACAO (deterministica, sem modelo)

  primitiva de movimento (motion_conditioning, ver motion_conditioner):
    environment/speak 0 · idle/turn/gesture 0,5 · reach/locomote 1 · fire 1,5 ·
    pursue/protect/escort 2 · contact 2,5 · restrain/takedown 3
  + 0,75 por pessoa alem da primeira (sujeito, co-sujeito, figurantes citados)
  + 0,5 por objeto manipulado (arma, mochila, algemas...)
  + 2 se o plano MUDA o estado do mundo (explode, cai, fecha...)
  close de FALA pontua so o rosto (a acao do texto fica fora do quadro e e sinalizada)
  + 1 se a acao e fisica e o plano dura menos de 1,8 s (nao da tempo de ler)
  + 0,5 se a camera se move
  simples < 2,5 <= media < 4,5 <= complexa

Saida: `shots/complexity_report.json`. Nunca bloqueia.

CLI:
  python -m script_pipeline.shot_complexity --run-dir DIR [--engine ltx|minimax|minimax-longtake|longcat]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline.cast_characters import personagens_citados  # noqa: E402

PESO_PRIMITIVA = {
    "environment": 0.0, "speak": 0.0, "idle": 0.5, "turn": 0.5, "gesture": 0.5,
    "reach": 1.0, "locomote": 1.0, "fire": 1.5, "pursue": 2.0, "protect": 2.0,
    "escort": 2.0, "contact": 2.5, "restrain": 3.0, "takedown": 3.0,
}
FISICAS = {"reach", "locomote", "fire", "pursue", "protect", "escort", "contact", "restrain", "takedown"}
CONTATO = {"protect", "escort", "contact", "restrain", "takedown"}
_OBJETOS = re.compile(
    r"\b(gun|pistol|weapon|rifle|knife|backpack|bag|helmet|handcuffs?|phone|key|briefcase|"
    r"sign|door|doors|window|glass|car|sedan|taxi|table|chair|"
    r"arma|pistola|faca|mochila|bolsa|capacete|algemas?|celular|chave|maleta|placa|porta|"
    r"portas|janela|vidro|carro|sed[aã]|t[aá]xi|mesa|cadeira)\b", re.IGNORECASE)
_MUDA_ESTADO = re.compile(
    r"\b(explod\w*|shatter\w*|collaps\w*|falls?|crash\w*|catches fire|clos(?:e|es|ing)|"
    r"breaks?|tears? off|removes?|drops?|handcuff\w*|explode|estilha[cç]\w*|desab\w*|despenc\w*|"
    r"cai|fecha\w*|quebra\w*|arranca\w*|algema\w*)\b", re.IGNORECASE)

LIMIAR_MEDIA = 2.5
LIMIAR_COMPLEXA = 4.5


def score_shot(shot: dict, cast: dict) -> dict:
    from script_pipeline.motion_conditioner import infer_primitive

    beat = str(shot.get("beat") or "")
    primitiva = ((shot.get("motion_conditioning") or {}).get("primitive")) or "idle"
    fala = shot.get("line_index") is not None
    fechado = str(shot.get("framing") or "") in ("close", "extreme_close")
    # Close de fala mostra so a REACAO (shot_plan._video_prompt): a acao do texto fica
    # fora do quadro e nao pesa aqui -- mas e sinalizada, porque precisa de plano proprio.
    primitiva_do_texto, _ = infer_primitive(beat, shot.get("subject") or "X",
                                            shot.get("co_subject") or "", framing="medium",
                                            speaking=False)
    reacao = fala and fechado
    pessoas = [p for p in (shot.get("subject"), shot.get("co_subject")) if p]
    if not reacao:
        for nome in personagens_citados(beat, cast):
            if nome not in pessoas:
                pessoas.append(nome)
    objetos = [] if reacao else sorted({m.group(1).lower() for m in _OBJETOS.finditer(beat)})
    muda_estado = (not reacao) and bool(_MUDA_ESTADO.search(beat))
    segundos = float(shot.get("seconds") or 0.0)
    motivos = []
    if reacao and primitiva_do_texto in FISICAS:
        motivos.append(f"acao do texto ({primitiva_do_texto}) fica FORA do quadro neste close de fala")
    score = PESO_PRIMITIVA.get(primitiva, 0.5)
    if primitiva in FISICAS:
        motivos.append(f"acao fisica ({primitiva})")
    if len(pessoas) > 1:
        score += 0.75 * (len(pessoas) - 1)
        motivos.append(f"{len(pessoas)} pessoas ({', '.join(pessoas)})")
    if objetos:
        score += 0.5 * len(objetos)
        motivos.append(f"objetos: {', '.join(objetos)}")
    if muda_estado:
        score += 2.0
        motivos.append("muda o estado do mundo (definir antes/depois)")
    if primitiva in FISICAS and segundos and segundos < 1.8:
        score += 1.0
        motivos.append(f"acao fisica em {segundos:.2f} s")
    if (shot.get("movement") or "static") != "static":
        score += 0.5
    nivel = ("complexa" if score >= LIMIAR_COMPLEXA else "media" if score >= LIMIAR_MEDIA
             else "simples")
    return {"primitiva": primitiva, "primitiva_do_texto": primitiva_do_texto, "pessoas": pessoas,
            "objetos": objetos, "muda_estado": muda_estado, "score": round(score, 2),
            "nivel": nivel, "motivos": motivos, "fala": fala, "fechado": fechado,
            "acao_fora_do_quadro": reacao and primitiva_do_texto in FISICAS,
            "contato": primitiva in CONTATO and len(pessoas) > 1}


def recomendacao(analise: dict, engine: str) -> list[str]:
    """O que fazer ANTES do still/video, e o que o motor escolhido consegue receber."""
    r = []
    if analise["acao_fora_do_quadro"]:
        r.append(f"a acao do texto ({analise['primitiva_do_texto']}) nao aparece neste close de "
                 "fala: confira se um plano vizinho a mostra; senao, a acao se perde")
    if analise["nivel"] == "simples":
        return r + ["still + prompt bastam"]
    if analise["nivel"] == "media":
        r.append("revisar o blocking 2D (shots/blocking_preview.png) e o still: todas as "
                 "pessoas do plano precisam estar nele, com a referencia de cada uma")
    else:
        r.append("PREVIS: bloquear em 3D de baixa resolucao (manequins) antes do still -- "
                 "spatial_pipeline (estado/objetos) ou motion_director (InterGen, dupla em contato)")
        r.append("ou DIVIDIR em acao / reacao / consequencia: cada plano com uma mudanca so")
    if analise["muda_estado"]:
        r.append("registrar o estado novo (destruido, fechado, caido...) para os planos seguintes")
    if analise["fala"] and not analise["fechado"] and analise["primitiva"] in FISICAS:
        r.append("fala + acao fisica no mesmo plano aberto: o lip-sync precisa do rosto e a acao "
                 "do corpo -- prefira acao sem fala e fala em close")
    if engine == "ltx" and analise["contato"]:
        r.append("LTX: guia de pose/profundidade (IC-LoRA union-control) so em plano SEM fala -- "
                 "fator de referencia 2 e recusado junto de audio_conditioning")
    if engine.startswith("minimax") and analise["nivel"] == "complexa":
        r.append("MiniMax: sem controle de pose (ControlNet bloqueado por VRAM) -- o previs so "
                 "entra pelo still; em long-take so o 1o plano do take recebe referencia, "
                 "prefira clipe avulso para este plano")
    if engine == "longcat" and not analise["fala"]:
        r.append("LongCat e so para fala: este plano vai para o LTX")
    return r


def audit(run_dir: Path, *, engine: str = "ltx", log=print) -> dict:
    plan = json.loads((run_dir / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    cast_path = run_dir / "characters" / "cast.json"
    cast = json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.exists() else {}
    planos = []
    for i, shot in enumerate(plan.get("shots") or []):
        a = score_shot(shot, cast)
        a["plano"] = i
        a["framing"] = shot.get("framing")
        a["beat"] = str(shot.get("beat") or "")[:160]
        a["recomendacao"] = recomendacao(a, engine)
        planos.append(a)
    contagem = {n: sum(1 for p in planos if p["nivel"] == n) for n in ("simples", "media", "complexa")}
    relatorio = {"engine": engine, "resumo": contagem,
                 "precisam_previs": [p["plano"] for p in planos if p["nivel"] == "complexa"],
                 "planos": planos}
    out = run_dir / "shots" / "complexity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[complexidade] {contagem['simples']} simples, {contagem['media']} media(s), "
        f"{contagem['complexa']} complexa(s) -> {out}")
    for p in planos:
        if p["nivel"] == "complexa":
            log(f"  plano {p['plano']} ({p['framing']}, score {p['score']}): {'; '.join(p['motivos'])}")
    return relatorio


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", default="ltx",
                    help="motor de video da corrida (muda as recomendacoes): ltx, minimax, "
                         "minimax-longtake, longcat")
    args = ap.parse_args(argv)
    audit(Path(args.run_dir).resolve(), engine=args.engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
