"""Previs 3D automatico: spec espacial gerado da decupagem, manequins no Blender, sem spec manual.

Pedido do usuario (2026-09-27, AUDITORIA_GERAL item 2): as pecas do previs ja existiam -- o
`spatial_pipeline` (WorldStore + Blender com manequins, camera real, profundidade, pose) e o
InterGen (movimento de duas pessoas) --, mas o spec espacial tinha de ser escrito a mao
(`voo702_spatial.py` e o unico exemplo). Este modulo escreve esse spec a partir do que a
decupagem ja decidiu e renderiza em baixa resolucao os planos que a etapa [M] complexidade
marcou como caros de errar.

Entradas (todas da corrida): `parse/shot_plan.json` (enquadramento, sujeito, parceiro, lado de
tela, beat), `characters/cast.json` (quem existe, figurantes, figurino), `parse/motion_plan.json`
(formacao por cena e primitiva por plano) e `shots/complexity_report.json` (nivel por plano).

O que e inferido -- e fica marcado como inferido no relatorio, para revisao:
- a locacao vira um cenario-modelo (rua / interior / interior estreito) com marcos citados no
  texto (carro, predio/entrada, mesas, faixa de pedestres, portas);
- a primitiva de movimento vira mudanca de ESTADO entre o quadro inicial e o final (derrubada:
  o alvo termina de brucos e quem derruba ajoelhado ao lado; algemar: pulsos nas costas e as
  algemas passam de mao; tiro: arma na mao direita e pose de mira; escolta: os dois andam para a
  saida; protecao: o corpo entre o protegido e a ameaca), mantendo a continuidade ENTRE planos
  (quem caiu continua no chao no plano seguinte);
- a camera segue o enquadramento do plano (spatial_planner.camera_for_shot, com a cabeca na pose
  real -- ajoelhado ou no chao a camera mira onde o rosto esta).

O que NAO e representado: explosao, fogo, vidro, fumaca (efeitos, nao corpos) -- vao como nota.
O previs e para bloquear CORPO e CAMERA; o texto do plano continua sendo o que o still pede.

Saidas: `world/previs_spec.json` (o spec; entrada valida para `spatial_pipeline --spec`),
`previs/` (projeto WorldStore com os quadros do Blender), `shots/previs_3d.png` (folha de
contato: quadro inicial, quadro final e esqueleto por plano), `shots/previs_3d.mp4` (movimento
em baixa resolucao, Workbench) e `shots/previs_3d.json` (relatorio com inferencias e avisos).

CLI:
    .venv/Scripts/python.exe -m script_pipeline.previs_spec --run-dir RUN [--niveis complexas]
        [--so-spec] [--sem-movimento] [--stills] [--intergen] [--width 960 --height 544]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline import mannequin_poses as mp  # noqa: E402
from script_pipeline.world_store import digest  # noqa: E402

SCHEMA = "previs-spec/v1"
NIVEIS = {"complexas": {"complexa"}, "medias": {"media", "complexa"},
          "todos": {"simples", "media", "complexa"}}
MOTION_FPS = 12
FFMPEG = shutil.which("ffmpeg") or "C:/ffmpeg/bin/ffmpeg.exe"
PARK = [0.0, 40.0, 0.0]  # onde fica quem ainda nao entrou em cena (fora do quadro, presente=False)
TIGHT = {"close", "extreme_close"}
LYING = {"prone", "prone_cuffed"}
DOWN = LYING | {"crouching", "kneeling"}

# ---------------------------------------------------------------- figurino -> cor do manequim
_CORES = [
    (r"navy|azul[- ]marinho", [.06, .09, .25]), (r"black|pret[oa]", [.05, .05, .06]),
    (r"white|branc[oa]", [.85, .85, .83]), (r"charcoal|gr[ae]y|cinza|silver|prata", [.33, .33, .35]),
    (r"blue|azul", [.10, .22, .55]), (r"red|crimson|burgundy|maroon|vermelh[oa]|bord[oô]", [.55, .06, .06]),
    (r"green|olive|verde|oliva", [.15, .33, .15]), (r"khaki|beige|tan|camel|bege|c[aá]qui", [.60, .52, .38]),
    (r"brown|marrom", [.35, .22, .12]), (r"yellow|amarel[oa]", [.80, .65, .10]),
    (r"orange|laranja", [.85, .40, .08]), (r"purple|violet|rox[oa]", [.35, .12, .45]),
    (r"pink|ros[ae]", [.85, .45, .55]),
]
_NAO_ROUPA = re.compile(r"\b(hair|eyes?|skin|beard|mustache|moustache|lips?|complexion|cabelo|olhos?|"
                        r"pele|barba|bigode|l[aá]bios?)\b", re.IGNORECASE)
PALETA = [[.55, .04, .03], [.03, .10, .46], [.10, .36, .12], [.62, .45, .05], [.36, .10, .42],
          [.05, .38, .40], [.50, .25, .08], [.30, .30, .32]]
PELE_NEUTRA = [.70, .52, .40]  # manequim: a cor da pele nao e inferida do roteiro


def cor_do_figurino(descriptor: str) -> list[float] | None:
    """Primeira cor citada num trecho de ROUPA do descritor (pula cabelo, olhos, pele)."""
    for trecho in re.split(r"[,;.]", descriptor or ""):
        if _NAO_ROUPA.search(trecho):
            continue
        melhor = None
        for padrao, cor in _CORES:
            m = re.search(rf"\b(?:{padrao})\b", trecho, re.IGNORECASE)
            if m and (melhor is None or m.start() < melhor[0]):
                melhor = (m.start(), cor)
        if melhor:
            return list(melhor[1])
    return None


# ---------------------------------------------------------------- objetos e marcos citados
# (regex, tipo, tamanho xyz, cor, segura na mao)
OBJETOS = [
    (r"gun|guns|pistol|handgun|weapon|rifle|firearm|arma|pistola|rev[oó]lver", "GUN", [.05, .22, .14], [.08, .08, .09], True),
    (r"backpack|bag|mochila|bolsa|sacola", "BACKPACK", [.32, .22, .45], [.25, .28, .15], True),
    (r"helmet|capacete", "HELMET", [.26, .30, .24], [.06, .06, .07], True),
    (r"handcuffs?|cuffs|algemas?", "HANDCUFFS", [.16, .05, .08], [.70, .70, .72], True),
    (r"phone|cellphone|smartphone|celular|telefone", "PHONE", [.08, .02, .15], [.05, .05, .05], True),
    (r"briefcase|maleta|pasta executiva", "BRIEFCASE", [.45, .12, .33], [.25, .15, .08], True),
    (r"knife|faca|punhal", "KNIFE", [.03, .25, .04], [.60, .60, .62], True),
    (r"sign|signboard|placa|letreiro", "SIGN", [1.4, .08, .9], [.55, .56, .58], False),
]
MARCOS = {
    "CAR": r"car|cars|sedan|limousine|limo|taxi|convoy|motorcade|vehicle|carro|sed[aã]|t[aá]xi|comboio|carreata|ve[ií]culo",
    "BUILDING": r"hotel|building|buildings|skyscraper|tower|window|windows|glass|facade|pr[eé]dio|edif[ií]cio|arranha-c[eé]u|janelas?|vidro|fachada",
    "ENTRANCE": r"entrance|doors?|lobby|gate|entrada|portas?|port[aã]o|saguão",
    "TABLES": r"tables?|caf[eé]|terrace|mesas?|terra[cç]o",
    "CROSSWALK": r"crossing|crosswalk|zebra|faixa de pedestres?|faixa",
}
_RUA = r"street|avenue|road|boulevard|plaza|square|sidewalk|crossing|parking|highway|rua|avenida|estrada|pra[cç]a|cal[cç]ada|estacionamento|rodovia|exterior|ext\."
_ESTREITO = r"cockpit|cabin|aircraft|airplane|plane|fuselage|train|wagon|van|elevator|corridor|hallway|cabine|avi[aã]o|trem|vag[aã]o|elevador|corredor"
_EFEITOS = re.compile(r"\b(explod\w*|explos\w*|fireball|fire|flames?|smoke|shatter\w*|glass shatters|debris|"
                      r"explo[dsç]\w*|fogo|chamas?|fuma[cç]a|estilha[cç]\w*|destro[cç]os)\b", re.IGNORECASE)
# sem o "some" (sumir) do portugues: casa com o "some" do ingles ("some pedestrians")
_SOME = re.compile(r"\b(disappears?|vanish\w*|flees?|fleeing|escapes?|desaparece|foge|fugindo|sumiu)\b",
                   re.IGNORECASE)
# ACHADO (review adversarial 2026-09-27): "high"/"above"/"upper"/"sky" sozinhos casam em frases
# comuns sem alvo elevado ("crouched above the curb") -- tirados; "alto"/"janela"/"telhado" etc.
# continuam por serem termos concretos de posicao, nao qualificadores genéricos.
_ALTO = re.compile(r"\b(window|windows|roof|rooftop|balcony|sniper|upper floor|upper window|"
                   r"from above|alto|janela|telhado|sacada|atirador|c[eé]u)\b", re.IGNORECASE)
_CORRE = re.compile(r"\b(run\w*|sprint\w*|dash\w*|rush\w*|races?|corre\w*|dispara)\b", re.IGNORECASE)
_LARGA = re.compile(r"\b(leaves?|leaving|drops?|dropping|plants?|planting|abandons?|sets? down|puts? down|"
                    r"deixa|larga|abandona|solta)\b", re.IGNORECASE)
_PEGA = re.compile(r"\b(picks? up|takes?|grabs?|seizes?|pega|agarra|apanha)\b", re.IGNORECASE)
_PUXA = re.compile(r"\b(pulls?|drags?|yanks?|puxa|arrasta)\b", re.IGNORECASE)
_CHUTA = re.compile(r"\b(kicks?|kicking|chuta|chutando)\b", re.IGNORECASE)
_LEVANTA = re.compile(r"\b(helps? \w+ (?:up|to (?:his|her|their) feet)|to (?:his|her|their) feet|gets? up|"
                      r"levanta|ajuda \w+ a (?:se )?levantar|p[oõ]e de p[eé])\b", re.IGNORECASE)
_QUEDA = re.compile(r"\b(falls?|falling|collaps\w*|crash\w* down|cai|desaba|despenca)\b", re.IGNORECASE)
_EMPURRA_CHAO = re.compile(r"\b(pushes|shoves|throws|knocks)\b.*\b(ground|floor|down)\b|\bempurra\b.*\bch[aã]o\b",
                           re.IGNORECASE)


def _tem(padrao: str, texto: str) -> bool:
    return re.search(rf"\b(?:{padrao})\b", texto or "", re.IGNORECASE) is not None


def _objetos_citados(texto: str) -> list[tuple]:
    return [o for o in OBJETOS if _tem(o[0], texto)]


# ---------------------------------------------------------------- utilidades geometricas
def _unit(v):
    n = math.hypot(v[0], v[1])
    return [v[0] / n, v[1] / n] if n > 1e-9 else [0.0, 0.0]


def _perp(v):
    return [-v[1], v[0]]


def _xy_add(p, v, k=1.0):
    return [round(p[0] + v[0] * k, 3), round(p[1] + v[1] * k, 3), p[2]]


def _clamp_stage(p, stage):
    """Mantem no palco e fora dos obstaculos (carro, mesas): quem cairia dentro de um e empurrado
    para a borda mais proxima."""
    (x0, x1), (y0, y1) = stage["bounds"]
    x, y = min(x1, max(x0, p[0])), min(y1, max(y0, p[1]))
    for ox0, ox1, oy0, oy1 in stage.get("obstacles", []):
        if ox0 < x < ox1 and oy0 < y < oy1:
            saidas = [(x - ox0, ox0, y), (ox1 - x, ox1, y), (y - oy0, x, oy0), (oy1 - y, x, oy1)]
            _, x, y = min(saidas)
    return [round(x, 3), round(y, 3), p[2]]


def _read(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- locacao-modelo
def _resolve_interior(flag: str, texto: str) -> bool:
    """`interior` do shot_plan e {"interior","exterior",""} -- "" quando a cabecalho da cena e
    hibrido ou ausente (shot_plan.py, ~110 planos assim nas corridas existentes). ACHADO (review
    adversarial 2026-09-27): tratar "" como exterior perdia cabines/cockpits/camaras estreitas.
    Sem flag, decide pelo TEXTO: narrow primeiro (cockpit etc. e sempre interior), depois rua."""
    valor = str(flag or "").strip().casefold()
    if valor in {"interior", "int", "true", "1"}:
        return True
    if valor in {"exterior", "ext", "false", "0"}:
        return False
    if _tem(_ESTREITO, texto):
        return True
    return not _tem(_RUA, texto)


def build_location(texto: str, interior: bool) -> tuple[dict, dict, list[str]]:
    """Cenario-modelo + marcos citados. Devolve (location, marcos {id: xyz}, inferencias)."""
    boxes, marcos, notas = [], {}, []
    estreito = interior and _tem(_ESTREITO, texto)
    if not interior:
        tipo = "rua" if _tem(_RUA, texto) else "exterior"
        boxes.append({"id": "GROUND", "position": [0, 6, -.1], "size": [40, 40, .2],
                      "color": [.20, .20, .21] if tipo == "rua" else [.30, .33, .24]})
        if tipo == "rua":
            # meio-fio baixo (4 cm): os pes do manequim ficam em z=0 em qualquer ponto do palco
            boxes.append({"id": "SIDEWALK", "position": [0, 8.6, -.06], "size": [40, 3.2, .2],
                          "color": [.46, .45, .43]})
        bounds = ((-5.5, 5.5), (-1.5, 9.4))
        if _tem(MARCOS["BUILDING"], texto) or _tem(MARCOS["ENTRANCE"], texto):
            boxes.append({"id": "BUILDING", "position": [0, 10.7, 4], "size": [40, 1, 8], "color": [.56, .53, .49]})
            marcos["BUILDING"] = [0.0, 10.2, 4.0]
            notas.append("fachada de predio inferida ao fundo (y=10,2 m)")
        if _tem(MARCOS["ENTRANCE"], texto) or _tem(r"hotel|lobby|saguão", texto):
            boxes.append({"id": "ENTRANCE", "position": [0, 10.15, 1.4], "size": [3.2, .12, 2.8],
                          "color": [.10, .11, .12]})
            marcos["ENTRANCE"] = [0.0, 9.9, 0.0]
            notas.append("entrada/portas inferidas no centro da fachada")
        if _tem(MARCOS["CROSSWALK"], texto):
            for i, x in enumerate([-2.5, -1.5, -.5, .5, 1.5, 2.5]):
                boxes.append({"id": f"CROSSWALK_{i}", "position": [x, 2.4, .006], "size": [.5, 4.2, .012],
                              "color": [.86, .86, .84]})
            notas.append("faixa de pedestres inferida sob a acao")
    elif estreito:
        tipo = "interior_estreito"
        boxes += [
            {"id": "FLOOR", "position": [0, 2, -.1], "size": [3.6, 14, .2], "color": [.34, .35, .37]},
            {"id": "LEFT_WALL", "position": [-1.8, 2, 1.2], "size": [.12, 14, 2.4], "color": [.72, .72, .70]},
            {"id": "RIGHT_WALL", "position": [1.8, 2, 1.2], "size": [.12, 14, 2.4], "color": [.72, .72, .70]},
            {"id": "BACK_WALL", "position": [0, 8.5, 1.2], "size": [3.6, .12, 2.4], "color": [.60, .60, .58]},
        ]
        bounds = ((-1.4, 1.4), (-1.0, 8.0))
    else:
        tipo = "interior"
        boxes += [
            {"id": "FLOOR", "position": [0, 2, -.1], "size": [11, 13, .2], "color": [.34, .36, .37]},
            {"id": "BACK_WALL", "position": [0, 6.5, 1.6], "size": [11, .15, 3.2], "color": [.69, .67, .60]},
            {"id": "LEFT_WALL", "position": [-5.5, 2.5, 1.6], "size": [.15, 8, 3.2], "color": [.57, .59, .56]},
            {"id": "RIGHT_WALL", "position": [5.5, 2.5, 1.6], "size": [.15, 8, 3.2], "color": [.57, .59, .56]},
        ]
        bounds = ((-4.8, 4.8), (-1.0, 5.8))
        if _tem(MARCOS["ENTRANCE"], texto):
            boxes.append({"id": "ENTRANCE", "position": [1.8, 6.38, 1.1], "size": [1.2, .12, 2.2],
                          "color": [.10, .11, .12]})
            marcos["ENTRANCE"] = [1.8, 5.9, 0.0]
            notas.append("porta inferida na parede do fundo, a direita")
    obstaculos, vistas = [], []  # pisada (para corpos) e volume (para a linha de visada)
    if _tem(MARCOS["CAR"], texto) and tipo != "interior_estreito":
        # a direita do fundo, paralelo a camera: deixa livre o caminho ate a entrada (x=0)
        boxes += [{"id": "CAR_BODY", "position": [3.4, 6.2, .55], "size": [4.6, 1.9, .8], "color": [.07, .07, .08]},
                  {"id": "CAR_CABIN", "position": [3.2, 6.2, 1.25], "size": [2.6, 1.7, .62], "color": [.12, .13, .15]}]
        marcos["CAR"] = [3.4, 6.2, 0.0]
        obstaculos.append((3.4 - 2.3 - .35, 3.4 + 2.3 + .35, 6.2 - .95 - .35, 6.2 + .95 + .35))
        vistas.append((3.4 - 2.3, 3.4 + 2.3, 6.2 - .95, 6.2 + .95, 1.56))
        notas.append("carro inferido ao fundo, a direita, paralelo a camera (x=3,4 m, y=6,2 m)")
    if _tem(MARCOS["TABLES"], texto) and tipo != "interior_estreito":
        for i, (x, y) in enumerate([(-3.6, 2.6), (3.6, 3.2)]):
            boxes.append({"id": f"TABLE_{i}", "position": [x, y, .37], "size": [.8, .8, .74], "color": [.40, .26, .14]})
            obstaculos.append((x - .7, x + .7, y - .7, y + .7))
            vistas.append((x - .4, x + .4, y - .4, y + .4, .74))
        marcos["TABLES"] = [-3.6, 2.6, 0.0]
        notas.append("mesas inferidas nas laterais")
    return {"boxes": boxes, "template": tipo}, {"marcos": marcos, "bounds": bounds, "tipo": tipo,
                                                 "obstacles": obstaculos, "occluders": vistas}, notas


# ---------------------------------------------------------------- entradas da corrida
def load_inputs(run: Path) -> dict:
    run = Path(run)
    plan = _read(run / "parse" / "shot_plan.json")
    if not plan or not plan.get("shots"):
        raise ValueError(f"{run}/parse/shot_plan.json ausente ou vazio -- rode a decupagem antes")
    cast = _read(run / "characters" / "cast.json", {}) or {}
    motion = _read(run / "parse" / "motion_plan.json")
    if not motion or len(motion.get("commands") or []) != len(plan["shots"]):
        # motion_plan desatualizado (ou ausente): recompila do plano, sem LLM e sem gravar.
        from script_pipeline.motion_conditioner import build_motion_score
        motion = build_motion_score(plan)
    complexity = _read(run / "shots" / "complexity_report.json")
    if complexity and len(complexity.get("planos") or []) == len(plan["shots"]):
        niveis = [p.get("nivel", "simples") for p in complexity["planos"]]
        scores = [p.get("score", 0) for p in complexity["planos"]]
    else:
        from script_pipeline.shot_complexity import score_shot
        analises = [score_shot(s, cast) for s in plan["shots"]]
        niveis = [a["nivel"] for a in analises]
        scores = [a["score"] for a in analises]
    commands = {c["shot_index"]: c for c in motion.get("commands") or []}
    formations = {sc.get("scene"): sc.get("formation") or {} for sc in motion.get("scenes") or []}
    return {"plan": plan, "cast": cast, "commands": commands, "formations": formations,
            "niveis": niveis, "scores": scores}


def _entity_ids(plan: dict, cast: dict) -> list[str]:
    from script_pipeline.cast_characters import personagens_citados
    nomes = []
    for shot in plan["shots"]:
        for nome in (shot.get("subject"), shot.get("co_subject"),
                     *personagens_citados(shot.get("beat") or "", cast)):
            if nome and nome not in nomes:
                nomes.append(nome)
    return nomes


def build_entities(plan: dict, cast: dict) -> tuple[dict, list[str]]:
    entities, notas = {}, []
    for i, nome in enumerate(_entity_ids(plan, cast)):
        info = cast.get(nome) or {}
        cor = cor_do_figurino(info.get("descriptor") or "")
        if cor is None:
            cor = PALETA[i % len(PALETA)]
            notas.append(f"{nome}: cor do figurino nao encontrada no descritor -- cor de paleta")
        entities[nome] = {"kind": "extra" if info.get("extra") else "character", "position": list(PARK),
                          "yaw": 0.0, "pose": "standing", "wardrobe_id": f"{nome}_W1", "color": cor,
                          "skin": list(PELE_NEUTRA), "present": False}
    return entities, notas


# ---------------------------------------------------------------- coreografia (estado por plano)
SEM_MUDANCA = {"speak", "gesture", "idle", "environment"}
# poses de ACAO: quem e citado num plano novo volta a ficar em pe no corte; deitado e ajoelhado
# sao ESTADO (continuam ate uma acao mudar -- levantar, ajudar a levantar)
TRANSITORIAS = {"running", "walking", "aiming", "aiming_high", "shielding", "reaching", "escorting",
                "crouching"}
COM_PARCEIRO = {"takedown", "restrain", "protect", "escort", "contact", "push_down"}


class Choreographer:
    """Simula o mundo plano a plano. `step(i)` devolve o estado no inicio e no fim do plano i;
    a continuidade entre planos vem de o estado final de um ser o inicio do seguinte.

    Plano de FALA em close mostra so a reacao (shot_plan._video_prompt), mas a acao que o texto
    descreve acontece na historia: ela e aplicada ao mundo FORA do quadro, depois do plano, e o
    plano seguinte ja comeca com o resultado (quem foi empurrado ao chao continua la)."""

    def __init__(self, inputs: dict, entities: dict):
        self.plan, self.cast = inputs["plan"], inputs["cast"]
        self.commands, self.formations = inputs["commands"], inputs["formations"]
        self.world = copy.deepcopy(entities)
        self.scene = None
        self.stage = None
        self.location = None
        self.location_id = None
        self.scene_notes: list[str] = []
        self.heading = None
        self.threat = None
        self.fugitive = None

    # -- helpers
    def _prop(self, obj: tuple, owner: str | None) -> str:
        pid = f"{obj[1]}_{owner}" if owner and obj[4] else obj[1]
        if pid not in self.world:
            self.world[pid] = {"kind": "prop", "position": list(PARK), "yaw": 0.0, "color": list(obj[3]),
                               "size": list(obj[2]), "present": False}
        return pid

    def _face(self, eid, target_xy, world=None):
        e = (world or self.world)[eid]
        d = [target_xy[0] - e["position"][0], target_xy[1] - e["position"][1]]
        if math.hypot(*d) > 1e-6:
            e["yaw"] = round(mp.facing_yaw(d), 2)

    def _clamp(self, p):
        return _clamp_stage(p, self.stage)

    def _anchor(self, eid) -> list[float]:
        # ACHADO (review adversarial 2026-09-27): usar a ancora do motion_plan (ou contar
        # "quantos ja estao presentes") sem checar colisao empilhava gente no mesmo ponto assim
        # que o palco clampava a profundidade -- `_formation()` do motion_conditioner poe TODO
        # mundo sem `screen_sides` explicito no mesmo lado ("left"), e o 5o+ nome excedia o
        # limite em Y e colava na borda. Agora a ancora do motion_plan e so a PRIMEIRA candidata;
        # se colidir com quem ja esta em cena, cai numa grade (profundidade x os dois lados) ate
        # achar um ponto livre.
        ocupados = [v["position"][:2] for v in self.world.values()
                   if v["kind"] != "prop" and v.get("present")]
        candidatos = []
        info = (self.formations.get(self.scene) or {}).get(eid)
        if info and info.get("anchor_m"):
            ax, ay = info["anchor_m"]
            candidatos.append([float(ax), 1.0 + float(ay), 0.0])
        candidatos += [[lado, 1.0 + 1.35 * linha, 0.0] for linha in range(8) for lado in (-1.2, 1.2)]
        for bruto in candidatos:
            candidato = self._clamp(bruto)
            if all(math.dist(candidato[:2], o) >= .6 for o in ocupados):
                return candidato
        return self._clamp([0.0, 1.0, 0.0])

    def _enter(self, eid):
        """Quem e citado no plano entra em cena (fora de quadro, no corte) na sua ancora."""
        e = self.world.get(eid)
        if not e or e["kind"] == "prop" or e.get("present"):
            return
        e["position"] = self._anchor(eid)
        e["present"] = True
        e["pose"] = "standing"
        # 3/4 para a camera, voltado para o centro -- a "trapaca" de palco classica.
        e["yaw"] = round(mp.facing_yaw([-1.0 if e["position"][0] > 0 else 1.0, -1.0]), 2)

    def _begin_scene(self, shot):
        scene = shot.get("scene")
        if scene == self.scene:
            return
        self.scene = scene
        beats = [str(s.get("beat") or "") for s in self.plan["shots"] if s.get("scene") == scene]
        texto = " ".join([str(shot.get("location") or "")] + beats)
        interior = _resolve_interior(shot.get("interior"), texto)
        location, stage, notas = build_location(texto, interior)
        self.location, self.stage = location, stage
        self.location_id = "LOC_" + digest({"location": shot.get("location"), "boxes": location["boxes"]})[:10]
        self.scene_notes = notas
        self.heading = None
        self.fugitive = None
        # Ameaca da cena (para quem protege): o carro que explode, ou o predio de onde atiram.
        self.threat = None
        marcos = stage["marcos"]
        if "CAR" in marcos and any(_EFEITOS.search(b) and _tem(MARCOS["CAR"], b) for b in beats):
            self.threat = ("carro que explode", marcos["CAR"])
        elif "BUILDING" in marcos and any(_ALTO.search(b) and re.search(r"\b(shot|shoots?|fires?|bullet|"
                                                                       r"sniper|tiro|atira|bala)\b", b, re.I)
                                          for b in beats):
            self.threat = ("atirador no predio", marcos["BUILDING"])
        # Corte de cena: ninguem continua no palco; cada um entra quando for citado.
        for e in self.world.values():
            if e.get("present") or e.get("attachment"):
                e["present"] = False
                e.pop("attachment", None)
                e["position"] = list(PARK)
                if e["kind"] != "prop":
                    e["pose"] = "standing"

    def _landmark_for(self, texto: str):
        for nome in ("CAR", "ENTRANCE", "BUILDING", "TABLES"):
            if nome in self.stage["marcos"] and _tem(MARCOS[nome], texto):
                return nome, self.stage["marcos"][nome]
        return None, None

    def _travel_heading(self, eid):
        """Direcao de deslocamento da cena: definida no primeiro deslocamento e mantida, para a
        perseguicao nao inverter o eixo de um plano para o outro. Se alguem foge (visto antes na
        cena), a perseguicao vai na direcao dele."""
        if self.heading is None:
            f = self.world.get(self.fugitive or "")
            if f and f.get("present") and self.fugitive != eid:
                dx = f["position"][0] - self.world[eid]["position"][0]
                self.heading = [1.0 if dx >= 0 else -1.0, 0.0]
            else:
                self.heading = [1.0 if self.world[eid]["position"][0] <= 0 else -1.0, 0.0]
        return self.heading

    def _kneel_beside(self, actor, target, came_from_xy, pose="kneeling"):
        """Quem imobiliza fica ao lado do tronco de quem esta no chao, virado para ele; se o lado
        ja estiver ocupado (outro agente), usa o lado oposto."""
        t = self.world[target]
        heading = mp.facing_vector(t["yaw"])  # de brucos: a cabeca aponta para o olhar
        side = _perp(heading)
        rel = [came_from_xy[0] - t["position"][0], came_from_xy[1] - t["position"][1]]
        if side[0] * rel[0] + side[1] * rel[1] < 0:
            side = [-side[0], -side[1]]
        torso = _xy_add(t["position"], heading, .35)
        pos = self._clamp(_xy_add(torso, side, .6))
        ocupado = any(math.dist(v["position"][:2], pos[:2]) < .5 for k, v in self.world.items()
                      if k not in (actor, target) and v["kind"] != "prop" and v.get("present"))
        if ocupado:
            pos = self._clamp(_xy_add(torso, side, -.6))
        a = self.world[actor]
        a["position"] = pos
        a["pose"] = pose
        self._face(actor, torso)

    def _knock_down(self, actor, target, *, cuffed=False, actor_pose="kneeling"):
        a, t = self.world[actor], self.world[target]
        came_from = list(a["position"])
        if t.get("pose") not in LYING:
            d = _unit([t["position"][0] - a["position"][0], t["position"][1] - a["position"][1]])
            if d == [0.0, 0.0]:
                d = [0.0, 1.0]
            t["position"] = self._clamp(_xy_add(t["position"], d, .3))
            t["yaw"] = round(mp.facing_yaw(d), 2)  # cai para a frente, para longe de quem empurra
        t["pose"] = "prone_cuffed" if cuffed else "prone"
        self._kneel_beside(actor, target, came_from, pose=actor_pose)

    def _text_primitive(self, beat: str, a: str) -> tuple[str, str]:
        """Primitiva e parceiro lidos do TEXTO (plano de fala perdeu o co-sujeito na decupagem)."""
        from script_pipeline.cast_characters import personagens_citados
        from script_pipeline.motion_conditioner import infer_primitive
        outros = [n for n in personagens_citados(beat, self.cast) if n != a and n in self.world]
        parceiro = outros[0] if outros else ""
        if parceiro and _EMPURRA_CHAO.search(beat):
            return "push_down", parceiro
        prim, _ = infer_primitive(beat, a or "X", parceiro, framing="medium", speaking=False)
        return prim, parceiro

    # -- mudancas no CORTE (pose de partida da acao), aplicadas ao mundo antes do quadro inicial
    def _pre(self, primitive, a, p, beat, inferred):
        if primitive in ("takedown", "push_down") and a and p:
            self.world[a]["pose"] = "running" if primitive == "takedown" else self.world[a]["pose"]
            self._face(a, self.world[p]["position"])
        elif primitive in ("speak", "contact") and a and p:
            self._face(a, self.world[p]["position"])
            self._face(p, self.world[a]["position"])
        elif primitive == "fire" and a:
            gun = self._prop(next(o for o in OBJETOS if o[1] == "GUN"), a)
            self.world[gun].update(present=True, attachment={"entity_id": a, "socket": "right_hand"})
            alto = bool(_ALTO.search(beat))
            self.world[a]["pose"] = "aiming_high" if alto else "aiming"
            if alto:
                self.world[a]["yaw"] = 180.0  # alvo alto, ao fundo (janela, telhado)
                inferred.append(f"{a} mira para o alto, voltado para o fundo do cenario")
            else:
                ax = self.world[a]["position"][0]
                self.world[a]["yaw"] = round(mp.facing_yaw([1.0 if ax <= 0 else -1.0, .35]), 2)
                inferred.append(f"{a} mira para o lado oposto da tela (alvo nao dito no texto)")
        # objeto que o sujeito vai largar: esta na mao dele desde o inicio
        if a and _LARGA.search(beat):
            for obj in _objetos_citados(beat):
                if obj[4]:
                    pid = self._prop(obj, a)
                    if not self.world[pid].get("present"):
                        self.world[pid].update(present=True, attachment={"entity_id": a, "socket": "right_hand"})
        # objeto grande que vai cair: comeca no alto, perto do sujeito
        if _QUEDA.search(beat):
            for obj in _objetos_citados(beat):
                if not obj[4]:
                    pid = self._prop(obj, None)
                    base = self.world[a]["position"] if a else [0.0, 3.0, 0.0]
                    self.world[pid].update(present=True, position=[round(base[0] + .4, 3),
                                                                   round(base[1] + .9, 3), 3.2], yaw=0.0)
                    self.world[pid].pop("attachment", None)
        # arma chutada: esta no chao, junto de quem foi dominado
        if _CHUTA.search(beat) and p and any(o[1] == "GUN" for o in _objetos_citados(beat)):
            gun = self._prop(next(o for o in OBJETOS if o[1] == "GUN"), p)
            g = self.world[gun]
            if not g.get("present") or g.get("attachment"):
                heading = mp.facing_vector(self.world[p]["yaw"])
                pos = _xy_add(self.world[p]["position"], heading, .95)
                g.update(present=True, position=[pos[0], pos[1], .03])
                g.pop("attachment", None)

    # -- o que muda DURANTE o plano (sobre self.world; `start` recebe ajustes do quadro inicial)
    def _act(self, primitive, a, p, beat, cmd, start, notes, inferred):
        nome_marco, marco = self._landmark_for(beat)
        if primitive in SEM_MUDANCA or not a:
            return
        if primitive in COM_PARCEIRO and not p:
            notes.append(f"primitiva '{primitive}' sem parceiro resolvido: nada a encenar")
            return
        w = self.world
        if primitive == "turn":
            if p:
                self._face(a, w[p]["position"])
            elif _ALTO.search(beat):
                w[a]["yaw"] = 180.0
                inferred.append(f"{a} olha para o fundo (alvo alto/distante)")
        elif primitive == "reach":
            w[a]["pose"] = "reaching"
            if p:
                self._face(a, w[p]["position"])
                if _PUXA.search(beat):
                    d = _unit([w[a]["position"][0] - w[p]["position"][0], w[a]["position"][1] - w[p]["position"][1]])
                    w[p]["position"] = self._clamp(_xy_add(w[p]["position"], d, .8))
                    inferred.append(f"{p} puxado 0,8 m na direcao de {a}")
            if _LARGA.search(beat):
                for obj in _objetos_citados(beat):
                    pid = f"{obj[1]}_{a}"
                    if pid in w and (w[pid].get("attachment") or {}).get("entity_id") == a:
                        f = mp.facing_vector(w[a]["yaw"])
                        pos = _xy_add(w[a]["position"], f, .6)
                        w[pid].pop("attachment", None)
                        w[pid]["position"] = [pos[0], pos[1], round(obj[2][2] / 2, 3)]
                        inferred.append(f"{obj[1].lower()} largado no chao a 0,6 m a frente de {a}")
            elif _PEGA.search(beat):
                for obj in _objetos_citados(beat):
                    if obj[4]:
                        pid = self._prop(obj, a)
                        w[pid].update(present=True, attachment={"entity_id": a, "socket": "right_hand"})
            if _SOME.search(beat):
                w[a]["present"] = False
                notes.append(f"{a} sai de cena no fim do plano (o quadro final ainda o mostra)")
        elif primitive in ("locomote", "pursue"):
            fugitivo = self.fugitive if primitive == "pursue" and self.fugitive not in (a, p) else None
            corre = primitive == "pursue" or bool(_CORRE.search(beat))
            if marco is not None and primitive == "locomote":
                lado = -1.6 if w[a]["position"][0] <= marco[0] else 1.6
                dest = [marco[0] + lado, marco[1] - 1.6, 0.0]
                vet = [dest[0] - w[a]["position"][0], dest[1] - w[a]["position"][1]]
                d, dist = _unit(vet), min(2.2, math.hypot(*vet))
                inferred.append(f"{a} vai na direcao de {nome_marco.lower()}")
            else:
                d = self._travel_heading(a)
                dist = 3.0 if primitive == "pursue" else 1.8
                inferred.append(f"{a} atravessa o quadro no eixo da cena ({'+' if d[0] > 0 else '-'}X)")
            if fugitivo and w.get(fugitivo, {}).get("present"):
                start[fugitivo]["pose"] = w[fugitivo]["pose"] = "running"
                start[fugitivo]["yaw"] = w[fugitivo]["yaw"] = round(mp.facing_yaw(d), 2)
                w[fugitivo]["position"] = self._clamp(_xy_add(w[fugitivo]["position"], d, dist))
                inferred.append(f"{fugitivo} foge a frente, no mesmo eixo")
            for eid in [a] + ([p] if p else []):
                start[eid]["pose"] = w[eid]["pose"] = "running" if corre else "walking"
                start[eid]["yaw"] = w[eid]["yaw"] = round(mp.facing_yaw(d), 2)
                w[eid]["position"] = self._clamp(_xy_add(w[eid]["position"], d, dist))
        elif primitive == "contact":
            dest = (cmd.get("target") or {}).get("destination")
            if dest and cmd.get("actor") == a:
                w[a]["position"] = self._clamp([float(dest[0]), 1.0 + float(dest[1]), 0.0])
            else:
                d = _unit([w[p]["position"][0] - w[a]["position"][0], w[p]["position"][1] - w[a]["position"][1]])
                w[a]["position"] = self._clamp(_xy_add(w[p]["position"], d, -.75))
            w[a]["pose"] = "reaching"
            self._face(a, w[p]["position"])
            self._face(p, w[a]["position"])
        elif primitive in ("takedown", "restrain", "push_down"):
            algema = _tem(OBJETOS[3][0], beat)
            cuffed = primitive == "restrain" and (algema or _tem(r"restrains?|rende", beat))
            chutada = f"GUN_{p}" if f"GUN_{p}" in w and w[f"GUN_{p}"].get("present") \
                and not w[f"GUN_{p}"].get("attachment") and _CHUTA.search(beat) else None
            self._knock_down(a, p, cuffed=cuffed, actor_pose="crouching" if primitive == "push_down" else "kneeling")
            if primitive == "restrain" and algema:
                cuffs = self._prop(OBJETOS[3], None)
                start[cuffs] = dict(copy.deepcopy(w[cuffs]), present=True,
                                    attachment={"entity_id": a, "socket": "right_hand"})
                w[cuffs].update(present=True, attachment={"entity_id": p, "socket": "right_hand"})
                inferred.append(f"algemas passam da mao de {a} para os pulsos de {p}")
            if chutada:
                heading = mp.facing_vector(w[p]["yaw"])
                pos = self._clamp(_xy_add(w[chutada]["position"], _perp(heading), 1.6))
                w[chutada]["position"] = [pos[0], pos[1], .03]
                inferred.append(f"arma de {p} chutada 1,6 m para o lado")
            if primitive == "push_down":
                inferred.append(f"{p} empurrado ao chao; {a} agachado cobrindo")
            else:
                inferred.append(f"{p} termina de brucos; {a} ajoelhado ao lado")
        elif primitive == "protect":
            alvo = w[p]
            if marco is not None:
                origem, nome = marco, nome_marco.lower()
            elif self.threat:
                nome, origem = self.threat
            else:
                origem, nome = None, None
            if origem is not None:
                ameaca = _unit([origem[0] - alvo["position"][0], origem[1] - alvo["position"][1]])
                inferred.append(f"ameaca vem de: {nome}")
            else:
                ameaca = [0.0, -1.0]
                inferred.append("ameaca nao dita no texto: assumida do lado da camera")
            w[a]["position"] = self._clamp(_xy_add(alvo["position"], ameaca, .5))
            w[a]["pose"] = "shielding"
            w[a]["yaw"] = round(mp.facing_yaw(ameaca), 2)
            if alvo.get("pose") not in LYING:
                alvo["pose"] = "crouching"
                alvo["yaw"] = round(mp.facing_yaw([-ameaca[0], -ameaca[1]]), 2)
        elif primitive == "escort":
            alvo = w[p]
            if alvo.get("pose") in DOWN and (_LEVANTA.search(beat) or alvo.get("pose") in LYING):
                self._face(a, alvo["position"])
                start[a]["yaw"] = w[a]["yaw"]
                w[a]["pose"] = "reaching"
                alvo["pose"] = "standing"
                self._face(p, w[a]["position"])
                inferred.append(f"{a} ajuda {p} a levantar")
            else:
                if "ENTRANCE" in self.stage["marcos"]:
                    saida = self.stage["marcos"]["ENTRANCE"]
                    inferred.append("rota de fuga: a entrada do predio")
                else:
                    saida = [alvo["position"][0], alvo["position"][1] + 4, 0.0]
                    inferred.append("rota de fuga nao dita: para o fundo do cenario")
                d = _unit([saida[0] - alvo["position"][0], saida[1] - alvo["position"][1]])
                if d == [0.0, 0.0]:
                    d = [0.0, 1.0]
                lado = _perp(d)
                start[a]["position"] = self._clamp(_xy_add(_xy_add(start[p]["position"], d, -.35), lado, .45))
                for eid, pose in ((a, "escorting"), (p, "walking")):
                    start[eid]["pose"] = w[eid]["pose"] = pose
                    start[eid]["yaw"] = w[eid]["yaw"] = round(mp.facing_yaw(d), 2)
                w[a]["position"] = self._clamp(_xy_add(start[a]["position"], d, 2.0))
                alvo["position"] = self._clamp(_xy_add(start[p]["position"], d, 2.0))

    def _falls(self, a, beat, inferred):
        """Objeto grande que cai (placa): do alto ao chao, em pe, ao lado de quem e ameacado."""
        if not _QUEDA.search(beat):
            return
        for obj in _objetos_citados(beat):
            pid = obj[1]
            if not obj[4] and pid in self.world and self.world[pid].get("present"):
                base = self.world[a]["position"] if a else [0.0, 3.0, 0.0]
                self.world[pid]["position"] = [round(base[0] + .4, 3), round(base[1] + .5, 3),
                                               round(obj[2][2] / 2, 3)]
                inferred.append(f"{pid.lower()} cai do alto e termina no chao ao lado de {a or 'centro'}")

    # -- um plano
    def step(self, index: int) -> dict:
        from script_pipeline.cast_characters import personagens_citados
        shot = self.plan["shots"][index]
        self._begin_scene(shot)
        cmd = self.commands.get(index) or {}
        beat = str(shot.get("beat") or cmd.get("beat_source") or "")
        subject, partner = shot.get("subject") or "", shot.get("co_subject") or ""
        primitive = (shot.get("motion_conditioning") or {}).get("primitive") or cmd.get("primitive") or "idle"
        speaking = shot.get("line_index") is not None or primitive == "speak"
        notes, inferred = [], []
        for nome in (subject, partner, *personagens_citados(beat, self.cast)):
            if nome in self.world:
                self._enter(nome)
                if self.world[nome].get("pose") in TRANSITORIAS:
                    self.world[nome]["pose"] = "standing"
        a = subject if subject in self.world else ""
        p = partner if partner in self.world and partner != a else ""
        if _SOME.search(beat):
            # quem foge: o parceiro de quem avista/persegue; o proprio sujeito se ele sai de cena
            outros = [n for n in personagens_citados(beat, self.cast) if n in self.world and n != a]
            self.fugitive = p or (outros[0] if outros else a) if primitive in ("turn", "pursue", "speak")                 else (a or self.fugitive)
        if not a and _QUEDA.search(beat):
            outros = [n for n in personagens_citados(beat, self.cast) if n in self.world]
            a = outros[0] if outros else ""
        if speaking:
            if a and p:
                self._pre("speak", a, p, beat, inferred)
            start = copy.deepcopy(self.world)
            end = copy.deepcopy(self.world)
            prim_txt, p_txt = self._text_primitive(beat, a)
            if a and prim_txt not in SEM_MUDANCA and prim_txt != "turn":
                self._pre(prim_txt, a, p_txt, beat, inferred)
                self._act(prim_txt, a, p_txt, beat, {}, copy.deepcopy(self.world), notes, inferred)
                self._falls(a, beat, inferred)
                inferred.append(f"acao do texto ({prim_txt}) acontece fora do quadro; o plano "
                                "seguinte ja comeca com o resultado")
            return {"start": start, "end": end, "primitive": primitive, "world_primitive": prim_txt,
                    "subject": a, "partner": p, "notes": notes, "inferred": inferred, "beat": beat}
        prim = primitive
        if p and _EMPURRA_CHAO.search(beat) and prim not in ("takedown", "restrain"):
            prim = "push_down"
        self._pre(prim, a, p, beat, inferred)
        start = copy.deepcopy(self.world)
        self._act(prim, a, p, beat, cmd, start, notes, inferred)
        self._falls(a, beat, inferred)
        if _EFEITOS.search(beat):
            notes.append("efeito (explosao/fogo/vidro/fumaca) nao representado no previs -- so corpos e camera")
        return {"start": start, "end": copy.deepcopy(self.world), "primitive": primitive, "world_primitive": prim,
                "subject": a, "partner": p, "notes": notes, "inferred": inferred, "beat": beat}


# ---------------------------------------------------------------- diffs -> operacoes do WorldStore
def diff_ops(old: dict, new: dict) -> list[dict]:
    ops = []
    for eid in sorted(new):
        a, b = old.get(eid), new[eid]
        if a is None:
            raise ValueError(f"entidade nova no meio do spec: {eid}")
        for field in ("present", "pose", "yaw", "position"):
            if b["kind"] == "prop" and field in ("pose",):
                continue
            if field == "position" and b["kind"] == "prop" and b.get("attachment"):
                continue
            va, vb = a.get(field), b.get(field)
            if va != vb:
                ops.append({"op": "set", "entity_id": eid, "field": field, "expected": copy.deepcopy(va),
                            "value": copy.deepcopy(vb)})
        att_a, att_b = a.get("attachment"), b.get("attachment")
        if b["kind"] == "prop" and att_a != att_b:
            if att_b:
                ops.append({"op": "transfer", "entity_id": eid, "from": (att_a or {}).get("entity_id"),
                            "to": att_b["entity_id"], "socket": att_b["socket"]})
            else:
                ops.append({"op": "detach", "entity_id": eid, "from": att_a["entity_id"],
                            "position": copy.deepcopy(b["position"])})
    # detach grava a posicao; um set de posicao do mesmo prop depois dele teria `expected` velho
    detached = {o["entity_id"] for o in ops if o["op"] == "detach"}
    return [o for o in ops if not (o["op"] == "set" and o["field"] == "position" and o["entity_id"] in detached)]


# ---------------------------------------------------------------- camera por plano
def base_camera(width: int, height: int) -> dict:
    return {"position": [0.0, -5.6, 1.65], "target": [0.0, 2.4, 1.1], "lens_mm": 30, "sensor_mm": 36,
            "width": int(width), "height": int(height), "near": .05, "far": 80}


LENTE = {"medium": 40, "ots": 45, "full": 32, "wide": 26}
UPPER = [mp.NOSE, mp.NECK, mp.R_SHOULDER, mp.R_ELBOW, mp.R_WRIST, mp.L_SHOULDER, mp.L_ELBOW,
         mp.L_WRIST, mp.R_HIP, mp.L_HIP, mp.R_EAR, mp.L_EAR]


def _rot2(v, graus):
    a = math.radians(graus)
    return [v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)]


def fit_camera(points: list[list[float]], axis_xy: list[float], lens: float, width: int, height: int,
               *, margin: float = .82, min_dist: float = 1.3) -> dict:
    """Camera na direcao `axis_xy` (do centro da acao PARA a camera) que enquadra todos os
    `points` (juntas no inicio e no fim do plano) com folga. Resolve a distancia minima em que
    cada ponto cabe no campo horizontal e vertical da lente."""
    n = _unit(axis_xy)
    if n == [0.0, 0.0]:
        n = [0.0, -1.0]
    xs, ys, zs = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
    center = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2]
    fw = [-n[0], -n[1]]
    right = [fw[1], -fw[0]]  # fw x cima, no plano do chao (mesma convencao de camera_geometry.project)
    tan_h = 18.0 / lens
    tan_v = tan_h * height / width
    dist = min_dist
    for p in points:
        rel = [p[0] - center[0], p[1] - center[1], p[2] - center[2]]
        x = abs(rel[0] * right[0] + rel[1] * right[1])
        depth = rel[0] * fw[0] + rel[1] * fw[1]
        dist = max(dist, x / (tan_h * margin) - depth, abs(rel[2]) / (tan_v * margin) - depth)
    target = [round(center[0], 3), round(center[1], 3), round(center[2], 3)]
    position = [round(center[0] + n[0] * dist, 3), round(center[1] + n[1] * dist, 3),
                round(center[2] + .15, 3)]
    return {"position": position, "target": target, "lens_mm": lens, "sensor_mm": 36,
            "width": int(width), "height": int(height), "near": .05, "far": 80}


def _linha_bloqueada(camera, ponto, occluders, samples=40) -> bool:
    cx, cy, cz = camera["position"]
    tx, ty, tz = ponto
    for i in range(1, samples):
        t = i / samples
        x, y, z = cx + (tx - cx) * t, cy + (ty - cy) * t, cz + (tz - cz) * t
        for x0, x1, y0, y1, top in occluders:
            if x0 < x < x1 and y0 < y < y1 and z < top:
                return True
    return False


def occluded(camera: dict, occluders: list, points: list[list[float]] | None = None,
             limite: float = .2) -> bool:
    """Um marco (carro, mesa) esconde parte relevante dos corpos? Testa a linha da camera ate
    cabeca, quadril, joelhos e pes de cada um (inicio e fim); bloqueado se mais de `limite`
    dessas linhas atravessa o volume de um marco. Sem `points`, so a linha ate o alvo."""
    pts = points or [camera["target"]]
    bloqueadas = sum(_linha_bloqueada(camera, p, occluders) for p in pts)
    return bloqueadas > (limite * len(pts) if points else 0)


def _pontos_chave(states: list[dict], ids: list[str]) -> list[list[float]]:
    pts = []
    for state in states:
        for i in ids:
            e = state.get(i)
            if e and e.get("present") and e["kind"] != "prop":
                j = mp.world_joints(e)
                pts.append(mp.head_center(j))
                pts.append([(j[mp.R_HIP][k] + j[mp.L_HIP][k]) / 2 for k in range(3)])
                pts.extend(j[k] for k in (mp.R_KNEE, mp.L_KNEE, mp.R_ANKLE, mp.L_ANKLE))
    return pts


def _pontos(states: list[dict], ids: list[str], so_tronco: bool) -> list[list[float]]:
    pts = []
    for state in states:
        for i in ids:
            e = state.get(i)
            if not e or not e.get("present"):
                continue
            if e["kind"] == "prop":
                if not e.get("attachment"):
                    pts.append(list(e["position"]))
                continue
            joints = mp.world_joints(e)
            if so_tronco:
                pts.extend(joints[j] for j in UPPER)
            else:
                pts.extend(joints)
            # ACHADO (review adversarial 2026-09-27): a malha da cabeca (blender_scene_worker)
            # vai ~0,16-0,18 m acima do centro entre as orelhas -- sem este ponto, o medio/aberto
            # cortava o topo da cabeca em 40/40 casos sinteticos (so o close ja tinha o "topo").
            centro = mp.head_center(joints)
            pts.append([centro[0], centro[1], centro[2] + .18])
    return pts


def camera_for(shot: dict, step: dict, active: list[str], base: dict, stage: dict,
               screen_sides: dict | None = None) -> tuple[dict, list[str]]:
    """Camera por enquadramento. Close: `spatial_planner.camera_for_shot` (o rosto na pose real,
    validado). Medio/aberto: enquadra o INICIO e o FIM da acao de todos os corpos ativos --
    numa acao a dois, de lado para o eixo da acao, com o sujeito no lado de tela do plano
    (regra dos 180 graus); nunca no eixo de uma arma apontada."""
    from script_pipeline.spatial_planner import camera_for_shot
    framing = str(shot.get("framing") or "wide").casefold()
    start, end = step["start"], step["end"]
    subject, partner = step["subject"], step["partner"]
    notes = []
    w, h = base["width"], base["height"]
    pessoas = [i for i in active if start[i]["kind"] != "prop"]
    if framing == "insert":
        # o detalhe e o que o TEXTO cita: objeto citado, senao marco citado, senao o primeiro marco
        citados = {o[1] for o in _objetos_citados(step["beat"])}
        props = [i for i in active if start[i]["kind"] == "prop" and i.split("_")[0] in citados]
        marco_citado = next((m for m in ("BUILDING", "ENTRANCE", "CAR", "TABLES")
                             if m in stage["marcos"] and _tem(MARCOS[m], step["beat"])), None)
        if props:
            e = start[props[0]] if start[props[0]].get("present") else end[props[0]]
            t = list(e["position"])
        elif marco_citado or stage["marcos"]:
            nome = marco_citado or next(iter(stage["marcos"]))
            t = stage["marcos"][nome]
            t = [t[0], t[1] - .5, max(1.2, t[2] * .4)]
            notes.append(f"insert sem objeto encenavel: camera no marco {nome.lower()}"
                         + ("" if marco_citado else " (nenhum marco citado no texto)"))
        else:
            t = [0.0, 3.0, 1.0]
        cam = dict(copy.deepcopy(base), target=[float(x) for x in t],
                   position=[t[0] + .3, t[1] - 2.4, t[2] + .35], lens_mm=50)
        return cam, notes
    if not pessoas:
        return copy.deepcopy(base), notes
    ent = start.get(subject) if subject else None
    aiming = bool(ent) and ent.get("pose") in ("aiming", "aiming_high")
    if framing in TIGHT and subject:
        f = mp.facing_vector(ent["yaw"])
        if aiming:
            # nunca no cano: 50 graus para o lado da camera padrao
            cands = [_rot2(f, 50), _rot2(f, -50)]
            axis = min(cands, key=lambda v: v[1])
            notes.append(f"{subject} mirando: camera a 50 graus do eixo da arma")
        elif f[1] > .2:
            axis = f
            notes.append(f"{subject} de costas para o eixo padrao: camera do lado do rosto")
        else:
            axis = _unit([f[0] * .55, f[1] * .55 - .45])
        if axis == [0.0, 0.0]:
            axis = [0.0, -1.0]
        # Do alto da cabeca aos ombros (o que o plano de close pede ao still), na lente do close
        # do planner. A distancia fixa do planner (1,45 m) cortava a cabeca em 16:9.
        j = mp.world_joints(ent)
        topo = mp.head_center(j)
        topo = [topo[0], topo[1], topo[2] + .22]
        if framing == "extreme_close":
            pts = [topo, j[mp.NOSE], j[mp.R_EAR], j[mp.L_EAR], [j[mp.NOSE][0], j[mp.NOSE][1], j[mp.NOSE][2] - .12]]
            lente = 85
        else:
            pts = [topo, j[mp.NOSE], j[mp.R_EAR], j[mp.L_EAR], j[mp.R_SHOULDER], j[mp.L_SHOULDER], j[mp.NECK]]
            lente = 72
        cam = fit_camera(pts, axis, lente, w, h, margin=.9, min_dist=.6)
        foco = mp.focus_point(ent)
        if math.dist(cam["target"], foco) > .4:
            planned = dict(shot, framing=framing, subject=subject, camera_axis=axis)
            return camera_for_shot(base, planned, start), notes
        return cam, notes
    muda = start != end
    so_tronco = framing in ("medium", "ots") and not muda and len(pessoas) == 1
    pontos = _pontos([start, end], active, so_tronco)
    distancia_min = 1.3
    if framing == "wide":
        # plano aberto mostra o LUGAR: o chao dos marcos (carro, entrada, mesas) entra no quadro
        for nome, (mx, my, _) in stage["marcos"].items():
            if nome != "BUILDING":
                pontos += [[mx - .8, my - .5, 0.0], [mx + .8, my - .5, 0.0], [mx, my, 1.4]]
        distancia_min = 6.0
    elif framing == "full":
        distancia_min = 3.2
    if framing in ("wide", "full") or not subject:
        axis = [0.0, -1.0]  # eixo de estabelecimento da cena
    elif partner and partner in pessoas:
        sa, sp = start[subject]["position"], start[partner]["position"]
        v = _unit([sp[0] - sa[0], sp[1] - sa[1]])
        if v == [0.0, 0.0]:
            axis = [0.0, -1.0]
        else:
            axis = min((_perp(v), [-_perp(v)[0], -_perp(v)[1]]), key=lambda c: c[1])
            lado = (screen_sides or {}).get(subject) or shot.get("screen_side")
            fw = [-axis[0], -axis[1]]
            right = [fw[1], -fw[0]]
            meio = [(sa[0] + sp[0]) / 2, (sa[1] + sp[1]) / 2]
            x_sujeito = (sa[0] - meio[0]) * right[0] + (sa[1] - meio[1]) * right[1]
            if lado in ("left", "right") and (x_sujeito > 0) != (lado == "right"):
                axis = [-axis[0], -axis[1]]
                notes.append(f"camera do outro lado do eixo para manter {subject} a {lado}")
            notes.append(f"acao a dois: camera de lado para o eixo {subject}-{partner}")
    else:
        f = mp.facing_vector(ent["yaw"])
        if aiming:
            axis = min([_rot2(f, 50), _rot2(f, -50)], key=lambda c: c[1])
            notes.append(f"{subject} mirando: camera a 50 graus do eixo da arma")
        elif f[1] > .2:
            axis = f
            notes.append(f"{subject} de costas para o eixo padrao: camera do lado do rosto")
        else:
            axis = _unit([f[0] * .55, f[1] * .55 - .45])
    lente = LENTE.get(framing, 40)
    if muda and framing in ("medium", "ots"):
        notes.append("acao com mudanca de estado: o medio abre para caber o inicio e o fim")
    cam = fit_camera(pontos, axis, lente, w, h, min_dist=distancia_min)
    chave = _pontos_chave([start, end], pessoas)
    if occluded(cam, stage.get("occluders", []), chave):
        # gira em torno da acao, do menor desvio ao maior, sem cruzar o eixo (mesmo lado de tela)
        for graus in (20, -20, 40, -40, 60, -60):
            alt = fit_camera(pontos, _rot2(axis, graus), lente, w, h, min_dist=distancia_min)
            if not occluded(alt, stage["occluders"], chave):
                notes.append(f"marco na frente da acao: camera girada {graus} graus")
                return alt, notes
        notes.append("AVISO: marco do cenario na frente da acao em todos os angulos tentados")
    return cam, notes


# ---------------------------------------------------------------- spec
def select_indices(inputs: dict, niveis: str | set) -> list[int]:
    alvo = NIVEIS[niveis] if isinstance(niveis, str) else set(niveis)
    return [i for i, n in enumerate(inputs["niveis"]) if n in alvo]


def build_spec(run: Path, *, niveis: str = "complexas", indices: list[int] | None = None,
               width: int = 480, height: int = 272, log=print) -> dict:
    run = Path(run)
    inputs = load_inputs(run)
    plan, cast = inputs["plan"], inputs["cast"]
    entities, notas_elenco = build_entities(plan, cast)
    # 1a passada so descobre os objetos que a historia cria (arma, algemas, mochila, placa): o
    # WorldStore exige que toda entidade exista no estado base, ausente ate entrar em cena.
    probe = Choreographer(inputs, entities)
    for index in range(len(plan["shots"])):
        probe.step(index)
    for eid, ent in probe.world.items():
        if eid not in entities:
            entities[eid] = {"kind": "prop", "position": list(PARK), "yaw": 0.0, "color": ent["color"],
                             "size": ent["size"], "present": False}
    chosen = set(indices if indices is not None else select_indices(inputs, niveis))
    choreo = Choreographer(inputs, entities)
    base = base_camera(width, height)
    fps = float(plan.get("fps") or 24)
    # Cada spec gera IDs de evento derivados da HISTORIA inteira: o mesmo plano com outro passado
    # vira outro evento (o WorldStore recusa reusar um ID com outro estado-pai).
    history = digest({"entities": entities, "plan": [(s.get("beat"), s.get("framing"), s.get("subject"),
                                                       s.get("co_subject")) for s in plan["shots"]]})
    emitted, shots, report_shots = copy.deepcopy(entities), [], []
    first_location = None
    for index, shot in enumerate(plan["shots"]):
        step = choreo.step(index)
        if index not in chosen:
            continue
        framing = str(shot.get("framing") or "wide").casefold()
        # UNIAO inicio+fim (nao so inicio): quem ENTRA durante o plano (ex.: figurante citado
        # so na 2a metade do beat) tambem precisa aparecer como ativo no aberto -- achado da
        # review adversarial 2026-09-27, mesmo ponto do bug de `state_for_shot`.
        present = [k for k in step["start"] if step["start"][k].get("present") or step["end"][k].get("present")]
        subject, partner = step["subject"], step["partner"]
        if framing in TIGHT and subject:
            ativos = [subject]
        elif framing in ("medium", "ots") and subject:
            ativos = [x for x in (subject, partner) if x]
        elif framing == "insert":
            ativos = []
        else:
            ativos = [k for k in present if step["start"][k]["kind"] != "prop"]
        # objetos na mao de quem aparece; objetos soltos so no plano aberto, ou quando o plano
        # os cita ou os move (a arma chutada, a placa que cai)
        citados = {o[1] for o in _objetos_citados(step["beat"])}
        for k, v in step["end"].items():
            ini = step["start"][k]
            if v["kind"] != "prop" or not (ini.get("present") or v.get("present")):
                continue
            donos = {(x.get("attachment") or {}).get("entity_id") for x in (ini, v)} - {None}
            if donos:
                if donos & set(ativos):
                    ativos.append(k)
            elif framing in ("wide", "full") or k.split("_")[0] in citados or ini != v:
                ativos.append(k)
        ativos = sorted(set(ativos))
        camera, cam_notes = camera_for(shot, step, ativos, base, choreo.stage,
                                       screen_sides=plan.get("screen_sides"))
        sid = shot.get("id") or f"PV_{index:03d}"
        setup_ops = diff_ops(emitted, step["start"])
        event_ops = diff_ops(step["start"], step["end"])
        history = digest([history, setup_ops, event_ops, choreo.location_id])
        unit = shot.get("source_unit_id") or f"PLAN_{index:03d}"
        spec_shot = {
            "id": sid, "scene": shot.get("scene"), "seconds": max(.5, float(shot.get("seconds") or 1.0)),
            "camera": camera, "action": step["beat"], "prompt": shot.get("storyboard_prompt") or step["beat"],
            "location_id": choreo.location_id, "location": choreo.location, "active_entities": ativos,
            "framing": framing, "subject": subject, "co_subject": partner,
            "screen_side": shot.get("screen_side"),
            "setup": {"id": f"SETUP_{sid}_{history[:10]}", "source_unit_id": unit, "operations": setup_ops}
            if setup_ops else None,
            "event": {"id": f"EV_{sid}_{history[:10]}", "source_unit_id": unit, "operations": event_ops}
            if event_ops else None,
            "previs": {"shot_index": index, "nivel": inputs["niveis"][index], "score": inputs["scores"][index],
                       "primitive": step["primitive"], "inferred": step["inferred"] + cam_notes,
                       "notes": step["notes"], "location_notes": choreo.scene_notes,
                       "template": choreo.location["template"]},
        }
        if framing in TIGHT and subject:
            from script_pipeline.spatial_planner import validate_spatial_shot
            validate_spatial_shot(spec_shot, step["start"])
        shots.append(spec_shot)
        report_shots.append(spec_shot["previs"])
        emitted = copy.deepcopy(step["end"])
        first_location = first_location or (choreo.location_id, choreo.location)
    spec = {"schema_version": 1, "generator": SCHEMA, "title": f"previs automatico -- {run.name}",
            "fps": fps, "location_id": (first_location or ("LOC_EMPTY", None))[0],
            "location": (first_location or (None, {"boxes": []}))[1], "entities": entities,
            "shots": shots, "source_run": str(run.resolve()),
            "inputs": {"niveis": niveis if indices is None else "indices", "notas_elenco": notas_elenco}}
    return spec


# ---------------------------------------------------------------- render (Blender)
def _blender() -> str | None:
    return shutil.which("blender")


def _project_dir(run: Path) -> Path:
    return Path(run) / "previs"


def render_keyframes(run: Path, spec: dict, *, log=print) -> Path:
    from script_pipeline.production_project import write
    from script_pipeline.spatial_pipeline import prepare, blocking
    project = _project_dir(run)
    project.mkdir(parents=True, exist_ok=True)
    if not (project / "projeto.json").exists():
        # marcador: evita que o prepare crie um projeto de producao ficticio aqui dentro
        write(project / "projeto.json", {"schema_version": 1, "kind": "previs",
                                         "source_run": str(Path(run).resolve())})
    prepare(project, spec)
    blocking(project)
    return project / "world" / "shot_bindings.json"


def _interpolated_states(initial: dict, final: dict, n: int) -> list[dict]:
    frames = []
    for i in range(n):
        t = mp.ease(i / max(1, n - 1))
        ents = {}
        for eid in sorted(set(initial["entities"]) | set(final["entities"])):
            a, b = initial["entities"].get(eid), final["entities"].get(eid)
            a = a or b
            b = b or a
            vis_a, vis_b = a.get("present", True), b.get("present", True)
            e = copy.deepcopy(a if t < .5 else b)
            e["present"] = vis_a if t < .5 else vis_b
            # objeto na mao segue o soquete (a pose interpolada do dono); so objeto solto nos dois
            # extremos desliza de uma posicao a outra -- a posicao gravada de objeto na mao e velha
            solto = e["kind"] != "prop" or not (a.get("attachment") or b.get("attachment"))
            if vis_a and vis_b and solto:
                e["position"] = [pa + (pb - pa) * t for pa, pb in zip(a["position"], b["position"])]
                e["yaw"] = mp.blend_yaw(a.get("yaw", 0), b.get("yaw", 0), t)
            if vis_a and vis_b and e["kind"] != "prop":
                e["pose"] = {"joints": mp.blend_joints(mp.local_joints(a.get("pose")),
                                                       mp.local_joints(b.get("pose")), t)}
            ents[eid] = e
        frames.append(dict(initial, entities=ents))
    return frames


def render_motion(project: Path, shot: dict, *, fps: int = MOTION_FPS, log=print,
                  joint_frames: list[dict] | None = None) -> Path | None:
    """Clipe de movimento em baixa resolucao (Workbench): interpola as poses do quadro inicial
    ao final -- ou usa `joint_frames` prontos (InterGen)."""
    from script_pipeline.world_store import WorldStore
    from script_pipeline.scene_composer import file_hash
    binding = shot["binding"]
    n = max(2, round(float(shot["seconds"]) * fps))
    with WorldStore(project) as store:
        initial, final = store.get(binding["initial"]), store.get(binding["final"])
        location = store.asset(initial["location_id"], initial["location_asset"])
    states = joint_frames or _interpolated_states(initial, final, n)
    worker = Path(__file__).with_name("blender_scene_worker.py")
    camera = dict(binding["camera"])
    key = digest({"states": states, "camera": camera, "location": location, "fps": fps,
                  "worker": file_hash(worker), "poses": file_hash(Path(mp.__file__))})
    out = project / "vfx" / "previs_motion" / key[:24]
    clip = out / "motion.mp4"
    if clip.exists():
        return clip
    out.mkdir(parents=True, exist_ok=True)
    job = {"output": str(out.resolve()), "camera": camera, "location": location, "sequence": states}
    (out / "job.json").write_text(json.dumps(job), encoding="utf-8")
    exe = _blender()
    with (out / "blender.log").open("w", encoding="utf-8") as fh:
        subprocess.run([exe, "--background", "--factory-startup", "--python", str(worker), "--",
                        str(out / "job.json")], stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=900)
    _label_frames(out, shot)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-framerate", str(fps), "-i", str(out / "frame_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(clip)], check=True)
    return clip


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label_frames(folder: Path, shot: dict):
    from PIL import Image, ImageDraw
    pv = shot.get("previs") or {}
    label = f"#{pv.get('shot_index')} {shot.get('framing')} {pv.get('primitive')} [{pv.get('nivel')}]"
    for png in sorted(folder.glob("frame_*.png")):
        img = Image.open(png).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
        draw.text((4, 2), label, fill=(255, 220, 90), font=_font(12))
        img.save(png)


def contact_sheet(run: Path, bindings: dict, *, log=print) -> Path:
    from PIL import Image, ImageDraw
    rows = []
    for shot in bindings["shots"]:
        controls = shot.get("controls") or {}
        if not controls:
            continue
        ini = Image.open(controls["initial"]["files"]["beauty"]).convert("RGB")
        fin = Image.open(controls["final"]["files"]["beauty"]).convert("RGB")
        pose = Image.open(controls["initial"]["files"]["pose"]).convert("RGB")
        over = Image.blend(ini, pose, .55)
        w, h = ini.size
        row = Image.new("RGB", (w * 3, h + 44), (12, 12, 16))
        for k, im in enumerate((ini, fin, over)):
            row.paste(im, (k * w, 44))
        pv = shot.get("previs") or {}
        draw = ImageDraw.Draw(row)
        draw.text((6, 4), f"#{pv.get('shot_index')}  {shot.get('framing')}  {pv.get('primitive')}  "
                          f"[{pv.get('nivel')} {pv.get('score')}]  {shot.get('seconds'):.2f}s", fill="white",
                  font=_font(14))
        draw.text((6, 24), str(shot.get("action") or "")[:150], fill=(190, 190, 190), font=_font(12))
        for k, t in enumerate(("inicio", "fim", "esqueleto (inicio)")):
            draw.text((k * w + 6, 46), t, fill=(255, 220, 90), font=_font(12))
        rows.append(row)
    if not rows:
        raise ValueError("nenhum plano renderizado")
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (0, 0, 0))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    out = Path(run) / "shots" / "previs_3d.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def concat_clips(run: Path, clips: list[Path]) -> Path | None:
    if not clips:
        return None
    out = Path(run) / "shots" / "previs_3d.mp4"
    lst = out.with_suffix(".txt")
    # caminhos ABSOLUTOS: o concat do ffmpeg resolve relativo a pasta da lista, nao ao cwd
    lst.write_text("".join(f"file '{str(Path(c).resolve()).replace(chr(92), '/')}'\n" for c in clips),
                   encoding="utf-8")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    lst.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------- InterGen (opcional)
INTERGEN_PROMPTS = {
    "takedown": "One person runs at the other person, tackles him and pushes him down to the ground.",
    "restrain": "One person kneels on the other person who lies on the ground and holds his arms behind his back.",
    "protect": "One person steps in front of the other person and protects him with his body.",
    "escort": "One person puts a hand on the other person's back and walks him forward quickly.",
    "contact": "Two people bump into each other.",
}


def intergen_frames(project: Path, shot: dict, joints, *, fps: int = MOTION_FPS) -> tuple[list[dict], dict]:
    """(T,2,22,3) do InterGen (Y para cima) -> estados por quadro com as juntas nos dois
    manequins do plano, e um resumo {compressao, papel}.

    MEDIDO 2026-09-27 (CERCO, planos 17/19): o InterGen NAO conhece o estado de continuidade
    (quem ja estava no chao comeca em pe) e gera ~10 s de movimento, comprimidos aqui para a
    duracao do plano. E uma SUGESTAO de coreografia para o clipe; os quadros-chave e o spec
    continuam sendo os da coreografia deterministica."""
    import numpy as np
    from script_pipeline.pose_video import smpl_to_coco18, resample_time
    from script_pipeline.camera_geometry import y_up_to_z_up
    from script_pipeline.world_store import WorldStore
    coco = smpl_to_coco18(np.asarray(joints, dtype=float))          # (T,2,18,3)
    coco = y_up_to_z_up(coco)
    n = max(2, round(float(shot["seconds"]) * fps))
    coco = resample_time(coco.transpose(1, 0, 2, 3), source_fps=20.0, target_fps=fps, target_frames=n)
    coco = coco.transpose(1, 0, 2, 3)                                  # (n,2,18,3)
    fonte = np.asarray(joints).shape[0]
    roots = (coco[:, :, mp.R_HIP] + coco[:, :, mp.L_HIP]) / 2
    prim = (shot.get("previs") or {}).get("primitive")
    if prim in ("takedown", "restrain"):
        # quem termina MAIS BAIXO e o alvo (derrubado/imobilizado); o outro age
        alvo = int(np.argmin(roots[-1, :, 2]))
        order, papel = [1 - alvo, alvo], "alvo = quem termina mais baixo"
    else:
        moved = np.linalg.norm(roots[-1, :, :2] - roots[0, :, :2], axis=-1)
        order, papel = [int(np.argmax(moved)), 1 - int(np.argmax(moved))], "sujeito = quem mais se desloca"
    with WorldStore(project) as store:
        initial = store.get(shot["binding"]["initial"])
    a, p = shot["subject"], shot["co_subject"]
    ea, ep = initial["entities"][a], initial["entities"][p]
    # alinhamento: escala pela altura do pescoco, gira o eixo sujeito->parceiro, translada o sujeito
    neck = float(np.mean(coco[0, :, mp.NECK, 2]))
    scale = 1.45 / neck if neck > .3 else 1.0
    coco = coco * scale
    src_a, src_p = roots[0, order[0], :2] * scale, roots[0, order[1], :2] * scale
    dst_a, dst_p = np.array(ea["position"][:2]), np.array(ep["position"][:2])
    ang = math.atan2(*(dst_p - dst_a)[::-1]) - math.atan2(*(src_p - src_a)[::-1])
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    xy = (coco[..., :2] - src_a) @ rot.T + dst_a
    coco = np.concatenate([xy, coco[..., 2:]], axis=-1)
    frames = []
    for t in range(coco.shape[0]):
        ents = copy.deepcopy(initial["entities"])
        for slot, eid in zip(order, (a, p)):
            world = coco[t, slot].tolist()
            right = np.array(world[mp.R_SHOULDER][:2]) - np.array(world[mp.L_SHOULDER][:2])
            # olhar = cima x direita anatomica = (-ry, rx) no plano do chao
            yaw = mp.facing_yaw([-right[1], right[0]]) if np.linalg.norm(right) > 1e-6 \
                else ents[eid].get("yaw", 0)
            root = [(world[mp.R_HIP][0] + world[mp.L_HIP][0]) / 2, (world[mp.R_HIP][1] + world[mp.L_HIP][1]) / 2, 0.0]
            local = mp.complete_face(mp.from_world(world, root, yaw))
            ents[eid].update(position=root, yaw=yaw, pose={"joints": local}, present=True)
        frames.append(dict(initial, entities=ents))
    info = {"compressao": round(fonte / 20.0 / float(shot["seconds"]), 2), "papel": papel,
            "aviso": "ignora o estado de continuidade; so o clipe usa o InterGen"}
    return frames, info


# ---------------------------------------------------------------- still (opcional)
def attach_to_plan(run: Path, bindings: dict, *, denoise: float = .65, log=print) -> int:
    """Liga o quadro INICIAL do previs ao still dos planos renderizados (FLUX img2img nos
    abertos/medios, ReferenceLatent nos closes -- o mesmo adaptador do spatial_pipeline).
    O video recebe so o still: o estado final do previs nao chega ao motor de video."""
    plan_path = Path(run) / "parse" / "shot_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    n = 0
    for shot in bindings["shots"]:
        index = (shot.get("previs") or {}).get("shot_index")
        if index is None or not shot.get("controls"):
            continue
        bundle = shot["controls"]["initial"]
        tight = str(shot.get("framing") or "") in TIGHT
        plan["shots"][index]["spatial"] = {
            "control_bundle": str(Path(bundle["files"]["beauty"]).with_name("control_bundle.json")),
            "fingerprint": bundle["fingerprint"], "denoise": float(denoise),
            "mode": "reference" if tight else "img2img", "adapter_version": 2,
            "binding": shot["binding"], "source": "previs_auto"}
        n += 1
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[previs-3d] {n} plano(s) com o blocking 3D ligado ao still (denoise {denoise})")
    return n


def detach_from_plan(run: Path) -> int:
    plan_path = Path(run) / "parse" / "shot_plan.json"
    plan = _read(plan_path)
    if not plan:
        return 0
    n = 0
    for shot in plan["shots"]:
        if (shot.get("spatial") or {}).get("source") == "previs_auto":
            shot.pop("spatial")
            n += 1
    if n:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return n


# ---------------------------------------------------------------- orquestracao
def run_previs(run: Path, *, niveis: str = "complexas", indices: list[int] | None = None,
               width: int = 960, height: int = 544, scale: float = .5, so_spec: bool = False,
               movimento: bool = True, stills: bool = False, denoise: float = .65,
               intergen: bool = False, log=print) -> dict:
    run = Path(run)
    # ACHADO (review adversarial 2026-09-27): so o `else` (sem --stills) no fim desta funcao
    # removia o vinculo previs->still de uma corrida anterior -- toda saida ANTES dele (sem
    # planos no nivel pedido, --so-spec, Blender ausente) deixava o vinculo velho no shot_plan
    # mesmo com --previs3d off/mudado. Remove aqui, no comeco; se `stills` for pedido de novo
    # mais abaixo, `attach_to_plan` religa so os planos desta selecao.
    removidos = detach_from_plan(run)
    if removidos:
        log(f"[previs-3d] {removidos} vinculo(s) previs->still de corrida anterior removido(s) "
            "antes desta selecao")
    w = max(64, int(round(width * scale / 16)) * 16)
    h = max(64, int(round(w * height / width)))
    if abs(w / h - width / height) > .001:
        # a proporcao precisa bater com a do still, ou o adaptador espacial recusa
        w, h = int(width * scale), int(height * scale)
    spec = build_spec(run, niveis=niveis, indices=indices, width=w, height=h, log=log)
    spec_path = run / "world" / "previs_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"schema": SCHEMA, "spec": str(spec_path), "niveis": niveis, "resolucao": [w, h],
              "planos": [dict(s["previs"], id=s["id"], framing=s["framing"], action=s["action"],
                              setup=len((s.get("setup") or {}).get("operations") or []),
                              eventos=len((s.get("event") or {}).get("operations") or []))
                         for s in spec["shots"]],
              "renderizado": False}
    log(f"[previs-3d] spec automatico: {len(spec['shots'])} plano(s) ({niveis}) -> {spec_path}")
    for s in spec["shots"]:
        pv = s["previs"]
        log(f"  #{pv['shot_index']} {s['framing']} {pv['primitive']} [{pv['nivel']}] "
            f"{len((s.get('event') or {}).get('operations') or [])} mudanca(s) no plano"
            + (f" -- {'; '.join(pv['inferred'])}" if pv["inferred"] else ""))
    out_report = run / "shots" / "previs_3d.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    if not spec["shots"]:
        log("[previs-3d] nenhum plano no nivel pedido -- nada a renderizar")
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    if so_spec or not _blender():
        if not so_spec:
            log("[previs-3d] Blender nao encontrado no PATH -- spec escrito, render pulado")
            report["aviso"] = "blender ausente"
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    bindings_path = render_keyframes(run, spec, log=log)
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    report["renderizado"] = True
    report["folha"] = str(contact_sheet(run, bindings, log=log))
    log(f"[previs-3d] folha de contato -> {report['folha']}")
    if movimento:
        project = _project_dir(run)
        joints_by_shot = {}
        if intergen:
            joints_by_shot = _run_intergen(bindings, log=log)
        clips = []
        for shot in bindings["shots"]:
            frames = None
            if shot["id"] in joints_by_shot:
                frames, info = intergen_frames(project, shot, joints_by_shot[shot["id"]])
                report.setdefault("intergen_info", {})[shot["id"]] = info
                log(f"[previs-3d] {shot['id']}: InterGen comprimido {info['compressao']}x na duracao "
                    f"do plano ({info['papel']}); {info['aviso']}")
            clip = render_motion(project, shot, joint_frames=frames, log=log)
            if clip:
                clips.append(clip)
        video = concat_clips(run, clips)
        report["video"] = str(video) if video else None
        report["intergen"] = sorted(joints_by_shot)
        log(f"[previs-3d] movimento em baixa resolucao -> {video}")
    if stills:
        # o detach do topo da funcao ja limpou vinculos anteriores; religa so a selecao atual
        report["stills_ligados"] = attach_to_plan(run, bindings, denoise=denoise, log=log)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _run_intergen(bindings: dict, *, log=print) -> dict:
    from script_pipeline.motion_director import run_intergen_batch
    import numpy as np
    pedidos = {}
    for shot in bindings["shots"]:
        prim = (shot.get("previs") or {}).get("primitive")
        if prim in INTERGEN_PROMPTS and shot.get("subject") and shot.get("co_subject"):
            pedidos[shot["id"]] = INTERGEN_PROMPTS[prim]
    if not pedidos:
        return {}
    try:
        paths = run_intergen_batch(list(pedidos.values()), log=log)
    except Exception as exc:  # InterGen e opcional: o previs segue com a interpolacao
        log(f"[previs-3d] InterGen indisponivel ({exc}); movimento por interpolacao")
        return {}
    return {sid: np.load(paths[prompt]) for sid, prompt in pedidos.items() if prompt in paths}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--niveis", choices=sorted(NIVEIS), default="complexas",
                    help="quais planos entram no previs, pelo nivel de shots/complexity_report.json")
    ap.add_argument("--planos", default=None, help="indices explicitos (ex.: 17,19,23); ignora --niveis")
    ap.add_argument("--width", type=int, default=960, help="largura do STILL da corrida (a proporcao vem daqui)")
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--scale", type=float, default=.5, help="resolucao do previs relativa ao still")
    ap.add_argument("--so-spec", action="store_true", help="so escreve world/previs_spec.json, sem Blender")
    ap.add_argument("--sem-movimento", action="store_true", help="so os quadros inicial/final, sem o clipe")
    ap.add_argument("--stills", action="store_true",
                    help="liga o quadro inicial do previs ao still desses planos (FLUX img2img/referencia)")
    ap.add_argument("--denoise", type=float, default=.65)
    ap.add_argument("--desligar-stills", action="store_true",
                    help="remove do shot_plan o vinculo de previs ligado antes por --stills")
    ap.add_argument("--intergen", action="store_true",
                    help="movimento de dupla em contato pelo InterGen (usa a GPU) em vez da interpolacao")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    if args.desligar_stills:
        print(f"[previs-3d] {detach_from_plan(run)} vinculo(s) removido(s)")
        return 0
    indices = [int(x) for x in args.planos.split(",") if x.strip()] if args.planos else None
    run_previs(run, niveis=args.niveis, indices=indices, width=args.width, height=args.height,
               scale=args.scale, so_spec=args.so_spec, movimento=not args.sem_movimento,
               stills=args.stills, denoise=args.denoise, intergen=args.intergen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
