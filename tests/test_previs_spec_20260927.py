"""Previs 3D automatico (MEMORIAL 3.135): poses do manequim com lado anatomico, spec gerado da
decupagem, continuidade entre planos, camera que enquadra a acao inteira. Sem GPU; o teste de
render real com Blender so roda com PREVIS_BLENDER_TEST=1."""
import copy
import json
import math
import os
import shutil

import pytest

from script_pipeline import mannequin_poses as mp
from script_pipeline import previs_spec as pv
from script_pipeline.camera_geometry import project, socket_position
from script_pipeline.world_store import WorldStore, apply_operations, validate_state


def _pessoa(pose="standing", yaw=0.0, pos=(0.0, 0.0, 0.0)):
    return {"kind": "character", "position": list(pos), "yaw": yaw, "pose": pose, "color": [.5, .5, .5]}


# ---------------------------------------------------------------- manequim
def test_lado_anatomico_e_rosto_para_menos_y():
    j = mp.world_joints(_pessoa())
    assert j[mp.R_SHOULDER][0] < 0 < j[mp.L_SHOULDER][0]      # direita anatomica em -X
    assert j[mp.R_WRIST][0] < 0 < j[mp.L_WRIST][0]
    assert j[mp.NOSE][1] < 0                                   # rosto para -Y (camera padrao)
    # olhando para +X (yaw 90), a direita anatomica fica em -Y
    j = mp.world_joints(_pessoa(yaw=90))
    assert j[mp.R_SHOULDER][1] < 0 < j[mp.L_SHOULDER][1]
    assert j[mp.NOSE][0] > 0


def test_em_pe_mantem_o_ponto_de_mira_antigo():
    e = _pessoa(pos=(1.0, 2.0, 0.0))
    assert mp.focus_point(e) == pytest.approx([1.0, 2.0, 1.55])


def test_todas_as_poses_validas_e_cabeca_no_lugar_certo():
    for nome in mp.POSES:
        mp.validate_pose(nome)
        assert len(mp.local_joints(nome)) == 18
    assert mp.head_center(mp.local_joints("prone"))[2] < .3
    assert mp.head_center(mp.local_joints("kneeling"))[2] < 1.3
    assert mp.head_center(mp.local_joints("standing"))[2] == pytest.approx(1.70)
    with pytest.raises(ValueError):
        mp.validate_pose("voando")


def test_soquete_da_mao_segue_a_pose():
    mira = _pessoa(pose="aiming")
    mao = socket_position(mira, "right_hand")
    assert mao[1] < -.4 and mao[2] > 1.3 and mao[0] < .05      # braco estendido a frente
    assert list(mao) == pytest.approx(mp.world_joints(mira)[mp.R_WRIST])


def test_ida_e_volta_de_referencial_e_rosto_reconstruido():
    e = _pessoa(yaw=37.0, pos=(1.5, -2.0, 0.0))
    world = mp.world_joints(e)
    local = mp.from_world(world, e["position"], e["yaw"])
    for got, want in zip(local, mp.local_joints("standing")):
        assert got == pytest.approx(want, abs=1e-9)
    smpl_like = copy.deepcopy(mp.local_joints("standing"))
    for k in (mp.NOSE, mp.R_EYE, mp.L_EYE, mp.R_EAR, mp.L_EAR):
        smpl_like[k] = [0.0, 0.0, 1.70]
    face = mp.complete_face(smpl_like)
    assert face[mp.R_EAR][0] < 0 < face[mp.L_EAR][0] and face[mp.NOSE][1] < 0
    for yaw in (0, 45, 90, 180, -120):
        assert mp.facing_yaw(mp.facing_vector(yaw)) % 360 == pytest.approx(yaw % 360)


# ---------------------------------------------------------------- WorldStore
def _estado(**entities):
    return {"schema_version": 1, "coordinates": "Z_UP_METERS", "location_id": "L", "location_asset": "v",
            "entities": entities}


def test_pose_validada_e_objeto_largado():
    with pytest.raises(ValueError):
        validate_state(_estado(A=_pessoa(pose="voando")))
    validate_state(_estado(A=_pessoa(pose={"joints": mp.local_joints("kneeling")})))
    st = _estado(A=_pessoa(), BAG={"kind": "prop", "position": [0, 0, 0], "color": [1, 1, 1],
                                   "attachment": {"entity_id": "A", "socket": "right_hand"}})
    novo = apply_operations(st, [{"op": "detach", "entity_id": "BAG", "from": "A", "position": [1, 2, .2]}])
    assert "attachment" not in novo["entities"]["BAG"] and novo["entities"]["BAG"]["position"] == [1, 2, .2]
    with pytest.raises(ValueError):
        apply_operations(st, [{"op": "detach", "entity_id": "BAG", "from": "B", "position": [0, 0, 0]}])


def test_close_de_quem_esta_no_chao_mira_a_cabeca():
    from script_pipeline.spatial_planner import camera_for_shot, validate_spatial_shot
    ents = {"A": _pessoa(pose="prone", pos=(0.0, 2.0, 0.0))}
    base = pv.base_camera(480, 272)
    cam = camera_for_shot(base, {"framing": "close", "subject": "A"}, ents)
    assert cam["target"][2] < .2
    validate_spatial_shot({"id": "X", "camera": cam, "framing": "close", "subject": "A"}, ents)


# ---------------------------------------------------------------- spec a partir de uma corrida
def _corrida(tmp_path):
    run = tmp_path / "run"
    (run / "parse").mkdir(parents=True)
    (run / "characters").mkdir()
    cast = {
        "SEO-YEON": {"descriptor": "Dark hair, wearing a black pinstripe suit", "line_count": 1},
        "HA-EUN": {"descriptor": "Ponytail, navy tactical jacket", "line_count": 0},
        "MIN-JUN": {"descriptor": "Short hair, charcoal-gray suit", "line_count": 0},
        "PRESIDENT": {"descriptor": "silver hair, navy-blue suit", "extra": True, "aliases": ["President"]},
        "TERRORIST": {"descriptor": "black leather jacket", "extra": True, "aliases": ["terrorist"]},
    }
    comum = {"scene": 1, "location": "SEOUL AVENUE, HOTEL ENTRANCE", "interior": "", "seconds": 1.6,
             "co_subject": "", "line_index": None, "screen_side": "left", "storyboard_prompt": "x"}
    shots = [
        dict(comum, framing="wide", subject="", beat="The convoy stops by the armored sedan"),
        dict(comum, framing="close", subject="SEO-YEON", line_index=0,
             beat="Seo-yeon pushes the President to the ground."),
        dict(comum, framing="medium", subject="SEO-YEON", co_subject="PRESIDENT",
             beat="Seo-yeon helps the President to his feet"),
        dict(comum, framing="medium", subject="HA-EUN", co_subject="TERRORIST",
             beat="Ha-eun takes down the terrorist on a pedestrian crossing"),
        dict(comum, framing="medium", subject="MIN-JUN", co_subject="TERRORIST", screen_side="right",
             beat="Min-jun kicks the weapon away and handcuffs the terrorist"),
        dict(comum, framing="medium", subject="SEO-YEON",
             beat="Seo-yeon fires her gun repeatedly at a high window"),
    ]
    for i, s in enumerate(shots):
        s["position"] = i
    plan = {"fps": 24, "shots": shots, "screen_sides": {"SEO-YEON": "left", "HA-EUN": "left",
            "MIN-JUN": "right", "PRESIDENT": "left", "TERRORIST": "right"}}
    (run / "parse" / "shot_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run / "characters" / "cast.json").write_text(json.dumps(cast), encoding="utf-8")
    return run


def _prepare(tmp_path, spec, nome="proj"):
    from script_pipeline.production_project import write
    from script_pipeline.spatial_pipeline import prepare
    proj = tmp_path / nome
    proj.mkdir(exist_ok=True)
    write(proj / "projeto.json", {"kind": "previs"})
    return proj, prepare(proj, spec)


def _estados(proj, shots):
    with WorldStore(proj) as st:
        return {s["previs"]["shot_index"]: (st.get(s["binding"]["initial"])["entities"],
                                           st.get(s["binding"]["final"])["entities"]) for s in shots}


def test_spec_gera_continuidade_e_passa_pelo_worldstore(tmp_path):
    run = _corrida(tmp_path)
    spec = pv.build_spec(run, indices=list(range(6)))
    assert [s["previs"]["shot_index"] for s in spec["shots"]] == list(range(6))
    ids = [op["id"] for s in spec["shots"] for op in (s.get("setup"), s.get("event")) if op]
    assert len(ids) == len(set(ids))
    proj, shots = _prepare(tmp_path, spec)
    est = _estados(proj, shots)
    # 1: a fala em close NAO mostra o empurrao; 2: o Presidente ja comeca no chao e levanta
    ini1, fim1 = est[1]
    assert set(k for k, v in fim1.items() if v.get("present")) == {"SEO-YEON"}
    ini2, fim2 = est[2]
    assert ini2["PRESIDENT"]["pose"] in pv.LYING and fim2["PRESIDENT"]["pose"] == "standing"
    # 3: derrubada -- alvo de brucos, quem derruba ajoelhado ao lado
    ini3, fim3 = est[3]
    assert ini3["HA-EUN"]["pose"] == "running" and fim3["TERRORIST"]["pose"] == "prone"
    assert fim3["HA-EUN"]["pose"] == "kneeling"
    assert math.dist(fim3["HA-EUN"]["position"][:2], fim3["TERRORIST"]["position"][:2]) < 1.2
    # 4: algemas passam de mao; o alvo continua no chao (continuidade) e termina algemado
    ini4, fim4 = est[4]
    assert ini4["TERRORIST"]["pose"] == "prone"
    assert ini4["HANDCUFFS"]["attachment"]["entity_id"] == "MIN-JUN"
    assert fim4["HANDCUFFS"]["attachment"]["entity_id"] == "TERRORIST"
    assert fim4["TERRORIST"]["pose"] == "prone_cuffed"
    assert ini4["GUN_TERRORIST"]["position"] != fim4["GUN_TERRORIST"]["position"]   # arma chutada
    # 5: tiro para o alto com a arma na mao direita
    ini5, _ = est[5]
    assert ini5["SEO-YEON"]["pose"] == "aiming_high"
    assert ini5["GUN_SEO-YEON"]["attachment"] == {"entity_id": "SEO-YEON", "socket": "right_hand"}
    # determinismo e idempotencia: o mesmo plano gera o mesmo spec e o prepare aceita de novo
    assert pv.build_spec(run, indices=list(range(6))) == spec
    _prepare(tmp_path, spec)


def test_so_os_planos_do_nivel_mas_com_a_historia_inteira(tmp_path):
    run = _corrida(tmp_path)
    spec = pv.build_spec(run, indices=[4])
    assert len(spec["shots"]) == 1
    # o plano 4 sozinho herda, via `setup`, o que aconteceu fora dele (derrubada do plano 3)
    proj, shots = _prepare(tmp_path, spec, "so4")
    ini, _ = _estados(proj, shots)[4]
    assert ini["TERRORIST"]["pose"] == "prone" and ini["TERRORIST"]["present"]


def _todos_no_quadro(camera, entidades):
    pts = [p for e in entidades for p in mp.world_joints(e)]
    xy, depth, visible = project(pts, camera)
    return bool(visible.all())


def test_camera_enquadra_inicio_e_fim_da_acao_e_respeita_o_lado(tmp_path):
    run = _corrida(tmp_path)
    spec = pv.build_spec(run, indices=[3])
    shot = spec["shots"][0]
    proj, shots = _prepare(tmp_path, spec, "cam")
    ini, fim = _estados(proj, shots)[3]
    cam = shot["camera"]
    assert _todos_no_quadro(cam, [ini["HA-EUN"], ini["TERRORIST"], fim["HA-EUN"], fim["TERRORIST"]])
    # HA-EUN e 'left' no plano: fica na metade esquerda do quadro no inicio
    xy, _, _ = project([mp.focus_point(ini["HA-EUN"])], cam)
    xy2, _, _ = project([mp.focus_point(ini["TERRORIST"])], cam)
    assert xy[0][0] < xy2[0][0]


def test_camera_nunca_no_eixo_da_arma(tmp_path):
    run = _corrida(tmp_path)
    spec = pv.build_spec(run, indices=[5])
    shot = spec["shots"][0]
    proj, shots = _prepare(tmp_path, spec, "tiro")
    ini, _ = _estados(proj, shots)[5]
    f = mp.facing_vector(ini["SEO-YEON"]["yaw"])
    pos = ini["SEO-YEON"]["position"]
    d = pv._unit([shot["camera"]["position"][0] - pos[0], shot["camera"]["position"][1] - pos[1]])
    angulo = math.degrees(math.acos(max(-1, min(1, f[0] * d[0] + f[1] * d[1]))))
    assert angulo >= 30


def test_fit_camera_e_oclusao():
    pts = [[x, y, z] for x in (-1, 1) for y in (2, 3) for z in (0, 1.8)]
    cam = pv.fit_camera(pts, [0, -1], 40, 480, 272)
    _, _, visible = project(pts, cam)
    assert visible.all()
    mesa = [(-.5, .5, -1.5, -.5, 2.5)]  # volume entre a camera e a acao
    assert pv.occluded(cam, mesa, pts)
    assert not pv.occluded(cam, [], pts)


def test_run_previs_so_spec_e_sem_blender(tmp_path, monkeypatch):
    run = _corrida(tmp_path)
    rep = pv.run_previs(run, niveis="todos", so_spec=True, log=lambda m: None)
    assert (run / "world" / "previs_spec.json").exists() and (run / "shots" / "previs_3d.json").exists()
    assert rep["renderizado"] is False and len(rep["planos"]) == 6
    monkeypatch.setattr(pv, "_blender", lambda: None)
    rep = pv.run_previs(run, niveis="todos", log=lambda m: None)
    assert rep.get("aviso") == "blender ausente"


def test_vinculo_do_still_e_so_do_previs(tmp_path):
    run = _corrida(tmp_path)
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    plan["shots"][0]["spatial"] = {"source": "manual"}
    (run / "parse" / "shot_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    bundle = {"files": {"beauty": str(tmp_path / "b" / "beauty.png")}, "fingerprint": "f"}
    bindings = {"shots": [{"previs": {"shot_index": 3}, "framing": "medium", "binding": {},
                           "controls": {"initial": bundle}},
                          {"previs": {"shot_index": 1}, "framing": "close", "binding": {},
                           "controls": {"initial": bundle}}]}
    assert pv.attach_to_plan(run, bindings, log=lambda m: None) == 2
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    assert plan["shots"][3]["spatial"]["mode"] == "img2img"
    assert plan["shots"][1]["spatial"]["mode"] == "reference"
    assert pv.detach_from_plan(run) == 2
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    assert plan["shots"][0]["spatial"] == {"source": "manual"} and "spatial" not in plan["shots"][3]


def test_cor_do_figurino_pula_cabelo():
    assert pv.cor_do_figurino("Dark-brown hair, wearing a black pinstripe suit") == [.05, .05, .06]
    assert pv.cor_do_figurino("silver-gray hair, navy-blue tailored suit") == [.06, .09, .25]
    assert pv.cor_do_figurino("") is None


@pytest.mark.skipif(not (shutil.which("blender") and os.environ.get("PREVIS_BLENDER_TEST")),
                    reason="render real: PREVIS_BLENDER_TEST=1 e Blender no PATH")
def test_render_real_no_blender(tmp_path):
    run = _corrida(tmp_path)
    rep = pv.run_previs(run, indices=[3], width=320, height=180, scale=.5, log=lambda m: None)
    assert rep["renderizado"] and (run / "shots" / "previs_3d.png").exists()
    assert (run / "shots" / "previs_3d.mp4").exists()


# ---------------------------------------------------------------- achados da review adversarial 2026-09-27
def test_medio_nao_corta_o_topo_da_cabeca():
    """Achado A (lens geometry): fit_camera so via joints COCO deixava o cocuruto fora do
    quadro no medio/aberto -- a malha da cabeca vai ~0,18 m acima do centro das orelhas."""
    e = dict(_pessoa(pos=(0.0, 1.0, 0.0)), present=True)
    st = {"A": e}
    cam, _ = pv.camera_for({"framing": "medium"}, {"start": st, "end": st, "subject": "A",
                           "partner": None, "beat": ""}, ["A"], pv.base_camera(160, 90),
                           {"marcos": {}, "occluders": []})
    topo = mp.head_center(mp.world_joints(e))
    topo = [topo[0], topo[1], topo[2] + .18]
    _, _, visible = project([topo], cam)
    assert bool(visible[0])


def test_presenca_semantica_sobrevive_a_lista_de_ativos(tmp_path):
    """Achado A (lens state): quem SAI de cena (present=False no evento) nao pode reaparecer
    no quadro final so por constar na lista `active_entities` do enquadramento aberto."""
    from script_pipeline.spatial_planner import state_for_shot
    st = {"schema_version": 1, "coordinates": "Z_UP_METERS", "location_id": "L", "location_asset": "v",
          "entities": {"A": dict(_pessoa(), present=False), "B": dict(_pessoa(), present=True)}}
    vista = state_for_shot(st, ["A", "B"])
    assert vista["entities"]["A"]["present"] is False
    assert vista["entities"]["B"]["present"] is True


def test_terrorista_que_some_nao_reaparece_no_quadro_final(tmp_path):
    run = _corrida(tmp_path)
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    plan["shots"].insert(2, dict(plan["shots"][0], framing="wide", subject="",
                                 beat="A motorcycle rider leaves a backpack near the car and disappears",
                                 position=2))
    for i, s in enumerate(plan["shots"]):
        s["position"] = i
    (run / "parse" / "shot_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    spec = pv.build_spec(run, indices=list(range(len(plan["shots"]))))
    proj, shots = _prepare(tmp_path, spec, "sumico")
    with WorldStore(proj) as st:
        shot = next(s for s in shots if s["previs"]["shot_index"] == 2)
        fim = st.get(shot["binding"]["final"])["entities"]
    assert fim["TERRORIST"]["present"] is False


def test_ancoras_nao_empilham_seis_pessoas_na_mesma_cena(tmp_path):
    """Achado B (lens choreo): a 5a/6a pessoa sem formacao batiam no limite do palco e
    ficavam todas no mesmo ponto."""
    run = tmp_path / "lotado"
    (run / "parse").mkdir(parents=True)
    (run / "characters").mkdir()
    nomes = [f"P{i}" for i in range(7)]
    cast = {n: {"descriptor": "gray suit", "line_count": 0} for n in nomes}
    shots = [dict(scene=1, location="ROOM", interior="interior", seconds=1.0, co_subject="",
                 line_index=None, screen_side="left", storyboard_prompt="x", framing="medium",
                 subject=n, beat=f"{n} stands.", position=i) for i, n in enumerate(nomes)]
    (run / "parse" / "shot_plan.json").write_text(json.dumps({"fps": 24, "shots": shots}), encoding="utf-8")
    (run / "characters" / "cast.json").write_text(json.dumps(cast), encoding="utf-8")
    spec = pv.build_spec(run, indices=list(range(7)))
    proj, out = _prepare(tmp_path, spec, "lotado_proj")
    with WorldStore(proj) as st:
        finais = {s["previs"]["shot_index"]: st.get(s["binding"]["final"])["entities"] for s in out}
    posicoes = [tuple(finais[i][nomes[i]]["position"][:2]) for i in range(7)]
    assert len(set(posicoes)) == 7


def test_interior_vazio_com_texto_estreito_nao_vira_exterior():
    """Achado B (lens choreo): `interior=""` (cabecalho hibrido/ausente) caia sempre em
    exterior; uma cabine/cockpit citada no texto precisa continuar interior estreito."""
    assert pv._resolve_interior("", "ANNA sits in the cockpit, hands on the controls.") is True
    loc, stage, _ = pv.build_location("cockpit fuselage", True)
    assert stage["tipo"] == "interior_estreito"
    assert pv._resolve_interior("", "") is True          # sem pista nenhuma: mais seguro que exterior
    assert pv._resolve_interior("", "wide open avenue, cars pass by") is False
    assert pv._resolve_interior("exterior", "cockpit") is False  # flag explicita manda


def test_alto_nao_casa_frase_comum_sem_alvo_elevado():
    """Achado C (lens choreo): "above"/"high"/"upper" sozinhos davam falso positivo."""
    assert not pv._ALTO.search("the gunman crouched above the curb")
    assert not pv._ALTO.search("a high-speed chase through the avenue")
    assert pv._ALTO.search("shots come from an upper window")
    assert pv._ALTO.search("o atirador esta no telhado")


def test_previs3d_off_ou_so_spec_remove_vinculo_antigo(tmp_path):
    """Achado B (lens pipeline): so o ramo 'sem --stills' no FIM de run_previs limpava o
    vinculo previs->still antigo -- --so-spec e 'sem planos no nivel' saiam antes dele."""
    run = _corrida(tmp_path)
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    plan["shots"][0]["spatial"] = {"source": "previs_auto", "control_bundle": "x", "fingerprint": "y"}
    (run / "parse" / "shot_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    pv.run_previs(run, niveis="todos", so_spec=True, log=lambda m: None)
    plan = json.loads((run / "parse" / "shot_plan.json").read_text(encoding="utf-8"))
    assert "spatial" not in plan["shots"][0]
