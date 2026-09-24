import numpy as np

from script_pipeline.motion_command_geometry import (
    ARRIVAL_EPS_M,
    PRIMITIVE_TO_CLIP,
    commands_for_actor,
    current_command,
    facing_for,
    formation_by_scene,
    idle_command_for_gap,
    maintain_heading,
    partner_anchor,
    resolve_command,
    scene_time_offsets,
    walk_target,
)


def test_action_primitives_map_to_motionbrick_capability_or_explicit_idle():
    assert PRIMITIVE_TO_CLIP["pursue"] == "walk"
    assert PRIMITIVE_TO_CLIP["takedown"] == "walk"
    assert PRIMITIVE_TO_CLIP["protect"] == "walk"
    assert PRIMITIVE_TO_CLIP["fire"] == "idle"  # sem clipe de arma no checkpoint publico


def _plan():
    return {
        "scenes": [{"scene": 1, "formation": {
            "ANA": {"anchor_m": [-1.2, 0.0], "facing": "toward_center"},
            "BIA": {"anchor_m": [1.2, 0.0], "facing": "toward_center"},
        }}],
        "commands": [
            {"shot_index": 0, "scene": 1, "start_s": 0.0, "duration_s": 2.0,
             "actor": "ANA", "partner": None, "primitive": "idle",
             "target": {"mode": "hold_anchor"}},
            {"shot_index": 1, "scene": 1, "start_s": 2.0, "duration_s": 3.0,
             "actor": "ANA", "partner": "BIA", "primitive": "locomote",
             "target": {"mode": "approach_partner", "destination": [1.2, 0.0]}},
            {"shot_index": 2, "scene": 1, "start_s": 5.0, "duration_s": 2.0,
             "actor": "ANA", "partner": "BIA", "primitive": "speak",
             "target": {"mode": "face_partner"}},
        ],
    }


def test_commands_for_actor_filters_and_sorts():
    plan = _plan()
    cmds = commands_for_actor(plan, "ANA")
    assert [c["shot_index"] for c in cmds] == [0, 1, 2]
    import pytest
    with pytest.raises(ValueError, match="ANA"):
        commands_for_actor(plan, "CARLOS")


def test_current_command_picks_by_sim_time_and_returns_none_after_last():
    # ACHADO 2026-09-17 (auditoria externa): a versao antiga devolvia o
    # ULTIMO comando pra sempre depois do fim -- current_command agora
    # devolve None (lacuna explicita); quem quer "segurar" chama
    # resolve_command/idle_command_for_gap.
    cmds = commands_for_actor(_plan(), "ANA")
    assert current_command(cmds, 0.5)["shot_index"] == 0
    assert current_command(cmds, 2.1)["shot_index"] == 1
    assert current_command(cmds, 999.0) is None


def test_current_command_returns_none_in_a_gap_between_commands():
    # Reproducao exata do achado da auditoria: comandos em [0,1) e [5,6),
    # em t=2 a versao antiga devolvia o comando de t=5 (do futuro).
    cmds = [
        {"shot_index": 0, "scene": 1, "start_s": 0.0, "duration_s": 1.0,
         "actor": "ANA", "partner": None, "primitive": "idle", "target": None},
        {"shot_index": 1, "scene": 1, "start_s": 5.0, "duration_s": 1.0,
         "actor": "ANA", "partner": "BIA", "primitive": "contact", "target": None},
    ]
    assert current_command(cmds, 2.0) is None


def test_current_command_returns_none_before_first_command():
    cmds = [{"shot_index": 0, "scene": 1, "start_s": 3.0, "duration_s": 1.0,
            "actor": "ANA", "partner": None, "primitive": "idle", "target": None}]
    assert current_command(cmds, 0.0) is None


def test_resolve_command_idles_in_gap_using_last_finished_command_scene():
    cmds = [
        {"shot_index": 0, "scene": 1, "start_s": 0.0, "duration_s": 1.0,
         "actor": "ANA", "partner": None, "primitive": "locomote", "target": None},
        {"shot_index": 1, "scene": 2, "start_s": 5.0, "duration_s": 1.0,
         "actor": "ANA", "partner": "BIA", "primitive": "contact", "target": None},
    ]
    resolved = resolve_command(cmds, 2.0)
    assert resolved["primitive"] == "idle"
    assert resolved["scene"] == 1  # cena do ultimo comando ja terminado, nao a futura
    assert resolved["partner"] is None


def test_resolve_command_idles_after_last_using_its_scene():
    cmds = commands_for_actor(_plan(), "ANA")
    resolved = resolve_command(cmds, 999.0)
    assert resolved["primitive"] == "idle"
    assert resolved["scene"] == 1


def test_idle_command_for_gap_before_first_uses_first_scene():
    cmds = [{"shot_index": 0, "scene": 7, "start_s": 3.0, "duration_s": 1.0,
            "actor": "ANA", "partner": None, "primitive": "idle", "target": None}]
    gap = idle_command_for_gap(cmds, 0.0)
    assert gap["scene"] == 7
    assert gap["primitive"] == "idle"


def test_scene_time_offsets_cumulative_in_scene_order():
    plan = {"scenes": [{"scene": "a", "duration_s": 3.0}, {"scene": "b", "duration_s": 1.5},
                       {"scene": "c", "duration_s": 2.0}]}
    assert scene_time_offsets(plan) == {"a": 0.0, "b": 3.0, "c": 4.5}


def test_commands_for_actor_offsets_start_s_across_scenes_without_overlap():
    # Reproducao exata do achado da auditoria: duas cenas de 2s da ANA, cada
    # uma comecando em start_s=0.0 RELATIVO a propria cena no JSON.
    plan = {
        "scenes": [{"scene": 1, "duration_s": 2.0, "formation": {}},
                  {"scene": 2, "duration_s": 2.0, "formation": {}}],
        "commands": [
            {"shot_index": 0, "scene": 1, "start_s": 0.0, "duration_s": 2.0,
             "actor": "ANA", "partner": None, "primitive": "idle", "target": None},
            {"shot_index": 1, "scene": 2, "start_s": 0.0, "duration_s": 2.0,
             "actor": "ANA", "partner": None, "primitive": "idle", "target": None},
        ],
    }
    cmds = commands_for_actor(plan, "ANA")
    assert cmds[0]["start_s"] == 0.0
    assert cmds[1]["start_s"] == 2.0  # deslocado pela duracao da cena 1 -- nao sobrepoe mais
    assert current_command(cmds, 1.0)["shot_index"] == 0
    assert current_command(cmds, 3.0)["shot_index"] == 1  # ANTES: nunca era selecionada (achado #1)


def test_partner_anchor_prefers_explicit_destination_then_formation():
    plan = _plan()
    formation = formation_by_scene(plan)[1]
    with_dest = {"partner": "BIA", "target": {"destination": [5.0, 5.0]}}
    assert partner_anchor(with_dest, formation) == [5.0, 5.0]
    without_dest = {"partner": "BIA", "target": {"mode": "face_partner"}}
    assert partner_anchor(without_dest, formation) == [1.2, 0.0]
    no_partner = {"partner": None, "target": {}}
    assert partner_anchor(no_partner, formation) is None


def test_walk_target_points_toward_partner_and_signals_arrival():
    formation = formation_by_scene(_plan())[1]
    command = {"partner": "BIA", "target": {"destination": [1.2, 0.0]}}
    direction, chegou = walk_target(command, np.array([-1.2, 0.0]), formation)
    assert not chegou
    assert direction[0] > 0.99  # anda reto na direcao +x, rumo a BIA
    direction, chegou = walk_target(command, np.array([1.15, 0.0]), formation)
    assert chegou and direction is None


def test_walk_target_does_not_double_count_min_separation_m():
    # ACHADO 2026-09-17: destino ja e o ponto de encontro (metade da separacao
    # embutida por _meeting_point) -- se walk_target tambem exigisse
    # min_separation_m inteiro pra "chegar", o andarilho pararia sem sair do
    # lugar (0,5 m de distancia real < 1,0 m de piso duplicado). So a
    # tolerancia de chegada (ARRIVAL_EPS_M) deve valer aqui.
    command = {"partner": "BIA", "target": {"destination": [0.5, 0.0], "min_separation_m": 1.0}}
    direction, chegou = walk_target(command, np.array([0.0, 0.0]), {})  # 0.5 m do destino
    assert not chegou and direction[0] > 0.99


def test_partner_anchor_prefers_live_position_over_static_destination():
    command = {"partner": "BIA", "target": {"destination": [5.0, 5.0]}}
    formation = {"BIA": {"anchor_m": [1.2, 0.0]}}
    live = {"BIA": np.array([3.0, 4.0])}
    assert partner_anchor(command, formation, live_positions=live) == [3.0, 4.0]
    # sem BIA em live_positions, cai pro destino estatico normalmente
    assert partner_anchor(command, formation, live_positions={}) == [5.0, 5.0]


def test_walk_target_brakes_on_live_proximity_even_before_reaching_destination():
    # Destino ao vivo esta longe (BIA em (5,0)), mas ela JA esta mais perto do
    # que o minimo seguro (+ margem de frenagem) agora -- freia na hora, nao
    # espera chegar no alvo.
    command = {"partner": "BIA", "target": {"destination": [5.0, 0.0]}}
    live = {"BIA": np.array([0.5, 0.0])}
    direction, chegou = walk_target(
        command, np.array([0.0, 0.0]), {}, live_positions=live, min_separation_m=1.0,
        brake_margin_m=0.0)
    assert chegou and direction is None
    # Fora do alcance do freio (mesmo com a margem), continua andando.
    direction, chegou = walk_target(
        command, np.array([-2.0, 0.0]), {}, live_positions=live, min_separation_m=1.0,
        brake_margin_m=0.0)
    assert not chegou and direction[0] > 0.99


def test_walk_target_brake_margin_starts_stopping_earlier():
    # Achado 2026-09-17: o clipe de caminhada tem momento e nao para na hora
    # que o modo muda -- a margem faz o freio disparar ANTES do minimo puro,
    # pra absorver esse momento (calibrada empiricamente, nao arbitraria).
    command = {"partner": "BIA", "target": {"destination": [5.0, 0.0]}}
    live = {"BIA": np.array([1.3, 0.0])}  # 1.3 m de distancia
    # min_separation=1.0 sem margem: ainda anda (1.3 > 1.0).
    direction, chegou = walk_target(command, np.array([0.0, 0.0]), {},
                                    live_positions=live, min_separation_m=1.0, brake_margin_m=0.0)
    assert not chegou
    # Com margem de 0.5, o gatilho sobe pra 1.5 -- 1.3 ja esta dentro, freia.
    direction, chegou = walk_target(command, np.array([0.0, 0.0]), {},
                                    live_positions=live, min_separation_m=1.0, brake_margin_m=0.5)
    assert chegou and direction is None


def test_walk_target_without_live_positions_ignores_min_separation_m():
    # Sem live_positions (um agente so, modo de sempre), min_separation_m nao
    # se aplica -- so a tolerancia de chegada contra o destino.
    command = {"partner": "BIA", "target": {"destination": [0.05, 0.0]}}
    direction, chegou = walk_target(command, np.array([0.0, 0.0]), {}, min_separation_m=1.0)
    assert chegou and direction is None  # 0.05 m < ARRIVAL_EPS_M, chega normalmente


def test_walk_target_none_destination_returns_none_without_arrival():
    command = {"partner": None, "target": {"mode": "short_path", "destination": None}}
    direction, chegou = walk_target(command, np.array([0.0, 0.0]), {})
    assert direction is None and not chegou


def test_facing_for_turns_toward_center_without_partner():
    command = {"partner": None, "target": {"mode": "turn_in_place"}}
    prev = np.tile(np.array([2.0, 0.0]), (3, 1))
    facing = facing_for(command, np.array([2.0, 0.0]), {}, prev)
    assert facing[0] < 0  # esta a direita do centro (x>0): encara a esquerda


def test_maintain_heading_continues_last_velocity_direction():
    prev = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    heading = maintain_heading(prev)
    assert heading[0] > 0.99 and abs(heading[1]) < 1e-6


def test_maintain_heading_falls_back_when_stationary():
    prev = np.tile(np.array([1.0, 1.0]), (4, 1))
    heading = maintain_heading(prev)
    assert np.allclose(heading, [1.0, 0.0, 0.0])


# -- ACHADO 2026-09-17 (auditoria externa) #5: escala aplicada duas vezes --

def test_partner_anchor_scales_static_anchor_but_not_live_position():
    formation = {"BIA": {"anchor_m": [2.0, 0.0]}}
    cmd = {"partner": "BIA", "target": None}
    assert partner_anchor(cmd, formation, anchor_scale_m=2.0) == [4.0, 0.0]
    live = {"BIA": [2.0, 0.0]}
    # ao vivo ja e metro de mundo -- NAO escala de novo (seria [4.0, 0.0] se escalasse)
    assert partner_anchor(cmd, formation, live_positions=live, anchor_scale_m=2.0) == [2.0, 0.0]


def test_partner_anchor_scales_json_destination():
    cmd = {"partner": "BIA", "target": {"destination": [3.0, 1.0]}}
    assert partner_anchor(cmd, {}, anchor_scale_m=2.0) == [6.0, 2.0]


def test_facing_for_does_not_double_scale_live_position():
    # Reproducao exata do achado: ator em x=3, parceiro AO VIVO em x=2,
    # escala=2 -- a versao com bug dava direcao +X (parceiro pareceria a
    # DIREITA quando na verdade esta a ESQUERDA).
    cmd = {"partner": "BIA", "target": None}
    root_xy = np.array([3.0, 0.0])
    prev = np.zeros((2, 2))
    live = {"BIA": np.array([2.0, 0.0])}
    facing = facing_for(cmd, root_xy, {}, prev, anchor_scale_m=2.0, live_positions=live)
    assert facing[0] < 0  # parceiro esta a esquerda (x=2 < x=3) -- direcao deve ser -X


def test_walk_target_does_not_double_scale_live_position_direction():
    cmd = {"partner": "BIA", "target": None}
    root_xy = np.array([3.0, 0.0])
    live = {"BIA": np.array([2.0, 0.0])}
    direction, chegou = walk_target(cmd, root_xy, {}, anchor_scale_m=2.0, live_positions=live)
    assert direction[0] < 0  # mesma logica: parceiro a esquerda, direcao -X
    assert not chegou


# -- ACHADO 2026-09-17 (auditoria externa) #6: freio so via o parceiro --

def test_walk_target_brakes_against_any_actor_not_just_designated_partner():
    cmd = {"partner": "BIA", "target": {"destination": [10.0, 0.0]}}
    root_xy = np.array([0.0, 0.0])
    live = {"BIA": np.array([5.0, 0.0]), "CARLOS": np.array([0.1, 0.0])}
    direction, chegou = walk_target(cmd, root_xy, {}, live_positions=live,
                                    min_separation_m=1.0, brake_margin_m=0.0, own_actor="ANA")
    assert chegou is True
    assert direction is None


def test_walk_target_brakes_even_without_designated_partner():
    # "locomote" sem parceiro (target/destino None de proposito) -- a versao
    # antiga nem entrava no bloco de freio porque so olhava command["partner"].
    cmd = {"partner": None, "target": None}
    root_xy = np.array([0.0, 0.0])
    live = {"CARLOS": np.array([0.1, 0.0])}
    direction, chegou = walk_target(cmd, root_xy, {}, live_positions=live,
                                    min_separation_m=1.0, brake_margin_m=0.0, own_actor="ANA")
    assert chegou is True


def test_walk_target_own_actor_excluded_from_brake_scan():
    cmd = {"partner": "BIA", "target": {"destination": [10.0, 0.0]}}
    root_xy = np.array([0.0, 0.0])
    live = {"ANA": np.array([0.0, 0.0]), "BIA": np.array([5.0, 0.0])}
    direction, chegou = walk_target(cmd, root_xy, {}, live_positions=live,
                                    min_separation_m=1.0, brake_margin_m=0.0, own_actor="ANA")
    assert chegou is False  # nao freia por causa da propria entrada em live_positions
