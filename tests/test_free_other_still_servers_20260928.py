"""Achado 2026-09-28: nada derrubava o Fish Speech (nem Z-Image/Qwen-Image-2.1) antes dos
stills -- so antes do estagio de VIDEO (render_shots.py, achado gemeo 2026-09-22). Medido AO
VIVO: com o Fish Speech ainda residente da etapa de TTS, passos de FLUX que uma placa livre
faz em ~5-9s levavam entre 120s e 470s. `gpu_watchdog.free_other_still_servers` centraliza a
limpeza; `ensure_comfyui_running`/`zimage_backend.ensure_server`/`qwen_image21*.ensure_server`
chamam sempre, mantendo so a porta do motor que vai gerar. Sem GPU."""
from script_pipeline import gpu_watchdog as gw


def test_derruba_todas_menos_a_que_vai_ser_usada(monkeypatch):
    chamadas = []
    monkeypatch.setattr(gw, "free_port", lambda porta, log=print: chamadas.append(porta) or False)
    gw.free_other_still_servers(gw.STILL_ENGINE_PORTS["fish_speech"], log=lambda m: None)
    assert gw.STILL_ENGINE_PORTS["fish_speech"] not in chamadas
    assert set(chamadas) == set(gw.STILL_ENGINE_PORTS.values()) - {gw.STILL_ENGINE_PORTS["fish_speech"]}


def test_sem_keep_port_derruba_tudo(monkeypatch):
    chamadas = []
    monkeypatch.setattr(gw, "free_port", lambda porta, log=print: chamadas.append(porta) or False)
    gw.free_other_still_servers(log=lambda m: None)
    assert set(chamadas) == set(gw.STILL_ENGINE_PORTS.values())


def test_minimax_e_longcat_ficam_de_fora(monkeypatch):
    """So entram em jogo no estagio de video (render_shots.py ja troca essas portas la);
    incluir aqui derrubaria um motor de VIDEO por engano se alguem chamar isto cedo demais."""
    assert 8189 not in gw.STILL_ENGINE_PORTS.values()
    assert 8190 not in gw.STILL_ENGINE_PORTS.values()


def test_ensure_comfyui_running_chama_a_limpeza_antes_de_checar_o_servidor(monkeypatch):
    from script_pipeline import generate_storyboards as gs
    chamadas = []
    monkeypatch.setattr(gw, "free_other_still_servers", lambda keep=None, log=print: chamadas.append(keep))
    monkeypatch.setattr(gs, "comfy_is_up", lambda server: True)
    monkeypatch.setattr(gs, "_start_stall_watch_once", lambda server, log=print: None)
    assert gs.ensure_comfyui_running("http://127.0.0.1:8188", log=lambda m: None) is True
    assert chamadas == [gw.STILL_ENGINE_PORTS["comfyui"]]


def test_zimage_ensure_server_chama_a_limpeza(monkeypatch):
    import zimage_backend as z
    chamadas = []
    monkeypatch.setattr(gw, "free_other_still_servers", lambda keep=None, log=print: chamadas.append(keep))
    monkeypatch.setattr(z, "server_is_up", lambda: True)
    z.ensure_server(log_cb=lambda m: None)
    assert chamadas == [z.ZIMAGE_PORT]
