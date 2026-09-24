import numpy as np
import pytest

from script_pipeline.motion_to_h3_controlnet import (
    h3_frame_count,
    render_h3_control_video,
    round_to_multiple_of_32,
)


# ACHADO 2026-09-17 (auditoria externa) #4: o adaptador arredondava pro
# 17n+5 mais PROXIMO, mas o node 131 (ComfyMathExpression) do workflow real
# arredonda pra CIMA -- "max(5, round(a*24)) + (5 - (... % 17)) % 17". Os
# casos abaixo replicam a formula exata (extraida do JSON do workflow).
@pytest.mark.parametrize("n_raw,expected", [
    (22, 22),      # ja e 17*1+5, fica igual
    (124, 124),    # exemplo do H3: 124 quadros ~5,17s a 24fps -- o node bate igual
    (5, 5),        # minimo do PROPRIO node e 5, nao 22
    (0, 5),        # abaixo do minimo, sobe pro minimo (nao pro 17n+5 minimo)
    (120, 124),    # sobe pro 17n+5 SEGUINTE (n=7), nao arredonda por proximidade
    (108, 124),    # ACHADO: a versao antiga dava 107 (mais perto); o node real pede 124
])
def test_h3_frame_count_matches_node_131_formula(n_raw, expected):
    assert h3_frame_count(n_raw) == expected


def test_h3_frame_count_always_ceils_to_17n_plus_5_and_never_below_5():
    for n_raw in range(0, 300):
        n_frames = h3_frame_count(n_raw)
        assert (n_frames - 5) % 17 == 0
        assert n_frames >= 5
        assert n_frames >= n_raw or n_raw < 5  # nunca arredonda pra BAIXO (exceto o piso de 5)


@pytest.mark.parametrize("value,expected", [(0, 32), (1, 32), (16, 32), (17, 32),
                                            (480, 480), (500, 512), (863, 864)])
def test_round_to_multiple_of_32(value, expected):
    assert round_to_multiple_of_32(value) == expected


def test_render_h3_control_video_enforces_constraints(tmp_path):
    ffmpeg = pytest.importorskip("shutil").which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    rng = np.random.default_rng(3)
    joints = rng.normal(size=(150, 2, 55, 3)) * 0.3
    out_path = tmp_path / "control_video.mp4"
    n, w, h = render_h3_control_video(joints, out_path, width=800, height=470,
                                      duration_s=5.0, source_fps=30.0)
    assert (n - 5) % 17 == 0
    assert n == 124  # 5.0s * 24fps = 120 -> sobe pro 17n+5 seguinte = 124
    assert w % 32 == 0 and h % 32 == 0
    assert out_path.exists()


def test_render_h3_control_video_requires_duration_or_num_frames(tmp_path):
    joints = np.zeros((10, 22, 3))
    with pytest.raises(ValueError):
        render_h3_control_video(joints, tmp_path / "x.mp4", width=480, height=480)
