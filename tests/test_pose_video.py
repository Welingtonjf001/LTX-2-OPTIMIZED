import numpy as np
import pytest

from script_pipeline.pose_video import (
    COCO18_COLORS,
    COCO18_EDGES,
    SMPL_TO_COCO18,
    _rgb_to_bgr_tuple,
    draw_coco18_frame,
    fit_transform,
    project_orthographic,
    render_pose_video,
    resample_time,
    smpl_to_coco18,
)


def test_smpl_to_coco18_shape_and_indices():
    joints = np.arange(22 * 3, dtype=np.float64).reshape(22, 3)
    coco = smpl_to_coco18(joints)
    assert coco.shape == (18, 3)
    np.testing.assert_array_equal(coco[0], joints[15])  # Nose <- head
    np.testing.assert_array_equal(coco[1], joints[12])  # Neck
    np.testing.assert_array_equal(coco[8], joints[2])   # RHip
    np.testing.assert_array_equal(coco[11], joints[1])  # LHip


def test_smpl_to_coco18_works_with_smplx_55_joints():
    joints = np.random.default_rng(0).normal(size=(5, 55, 3))
    coco = smpl_to_coco18(joints)
    assert coco.shape == (5, 18, 3)
    np.testing.assert_array_equal(coco[:, 0], joints[:, 15])


def test_smpl_to_coco18_rejects_too_few_joints():
    with pytest.raises(ValueError):
        smpl_to_coco18(np.zeros((10, 3)))


def test_coco18_edges_reference_valid_indices():
    for i, j in COCO18_EDGES:
        assert 0 <= i < 18 and 0 <= j < 18
    assert len(SMPL_TO_COCO18) == 18


def test_project_orthographic_identity_at_zero_rotation():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    proj = project_orthographic(points, azimuth_deg=0.0, elevation_deg=0.0)
    np.testing.assert_allclose(proj, points[..., :2])


def test_project_orthographic_azimuth_90_maps_z_to_x():
    # xr = x*cos(az) - z*sin(az): at az=90 a point on +Z lands on -X.
    points = np.array([[0.0, 0.0, 1.0]])
    proj = project_orthographic(points, azimuth_deg=90.0)
    np.testing.assert_allclose(proj, [[-1.0, 0.0]], atol=1e-10)


def test_fit_transform_centers_and_fills_frame():
    points = np.array([[-1.0, -1.0], [1.0, 1.0]])
    to_pixels = fit_transform(points, width=200, height=100, margin=0.0)
    center_px = to_pixels(np.array([0.0, 0.0]))
    np.testing.assert_allclose(center_px, [100.0, 50.0])
    # (-1,-1) is the lower-left in world space -> lower-left in image (y grows down).
    corner_px = to_pixels(np.array([-1.0, -1.0]))
    assert corner_px[0] < 100.0
    assert corner_px[1] > 50.0


def test_fit_transform_rejects_all_nan():
    with pytest.raises(ValueError):
        fit_transform(np.full((3, 2), np.nan), width=100, height=100)


def test_resample_time_preserves_duration_and_endpoints():
    t = 10
    joints = np.zeros((1, t, 18, 3))
    joints[0, :, 0, 0] = np.arange(t)
    out = resample_time(joints, source_fps=30.0, target_fps=24.0)
    expected_frames = round(t * 24.0 / 30.0)
    assert out.shape == (1, expected_frames, 18, 3)
    np.testing.assert_allclose(out[0, 0, 0, 0], 0.0)
    np.testing.assert_allclose(out[0, -1, 0, 0], t - 1, atol=1e-9)


def test_resample_time_single_frame_repeats():
    joints = np.ones((2, 1, 18, 3))
    out = resample_time(joints, source_fps=30.0, target_fps=24.0, target_frames=5)
    assert out.shape == (2, 5, 18, 3)
    np.testing.assert_allclose(out, 1.0)


def test_render_pose_video_single_actor_smoke(tmp_path):
    ffmpeg = pytest.importorskip("shutil").which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    rng = np.random.default_rng(1)
    joints = rng.normal(size=(6, 22, 3)) * 0.3
    joints[:, 0] = 0.0  # pelvis parado, so os membros oscilam
    out_path = tmp_path / "preview.mp4"
    n = render_pose_video(joints, out_path, width=128, height=128, fps=24.0,
                          source_fps=24.0)
    assert n == 6
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_render_pose_video_two_actors_uses_TAJ3_convention(tmp_path):
    ffmpeg = pytest.importorskip("shutil").which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    rng = np.random.default_rng(2)
    t, a = 4, 2
    joints = rng.normal(size=(t, a, 55, 3)) * 0.3
    out_path = tmp_path / "two_actors.mp4"
    n = render_pose_video(joints, out_path, width=96, height=96, fps=24.0,
                          source_fps=30.0, num_frames=8)
    assert n == 8
    assert out_path.stat().st_size > 500


def test_render_pose_video_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        render_pose_video(np.zeros((3, 3)), tmp_path / "x.mp4", width=64, height=64)


@pytest.mark.parametrize("kwargs", [
    dict(fps=0.0), dict(fps=-1.0), dict(fps=float("nan")), dict(fps=float("inf")),
    dict(source_fps=0.0), dict(source_fps=float("nan")),
])
def test_render_pose_video_rejects_invalid_fps(tmp_path, kwargs):
    joints = np.zeros((3, 22, 3))
    with pytest.raises(ValueError):
        render_pose_video(joints, tmp_path / "x.mp4", width=64, height=64, **kwargs)


@pytest.mark.parametrize("width,height", [(0, 64), (64, 0), (-1, 64)])
def test_render_pose_video_rejects_invalid_dimensions(tmp_path, width, height):
    joints = np.zeros((3, 22, 3))
    with pytest.raises(ValueError):
        render_pose_video(joints, tmp_path / "x.mp4", width=width, height=height)


def test_render_pose_video_rejects_wrong_last_dimension(tmp_path):
    joints = np.zeros((3, 22, 2))  # deveria ser (...,3)
    with pytest.raises(ValueError):
        render_pose_video(joints, tmp_path / "x.mp4", width=64, height=64)


def test_render_pose_video_rejects_zero_frames(tmp_path):
    joints = np.zeros((0, 22, 3))
    with pytest.raises(ValueError):
        render_pose_video(joints, tmp_path / "x.mp4", width=64, height=64)


# -- ACHADO 2026-09-17 (auditoria externa) #3: RGB/BGR trocados --

def test_rgb_to_bgr_tuple_reverses_channel_order():
    assert _rgb_to_bgr_tuple([255, 0, 0]) == (0, 0, 255)
    assert _rgb_to_bgr_tuple(np.array([10, 20, 30], dtype=np.uint8)) == (30, 20, 10)


def test_drawn_pixel_matches_intended_rgb_color_not_swapped():
    # Reproducao exata do achado: um segmento vermelho ([255,0,0] na tabela)
    # tem que sair vermelho no CANVAS (que cv2 sempre guarda como B,G,R --
    # o mesmo layout que o pix_fmt bgr24 do ffmpeg espera), nao azul.
    canvas = np.zeros((40, 40, 3), dtype=np.uint8)
    i, j = COCO18_EDGES[0]
    points = np.full((18, 2), -1000.0)  # fora do canvas, exceto o membro testado
    points[i] = [10, 20]
    points[j] = [30, 20]
    draw_coco18_frame(canvas, [points], thickness=2, radius=0)
    pixel_bgr = canvas[20, 20]
    intended_rgb = COCO18_COLORS[0]
    np.testing.assert_array_equal(pixel_bgr, intended_rgb[::-1])  # canvas = B,G,R


def test_joint_circles_use_per_joint_color_not_flat_white():
    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    points = np.full((18, 2), -1000.0)
    points[0] = [10, 10]  # Nose -> COCO18_COLORS[0] = [255,0,0] (vermelho)
    draw_coco18_frame(canvas, [points], radius=3)
    pixel_bgr = canvas[10, 10]
    assert tuple(pixel_bgr) != (255, 255, 255)  # nao mais branco fixo
    np.testing.assert_array_equal(pixel_bgr, COCO18_COLORS[0][::-1])
