"""Poses do manequim de blocking: esqueleto COCO-18 no referencial LOCAL do personagem.

Um so lugar decide onde fica cada junta, e tres consumidores leem daqui:

- `blender_scene_worker.py` constroi a malha (torso, cabeca, membros) a partir das juntas;
- `camera_geometry.socket_position` pendura objetos na mao (a mao segue a pose);
- `spatial_planner.camera_for_shot` mira a cabeca (em pe, ajoelhado ou no chao).

Referencial local: Z para cima, em metros; o personagem OLHA para -Y (yaw 0 = de frente para a
camera padrao, que fica em -Y); o lado DIREITO anatomico do personagem e -X. `yaw` gira o
referencial em torno de Z (graus); a direcao do olhar no mundo e (sin(yaw), -cos(yaw)).

BUGFIX 2026-09-27: o manequim antigo punha o ombro "direito" (indice 2 do COCO) em +X, isto e,
no lado ESQUERDO anatomico de quem olha para -Y. O `pose.png` saia espelhado (esqueleto com
rosto visivel e membros de costas -- configuracao impossivel para o OpenPose/union-control) e o
soquete `right_hand` pendurava o objeto na mao esquerda. Aqui a convencao e anatomica.

Uma pose e um NOME desta biblioteca ou {"joints": [[x, y, z] x 18]} em coordenadas locais
(pose derivada do InterGen, por exemplo). Sem `pose`, o personagem esta em pe (`standing`).
Puro Python: roda dentro do Blender e no processo do pipeline sem dependencia nova."""
from __future__ import annotations

import math

# COCO-18: 0 nariz, 1 pescoco, 2 ombro D, 3 cotovelo D, 4 pulso D, 5 ombro E, 6 cotovelo E,
# 7 pulso E, 8 quadril D, 9 joelho D, 10 tornozelo D, 11 quadril E, 12 joelho E, 13 tornozelo E,
# 14 olho D, 15 olho E, 16 orelha D, 17 orelha E.
NOSE, NECK, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST = range(8)
R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE, R_EYE, L_EYE, R_EAR, L_EAR = range(8, 18)
SOCKET_JOINT = {"right_hand": R_WRIST, "left_hand": L_WRIST}
FOCUS_DROP_M = 0.15  # a camera mira 15 cm abaixo do centro da cabeca (rosto/peito alto)


def _v(a):
    return [float(a[0]), float(a[1]), float(a[2])]


def _add(a, b, k=1.0):
    return [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _norm(a):
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if n < 1e-9:
        raise ValueError("vetor nulo")
    return [a[0] / n, a[1] / n, a[2] / n]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _head(center, forward, neck):
    """Nariz, olhos e orelhas a partir do centro da cabeca e da direcao do rosto."""
    f = _norm(forward)
    up = _sub(center, neck)
    up = _norm(_sub(up, [x * _dot(up, f) for x in f]))
    right = _cross(f, up)  # olhando para f com a cabeca em up, a direita anatomica
    nose = _add(_add(center, f, .12), up, -.02)
    r_eye = _add(_add(_add(center, f, .113), right, .046), up, .01)
    l_eye = _add(_add(_add(center, f, .113), right, -.046), up, .01)
    return nose, r_eye, l_eye, _add(center, right, .12), _add(center, right, -.12)


def _pose(neck, r_sh, r_el, r_wr, l_sh, l_el, l_wr, r_hip, r_kn, r_an, l_hip, l_kn, l_an,
          head_center, head_forward):
    nose, r_eye, l_eye, r_ear, l_ear = _head(head_center, head_forward, neck)
    return [nose, _v(neck), _v(r_sh), _v(r_el), _v(r_wr), _v(l_sh), _v(l_el), _v(l_wr),
            _v(r_hip), _v(r_kn), _v(r_an), _v(l_hip), _v(l_kn), _v(l_an), r_eye, l_eye, r_ear, l_ear]


_WALK_LEGS = ((-.12, 0, .94), (-.13, -.22, .54), (-.13, -.30, .12),
              (.12, 0, .94), (.13, .12, .52), (.13, .32, .16))

POSES: dict[str, list[list[float]]] = {
    # Mesmas medidas do manequim anterior (ombros a 1,40 m, cabeca a 1,70 m), lado anatomico.
    "standing": _pose((0, 0, 1.45), (-.23, 0, 1.40), (-.31, 0, 1.15), (-.30, -.08, 1.00),
                      (.23, 0, 1.40), (.31, 0, 1.15), (.30, -.08, 1.00),
                      (-.12, 0, .95), (-.13, 0, .53), (-.13, 0, .13),
                      (.12, 0, .95), (.13, 0, .53), (.13, 0, .13), (0, 0, 1.70), (0, -1, 0)),
    "walking": _pose((0, -.04, 1.44), (-.23, -.03, 1.39), (-.28, .12, 1.16), (-.27, .20, .98),
                     (.23, -.03, 1.39), (.28, -.18, 1.16), (.26, -.28, 1.00), *_WALK_LEGS,
                     (0, -.05, 1.69), (0, -1, 0)),
    "running": _pose((0, -.18, 1.38), (-.22, -.16, 1.34), (-.27, .10, 1.12), (-.25, -.05, 1.00),
                     (.22, -.16, 1.34), (.27, -.36, 1.14), (.20, -.40, 1.34),
                     (-.12, 0, .92), (-.13, -.38, .70), (-.13, -.30, .30),
                     (.12, 0, .92), (.13, .18, .50), (.13, .45, .30), (0, -.24, 1.62), (0, -1, -.15)),
    # Duas maos na arma, na altura do ombro, apontando para a frente (-Y local).
    "aiming": _pose((0, 0, 1.45), (-.23, 0, 1.40), (-.18, -.30, 1.38), (-.06, -.58, 1.40),
                    (.23, 0, 1.40), (.14, -.30, 1.33), (-.02, -.56, 1.38),
                    (-.12, 0, .93), (-.18, -.08, .52), (-.22, 0, .12),
                    (.12, 0, .93), (.18, -.12, .52), (.22, -.08, .12), (0, -.02, 1.70), (0, -1, 0)),
    # Mira para o alto (janela, telhado): bracos a ~35 graus acima do horizonte.
    "aiming_high": _pose((0, .02, 1.45), (-.23, .02, 1.40), (-.17, -.24, 1.55), (-.06, -.44, 1.75),
                         (.23, .02, 1.40), (.13, -.24, 1.50), (-.02, -.42, 1.72),
                         (-.12, 0, .93), (-.18, -.08, .52), (-.22, 0, .12),
                         (.12, 0, .93), (.18, -.12, .52), (.22, -.08, .12),
                         (0, .01, 1.70), (0, -1, .55)),
    "crouching": _pose((0, -.20, 1.08), (-.22, -.18, 1.04), (-.28, -.35, .85), (-.22, -.45, .70),
                       (.22, -.18, 1.04), (.28, -.35, .85), (.22, -.45, .70),
                       (-.12, .05, .62), (-.14, -.30, .50), (-.14, -.15, .10),
                       (.12, .05, .62), (.14, -.30, .50), (.14, -.15, .10), (0, -.30, 1.30), (0, -1, -.2)),
    # Joelho direito no chao, pe esquerdo plantado, maos para baixo e para a frente (imobilizando).
    "kneeling": _pose((0, -.15, .99), (-.22, -.13, .95), (-.25, -.35, .78), (-.18, -.52, .60),
                      (.22, -.13, .95), (.25, -.35, .78), (.18, -.52, .60),
                      (-.12, .08, .52), (-.13, .05, .10), (-.13, .45, .08),
                      (.12, .08, .52), (.14, -.32, .50), (.14, -.34, .08), (0, -.24, 1.22), (0, -.8, -.6)),
    # Deitado de brucos: cabeca para -Y local, rosto para o chao, quadril na posicao da entidade.
    "prone": _pose((0, -.50, .17), (-.23, -.46, .17), (-.38, -.62, .09), (-.35, -.85, .07),
                   (.23, -.46, .17), (.38, -.62, .09), (.35, -.85, .07),
                   (-.12, 0, .15), (-.13, .42, .11), (-.13, .84, .10),
                   (.12, 0, .15), (.13, .42, .11), (.13, .84, .10), (0, -.73, .18), (0, 0, -1)),
    # De brucos com os pulsos juntos nas costas (algemado).
    "prone_cuffed": _pose((0, -.50, .17), (-.23, -.46, .17), (-.28, -.22, .24), (-.06, .02, .27),
                          (.23, -.46, .17), (.28, -.22, .24), (.06, .02, .27),
                          (-.12, 0, .15), (-.13, .42, .11), (-.13, .84, .10),
                          (.12, 0, .15), (.13, .42, .11), (.13, .84, .10), (0, -.73, .18), (0, 0, -1)),
    # Bracos abertos, joelhos levemente flexionados: corpo usado como escudo.
    "shielding": _pose((0, -.05, 1.40), (-.23, -.04, 1.36), (-.48, -.06, 1.30), (-.66, -.10, 1.18),
                       (.23, -.04, 1.36), (.48, -.06, 1.30), (.66, -.10, 1.18),
                       (-.13, 0, .90), (-.20, -.10, .50), (-.24, 0, .12),
                       (.13, 0, .90), (.20, -.10, .50), (.24, 0, .12), (0, -.08, 1.64), (0, -1, 0)),
    # Braco direito estendido para a frente e para baixo (ajudar a levantar, puxar, pegar).
    "reaching": _pose((0, -.10, 1.43), (-.22, -.09, 1.38), (-.22, -.36, 1.22), (-.18, -.60, 1.05),
                      (.23, -.09, 1.38), (.31, -.05, 1.13), (.30, -.12, .98),
                      (-.12, 0, .94), (-.13, -.06, .53), (-.13, 0, .13),
                      (.12, 0, .94), (.13, .02, .53), (.13, .06, .13), (0, -.14, 1.67), (0, -1, -.15)),
    # Andando com a mao direita nas costas/ombro de quem e conduzido (a frente e a direita).
    "escorting": _pose((0, -.04, 1.44), (-.23, -.03, 1.39), (-.30, -.25, 1.25), (-.35, -.45, 1.20),
                       (.23, -.03, 1.39), (.28, -.18, 1.16), (.26, -.28, 1.00), *_WALK_LEGS,
                       (0, -.05, 1.69), (0, -1, 0)),
    "seated": _pose((0, .02, 1.00), (-.23, .02, .95), (-.28, -.10, .72), (-.20, -.35, .55),
                    (.23, .02, .95), (.28, -.10, .72), (.20, -.35, .55),
                    (-.12, 0, .50), (-.13, -.45, .50), (-.13, -.47, .08),
                    (.12, 0, .50), (.13, -.45, .50), (.13, -.47, .08), (0, 0, 1.25), (0, -1, 0)),
}


def validate_pose(pose) -> None:
    if pose is None or (isinstance(pose, str) and pose in POSES):
        return
    if isinstance(pose, dict) and isinstance(pose.get("joints"), list) and len(pose["joints"]) == 18:
        for joint in pose["joints"]:
            if not (isinstance(joint, list) and len(joint) == 3 and all(
                    isinstance(c, (int, float)) and math.isfinite(c) for c in joint)):
                raise ValueError("pose joints must be 18 finite [x, y, z] lists")
        return
    raise ValueError(f"Unknown mannequin pose: {pose!r} (known: {sorted(POSES)})")


def local_joints(pose) -> list[list[float]]:
    validate_pose(pose)
    if pose is None:
        pose = "standing"
    source = POSES[pose] if isinstance(pose, str) else pose["joints"]
    return [list(map(float, j)) for j in source]


def _rotate(local, yaw_deg):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return [c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]]


def to_world(entity, local):
    x, y, z = entity["position"]
    r = _rotate(local, entity.get("yaw", 0))
    return [x + r[0], y + r[1], z + r[2]]


def world_joints(entity) -> list[list[float]]:
    return [to_world(entity, j) for j in local_joints(entity.get("pose"))]


def from_world(world_points, position, yaw_deg) -> list[list[float]]:
    """Inverso de `to_world`: juntas no mundo -> referencial local de uma entidade."""
    out = []
    for p in world_points:
        d = [p[0] - position[0], p[1] - position[1], p[2] - position[2]]
        out.append(_rotate(d, -yaw_deg))
    return out


def head_center(joints) -> list[float]:
    return [(joints[R_EAR][i] + joints[L_EAR][i]) / 2 for i in range(3)]


def focus_point(entity) -> list[float]:
    """Ponto que a camera mira: 15 cm abaixo do centro da cabeca, onde quer que ela esteja.
    Em pe e yaw 0 da exatamente posicao + 1,55 m (o valor fixo antigo do planner)."""
    center = head_center(world_joints(entity))
    return [center[0], center[1], center[2] - FOCUS_DROP_M]


def socket_world(entity, socket) -> list[float]:
    if socket not in SOCKET_JOINT:
        raise ValueError("Unknown attachment socket")
    return world_joints(entity)[SOCKET_JOINT[socket]]


def facing_yaw(direction_xy) -> float:
    """Yaw (graus) que faz o personagem olhar na direcao XY dada."""
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    if math.hypot(dx, dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, -dy))


def facing_vector(yaw_deg) -> list[float]:
    a = math.radians(yaw_deg)
    return [math.sin(a), -math.cos(a)]


def complete_face(joints) -> list[list[float]]:
    """Esqueletos vindos do SMPL (InterGen) repetem a cabeca em nariz/olhos/orelhas. Sem
    orelhas distintas a malha nao tem orientacao de rosto; reconstroi pela linha dos ombros."""
    joints = [list(map(float, j)) for j in joints]
    if math.dist(joints[R_EAR], joints[L_EAR]) > 1e-3:
        return joints
    center, neck = joints[NOSE], joints[NECK]
    right = _norm(_sub(joints[R_SHOULDER], joints[L_SHOULDER]))
    up = _norm(_sub(center, neck))
    forward = _cross(up, right)
    nose, r_eye, l_eye, r_ear, l_ear = _head(center, forward, neck)
    joints[NOSE], joints[R_EYE], joints[L_EYE], joints[R_EAR], joints[L_EAR] = nose, r_eye, l_eye, r_ear, l_ear
    return joints


def ease(t: float) -> float:
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3 - 2 * t)


def blend_joints(a, b, t) -> list[list[float]]:
    return [[pa + (pb - pa) * t for pa, pb in zip(ja, jb)] for ja, jb in zip(a, b)]


def blend_yaw(a, b, t) -> float:
    delta = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return float(a) + delta * t
