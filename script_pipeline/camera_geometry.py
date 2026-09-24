"""Shared Z-up, metre, perspective camera contract (Blender local -Z, up +Y)."""
import math
import numpy as np


def validate_camera(camera):
    for key in ('position', 'target'):
        value = camera.get(key)
        if not isinstance(value, list) or len(value) != 3 or not all(math.isfinite(x) for x in value):
            raise ValueError(f'Invalid camera {key}')
    for key in ('lens_mm', 'sensor_mm', 'width', 'height'):
        if not math.isfinite(camera.get(key, 0)) or camera.get(key, 0) <= 0:
            raise ValueError(f'Invalid camera {key}')
    direction = np.subtract(camera['target'], camera['position'])
    if np.linalg.norm(direction) < 1e-8 or np.linalg.norm(np.cross(direction, [0, 0, 1])) < 1e-8:
        raise ValueError('Degenerate camera direction')


def project(points, camera):
    validate_camera(camera)
    forward = np.subtract(camera['target'], camera['position']).astype(float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points, dtype=float) - camera['position']
    depth = delta @ forward
    focal = camera['lens_mm'] / camera['sensor_mm'] * camera['width']
    safe = np.where(depth > 0, depth, np.nan)
    xy = np.stack([camera['width'] / 2 + focal * (delta @ right) / safe,
                   camera['height'] / 2 - focal * (delta @ up) / safe], axis=-1)
    visible = (depth > camera.get('near', .05)) & (depth < camera.get('far', 100))
    visible &= (xy[..., 0] >= 0) & (xy[..., 0] < camera['width']) & (xy[..., 1] >= 0) & (xy[..., 1] < camera['height'])
    return xy, depth, visible


def y_up_to_z_up(points):
    points = np.asarray(points)
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def socket_position(entity, socket):
    local = np.array([-.30 if socket == 'left_hand' else .30, -.08, 1.00])
    angle = math.radians(entity.get('yaw', 0))
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(entity['position']) + np.array([c*local[0]-s*local[1], s*local[0]+c*local[1], local[2]])


def keyframe_indices(editorial_frames):
    if editorial_frames < 9:
        raise ValueError('Spatial video needs at least 9 frames')
    generated = 1 + math.ceil((editorial_frames - 1) / 8) * 8
    # Endpoint occurs within editorial trim, never solely in discarded padding.
    return generated, ((editorial_frames - 1) // 8) * 8

