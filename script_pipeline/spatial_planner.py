"""Generic spatial shot planning from editorial framing and world entities."""
from __future__ import annotations

import copy
import math


TIGHT_FRAMINGS = {"close", "extreme_close"}


def _entity_point(entity):
    """Onde a camera mira: o rosto na pose ATUAL (em pe da posicao + 1,55 m, como antes;
    ajoelhado ou no chao, a cabeca de verdade -- ver mannequin_poses.focus_point)."""
    from script_pipeline.mannequin_poses import focus_point
    if entity.get("kind") == "prop":
        x, y, z = entity["position"]
        return [float(x), float(y), float(z)]
    return focus_point(entity)


def camera_for_shot(base_camera, shot, entities):
    """Return a deterministic camera matching framing and the named subject.

    Entities face local -Y in the procedural renderer, so the default camera is
    placed on that side. Projects with a different stage orientation may supply
    ``camera_axis`` (a normalized XY vector from subject toward camera).
    """
    camera = copy.deepcopy(base_camera)
    framing = str(shot.get("framing") or "wide").casefold()
    subject_id = shot.get("subject") or ""
    subject = entities.get(subject_id)
    if not subject or framing in {"wide", "full", "insert"}:
        return camera

    target = _entity_point(subject)
    axis = shot.get("camera_axis", [0.0, -1.0])
    length = math.hypot(float(axis[0]), float(axis[1]))
    if length < 1e-6:
        raise ValueError("camera_axis cannot be zero")
    ax, ay = float(axis[0]) / length, float(axis[1]) / length
    settings = {
        "extreme_close": (1.05, 85),
        "close": (1.45, 72),
        "medium": (2.55, 55),
        "ots": (2.8, 58),
    }
    distance, lens = settings.get(framing, (3.4, 48))
    side = str(shot.get("screen_side") or "").casefold()
    lateral = -.22 if side == "left" else (.22 if side == "right" else 0.0)
    camera["position"] = [target[0] + ax * distance + lateral,
                          target[1] + ay * distance,
                          target[2] + (0.08 if framing in TIGHT_FRAMINGS else 0.18)]
    camera["target"] = target
    camera["lens_mm"] = lens
    return camera


def active_entities_for_shot(shot, entities, *, location_cast=None):
    """Choose visible entities without deleting them from the semantic world.

    A spec may provide ``active_entities`` explicitly. Otherwise tight shots
    contain only their subject; open shots use the cast declared for the
    location. Props can be selected independently with ``active_props``.
    """
    explicit = shot.get("active_entities")
    framing = str(shot.get("framing") or "").casefold()
    if explicit is not None:
        active = set(explicit)
    elif framing in TIGHT_FRAMINGS and shot.get("subject"):
        active = {shot["subject"]}
    elif framing in {"wide", "full"} or not (shot.get("subject") or shot.get("co_subject")):
        active = set(entities) if location_cast is None else set(location_cast)
        if shot.get("subject"):
            active.add(shot["subject"])
        if shot.get("co_subject"):
            active.add(shot["co_subject"])
    else:
        # Medium/OTS coverage needs the named blocking participants, not every
        # person and prop known to the location.  Keeping the whole location
        # cast visible was the source of duplicated people and carts.
        active = {x for x in (shot.get("subject"), shot.get("co_subject")) if x}
    active.update(shot.get("active_props") or [])
    unknown = active.difference(entities)
    if unknown:
        raise ValueError(f"Unknown active entities: {sorted(unknown)}")
    return sorted(active)


def state_for_shot(state, active_entities):
    """Visibilidade e a INTERSECAO de "quem o enquadramento mostra" com "quem existe
    semanticamente" -- nao so o primeiro. ACHADO (review adversarial 2026-09-27): sobrescrever
    `present` so pelo enquadramento fazia um evento que tira alguem de cena (`present=False`)
    ser ignorado sempre que esse alguem ainda estava na lista `active_entities` do plano (ex.:
    plano aberto que lista todo o elenco da locacao) -- ele reaparecia no quadro final."""
    result = copy.deepcopy(state)
    active = set(active_entities)
    for entity_id, entity in result["entities"].items():
        entity["present"] = entity_id in active and entity.get("present", True)
    return result


def validate_spatial_shot(shot, entities):
    camera = shot["camera"]
    framing = str(shot.get("framing") or "wide").casefold()
    subject = shot.get("subject") or ""
    if framing in TIGHT_FRAMINGS:
        if not subject or subject not in entities:
            raise ValueError(f"Tight shot {shot.get('id')} requires a known subject")
        active = set(shot.get("active_entities") or [])
        if active and subject not in active:
            raise ValueError(f"Subject {subject} is hidden in {shot.get('id')}")
        target = camera["target"]
        expected = _entity_point(entities[subject])
        if math.dist(target, expected) > .45:
            raise ValueError(f"Camera for {shot.get('id')} does not target {subject}")
        if camera["lens_mm"] < 55:
            raise ValueError(f"Tight shot {shot.get('id')} needs lens >= 55mm")
    return True
