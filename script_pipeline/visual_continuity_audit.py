"""Gate visual multimodal para stills e clipes.

Ao contrario de ``continuity_audit.py`` (que valida o contrato textual), este
modulo entrega os pixels ao Qwen3-VL e bloqueia a proxima etapa quando a imagem
nao confirma o contrato. A regra de aeronave continua opt-in: so e perguntada
quando o plano carrega ``continuity_contract.aircraft``.

Stills: imagem gerada + referencias nominais dos personagens visiveis.
Video: primeiro, meio e ultimo quadro + referencias nominais.

O gate e fail-closed. Se o runtime disser que nao recebeu a imagem, devolver
JSON invalido ou ficar indisponivel, o plano nao e aprovado.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "qwen3-vl:30b"
DEFAULT_PERCEPTION_MODEL = "qwen3-vl:30b"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_json(text: str) -> dict:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("resposta visual nao e um objeto JSON")
    return value


def _ollama_json(model: str, prompt: str, images: list[Path], *, timeout: int = 180) -> dict:
    """Uma repeticao quando a resposta vem truncada/invalida (visto: 5 de 68 planos com
    "Unterminated string" e 800 tokens de teto)."""
    try:
        return _ollama_json_once(model, prompt, images, timeout=timeout)
    except ValueError:
        return _ollama_json_once(model, prompt, images, timeout=timeout, retry=True)


def _ollama_json_once(model: str, prompt: str, images: list[Path], *, timeout: int = 180,
                      retry: bool = False) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": "10m",
        "options": ({"temperature": 0.3, "seed": 7, "num_ctx": 8192, "num_predict": 3000} if retry else
                    {"temperature": 0, "num_ctx": 8192, "num_predict": 2000}),
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [_image_b64(path) for path in images],
        }],
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = body.get("message") or {}
    # Algumas builds thinking do Qwen3-VL devolvem JSON em `thinking` mesmo
    # com think=false e deixam `content` vazio. Aceitar esse campo evita
    # transformar uma resposta valida em erro de parse.
    text = message.get("content") or message.get("thinking") or ""
    return _parse_json(text)


def _cast(run_dir: Path) -> dict:
    return _read_json(run_dir / "characters" / "cast.json", {})


def _reference_paths(run_dir: Path, shot: dict) -> list[Path]:
    cast = _cast(run_dir)
    refs: list[Path] = []
    for name in (shot.get("subject"), shot.get("co_subject")):
        raw = (cast.get(str(name)) or {}).get("reference_image") if name else None
        if raw:
            path = Path(raw)
            if path.exists() and path not in refs:
                refs.append(path)
    return refs


def _contract(shot: dict, *, stage: str, target_count: int, ref_count: int,
              location_ref_count: int = 0) -> dict:
    contract = shot.get("continuity_contract") or {}
    subjects = [x for x in (shot.get("subject"), shot.get("co_subject")) if x]
    speaker = shot.get("subject") if shot.get("line_index") is not None else None
    return {
        "stage": stage,
        "shot_index": shot.get("index"),
        "scene": shot.get("scene"),
        "location_id": shot.get("location_id"),
        "framing": shot.get("framing"),
        "expected_subjects": subjects,
        "speaker_that_must_remain_visible": speaker,
        "persistent_objects": contract.get("persistent_objects") or [],
        "aircraft_rule": contract.get("aircraft"),
        "visual_intent": shot.get("video_prompt") or shot.get("storyboard_prompt") or "",
        "target_images": target_count,
        "reference_images_after_targets": ref_count,
        "location_reference_images_after_character_refs": location_ref_count,
    }


def _perception_prompt(*, stage: str, target_count: int, ref_count: int,
                       location_ref_count: int = 0) -> str:
    return (
        "Perform neutral pixel observation. Do not infer a screenplay, desired result, or continuity requirement. "
        f"The first {target_count} image(s) are {'chronological frames from one video clip' if stage == 'video' else 'one generated still'}. "
        f"The next {ref_count} image(s), if any, are portrait references for identity comparison. "
        f"The final {location_ref_count} image(s), if any, are location references from an earlier wide shot; "
        "compare fixed architecture and spatial layout only, allowing the target camera angle to crop details; "
        "for exterior or sky references compare the aircraft design (same airframe, wing shape, tail colour) "
        "and ignore clouds, weather and camera angle. "
        "Return JSON only. Required keys: image_received (boolean), target_images_seen (integer), "
        "reference_images_seen (integer), targets (array with one object per target image). "
        "Each target object must contain: location_description, framing, primary_people_description, "
        "visible_objects (array), text_detected (boolean), visible_text (array), "
        "text_type (exactly one of none, diegetic_instrument, subtitle_caption, watermark_logo, other), "
        "aircraft_visible (boolean), "
        "aircraft_state (exactly one of airborne, on_ground, unclear, absent), ground_or_runway_visible (boolean), "
        "wheels_visible (boolean), wheels_touching_surface (boolean). "
        "Also return identity_matches (one boolean per reference: true if the SAME person as that reference appears "
        "anywhere in the target image(s); different references may be different people), same_location_across_targets (boolean), same_primary_people_across_targets (boolean), "
        "same_location_as_reference (boolean; true when no location reference was supplied), "
        "primary_person_visible_in_every_target (boolean), and reference_subject_prominent_in_every_target (boolean). "
        "The last field is true only if the referenced speaking subject remains the clear primary subject in every "
        "target: face clearly visible, not reduced to background, not mostly occluded, and not cropped at the frame edge. "
        "Keep every array to at most 8 short items and never repeat an item. Describe only evidence actually visible."
    )


def _evaluation_prompt(shot: dict, *, stage: str, target_count: int,
                       ref_count: int, perception: dict, location_ref_count: int = 0) -> str:
    requirements = _contract(shot, stage=stage, target_count=target_count, ref_count=ref_count,
                             location_ref_count=location_ref_count)
    return (
        "You are the decision pass of a film continuity gate. Compare the neutral pixel observations with the "
        "requirements. Do not reinterpret or contradict the observations to satisfy the requirements. "
        "Do not invent an expected state, position or action for an object. If a persistent object is visible, "
        "its presence satisfies the inventory unless visual_intent or aircraft_rule explicitly requires a state. "
        "Do not require a service cart to be stowed, a person to be seated, or a light to be on/off unless the "
        "requirements say so. Framing labels are approximate and framing mismatch alone must not fail the shot. "
        "Small diegetic environmental text that a real photo of this kind of scene would plausibly contain "
        "(street signs, storefronts, license plates, badges, screens in the background) does NOT by itself fail "
        "visual_pass, even if it is garbled or not real words -- classify it as text_type diegetic_instrument and "
        "do not let its mere presence drive visual_pass to false. Only reject clearly invented overlay text: "
        "subtitles, captions, watermarks, logos, or large/prominent lettering that dominates the frame or reads "
        "as a title card. "
        "For video, subject, location and required objects must remain consistent across every target frame. "
        "A close-up may crop background objects, but must not replace the location or the speaking subject. "
        "Return one JSON object only with keys: image_received (boolean), target_images_seen (integer), "
        "visual_pass (boolean), location_match (boolean), framing_match (boolean), subjects_match (boolean), "
        "speaker_remains_visible (boolean), persistent_objects_match (boolean), aircraft_state_match (boolean), "
        "forbidden_text_detected (boolean), identity_match (boolean), confidence (number 0..1), reasons (array of short strings).\n"
        "Requirements:\n" + json.dumps(requirements, ensure_ascii=False) +
        "\nNeutral pixel observations:\n" + json.dumps(perception, ensure_ascii=False)
    )


# Presets de locacao por nome (Voo 702). Outro roteiro pode fornecer
# `continuity_contract["location_keywords"]`; sem nenhum dos dois, a locacao e
# decidida pela consistencia entre quadros + referencia + passe de decisao.
LOCATION_KEYWORDS = {
    "loc_cabin": ("cabin", "airplane interior", "aircraft interior"),
    "loc_cockpit": ("cockpit", "flight deck"),
}
EXTERIOR_LOCATIONS = {"loc_sky", "loc_exterior", "exterior"}
# Descricao de OUTRO ambiente reprova mesmo que uma palavra da locacao tambem apareca.
LOCATION_FORBIDDEN = {
    "loc_cabin": ("cockpit", "flight deck"),
    "loc_cockpit": ("passenger cabin", "passenger seats", "seat rows"),
}
FACE_IDENTITY_THRESHOLD = 0.35  # o mesmo limiar validado em clip_identity_audit

# Frases de oclusao/recorte do falante. "cropped" sozinho NAO entra: descreve
# cabelo/roupa ("cropped hair", "cropped jacket") e reprovava falante visivel.
_OCCLUSION = re.compile(
    r"partially visible|in the background|mostly occluded|partly occluded|"
    r"cropped (?:at|by|out|off)|cut off|out of frame|off-?screen|turned away|back to the camera|"
    r"only the back", re.I)


_GENERIC_WORDS = {"same", "fixed", "throughout", "flight", "with", "from", "that", "this", "inside",
                  "seated", "right", "left", "front", "forward", "aft", "single", "central", "rows",
                  "large", "small", "open", "closed", "near", "behind"}


def _object_visible(obj: str, visible_text: str) -> bool:
    """Casamento por palavra significativa (>= 4 letras): 'overhead bins' ~ 'overhead compartments'."""
    words = [w for w in re.findall(r"[a-z]{4,}", obj.casefold()) if w not in _GENERIC_WORDS]
    return any(w.rstrip("s") in visible_text for w in words)


def _all_targets_match(targets: list, keywords: tuple) -> bool:
    return bool(targets) and all(
        isinstance(item, dict)
        and any(k in str(item.get("location_description", "")).casefold() for k in keywords)
        for item in targets)


def face_check(frames: list, references, *, get_app=None) -> dict | None:
    """Trava NUMERICA de rosto (insightface/ArcFace, CPU): deterministica e barata.

    Devolve {frames, frames_with_face, min_similarity} ou None quando o
    insightface nao esta disponivel (o gate segue so com o VLM). Para CADA foto
    nominal usa o rosto do quadro que MAIS se parece com ela (a pessoa pode nao ser
    a maior do quadro: dois personagens, passageiros ao fundo) e o pior quadro;
    `min_similarity` e o pior entre as fotos. None se nao ha referencia."""
    if references is not None and not isinstance(references, (list, tuple)):
        references = [references]
    try:
        import cv2
        import numpy as np
        if get_app is None:
            from script_pipeline.consistency_audit import _get_app as get_app
        app = get_app()
    except Exception:
        return None
    ref_embs = []
    for ref in references or []:
        try:
            from script_pipeline.consistency_audit import face_embedding
            emb = face_embedding(str(ref))
        except Exception:
            emb = None
        if emb is not None:
            ref_embs.append(emb)
    with_face = 0
    per_ref = [[] for _ in ref_embs]
    for frame in frames:
        img = cv2.imdecode(np.fromfile(str(frame), dtype=np.uint8), cv2.IMREAD_COLOR)
        faces = app.get(img) if img is not None else []
        if not faces:
            continue
        with_face += 1
        for i, emb in enumerate(ref_embs):
            per_ref[i].append(max(float(np.dot(f.normed_embedding, emb)) for f in faces))
    worst = [min(v) for v in per_ref if v]
    return {"frames": len(frames), "frames_with_face": with_face,
            "min_similarity": round(min(worst), 3) if worst else None}


def _apply_perception_locks(raw: dict, perception: dict, *, aircraft_required: bool,
                            aircraft_must_be_visible: bool,
                            identity_required: bool, location_id: str = "",
                            framing: str = "", speaker_required: bool = False,
                            location_reference_required: bool = False,
                            location_keywords: tuple | list | None = None,
                            expected_refs: int | None = None,
                            face: dict | None = None,
                            persistent_objects: list | None = None,
                            expected_targets: int | None = None) -> dict:
    """Facts with simple physical meaning must not be overturned by pass 2.

    This prevents requirement-confirmation bias: perception is completed
    before the contract is disclosed, then simple physical facts are locked.

    Correcoes 2026-09-19 (auditoria): a locacao agora vale para CADA quadro
    (antes bastava UM quadro dizer "cabin"); deriva declarada pela percepcao
    (`same_*_across_targets` False) bloqueia; identidade exige um booleano por
    referencia; "cropped" nao reprova mais falante; `face` (numerico) so
    APERTA a decisao, nunca a afrouxa.
    """
    raw = dict(raw)
    # Identidade por rosto (foto nominal) so e confiavel com o rosto grande: em plano
    # medio/aberto o VLM e o ArcFace reprovavam personagens corretos (visto no Voo 702:
    # HA-EUN/MIN-JUN certos, similaridade 0,12). Fora de close a identidade e verificada
    # pelo contrato de figurino/cast no prompt e pela auditoria facial dos stills.
    strict_identity = framing.casefold() in {"close", "extreme_close"}
    if identity_required and not strict_identity:
        raw["subjects_match"] = True
        raw["identity_match"] = True
        identity_required = False
    targets = perception.get("targets") if isinstance(perception.get("targets"), list) else []
    if expected_targets:
        # O VLM as vezes lista as FOTOS DE REFERENCIA como alvos extras: so os N primeiros
        # sao os quadros de verdade (visto: 2 "alvos" num still -> falsa deriva).
        targets = targets[:expected_targets]
    multi = bool(expected_targets and expected_targets > 1) if expected_targets else len(targets) > 1
    raw["image_received"] = perception.get("image_received") is True
    raw["target_images_seen"] = perception.get("target_images_seen")
    location_id = location_id.casefold()
    keywords = tuple(k.casefold() for k in (location_keywords or LOCATION_KEYWORDS.get(location_id, ())))
    if keywords:
        raw["location_match"] = _all_targets_match(targets, keywords)
        forbidden_places = LOCATION_FORBIDDEN.get(location_id, ())
        if forbidden_places and any(
                any(f in str(item.get("location_description", "")).casefold() for f in forbidden_places)
                for item in targets if isinstance(item, dict)):
            raw["location_match"] = False
    elif location_id in EXTERIOR_LOCATIONS:
        raw["location_match"] = bool(targets) and all(
            isinstance(item, dict)
            and item.get("ground_or_runway_visible") is not True
            and (item.get("aircraft_visible") is True or "sky" in str(item.get("location_description", "")).casefold())
            for item in targets
        )
    if location_reference_required:
        raw["location_match"] = perception.get("same_location_as_reference") is True
    if multi and len(targets) > 1:
        if perception.get("same_location_across_targets") is False:
            raw["location_match"] = False
        if perception.get("same_primary_people_across_targets") is False:
            raw["subjects_match"] = False
            raw["visual_pass"] = False
    # Objetos persistentes: um quadro nao mostra o inventario inteiro (o passe de
    # decisao reprovava 23 de 68 stills por "corredor 3-3, galley traseiro, bagageiro
    # 19 ausentes"). Regra determinista: fora de close/inserto, PELO MENOS UM objeto do
    # inventario da locacao precisa estar visivel; a ausencia parcial e so aviso.
    if framing.casefold() in {"close", "extreme_close", "insert"} or not persistent_objects:
        raw["persistent_objects_match"] = True
    else:
        visible = " ".join(
            " ".join(str(o) for o in (item.get("visible_objects") or [])) + " " +
            str(item.get("location_description", ""))
            for item in targets if isinstance(item, dict)).casefold()
        seen = [obj for obj in persistent_objects if _object_visible(str(obj), visible)]
        raw["persistent_objects_match"] = bool(seen)
        raw["persistent_objects_missing"] = [str(o) for o in persistent_objects if o not in seen]
    # Texto proibido e decidido SO pela percepcao: legenda, marca d'agua ou texto
    # ilegivel inventado. Placa/instrumento da cena (EXIT, rotulos) e diegetico e passa
    # (o passe de decisao reprovava "EXIT" e "warning label").
    forbidden = False
    for item in targets:
        if not isinstance(item, dict):
            continue
        kind = item.get("text_type")
        if kind in {"subtitle_caption", "watermark_logo"} or (kind == "other" and item.get("visible_text")):
            forbidden = True
    raw["forbidden_text_detected"] = forbidden
    if aircraft_required:
        if aircraft_must_be_visible:
            aircraft_ok = bool(targets) and all(
                isinstance(item, dict)
                and item.get("aircraft_state") == "airborne"
                and item.get("ground_or_runway_visible") is not True
                and item.get("wheels_touching_surface") is not True
                for item in targets
            )
        else:
            # De DENTRO o estado do aviao nao e observavel: o VLM devolve "on_ground" para
            # qualquer cabine parada (visto em 4 stills com nuvens pela janela). So
            # evidencia de chao (pista/solo visivel, rodas tocando) bloqueia.
            aircraft_ok = bool(targets) and all(
                isinstance(item, dict)
                and item.get("ground_or_runway_visible") is not True
                and item.get("wheels_touching_surface") is not True
                for item in targets
            )
        raw["aircraft_state_match"] = aircraft_ok
        if not aircraft_ok:
            raw["visual_pass"] = False
    if identity_required:
        matches = perception.get("identity_matches")
        identity_ok = (isinstance(matches, list) and bool(matches)
                       and (expected_refs is None or len(matches) == expected_refs)
                       and matches[0] is True)  # close: o SUJEITO; o co-sujeito pode estar fora do quadro
        similarity = (face or {}).get("min_similarity")
        if similarity is not None:
            # Com o rosto medido (ArcFace, limiar validado em clip_identity_audit) a decisao
            # e numerica: o booleano do VLM reprovava closes claramente corretos (Ji-ho com
            # similaridade 0,57, Ha-eun 0,37) e nao e reprodutivel.
            identity_ok = bool(matches) and similarity >= FACE_IDENTITY_THRESHOLD
            if not identity_ok:
                raw["face_similarity_below_threshold"] = similarity
        raw["identity_match"] = identity_ok
        raw["subjects_match"] = identity_ok
        if speaker_required:
            close_ok = True
            if framing.casefold() == "close":
                close_ok = bool(targets) and all(
                    "close" in str(item.get("framing", "")).casefold()
                    for item in targets if isinstance(item, dict)
                )
            descriptions = [str(item.get("primary_people_description", ""))
                            for item in targets if isinstance(item, dict)]
            unobscured = bool(descriptions) and not any(_OCCLUSION.search(d) for d in descriptions)
            face_ok = True
            if face is not None:
                face_ok = face.get("frames_with_face") == face.get("frames")
                raw["face_in_every_frame"] = face_ok
            raw["speaker_remains_visible"] = (
                identity_ok
                and perception.get("reference_subject_prominent_in_every_target") is True
                and close_ok and unobscured and face_ok
            )
        if not identity_ok:
            raw["visual_pass"] = False
    if not identity_required:
        # Sem veredito por rosto (nao-close, ou sem foto nominal): o passe de decisao nao
        # consegue NOMEAR pessoas e reprovava "expected no subjects but observed passengers"
        # (22 de 68 stills). So a deriva declarada entre quadros reprova.
        raw["subjects_match"] = not (multi and len(targets) > 1 and
                                     perception.get("same_primary_people_across_targets") is False)
    if not identity_required and speaker_required and face is not None:
        # Fala sem foto nominal: ainda assim o rosto tem que estar em todos os quadros.
        face_ok = face.get("frames_with_face") == face.get("frames")
        raw["face_in_every_frame"] = face_ok
        raw["speaker_remains_visible"] = bool(raw.get("speaker_remains_visible")) and face_ok
    return raw


def _decision(raw: dict, *, expected_targets: int, aircraft_required: bool,
              speaker_required: bool, identity_required: bool) -> tuple[bool, list[str]]:
    reasons = [str(x) for x in (raw.get("reasons") or [])]
    required = {
        "image_received": raw.get("image_received") is True,
        "target_images_seen": (isinstance(raw.get("target_images_seen"), int)
                               and raw.get("target_images_seen") >= expected_targets),
        # O modelo emite um veredito global e critérios estruturados. Ignorar
        # visual_pass/framing_match permitia registrar pass=True mesmo quando
        # o próprio relatório descrevia enquadramento incompatível.
        "visual_pass": raw.get("visual_pass") is True,
        "location_match": raw.get("location_match") is True,
        "framing_match": raw.get("framing_match") is True,
        "subjects_match": raw.get("subjects_match") is True,
        "persistent_objects_match": raw.get("persistent_objects_match") is True,
        "forbidden_text_detected": raw.get("forbidden_text_detected") is False,
    }
    if aircraft_required:
        required["aircraft_state_match"] = raw.get("aircraft_state_match") is True
    if speaker_required:
        required["speaker_remains_visible"] = raw.get("speaker_remains_visible") is True
    if identity_required:
        required["identity_match"] = raw.get("identity_match") is True
    failed = [name for name, ok in required.items() if not ok]
    if failed:
        reasons.append("failed checks: " + ", ".join(failed))
    return not failed, reasons


def _still_path(run_dir: Path, manifest: dict, index: int) -> Path | None:
    item = manifest.get(str(index)) or {}
    name = item.get("file")
    path = run_dir / "shots" / "stills" / str(name) if name else None
    return path if path and path.exists() else None


def _location_reference(run_dir: Path, shot: dict, target: Path | None = None) -> Path | None:
    refs = _read_json(run_dir / "shots" / "location_refs.json", {})
    key = str(shot.get("location_id") or shot.get("scene"))
    if key.casefold() in EXTERIOR_LOCATIONS:
        return None  # nuvens/clima diferem por natureza; a comparacao so gerava falso bloqueio
    raw = refs.get(key)
    path = Path(str(raw)) if raw else None
    if not path or not path.exists():
        return None
    try:
        if target is not None and path.resolve() == target.resolve():
            return None
    except OSError:
        pass
    # Closes e insertos podem não mostrar arquitetura suficiente para uma
    # comparação honesta. Nos demais ângulos, a referência aberta trava a
    # geometria que o rótulo genérico LOC_CABIN sozinho não distingue.
    if str(shot.get("framing") or "").casefold() in {"close", "extreme_close", "insert"}:
        return None
    return path


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _video_frames(path: Path, directory: Path) -> list[Path]:
    duration = _probe_duration(path)
    points = [min(max(0.05, duration * ratio), max(0.05, duration - 0.05))
              for ratio in (0.08, 0.50, 0.92)]
    frames = []
    for idx, point in enumerate(points):
        out = directory / f"frame_{idx}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(point),
             "-i", str(path), "-frames:v", "1", str(out)], check=True,
        )
        frames.append(out)
    return frames


def _clip_map(run_dir: Path) -> dict[int, Path]:
    clips = _read_json(run_dir / "scenes" / "clips.json", [])
    result = {}
    for item in clips:
        match = re.search(r"shot(\d+)$", str(item.get("id", "")))
        path = Path(str(item.get("video_path", "")))
        if match and path.exists():
            result[int(match.group(1))] = path
    return result


FACE_FRAMINGS = {"close", "extreme_close", "medium", "medium_close", "ots"}
MAX_CONSECUTIVE_ERRORS = 3


def _report_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "shots" / ("visual_stills_audit.json" if stage == "stills"
                                else "visual_video_audit.json")


def merge_results(previous: list, fresh: list) -> list:
    """Resultados novos SUBSTITUEM os de mesmo plano; os demais ficam.

    ACHADO 2026-09-19: rodar com `--only-shots` regravava o relatorio so com
    os planos pedidos (e com status "ok"), apagando a auditoria completa."""
    por_plano = {int(r["shot"]): r for r in previous}
    for r in fresh:
        por_plano[int(r["shot"])] = r
    return [por_plano[k] for k in sorted(por_plano)]


def build_status(results: list, plan_indices: list) -> str:
    """ok = TODO plano do plano de tomadas tem resultado e todos passaram;
    partial = faltam planos sem resultado (auditoria incompleta) e nada falhou;
    blocked = algum plano reprovado."""
    by_shot = {int(r["shot"]): r for r in results}
    if any(not r.get("pass") for r in by_shot.values()):
        return "blocked"
    if any(i not in by_shot for i in plan_indices):
        return "partial"
    return "ok"


def audit(run_dir: Path, *, stage: str, model: str = DEFAULT_MODEL,
          perception_model: str = DEFAULT_PERCEPTION_MODEL,
          only_shots: set[int] | None = None, timeout: int = 180) -> dict:
    run_dir = run_dir.resolve()
    plan = _read_json(run_dir / "parse" / "shot_plan.json", {})
    # Arquivos de mídia são nomeados pela ordem atual do plano (shot000,
    # shot001...). ``index`` pode conservar a posição no plano-fonte quando um
    # teste/editorial usa apenas um trecho. Auditar pelo índice-fonte lia um
    # still antigo e podia aprovar/reprovar a imagem errada.
    all_shots = []
    for position, source in enumerate(plan.get("shots", [])):
        item = dict(source)
        item["source_index"] = item.get("index", position)
        item["index"] = position
        all_shots.append(item)
    shots = all_shots
    if only_shots is not None:
        shots = [shot for shot in all_shots if int(shot.get("index", -1)) in only_shots]
    manifest = _read_json(run_dir / "shots" / "stills" / "stills.json", {})
    clips = _clip_map(run_dir) if stage == "video" else {}
    results = []
    consecutive_errors = 0

    for shot in shots:
        index = int(shot.get("index", -1))
        refs = _reference_paths(run_dir, shot)
        contract = shot.get("continuity_contract") or {}
        speaker_required = stage == "video" and shot.get("line_index") is not None
        framing = str(shot.get("framing") or "")
        want_face = bool(refs) and (speaker_required or framing.casefold() in FACE_FRAMINGS)
        face = None
        try:
            if stage == "stills":
                target = _still_path(run_dir, manifest, index)
                if target is None:
                    raise FileNotFoundError(f"still ausente para plano {index}")
                targets = [target]
                location_ref = _location_reference(run_dir, shot, target)
                location_refs = [location_ref] if location_ref else []
                perception = _ollama_json(
                    perception_model, _perception_prompt(stage=stage, target_count=1, ref_count=len(refs),
                                                         location_ref_count=len(location_refs)),
                    targets + refs + location_refs, timeout=timeout)
                raw = _ollama_json(
                    model, _evaluation_prompt(shot, stage=stage, target_count=1,
                                              ref_count=len(refs), perception=perception,
                                              location_ref_count=len(location_refs)),
                    [], timeout=timeout)
                if want_face:
                    face = face_check(targets, refs[:1])
            else:
                clip = clips.get(index)
                if clip is None:
                    raise FileNotFoundError(f"clipe ausente para plano {index}")
                location_ref = _location_reference(run_dir, shot)
                location_refs = [location_ref] if location_ref else []
                with tempfile.TemporaryDirectory(prefix=f"visual_audit_{index:03d}_") as temp:
                    targets = _video_frames(clip, Path(temp))
                    perception = _ollama_json(
                        perception_model, _perception_prompt(stage=stage, target_count=3, ref_count=len(refs),
                                                             location_ref_count=len(location_refs)),
                        targets + refs + location_refs, timeout=timeout)
                    raw = _ollama_json(
                        model, _evaluation_prompt(shot, stage=stage, target_count=3,
                                                  ref_count=len(refs), perception=perception,
                                                  location_ref_count=len(location_refs)),
                        [], timeout=timeout)
                    if want_face or speaker_required:
                        face = face_check(targets, refs[:1])
            aircraft_required = bool(contract.get("aircraft"))
            location_id = str(shot.get("location_id") or "").casefold()
            aircraft_must_be_visible = location_id in EXTERIOR_LOCATIONS
            raw = _apply_perception_locks(raw, perception,
                                          aircraft_required=aircraft_required,
                                          aircraft_must_be_visible=aircraft_must_be_visible,
                                          identity_required=bool(refs),
                                          location_id=str(shot.get("location_id") or ""),
                                          framing=framing, speaker_required=speaker_required,
                                          location_reference_required=bool(location_refs),
                                          location_keywords=contract.get("location_keywords"),
                                          expected_refs=len(refs), face=face,
                                          persistent_objects=contract.get("persistent_objects"),
                                          expected_targets=len(targets))
            ok, reasons = _decision(
                raw,
                expected_targets=len(targets),
                aircraft_required=aircraft_required,
                speaker_required=speaker_required,
                identity_required=bool(refs),
            )
            warnings = _entry_warnings(targets, face)
            entry = {"shot": index, "pass": ok, "reasons": reasons, "face": face, "warnings": warnings,
                     "perception": perception, "model_output": raw}
            consecutive_errors = 0
            if perception.get("image_received") is not True:
                entry["global_failure"] = True
        except (OSError, ValueError, KeyError, subprocess.SubprocessError,
                urllib.error.URLError, TimeoutError) as exc:
            ok = False
            entry = {"shot": index, "pass": False, "reasons": [f"auditor error: {exc}"],
                     "model_output": {}, "auditor_error": True}
            consecutive_errors += 1
        results.append(entry)
        print(f"[visual_audit:{stage}] plano {index:03d}: {'OK' if ok else 'BLOQUEADO'}", flush=True)
        # Falha GLOBAL (o runtime nem entregou a imagem, ou o Ollama caiu varias
        # vezes seguidas): nao gaste dezenas de chamadas. Um erro isolado de
        # UM plano (arquivo ausente, JSON invalido) nao interrompe os demais.
        if entry.get("global_failure") or consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            break

    out = _report_path(run_dir, stage)
    previous = _read_json(out, {}).get("results", []) if only_shots is not None else []
    merged = merge_results(previous, results)
    plan_indices = [int(s.get("index", -1)) for s in all_shots]
    blocking = [r for r in merged if not r.get("pass")]
    report = {
        "status": build_status(merged, plan_indices),
        "stage": stage,
        "decision_model": model,
        "perception_model": perception_model,
        "audited": len(merged),
        "expected": len(all_shots),
        "blocking": blocking,
        "warnings": {str(r["shot"]): r["warnings"] for r in merged if r.get("warnings")},
        "results": merged,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[visual_audit:{stage}] status={report['status']}; "
          f"{report['audited']}/{report['expected']} auditado(s); "
          f"{len(blocking)} bloqueio(s); percepcao={perception_model}; decisao={model}")
    print(f"[visual_audit:{stage}] relatorio: {out}")
    return report


def _entry_warnings(targets: list, face: dict | None) -> list[str]:
    """Avisos que NAO bloqueiam mas qualificam a aprovacao (visto no teste de cabine: closes
    aprovados a 384x256 com ArcFace 0,32-0,39, margem de 0,03 sobre o limiar)."""
    out = []
    sim = (face or {}).get("min_similarity")
    if sim is not None and FACE_IDENTITY_THRESHOLD <= sim < FACE_IDENTITY_THRESHOLD + 0.05:
        out.append(f"similaridade facial marginal ({sim}, limiar {FACE_IDENTITY_THRESHOLD})")
    try:
        from PIL import Image
        width = min(Image.open(str(t)).width for t in targets)
        if width < 640:
            out.append(f"imagem de baixa resolucao ({width}px de largura): identidade e texto pouco confiaveis")
    except Exception:
        pass
    return out


def blocked_shots(run_dir: Path, stage: str) -> list[int]:
    """Planos reprovados no ultimo relatorio (para o laco de regeneracao)."""
    report = _read_json(_report_path(Path(run_dir), stage), {})
    return sorted(int(r["shot"]) for r in report.get("blocking", []))


def _shot_set(raw: str | None) -> set[int] | None:
    """"0-2,5" -> {0,1,2,5} (o formato que o laco de regeneracao emite)."""
    if not raw:
        return None
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", required=True, choices=["stills", "video"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--perception-model", default=DEFAULT_PERCEPTION_MODEL)
    parser.add_argument("--only-shots")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit(Path(args.run_dir), stage=args.stage, model=args.model,
                       perception_model=args.perception_model,
                       only_shots=_shot_set(args.only_shots), timeout=args.timeout)
        return 0 if args.report_only or report.get("status") == "ok" else 2
    finally:
        # O Qwen3-VL ocupa ~22 GB na RTX 3090. Mantê-lo pelos 10 minutos de
        # keep_alive depois do gate impede o estágio de vídeo de carregar.
        # O servidor permanece ativo; apenas os modelos usados saem da VRAM.
        from script_pipeline.ollama_runtime import unload_models
        unload_models([args.model, args.perception_model], log=print)


if __name__ == "__main__":
    raise SystemExit(main())
