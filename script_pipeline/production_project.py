"""Persistent production project. Source, editorial decisions and engine runs are separate.

The source ledger is authoritative: LLM annotations may enrich but never remove beats.
Coverage means text assigned to a shot, not proof of visual execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(p)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def project_for(run):
    config = read(Path(run) / "production.json", {})
    return Path(config["project_dir"]) if config else None


def source_ledger(text):
    from script_pipeline.parse_screenplay import _strip_markdown, parse_structure, _looks_like_cue
    clean = _strip_markdown(text)
    clean = re.sub(r"(?m)^\s*---+\s*$", "", clean)
    clean = re.sub(r"(?m)^\*([^*\n]+)\*$", r"\1", clean)
    clean = re.sub(r"(?m)^\s*\*\s+", "", clean).replace("*", "")
    headings = list(re.finditer(r"(?m)^(?:INT\.|EXT\.|INT/EXT\.)[^\n]+", clean))
    if not headings:
        raise ValueError("Projeto mestre exige roteiro estruturado com INT./EXT.; converta a prosa primeiro.")
    times = re.findall(r"CENA\s+\d+[^\n]*?\((\d+):(\d+)\s*[-–—]\s*(\d+):(\d+)\)", clean, re.I)
    scenes, script_blocks = [], []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(clean)
        block = clean[heading.start():end]
        block = re.sub(r"(?m)^CENA\s+\d+[^\n]*$", "", block, flags=re.I).strip()
        parsed = parse_structure(block)[0].to_dict()
        parsed["index"] = i + 1
        sid = "SC_" + digest(heading.group() + f":{i}")[:12]
        units = []
        di = 0
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        for paragraph in paragraphs[1:]:
            if paragraph.startswith("[CORTE"):
                continue
            first = paragraph.splitlines()[0]
            cue = _looks_like_cue(first, "dialogue")
            if cue and di < len(parsed["dialogue"]) and cue[0] == parsed["dialogue"][di]["character"]:
                line = parsed["dialogue"][di]
                unit = {"type": "dialogue", "line_index": di, "text": line["text"], "character": line["character"]}
                di += 1
            else:
                unit = {"type": "action", "text": " ".join(paragraph.split())}
            unit["id"] = sid + "_" + digest([unit, len(units)])[:10]
            units.append(unit)
        if di != len(parsed["dialogue"]):
            raise ValueError(f"Cena {i + 1}: falas fora do formato de blocos; revise o roteiro.")
        target = None
        if len(times) == len(headings):
            a, b, c, d = map(int, times[i])
            target = c * 60 + d - a * 60 - b
            if target <= 0:
                raise ValueError("Intervalo de cena inválido")
        parsed.update(id=sid, units=units, target_seconds=target, sequence_id="SEQ_001", act_id="ACT_001")
        scenes.append(parsed)
        script_blocks.append(block)
    return scenes, "\n\n".join(script_blocks)


def create_project(root, run, text, title, target_seconds=300):
    root, run = Path(root).resolve(), Path(run).resolve()
    if (root / "projeto.json").exists():
        raise ValueError("O projeto já existe. Retome pela corrida vinculada.")
    scenes, normalized = source_ledger(text)
    for directory in ("roteiro/versoes", "biblia", "assets/personagens", "assets/locacoes", "sequencias",
                      "renders", "editorial", "audio/dialogos", "audio/ambientes", "audio/foley",
                      "audio/efeitos", "audio/musica", "vfx", "entregas"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "roteiro/original.txt").write_text(text, encoding="utf-8")
    (root / "roteiro/roteiro.txt").write_text(normalized, encoding="utf-8")
    write(root / "roteiro/source.json", scenes)
    locations = {}
    characters = {}
    for scene in scenes:
        name = scene["location"]
        lid = "LOC_" + digest(name)[:12]
        scene["location_id"] = lid
        locations.setdefault(lid, {"name": name, "geometry": "", "references": {}, "fixed_props": [], "version": 1})
        for name in scene["characters"]:
            characters.setdefault(name, {"id": "CHR_" + digest(name)[:12], "wardrobe": "", "reference_image": None})
    write(root / "roteiro/source.json", scenes)
    write(root / "biblia/locacoes.json", locations)
    write(root / "biblia/personagens.json", characters)
    write(root / "biblia/continuidade.json", {s["id"]: {"location_id": s["location_id"], "story_day": None,
          "time_of_day": s["time_of_day"], "season": None, "weather": None, "before": {}, "after": {},
          "prompt": ""} for s in scenes})
    write(root / "projeto.json", {"schema_version": 1, "id": str(uuid.uuid4()), "title": title,
          "revision": 1, "source_hash": digest(text), "run_dir": str(run), "target_seconds": target_seconds,
          "acts": [{"id": "ACT_001", "sequences": [{"id": "SEQ_001", "scenes": [s["id"] for s in scenes]}]}],
          "max_attempts": 3, "generation_budget_seconds": target_seconds * 3})
    write(run / "production.json", {"project_dir": str(root)})
    return str(root / "roteiro/roteiro.txt")


def restore_source(run, engine=None):
    """Restore ordered source beats after enrichment; reuse exact matching visual annotations."""
    root = project_for(run)
    if root is None:
        return
    run = Path(run)
    source = read(root / "roteiro/source.json")
    path = run / "parse/scenes_enriched.json"
    data = read(path, read(run / "parse/scenes.json"))
    scenes = data.get("scenes", []) if isinstance(data, dict) else data
    if len(scenes) != len(source):
        raise ValueError("Enriquecimento alterou a quantidade de cenas")
    for scene, original in zip(scenes, source):
        if [(d["character"], d["text"]) for d in scene["dialogue"]] != [(d["character"], d["text"]) for d in original["dialogue"]]:
            raise ValueError(f"Falas alteradas na cena {original['index']}")
        old = scene.get("shot_list", [])
        translations = {}
        # O parser enriquecido ja costuma devolver exatamente uma descricao
        # visual para cada unidade da fonte, na mesma ordem e com o mesmo
        # tipo. Reaproveitar essa cobertura evita uma segunda chamada LLM por
        # cena (7 chamadas redundantes no Voo 702). Se qualquer tipo/contagem
        # divergir, cai no tradutor source-aware abaixo, que continua sendo o
        # fallback seguro para parses incompletos.
        aligned = (len(old) == len(original["units"]) and all(
            str(item.get("type")) == str(unit.get("type")) and
            isinstance(item.get("visual"), str) and item["visual"].strip()
            for item, unit in zip(old, original["units"])
        ))
        if aligned:
            translations = {
                unit["id"]: {"id": unit["id"], "visual": item["visual"],
                             "actor": item.get("actor")}
                for item, unit in zip(old, original["units"])
            }
        if engine:
            state = read(root / 'biblia/continuidade.json', {}).get(original['id'], {})
            cache = root / 'roteiro/versoes' / ('visual_' + digest([original['units'], original['dialogue'], state, engine])[:16] + '.json')
            translations = translations or read(cache, {})
            if not translations:
                from script_pipeline.story_structure import _call_ollama
                response = _call_ollama(
                    'You are a film storyboard translator. Return JSON {"units": [{"id": "exact source ID", "visual": "complete English visual description", "actor": "exact character name or null"}]}. '
                    'Include EVERY source unit, preserve all actions, names and order. Do not invent story events. Dialogue stays in the audio; visual describes the visible performance only.',
                    json.dumps({'characters': scene['characters'], 'location': scene['location'], 'units': original['units'],
                                'original_dialogue_and_stage_directions': original['dialogue'], 'continuity': state}, ensure_ascii=False), engine)
                values = (response or {}).get('units', [])
                if {u.get('id') for u in values} == {u['id'] for u in original['units']} and all(isinstance(u.get('visual'), str) and u['visual'].strip() for u in values):
                    translations = {u['id']: u for u in values}
                    write(cache, translations)
        scene["shot_list"] = []
        for unit in original["units"]:
            translated = translations.get(unit['id'], {})
            beat = translated.get('visual') or unit['text']
            if unit['type'] == 'dialogue' and translated:
                scene['dialogue'][unit['line_index']]['beat_visual'] = beat
            scene["shot_list"].append({"type": unit["type"], "line_index": unit.get("line_index"),
                "visual": beat, "actor": translated.get('actor'), "source_unit_id": unit["id"]})
    write(path, data)


def conform_plan(run):
    """Frame-accurate editorial duration; never speed up or shorten a dialogue to fit."""
    root = project_for(run)
    if root is None:
        return
    run = Path(run)
    source = read(root / "roteiro/source.json")
    continuity = read(root / "biblia/continuidade.json", {})
    locations = read(root / 'biblia/locacoes.json', {})
    plan = read(run / "parse/shot_plan.json")
    fps = float(plan.get("fps", 24))
    output, missing = [], []
    id_occurrences = {}
    for scene in source:
        shots = [s for s in plan["shots"] if s["scene"] == scene["index"]]
        units = scene["units"]
        actions = iter(u for u in units if u["type"] == "action")
        for shot in shots:
            if shot.get("line_index") is not None:
                unit = next((u for u in units if u["type"] == "dialogue" and u["line_index"] == shot["line_index"]), None)
            elif shot.get("type") == "action":
                unit = next(actions, None)
            else:
                unit = None
            # Source-aware planners must carry exact IDs. Legacy plans may use positional mapping.
            if "source_unit_id" not in shot:
                shot["source_unit_id"] = unit["id"] if unit else None
        covered = {s["source_unit_id"] for s in shots}
        missing.extend(u["id"] for u in units if u["id"] not in covered)
        target = scene.get("target_seconds")
        if target:
            target_frames = round(target * fps)
            speech = [s for s in shots if s.get("line_index") is not None]
            action = [s for s in shots if s.get("line_index") is None]
            speech_frames = sum(math.ceil(s["seconds"] * fps) for s in speech)
            available = target_frames - speech_frames
            if available < len(action) * round(fps) or not action:
                raise ValueError(f"Cena {scene['index']}: falas não cabem no tempo previsto sem cortes.")
            weights = sum(s["seconds"] for s in action)
            remaining = available
            for i, shot in enumerate(action):
                frames = remaining if i == len(action) - 1 else round(available * shot["seconds"] / weights)
                shot["seconds"] = frames / fps
                remaining -= frames
            for shot in speech:
                shot["seconds"] = math.ceil(shot["seconds"] * fps) / fps
        state = continuity.get(scene["id"], {})
        location = locations.get(state.get('location_id', scene['location_id']), {})
        scene_characters = sorted({str(s.get('subject')) for s in shots if s.get('subject')} |
                                  {str(s.get('co_subject')) for s in shots if s.get('co_subject')})
        # Aircraft continuity is opt-in per project/scene. Voo 702 supplies
        # `aircraft_contract` in voo702_setup.py; ordinary scripts receive no
        # aviation rule and are audited only for their own location/inventory.
        aircraft_contract = state.get('aircraft_contract')
        persistent_objects = state.get('persistent_objects') or []
        character_contract = ('Scene roster (keep identities and secondary characters stable): ' +
                              ', '.join(scene_characters) if scene_characters else '')
        contract_parts = [p for p in (aircraft_contract, character_contract) if p]
        contract = 'Continuity contract: ' + '. '.join(contract_parts)
        if persistent_objects:
            contract += '. Persistent scene objects already established and must not disappear or relocate: ' + ', '.join(persistent_objects)
        for shot in shots:
            count = max(1, math.ceil(shot["seconds"] / 6)) if shot.get("line_index") is None else 1
            total_frames = round(shot["seconds"] * fps)
            for part in range(count):
                item = copy.deepcopy(shot)
                frames = total_frames // count + (1 if part < total_frames % count else 0)
                item.update(seconds=frames / fps, frames=1 + math.ceil((frames - 1) / 8) * 8,
                    editorial_frames=frames, scene_id=scene["id"], sequence_id=scene["sequence_id"],
                    location_id=state.get("location_id", scene["location_id"]), continuity=state,
                    source_part=part, required=True)
                # More than one editorial shot can cover the same source unit
                # (coverage/reaction angles and split dialogue). The old key
                # collapsed them to one ID, making spatial bindings ambiguous.
                semantic_id = (scene["id"], item["source_unit_id"], item.get("type"), part)
                occurrence = id_occurrences.get(semantic_id, 0)
                id_occurrences[semantic_id] = occurrence + 1
                item["id"] = "SH_" + digest([*semantic_id, occurrence])[:16]
                item["index"] = len(output)
                context = ' '.join(filter(None, [location.get('geometry'), state.get('prompt')]))
                item['location_reference'] = location.get('references', {}).get('front')
                if context:
                    for field in ("video_prompt", "storyboard_prompt"):
                        item[field] += " Continuity context (show only the current beat, not every event): " + context
                item["continuity_contract"] = {
                    "location_id": item["location_id"],
                    "aircraft": aircraft_contract,
                    "scene_characters": scene_characters,
                    "persistent_objects": persistent_objects,
                    "visible_inventory_policy": "Objects established in this scene stay present in any angle that can show them; a close-up may crop them, but they are not removed or relocated.",
                }
                item["storyboard_prompt"] += " " + contract + ". Objects established in this scene stay present in any angle that can show them; a close-up may crop them."
                item["video_prompt"] += " " + contract + ". Objects established in this scene stay present in any angle that can show them; a close-up may crop them."
                output.append(item)
    report = {"source_units": sum(len(s["units"]) for s in source), "missing": missing,
              "complete": not missing, "meaning": "Cobertura textual; execução visual exige revisão."}
    write(root / "editorial/cobertura.json", report)
    if missing:
        raise ValueError(f"Cobertura incompleta: {missing}")
    plan.update(shots=output, total_shots=len(output), total_seconds=sum(s["seconds"] for s in output))
    plan_hash = digest(plan)
    write(root / f"roteiro/versoes/plano_{plan_hash[:16]}.json", plan)
    write(run / "parse/shot_plan.json", plan)
    previous = read(root / "editorial/timeline.json", {})
    previous_shots = {s["id"]: s for s in previous.get("shots", [])}
    timeline, cursor = [], 0
    for shot in output:
        fingerprint = digest(shot)
        old = previous_shots.get(shot["id"], {})
        same = old.get("fingerprint") == fingerprint
        timeline.append({"id": shot["id"], "index": shot["index"], "source_unit_id": shot["source_unit_id"],
            "start_frame": cursor, "duration_frames": shot["editorial_frames"], "trim_in": 0,
            "fingerprint": fingerprint, "approval": old.get("approval", "pending") if same else "pending",
            "selected_take": old.get("selected_take") if same else None, "takes": old.get("takes", []),
            "stale": bool(old) and not same})
        cursor += shot["editorial_frames"]
    write(root / "editorial/timeline.json", {"fps": fps, "duration_frames": cursor, "shots": timeline})
    write(root / "sequencias/production_queue.json", [{"shot_id": s["id"], "location_id": s["location_id"],
          "depends_on": [s["location_id"], s["scene_id"]], "estimated_seconds": s["seconds"]} for s in output])
    if not (root / "audio/cues.json").exists():
        write(root / "audio/cues.json", {name: [] for name in ("dialogos", "ambientes", "foley", "efeitos", "musica")})
    old_jobs = {j['shot_id']: j for j in read(root / 'vfx/jobs.json', [])}
    write(root / "vfx/jobs.json", [old_jobs.get(s['id'], {"shot_id": s["id"], "status": "unassigned", "camera": {},
          "passes": {"beauty": None, "depth": None, "mask": None}, "scene_file": None}) for s in output])
    print(f"[projeto] {len(output)} planos; {plan['total_seconds']:.2f}s; cobertura {report['source_units']}/{report['source_units']}", flush=True)


def apply_voice_overrides(run):
    root = project_for(run)
    if not root:
        return
    overrides = read(root / 'biblia/voz.json', {})
    path = Path(run) / 'parse/scenes_enriched.json'
    data = read(path)
    scenes = data.get('scenes', []) if isinstance(data, dict) else data or []
    for scene in scenes:
        for index, line in enumerate(scene.get('dialogue', [])):
            emotion = overrides.get(f"{scene['index']}:{index}")
            if emotion:
                line['emotion'] = emotion
                line['parenthetical'] = emotion
    if data:
        write(path, data)


def sync_media(run):
    root = project_for(run)
    if not root or not (root / "editorial/timeline.json").exists():
        return
    run = Path(run)
    timeline = read(root / "editorial/timeline.json")
    clips = read(run / "scenes/clips.json", [])
    for entry in timeline["shots"]:
        clip = next((c for c in clips if c["id"].endswith(f"shot{entry['index']:03d}")), None)
        if clip and clip.get("ok") and Path(clip.get("video_path", "")).is_file():
            media = Path(clip["video_path"])
            take_id = digest([entry["fingerprint"], media.stat().st_size, media.stat().st_mtime_ns])[:16]
            if not any(t["id"] == take_id for t in entry["takes"]):
                dest = root / "renders" / entry["id"] / f"{take_id}.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(media, dest)
                entry["takes"].append({"id": take_id, "media": str(dest), "fingerprint": entry["fingerprint"]})
                entry["selected_take"] = take_id
                entry["approval"] = "pending"
    write(root / "editorial/timeline.json", timeline)
    export_otio(root)


def sync_assets(run):
    root = project_for(run)
    if not root:
        return
    run = Path(run)
    cast = read(run / 'characters/cast.json', {})
    bible = read(root / 'biblia/personagens.json', {})
    for name, character in cast.items():
        bible.setdefault(name, {'id': 'CHR_' + digest(name)[:12]})
        bible[name]['casting'] = character
        source = character.get('reference_image')
        if source and Path(source).is_file():
            dest = root / 'assets/personagens' / (bible[name]['id'] + Path(source).suffix)
            shutil.copy2(source, dest)
            bible[name]['reference_image'] = str(dest)
    write(root / 'biblia/personagens.json', bible)
    locations = read(root / 'biblia/locacoes.json', {})
    for lid, source in read(run / 'shots/location_refs.json', {}).items():
        if lid in locations and Path(source).is_file():
            dest = root / 'assets/locacoes' / (lid + Path(source).suffix)
            if not dest.exists():
                shutil.copy2(source, dest)
            locations[lid].setdefault('generated_candidates', [])
            if str(dest) not in locations[lid]['generated_candidates']:
                locations[lid]['generated_candidates'].append(str(dest))
    write(root / 'biblia/locacoes.json', locations)


def export_otio(root):
    import opentimelineio as otio
    root = Path(root)
    edit = read(root / "editorial/timeline.json")
    timeline = otio.schema.Timeline(name=read(root / "projeto.json")["title"])
    track = otio.schema.Track(name="Picture")
    timeline.tracks.append(track)
    for entry in edit["shots"]:
        take = next((t for t in entry["takes"] if t["id"] == entry["selected_take"]), None)
        ref = otio.schema.ExternalReference(target_url=Path(take["media"]).as_uri()) if take else otio.schema.MissingReference()
        clip = otio.schema.Clip(name=entry["id"], media_reference=ref,
            source_range=otio.opentime.TimeRange(otio.opentime.RationalTime(entry["trim_in"], edit["fps"]),
                                                otio.opentime.RationalTime(entry["duration_frames"], edit["fps"])))
        clip.metadata["production"] = {"approval": entry["approval"], "source_unit_id": entry["source_unit_id"] or ""}
        track.append(clip)
    otio.adapters.write_to_file(timeline, str(root / "editorial/timeline.otio"))


def validate_timeline(edit):
    if not isinstance(edit.get('fps'), (int, float)) or edit['fps'] <= 0:
        raise ValueError('FPS inválido')
    ids, cursor = set(), 0
    for shot in edit['shots']:
        if shot['id'] in ids or shot['duration_frames'] <= 0 or shot['trim_in'] < 0:
            raise ValueError('ID repetido, duração inválida ou entrada negativa')
        if shot['start_frame'] != cursor:
            raise ValueError('Timeline deve ser contínua e começar em zero')
        if shot['approval'] not in ('pending', 'approved', 'rejected'):
            raise ValueError('Estado de aprovação inválido')
        if shot['approval'] == 'approved':
            take = next((t for t in shot['takes'] if t['id'] == shot['selected_take']), None)
            if not take or take.get('fingerprint') != shot['fingerprint'] or not Path(take['media']).is_file():
                raise ValueError('Tomada ausente ou obsoleta não pode ser aprovada')
        cursor += shot['duration_frames']
        ids.add(shot['id'])
    if edit['duration_frames'] != cursor:
        raise ValueError('Duração total não corresponde à timeline')


def reserve_generation(run, shot):
    root = project_for(run)
    if not root:
        return
    config = read(root / 'projeto.json')
    path = root / 'renders/attempts.json'
    attempts = read(path, [])
    used = [a for a in attempts if a['shot_id'] == shot['id']]
    if len(used) >= config['max_attempts']:
        raise ValueError(f"Limite de tentativas atingido: {shot['id']}")
    if sum(a['seconds'] for a in attempts) + shot['seconds'] > config['generation_budget_seconds']:
        raise ValueError('Orçamento de segundos de geração atingido')
    from datetime import datetime, timezone
    attempts.append({'shot_id': shot['id'], 'seconds': shot['seconds'], 'started': datetime.now(timezone.utc).isoformat()})
    write(path, attempts)
