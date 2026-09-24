"""Versioned spatial state. SQLite is authoritative; rendering never mutates it."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_state(state):
    if state.get('schema_version') != 1 or state.get('coordinates') != 'Z_UP_METERS':
        raise ValueError('Unsupported spatial schema/coordinate convention')
    if not state.get('location_id') or not state.get('location_asset'):
        raise ValueError('Location ID and asset are required')
    entities = state.get('entities', {})
    if not isinstance(entities, dict):
        raise ValueError('entities must be a mapping')
    for eid, entity in entities.items():
        if not eid or entity.get('kind') not in ('character', 'extra', 'prop'):
            raise ValueError(f'Invalid entity {eid}')
        position = entity.get('position')
        if not isinstance(position, list) or len(position) != 3 or not all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in position):
            raise ValueError(f'Invalid position: {eid}')
        if not math.isfinite(entity.get('yaw', 0)):
            raise ValueError(f'Invalid yaw: {eid}')
        attachment = entity.get('attachment')
        if attachment:
            owner = entities.get(attachment.get('entity_id'))
            if entity['kind'] != 'prop' or not owner or owner['kind'] not in ('character', 'extra'):
                raise ValueError(f'Invalid attachment: {eid}')
            if attachment.get('socket') not in ('left_hand', 'right_hand'):
                raise ValueError('Unknown attachment socket')
    canonical(state)  # also rejects nested NaN/Infinity


def apply_operations(state, operations):
    result = copy.deepcopy(state)
    if not isinstance(operations, list) or not operations:
        raise ValueError('An event needs operations')
    for op in operations:
        entity = result['entities'].get(op.get('entity_id'))
        if entity is None:
            raise ValueError('Unknown entity')
        if op.get('op') == 'transfer':
            previous = entity.get('attachment')
            if (previous or {}).get('entity_id') != op.get('from'):
                raise ValueError('Transfer precondition failed')
            entity['attachment'] = {'entity_id': op['to'], 'socket': op['socket']}
        elif op.get('op') == 'set':
            field = op.get('field')
            if field not in ('position', 'yaw', 'pose', 'wardrobe_id', 'present'):
                raise ValueError(f'Forbidden state field: {field}')
            if entity.get(field) != op.get('expected'):
                raise ValueError(f'Stale precondition: {field}')
            entity[field] = copy.deepcopy(op['value'])
        else:
            raise ValueError('Unknown event operation')
    validate_state(result)
    return result


class WorldStore:
    def __init__(self, project):
        self.root = Path(project) / 'world'
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'world.sqlite', timeout=30)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY);
          INSERT OR IGNORE INTO schema_migrations VALUES(1);
          CREATE TABLE IF NOT EXISTS assets(id TEXT, version TEXT, payload TEXT NOT NULL,
            PRIMARY KEY(id,version));
          CREATE TABLE IF NOT EXISTS scene_states(hash TEXT PRIMARY KEY, parent TEXT,
            payload TEXT NOT NULL, FOREIGN KEY(parent) REFERENCES scene_states(hash));
          CREATE TABLE IF NOT EXISTS continuity_events(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
            parent TEXT NOT NULL, result TEXT NOT NULL, payload TEXT NOT NULL,
            FOREIGN KEY(parent) REFERENCES scene_states(hash), FOREIGN KEY(result) REFERENCES scene_states(hash));
          CREATE TABLE IF NOT EXISTS shot_bindings(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def register_asset(self, asset_id, asset):
        version = digest(asset)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO assets VALUES(?,?,?)', (asset_id, version, canonical(asset)))
        return version

    def asset(self, asset_id, version):
        row = self.db.execute('SELECT payload FROM assets WHERE id=? AND version=?', (asset_id, version)).fetchone()
        if not row:
            raise ValueError(f'Unknown asset {asset_id}@{version}')
        return json.loads(row[0])

    def _insert(self, state, parent):
        validate_state(state)
        self.asset(state['location_id'], state['location_asset'])
        # Include ancestry, so a deliberate return to an earlier layout remains a new state.
        key = digest({'parent': parent, 'state': state})
        self.db.execute('INSERT OR IGNORE INTO scene_states VALUES(?,?,?)', (key, parent, canonical(state)))
        return key

    def create(self, state):
        with self.db:
            key = self._insert(state, None)
        self.export(key)
        return key

    def get(self, key):
        row = self.db.execute('SELECT payload FROM scene_states WHERE hash=?', (key,)).fetchone()
        if not row:
            raise ValueError(f'Unknown state {key}')
        return json.loads(row[0])

    def apply(self, parent, event):
        if not event.get('id') or not event.get('source_unit_id'):
            raise ValueError('Event and source unit IDs are required')
        semantic = {k:event[k] for k in ('id','source_unit_id','operations')}
        fingerprint = digest({'parent': parent, 'event': semantic})
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT fingerprint,result,parent,payload FROM continuity_events WHERE id=?', (event['id'],)).fetchone()
            if old:
                if old[0] != fingerprint:
                    old_event = json.loads(old[3])
                    old_semantic = {k:old_event[k] for k in ('id','source_unit_id','operations')}
                    if old[2] != parent or old_semantic != semantic:
                        raise ValueError('Event ID reused with different content or parent')
                key = old[1]
            else:
                state = apply_operations(self.get(parent), event['operations'])
                key = self._insert(state, parent)
                self.db.execute('INSERT INTO continuity_events VALUES(?,?,?,?,?)',
                                (event['id'], fingerprint, parent, key, canonical(event)))
        self.export(key)
        return key

    def export(self, key):
        directory = self.root / 'snapshots'
        directory.mkdir(exist_ok=True)
        path = directory / f'{key}.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(canonical(self.get(key)), encoding='utf-8')
        temporary.replace(path)
        return path

    def bind(self, shot_id, initial, final, camera, **metadata):
        self.get(initial)
        self.get(final)
        from script_pipeline.camera_geometry import validate_camera
        validate_camera(camera)
        binding = dict(shot_id=shot_id, initial=initial, final=final, camera=camera, **metadata)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO shot_bindings VALUES(?,?,?)',
                            (shot_id, digest(binding), canonical(binding)))
        return binding
