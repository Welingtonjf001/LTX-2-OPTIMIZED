# -*- coding: utf-8 -*-
"""Limpador de cache de stills da decupagem_ui (2026-09-13). Sem GPU, sem rede:
monta uma corrida falsa em tmp e confere o que some e o que fica."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import decupagem_ui as ui  # noqa: E402


def _corrida(tmp: Path) -> Path:
    run = tmp / "corrida"
    st, cl, ls = run / "shots" / "stills", run / "shots" / "clips", run / "lipsync"
    for d in (st, cl, ls):
        d.mkdir(parents=True)
    man = {}
    for i in range(3):
        (st / f"shot{i:03d}_close.png").write_bytes(b"png")
        man[str(i)] = {"key": "x"}
        (cl / f"shot{i:03d}.mp4").write_bytes(b"mp4")
        (cl / f"shot{i:03d}.key").write_text("k")
        (ls / f"shot{i:03d}.key").write_text("k")
        (ls / f"shot{i:03d}_synced.mp4").write_bytes(b"mp4")
    (st / "stills.json").write_text(json.dumps(man))
    (run / "shots" / "animatic.mp4").write_bytes(b"mp4")
    return run


def test_limpa_so_a_selecao_e_derivados(tmp_path, monkeypatch):
    run = _corrida(tmp_path)
    monkeypatch.setattr(ui, "RUNS_DIR", tmp_path)
    msg = ui.limpar_cache_stills("corrida", "1", True, False)
    cl, ls = run / "shots" / "clips", run / "lipsync"
    assert not (run / "shots" / "stills" / "shot001_close.png").exists()
    assert (run / "shots" / "stills" / "shot000_close.png").exists()
    assert not (cl / "shot001.key").exists() and (cl / "shot001.mp4").exists()
    assert (cl / "shot000.key").exists()
    assert not (ls / "shot001.key").exists() and not (ls / "shot001_synced.mp4").exists()
    assert (ls / "shot002.key").exists()
    assert not (run / "shots" / "animatic.mp4").exists()
    assert "1" not in json.loads((run / "shots" / "stills" / "stills.json").read_text())
    assert "Derivados" in msg


def test_sem_derivados_mantem_chaves(tmp_path, monkeypatch):
    run = _corrida(tmp_path)
    monkeypatch.setattr(ui, "RUNS_DIR", tmp_path)
    ui.limpar_cache_stills("corrida", "todos", False, False)
    assert not list((run / "shots" / "stills").glob("*.png"))
    assert (run / "shots" / "clips" / "shot000.key").exists()
    assert (run / "shots" / "animatic.mp4").exists()
