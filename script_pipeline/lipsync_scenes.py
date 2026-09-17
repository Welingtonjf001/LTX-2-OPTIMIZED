"""Stage [6]: lip-sync each rendered dialogue clip to its own dry TTS line, via the
project's already-existing tensorxx_ge/lipsync.py (LatentSync 1.6 first, Wav2Lip
fallback -- unmodified, no fork needed: it's already generic to "one face, one dry
audio track per call", which is exactly what a per-line clip is).

Action-only clips (no dialogue, no audio) pass through untouched -- there is nothing
to sync.

CLI: python -m script_pipeline.lipsync_scenes --run-dir DIR [--engine auto]
Reads <run>/scenes/clips.json, writes <run>/lipsync/scene_XX[_line_YY]_synced.mp4 and
<run>/lipsync/synced_clips.json (id -> final video path, consumed by assemble_final.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_clips(run_dir: Path) -> list[dict]:
    clips_path = run_dir / "scenes" / "clips.json"
    if not clips_path.exists():
        raise FileNotFoundError(f"{clips_path} not found -- run render_scenes first.")
    return json.loads(clips_path.read_text(encoding="utf-8"))


def _dubit_clip(clip: dict, output_path: Path, *, run_dir: Path, args, log) -> str | None:
    """Refaz a boca do clipe CRU do LTX com o IC-LoRA DubIt (ltx25_backend.generate_v2v).

    Em `congelar` a fala do TTS entra congelada e o modelo so redesenha a boca sobre
    ela: voz e duracao ficam as do TTS. Em `gerar` o modelo gera a fala a partir do
    texto (voz do modelo, com o TTS como referencia de timbre) -- o texto vai citado no
    prompt, que e o que o workflow oficial pede."""
    import ltx25_backend
    import ltx_loras
    lora = ltx_loras.BY_KEY["dubit-2.3"].local
    if ltx_loras.installed_path(lora) is None:
        log(f"{clip['id']}: LoRA DubIt nao instalado (python ltx_loras.py download dubit-2.3)")
        return None
    prompt = ""
    try:
        plan = json.loads((run_dir / "parse" / "shot_plan.json").read_text(encoding="utf-8"))["shots"]
        prompt = plan[int(clip["id"].rsplit("shot", 1)[1])]["video_prompt"]
    except (OSError, ValueError, KeyError, IndexError):
        pass
    if args.dubit_audio == "gerar":
        try:
            falas = json.loads((run_dir / "dialogue" / "lines.json").read_text(encoding="utf-8"))
            texto = next(e["text"] for e in falas if e["line_index"] == clip.get("line_index")
                         and e["scene_index"] == clip.get("scene_index"))
            prompt = f'{prompt} The character speaks in Brazilian Portuguese, saying: "{texto}"'
        except (OSError, StopIteration, KeyError):
            pass
    try:
        return ltx25_backend.generate_v2v(
            prompt, clip["video_path"], str(output_path), lora=lora,
            strength=args.dubit_strength, guide_strength=args.dubit_guide_strength,
            audio=args.dubit_audio, audio_path=clip["audio_path"],
            log_cb=lambda m: log(f"    [dubit] {m}"))
    except Exception as e:
        log(f"{clip['id']}: DubIt falhou ({type(e).__name__}: {str(e)[:300]})")
        return None


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--engine", default="auto", choices=["auto", "latentsync", "wav2lip", "dubit", "none"],
                        help="dubit = IC-LoRA DubIt do LTX refaz a boca sobre a fala (ver "
                             "ltx25_backend.generate_v2v); none = nao sincroniza, mantem o clipe do LTX")
    parser.add_argument("--no-syncnet", action="store_true",
                        help="nao medir com o SyncNet do LatentSync (fica so o proxy de correlacao)")
    parser.add_argument("--dubit-strength", type=float, default=1.0, help="forca do LoRA DubIt no modelo")
    parser.add_argument("--dubit-guide-strength", type=float, default=1.0,
                        help="quanto o clipe original prende a imagem (1 = fiel, menos = mais livre)")
    parser.add_argument("--dubit-audio", default="congelar", choices=["congelar", "gerar"],
                        help="congelar = mantem a voz do TTS; gerar = o modelo gera a fala do texto "
                             "(voz do modelo, com o TTS so como referencia)")
    parser.add_argument("--dubit-fallback", default="auto", choices=["auto", "none"],
                        help="se o DubIt falhar num plano: auto = LatentSync/Wav2Lip; none = clipe original")
    args = parser.parse_args(argv)

    from script_pipeline import run_folder
    from script_pipeline.lipsync_audit import audit_lipsync
    from tensorxx_ge.lipsync import apply_lipsync, lipsync_status

    run_dir = Path(args.run_dir).resolve()
    clips = _load_clips(run_dir)
    lipsync_dir = run_folder.subdir(run_dir, "lipsync")

    log = lambda msg: run_folder.append_log(run_dir, msg)  # noqa: E731
    status = lipsync_status()
    log(f"lipsync_scenes: LatentSync {'disponivel' if status.latentsync_available else 'indisponivel'}, "
        f"Wav2Lip {'disponivel' if status.wav2lip_available else 'indisponivel'}.")

    synced_manifest = []
    audit_report = {}
    failures = 0
    for clip in clips:
        if not clip.get("ok") or not clip.get("video_path"):
            log(f"{clip['id']}: sem video renderizado; pulando lip-sync.")
            synced_manifest.append({**clip, "final_video_path": None, "lipsync_applied": False})
            failures += 1
            continue

        if not clip.get("audio_path"):
            if clip.get("minimax_native_speech"):
                # Fix #6 (avaliacao visual 2026-09-17): fala do MiniMax e nativa,
                # embutida no proprio clipe -- nao ha WAV de TTS para o Wav2Lip/
                # LatentSync sincronizar contra (por isso audio_path e None), mas
                # ha fala de verdade, e ela nunca era auditada. O SyncNet aceita
                # audio=None e usa o audio JA EMBUTIDO no video -- mede se a boca
                # bate com a PROPRIA fala do MiniMax, o que da sinal de sincronia
                # real ainda que NAO verifique se o TEXTO falado bate com o
                # roteiro (isso exigiria transcricao/ASR, fora de escopo aqui).
                final_video = clip["video_path"]
                synced_manifest.append({**clip, "final_video_path": final_video, "lipsync_applied": False})
                log(f"{clip['id']}: fala nativa do MiniMax (sem TTS de referencia); "
                    "medindo sincronia contra o audio embutido no proprio clipe.")
                veredito = {"score": None, "faces_detected": None, "frames_sampled": None,
                            "motivo": "MiniMax: sem TTS de referencia -- so sync labial medido, "
                                      "conteudo da fala nao verificado.",
                            "audio_source": "minimax_native"}
                if not args.no_syncnet:
                    try:
                        from script_pipeline.syncnet_audit import LIMIAR_CONF, syncnet_score
                        sn = syncnet_score(final_video)
                        veredito["syncnet"] = sn
                        if sn.get("conf") is not None:
                            marca_sn = "OK" if sn["conf"] >= LIMIAR_CONF else "FRACO"
                            log(f"  SyncNet (audio nativo MiniMax): conf {sn['conf']:.2f} "
                                f"({marca_sn}), dist {sn['min_dist']:.2f}, offset "
                                f"{sn['av_offset']} quadro(s) -- so sincronia, sem checar "
                                "conteudo da fala.")
                        else:
                            log(f"  SyncNet: nao mediu ({sn.get('motivo')})")
                    except Exception as e:
                        veredito["syncnet"] = {"conf": None, "motivo": f"{type(e).__name__}: {e}"}
                audit_report[clip["id"]] = veredito
                continue
            # Action-only clip: nothing to sync, pass through as-is.
            synced_manifest.append({**clip, "final_video_path": clip["video_path"], "lipsync_applied": False})
            log(f"{clip['id']}: sem fala (cena de acao); mantendo clipe original.")
            continue

        if args.engine == "none":
            # Opcao explicita de NAO usar lip-sync: o LTX ja gerou a boca sobre a fala
            # condicionada. O clipe segue com a propria faixa (voz + trilha).
            synced_manifest.append({**clip, "final_video_path": clip["video_path"], "lipsync_applied": False})
            log(f"{clip['id']}: lip-sync desligado (--engine none); mantendo clipe do LTX.")
            continue

        if args.engine != "dubit" and not status.available:
            log(f"{clip['id']}: nenhum motor de lip-sync disponivel; mantendo clipe original.")
            synced_manifest.append({**clip, "final_video_path": clip["video_path"], "lipsync_applied": False})
            continue

        output_path = lipsync_dir / f"{clip['id']}_synced.mp4"
        if args.engine == "dubit":
            result = _dubit_clip(clip, output_path, run_dir=run_dir, args=args, log=log)
            if result is None and args.dubit_fallback == "auto" and status.available:
                log(f"{clip['id']}: DubIt falhou; caindo para LatentSync/Wav2Lip.")
                result = apply_lipsync(clip["video_path"], clip["audio_path"], str(output_path),
                                       work_dir=str(lipsync_dir), engine="auto", log=log)
        else:
            result = apply_lipsync(
                clip["video_path"], clip["audio_path"], str(output_path),
                work_dir=str(lipsync_dir), engine=args.engine, log=log,
            )
        final_video = str(result) if result is not None else clip["video_path"]
        if result is not None:
            synced_manifest.append({**clip, "final_video_path": final_video, "lipsync_applied": True})
            log(f"{clip['id']}: lip-sync ok -> {result}")
        else:
            log(f"{clip['id']}: lip-sync falhou; mantendo clipe original sem sincronia labial.")
            synced_manifest.append({**clip, "final_video_path": final_video, "lipsync_applied": False})

        # AUDITORIA DE QUALIDADE (2026-09-09, pedido do usuario): o bloco
        # acima so confere se o processo TECNICO rodou sem excecao -- isto
        # aqui mede se a boca do video final realmente se move em sincronia
        # com o audio, aplicado ou nao (um fallback sem sync tambem entra na
        # medicao, e deve medir baixo -- e o comportamento esperado). Nunca
        # bloqueia a corrida: so registra.
        try:
            veredito = audit_lipsync(final_video, clip["audio_path"], log=log)
        except Exception as e:
            veredito = {"score": None, "faces_detected": 0, "frames_sampled": 0,
                        "motivo": f"auditoria falhou: {type(e).__name__}: {e}"}
        # SyncNet (LSE-C/LSE-D), 2026-09-13: a medida padrao da area, com o avaliador que o
        # LatentSync ja traz. O proxy acima mede mal rosto a 3/4 e fala curta expressiva --
        # MEDIDO no mesmo dia: shot005 do "O Primeiro Tour" deu -0,29 no proxy e conf 8,5
        # no SyncNet (boa sincronia). Registra os dois; o SyncNet e o que decide.
        if not args.no_syncnet:
            try:
                from script_pipeline.syncnet_audit import LIMIAR_CONF, syncnet_score
                sn = syncnet_score(final_video, clip["audio_path"])
                veredito["syncnet"] = sn
                if sn.get("conf") is not None:
                    marca_sn = "OK" if sn["conf"] >= LIMIAR_CONF else "FRACO"
                    log(f"  SyncNet: conf {sn['conf']:.2f} ({marca_sn}), dist {sn['min_dist']:.2f}, "
                        f"offset {sn['av_offset']} quadro(s)")
                else:
                    log(f"  SyncNet: nao mediu ({sn.get('motivo')})")
            except Exception as e:
                veredito["syncnet"] = {"conf": None, "motivo": f"{type(e).__name__}: {e}"}
        audit_report[clip["id"]] = veredito
        if veredito["score"] is None:
            log(f"  auditoria de sync: nao foi possivel medir ({veredito['motivo']}).")
        elif veredito["frames_sampled"] < 15:
            # MEDIDO 2026-09-09: um plano curto (shot008, 1,47s, 11 quadros
            # amostrados) mediu correlacao negativa mesmo com o LatentSync
            # tendo rodado -- amostra pequena demais pra confiar no numero.
            # Nao classifica OK/SUSPEITO aqui; so avisa que a base e curta.
            log(f"  auditoria de sync: correlacao {veredito['score']:.3f}, mas so "
                f"{veredito['frames_sampled']} quadro(s) amostrado(s) -- baixa confianca "
                f"(plano curto), nao classificado como OK/suspeito.")
        else:
            # Calibrado com 5 clipes reais (2026-09-09): planos com lip-sync
            # aplicado mediram 0.29-0.34; planos sem sync (fallback pro
            # clipe cru) mediram -0.23 a -0.05 -- 0.15 separa os dois grupos
            # com folga, sem exigir uma correlacao alta (o proxy e' ruidoso).
            marca = "OK" if veredito["score"] >= 0.15 else "SUSPEITO -- boca pode nao acompanhar a fala"
            log(f"  auditoria de sync: correlacao boca-audio {veredito['score']:.3f} ({marca}).")

    (lipsync_dir / "synced_clips.json").write_text(
        json.dumps(synced_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (lipsync_dir / "lipsync_audit.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    # O resumo decide pelo SyncNet quando ele mediu. MEDIDO 2026-09-13 (teste_auditoria):
    # o proxy marcou os 3 planos como suspeitos (0,21 / 0,03 / -0,16) e o SyncNet aprovou
    # os 3 (conf 8,95 / 4,38 / 9,73, offset 0) -- o de -0,16 era o de MELHOR sincronia.
    from script_pipeline.syncnet_audit import LIMIAR_CONF
    confs = [v["syncnet"]["conf"] for v in audit_report.values()
             if (v.get("syncnet") or {}).get("conf") is not None]
    if confs:
        fracos = sum(1 for c in confs if c < LIMIAR_CONF)
        log(f"lipsync_scenes: SyncNet -- {len(confs)}/{len(audit_report)} clipe(s) medido(s), "
            f"{fracos} fraco(s) (conf < {LIMIAR_CONF}). Relatorio: {lipsync_dir / 'lipsync_audit.json'}")
    medidos = [v["score"] for v in audit_report.values() if v["score"] is not None]
    suspeitos = sum(1 for s in medidos if s < 0.3)
    if audit_report and not confs:
        log(f"lipsync_scenes: auditoria de sync (proxy, sem SyncNet) -- {len(medidos)}/{len(audit_report)} "
            f"clipe(s) medido(s), {suspeitos} suspeito(s) (correlacao < 0.3). "
            f"Relatorio: {lipsync_dir / 'lipsync_audit.json'}")
    ok_count = sum(1 for c in synced_manifest if c.get("final_video_path"))
    log(f"lipsync_scenes: {ok_count}/{len(clips)} clipe(s) com video final disponivel "
        f"({sum(1 for c in synced_manifest if c.get('lipsync_applied'))} com lip-sync aplicado).")

    if ok_count == len(clips):
        run_folder.mark_stage_complete(run_dir, "lipsync")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
