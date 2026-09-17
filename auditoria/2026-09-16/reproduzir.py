"""Reproduções dos defeitos atuais: PASS significa que o defeito foi observado.

Executar com .venv/Scripts/python.exe auditoria/2026-09-16/reproduzir.py.
Somente diretórios temporários e dependências simuladas; sem modelos/GPU/rede.
"""
import ast
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_function(rel, name, namespace):
    tree = ast.parse((ROOT / rel).read_text(encoding='utf-8-sig'))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(module, rel, 'exec'), namespace)
    return namespace[name]


class Reproductions(unittest.TestCase):
    def test_01_five_interfaces_lose_vocal_fallback(self):
        files = ['music_maker_ui_v2.py', 'music_maker_ui_v2_25.py',
                 'music_maker_ui_v3.py', 'music_maker_ui_v3_25.py', 'music_maker_ui_gguf.py']
        with tempfile.TemporaryDirectory() as td:
            for rel in files:
                with self.subTest(script=rel):
                    asr = Mock()
                    env = dict(os=os, AUDIO_CLIPS_DIR=td, CURRENT_LOG='',
                               load_audio_compatible=lambda _: (None, 24000, 3),
                               load_lyrics_segments=lambda *_: ([], None, None),
                               get_or_create_vocal_stem=Mock(side_effect=RuntimeError('Demucs unavailable')),
                               transcribe_lyrics_for_audio=asr,
                               gr=SimpleNamespace(update=lambda **kw: kw))
                    fn = load_function(rel, 'slice_audio', env)
                    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                        result = fn('audio.wav', '', 24, 121)
                    self.assertIn('CURRENT_LOG', result[0])
                    asr.assert_not_called()

    def test_02_unicode_concat_path_fails(self):
        from script_pipeline import assemble_final as af
        with tempfile.TemporaryDirectory() as td:
            with patch.object(af, '_has_audio_stream', return_value=True), \
                 patch.object(af, '_normalize_audio_for_concat', side_effect=lambda p, *a, **k: p):
                with self.assertRaises(UnicodeEncodeError):
                    af.concat_videos([str(Path(td) / 'cena_ação.mp4')], Path(td)/'out.mp4', work_dir=Path(td)/'work', log=lambda _: None)

    def test_03_import_without_face_missing_parent(self):
        import numpy as np
        from PIL import Image
        from script_pipeline import import_reference as ir
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)/'source.png'
            Image.new('RGB', (32, 32)).save(source)
            with patch.object(ir, '_detect_face_bbox', return_value=(None, np.zeros((32,32,3), dtype=np.uint8))):
                with self.assertRaises(FileNotFoundError):
                    ir.import_reference_photo(str(source), Path(td)/'new_refs'/'ref.png')

    def chain_env(self, td):
        return dict(Path=Path, shot={'frames':401, 'video_prompt':'dialogue'}, fps=24,
                    minimax_chain_max_seconds=6, ltx_chain_max_seconds=6,
                    clip_path=Path(td)/'shot000.mp4', seed=1234, i=0,
                    minimax_aspect_ratio=None, minimax_megapixels=None, minimax_turbo=True,
                    minimax_h3_backend=SimpleNamespace(generate=Mock(), DEFAULT_ASPECT='16:9', DEFAULT_MEGAPIXELS=0.4),
                    ltx25_backend=SimpleNamespace(generate=Mock()),
                    _extrair_ultimo_frame=lambda *_: True,
                    prompt_video='dialogue', width=960, height=544, video_loras=[], ic_spec=None, wav_cond='voice.wav')

    def test_04_minimax_last_frame_discarded_with_two_refs(self):
        from script_pipeline import assemble_final as af
        with tempfile.TemporaryDirectory() as td:
            env = self.chain_env(td)
            fn = load_function('script_pipeline/render_shots.py', '_minimax_chain_generate', env)
            with patch.object(af, 'concat_videos', return_value=True):
                fn(['still.png', 'sheet.png'], None, Path(td)/'out.mp4', lambda _: None)
            calls = env['minimax_h3_backend'].generate.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1].kwargs['ref_images'], ['still.png', 'sheet.png'])

    def test_05_ltx_chain_shortens_and_loses_audio(self):
        from script_pipeline import assemble_final as af
        with tempfile.TemporaryDirectory() as td:
            env = self.chain_env(td)
            fn = load_function('script_pipeline/render_shots.py', '_ltx_chain_generate', env)
            with patch.object(af, 'concat_videos', return_value=True):
                fn('still.png', Path(td)/'out.mp4', lambda _: None)
            calls = env['ltx25_backend'].generate.call_args_list
            self.assertEqual(sum(c.kwargs['num_frames'] for c in calls), 387)
            self.assertEqual([c.kwargs['audio_conditioning'] for c in calls], ['voice.wav', None, None])

    def test_06_chain_ignores_concat_failure(self):
        from script_pipeline import assemble_final as af
        with tempfile.TemporaryDirectory() as td:
            for name in ['_ltx_chain_generate', '_minimax_chain_generate']:
                env = self.chain_env(td)
                fn = load_function('script_pipeline/render_shots.py', name, env)
                dest = Path(td)/'out.mp4'
                args = ('still.png', dest, lambda _: None) if 'ltx' in name else ([], None, dest, lambda _: None)
                with patch.object(af, 'concat_videos', return_value=False):
                    self.assertIsNone(fn(*args))
                self.assertFalse(dest.exists())

    def test_07_manifest_calls_missing_video_ok(self):
        from script_pipeline.render_shots_stage import build_clips_manifest
        plan = {'shots':[{'scene':1}]}
        result = build_clips_manifest([{'shot':0, 'clip':'nonexistent.mp4'}], plan, {})
        self.assertTrue(result[0]['ok'])

    def test_08_audio_cache_collides_for_different_content(self):
        fn = load_function('script_pipeline/render_shots.py', '_audio_key', {'Path':Path})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'voice.wav'
            path.write_bytes(b'AAAA')
            before = fn(str(path))
            path.write_bytes(b'BBBB')
            self.assertEqual(before, fn(str(path)))

    def test_09_reference_cache_ignores_directory_and_content(self):
        import hashlib
        fn = load_function('script_pipeline/render_shots.py', '_still_key', {'Path':Path, 'hashlib':hashlib})
        shot = {'storyboard_prompt':'portrait', 'framing':'close'}
        a = fn(shot, 'cast_a/ref.png', 960, 544, 'model.safetensors')
        b = fn(shot, 'cast_b/ref.png', 960, 544, 'model.safetensors')
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
