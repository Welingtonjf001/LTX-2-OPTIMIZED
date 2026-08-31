"""Screenplay -> video pipeline (v1).

Turns a plain-text screenplay into a movie: parses scenes/characters/dialogue,
generates a storyboard image per scene, synthesizes character voices, renders
each scene in LTX conditioned on its storyboard image, lip-syncs dialogue,
layers ambient sound + reverb, and assembles the final cut.

Companion to the music-video pipeline (music_maker_ui_v2/v3.py,
tensorxx_ge/webui_v2/v3.py) -- reuses its lip-sync (tensorxx_ge/lipsync.py),
room-reverb (audio_fx.py) and run-folder/manifest conventions, but is driven
by a screenplay instead of a song.

See CLAUDE.md-adjacent docs / the plan for the full architecture. Driven by
``screenplay_to_video.py`` at the project root.
"""
