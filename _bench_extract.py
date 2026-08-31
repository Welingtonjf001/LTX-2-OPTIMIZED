# -*- coding: utf-8 -*-
"""Compara motores de LLM na tarefa de extracao prosa -> roteiro, com criterios
objetivos (nao impressao): falas preservadas COMO DIALOGO PARSEADO, planos
recuperados, e tempo."""
import sys, time
sys.path.insert(0, ".")
from script_pipeline.prose_to_screenplay import convert, extract_quoted_dialogue, check_dialogue_preserved
from script_pipeline.parse_screenplay import parse_structure

SRC = open("outputs/screenplay/20260823_231415_storyplay_20260823_231415/input/storyplay_20260823_231415.txt",
            encoding="utf-8").read()
esperado = len(extract_quoted_dialogue(SRC))
print(f"origem: {esperado} falas entre aspas\n")

for engine in sys.argv[1:]:
    t0 = time.time()
    try:
        sp, missing = convert(SRC, engine=engine, log=lambda m: None)
    except Exception as e:
        print(f"{engine:24s} ERRO: {e}"); continue
    dt = time.time() - t0
    if not sp:
        print(f"{engine:24s} FALHOU ({dt:.0f}s)"); continue
    kept, miss = check_dialogue_preserved(SRC, sp)
    scenes = parse_structure(sp)
    falas = sum(len(s.dialogue) for s in scenes)
    linhas = len([l for l in sp.splitlines() if l.strip()])
    print(f"{engine:24s} {dt:6.0f}s  falas_verbatim={len(kept)}/{esperado}  "
          f"parseadas={falas}  cenas={len(scenes)}  linhas={linhas}")
    open(f"_bench_{engine.replace(':','_').replace('.','_')}.txt","w",encoding="utf-8").write(sp)
