# Auditoria geral dos scripts — 2026-09-27

Ponto de partida: a auditoria externa do CERCO EM SEUL (`.audit_video/seul_20260927/avaliacao.md`)
e as perguntas sobre o que o roteiro precisa indicar, lacunas, pré-visualização 3D e efeito nos
motores. Método: leitura do código de todos os caminhos que geram vídeo, reprodução dos defeitos
com os artefatos reais das corridas, testes sem GPU (349 passando) e validação com GPU/LLM real
onde o defeito dependia do modelo. Histórico e medições: `MEMORIAL.md` §3.131-3.134.

## 1. Respostas

**O roteiro deve vir mais completo? O que falta indicar?** Sim. Cada plano é gerado sem
memória dos outros; o que o texto não diz, o modelo inventa, e inventa diferente em cada plano.
O que precisa estar escrito, com exemplos e um modelo de cena, está em
[`script_pipeline/GUIA_ROTEIRO.md`](script_pipeline/GUIA_ROTEIRO.md):

- lugar nomeável e geografia fixa;
- **todos** que aparecem, inclusive quem não fala, com aparência e gênero explícitos;
- ação escrita como agente + verbo físico + alvo + resultado;
- posição e direção;
- objetos e estado que persiste;
- causa, ação, reação e consequência, sem reiniciar o evento na fala;
- tempo;
- atuação separada da emoção;
- estilo visual separado de som.

**O parse/LLM deve verificar lacunas?** Sim, e agora verifica. Etapa nova `[L]`
(`screenplay_gaps.py`), antes da decupagem, nos dois caminhos (decupagem e screenplay):

- **Parte determinística:** pessoa que age sem cadastro, ação sem agente, contato/tiro sem
  alvo, fala que repete a ação vizinha, fala sobre ação física, aparência incompleta, voz por
  palpite e mudanças de estado a carregar.
- **Parte com LLM:** sugestões por ação, marcadas como inferidas.
- Saída em `parse/lacunas.md`. Só relata; `--lacunas bloquear` para a corrida.

No CERCO original a etapa apontou 11 lacunas críticas, as mesmas da auditoria externa. Na v2,
com os figurantes cadastrados, sobrou uma: "Seo-yeon fires her gun" sem direção. O roteiro
dizia "para o alto", e o enriquecimento perdeu isso.

**Cenas complexas precisam de prévia 3D?** Algumas, e medir antes é o que diz quais. A etapa
nova `[M] complexidade` (`shot_complexity.py`) pontua cada plano pela primitiva de movimento,
pelas pessoas, pelos objetos, pela mudança de estado, pela pressão de tempo e pela câmera, e
recomenda um de três caminhos:

| Nível | Recomendação |
|---|---|
| simples | still + prompt |
| média | revisar o blocking 2D, que já existe |
| complexa | previs 3D de baixa resolução, ou dividir o plano em ação / reação / consequência |

No CERCO v2: 14 simples, 8 médias, 3 complexas (derrubada, algemação, tiro com as portas
fechando). As complexas coincidem com os planos-sentinela da auditoria externa. Os closes de
fala cuja ação ficou fora do quadro saem marcados.

**O sistema deve renderizar polígonos e movimento em baixa resolução? Sim, e agora faz sozinho**
(implementado no mesmo retorno, `script_pipeline/previs_spec.py`, MEMORIAL §3.135). A peça 3D já
existia:

- `spatial_pipeline`: Blender com manequins, câmera real, profundidade, pose e máscaras.
- `motion_director`: InterGen, para duas pessoas em contato.

mas exigia um spec escrito à mão (`SPATIAL_PIPELINE.md`). O `previs_spec.py` gera esse spec
automaticamente do `shot_plan` + elenco + figurantes + `motion_plan` + `complexity_report.json`:
entidades do elenco (cor do manequim do figurino), âncoras pelo lado de tela, primitivas do
movimento viram mudança de estado com continuidade entre planos, objetos citados. Renderiza no
Blender (CPU, sem difusão) só os planos do nível pedido — folha `shots/previs_3d.png` e clipe
`shots/previs_3d.mp4`, revisáveis antes dos stills. Ligado em `run_decupagem`
(`--previs3d {complexas,medias,todos,off}`, padrão `complexas`) e na `decupagem_ui`. **Não
incluído**: still/vídeo condicionados automaticamente pelo previs em produção real (o mecanismo
de conexão existe, `--previs3d-stills`, mas não foi testado ponta a ponta no LTX/MiniMax).

**Como isso afeta os motores?**

| Motor | Recebe o previs por | Limite |
|---|---|---|
| FLUX (stills) | img2img / ReferenceLatent do blocking (`spatial_pipeline`) | caminho espacial só no FLUX |
| LTX 2.5 / 2.3 | still de partida + guia de pose/profundidade (IC-LoRA union-control) | fator de referência 2 é recusado junto de `audio_conditioning`: guia de pose só em plano **sem fala**; MSR 2.5 ocupa o mesmo ponto do grafo que o IC-LoRA (identidade **ou** controle, por plano) |
| MiniMax H3 | só o still/referência | ControlNet bloqueado por VRAM; no long-take só o 1º plano do take recebe referência, então plano complexo vai melhor como clipe avulso |
| LongCat | — | só fala; closes não precisam de previs |

Custo: um previs 3D em baixa resolução leva minutos de Blender por plano; um vídeo refeito
leva 5–40 min por plano. Só compensa onde o risco de refazer é real, e é para isso que a
medição existe.

## 2. Achados e correções

Gravidade: **A** = altera o filme em silêncio; **B** = degrada qualidade/diagnóstico; **C** = dívida.

| # | Onde | Achado | Grav. | Status |
|---|---|---|---|---|
| 1 | `shot_plan.py` (`_enquadre_sem_sujeito`) | ação sem sujeito cadastrado virava `insert` por exclusão (explosão e multidão como "detalhe sem rosto") | A | corrigido |
| 2 | `cast_characters.py` + `shot_plan.py` | figurantes que agem (Presidente, terrorista) fora do elenco: cada still inventava outra pessoa | A | corrigido: `extra`/`aliases`, detecção via LLM ou lista de papéis |
| 3 | `shot_plan.py` | descrição do parceiro omitida depois do primeiro plano próprio | A | corrigido |
| 4 | `shot_plan._video_prompt` | close de fala executava a ação física do beat (Ha-eun vestiu o capacete do fugitivo) | A | corrigido: reação com a ação fora do quadro |
| 5 | 5 chamadores | emoção literal: "alegre" virou sorriso na derrubada | B | corrigido: `emocao_para_video` único (shot_plan, story_editor, prompt_polish, render_shots, render_scenes) |
| 6 | `shot_plan._look_e_interior` | trilha, efeitos sonoros e câmera da direção de arte em **todo** prompt | B | corrigido: `_arte_visual` |
| 7 | `run_decupagem.py` | **movimento não aplicado**: `--ate animatic` regerava o plano sem movimento e o vídeo herdava | A | corrigido: aplica sempre |
| 8 | `motion_conditioner.py` | algemar, conduzir, chutar, "takes down", saltar, deixar objeto e avistar caíam em `idle` | A | corrigido: primitivas `restrain`/`escort` |
| 9 | `shot_plan.enrich_camera_style` | com `--camera-llm`, a reconstrução perdia o parceiro (existia desde 16/09) | A | corrigido |
| 10 | `story_editor.py` | reconstrução tratava plano de fala como ação muda (`speaking` vinha de `quote`) e perdia o parceiro | A | corrigido |
| 11 | `prompt_polish.py` | usava a emoção de ação muda em plano de fala | B | corrigido |
| 12 | `lipsync_scenes.py` | lip-sync que falha ou sem rosto: a fala do roteiro sumia | A | corrigido: `_mux_tts` |
| 13 | `cast_characters.assign_voices` | gênero por alternância (Ha-eun masculina) e voz infantil para adulto | A | corrigido: foto > LLM > texto; idade pelo roteiro; `--revoice`; `voice_locked` |
| 14 | `render_scenes.py` (screenplay, UI 7810) | clipes de ação sem aparência de ninguém; fala sobre ação física; emoção literal | A | corrigido com as mesmas regras |
| 15 | `generate_storyboards.build_prompt` (`storyplay25`, stills sem prompt próprio) | figurantes nunca entravam | B | corrigido |
| 16 | `render_shots.py` (MiniMax long-take) | take de 9 planos morto pelo watchdog; um take que falha perdia todos os planos | A | corrigido: teto de 4 planos por take + fallback avulso |
| 17 | `visual_continuity_audit.py` | Ollama 0.34.4 força raciocínio no Qwen3-VL e o JSON saía cortado: o gate parava em 3 de 25 planos e mostrava "bloqueado" | A | corrigido: `num_predict` 8192 / `num_ctx` 16384; relatório separa `rejected` de `auditor_errors`. **Resíduo:** 2/25 ainda falham de forma intermitente |
| 18 | `verify_output.py` | "0 problemas" com gates bloqueados e identidade reprovada | B | corrigido: aviso `GATES_NOT_APPROVED` |
| 19 | `screenplay_ui.py` | duplica a montagem das etapas do `screenplay_to_video` | C | parcial: a auditoria de lacunas foi ligada nos dois; a duplicação continua |
| 20 | gate visual | stills da v2: 21 de 23 reprovados, parte por excesso de literalidade ("broche não visível") ou contradição | B | **aberto**: calibrar o contrato do gate |
| 21 | duração | filme sai mais curto que o plano (v1 −3,4 s; v2 −2,5 s) entre lip-sync, mix e a grade 8k+1 | B | **aberto** |
| 22 | `parse_screenplay` (enriquecimento) | `shot_list` só tem `action` + `actor`: sem alvo, objetos, posição, estado | B | **aberto**: a etapa [L] relata, mas o contrato do parse não mudou |

## 3. Quem gera vídeo e o que herda

| Caminho | Herda automaticamente | Ajuste feito |
|---|---|---|
| `run_decupagem` / `decupagem_ui` (7913) | tudo: [L], [M] complexidade, movimento sempre, elenco, vozes, lip-sync, gate, verify | a UI chama o `run_decupagem`, sem mudança de interface; opções novas só por CLI (`--lacunas`, `--lacunas-sem-llm`, `--qwen-lora`) |
| `render_shots_stage` / `render_shots` | decupagem, figurantes, teto do long-take | — |
| `screenplay_to_video` / `screenplay_ui` (7810) / `render_scenes` | elenco, vozes, lip-sync (compartilhados) | regras de prompt portadas (#14); [L] ligada nos dois |
| `storyplay25` (7911) | figurantes no `build_prompt` | — |
| `motion_director` | figurantes viram co-sujeito, mais planos elegíveis ao InterGen | — |
| `continuous_chain`, music makers, `web_ui`/`film_maker` | não usam a decupagem | fora de escopo |

A `screenplay_ui` e as opções novas da `decupagem_ui` **não foram testadas no navegador**. Só
o laço de trabalho mudou, sem componente visual novo.

## 4. Próximos passos, por retorno

1. **Contrato do parse com eventos** (#22): o `shot_list` enriquecido passa a trazer agente,
   alvo, objetos, posição e estado inicial/final por item. A decupagem e o movimento passam a
   ler campos, não regex. Pré-requisito do item 2.
2. ~~**Previs 3D rascunho automático** para os planos complexos~~ — **feito neste retorno**
   (`previs_spec.py`, MEMORIAL §3.135). Falta testar `--previs3d-stills` ponta a ponta com GPU
   real (still condicionado pelo blocking do previs) e decidir automaticamente quando usar o
   previs em produção (hoje é opt-in por `--previs3d`).
3. **Calibrar o gate visual** (#20): separar "defeito que o espectador vê" de "detalhe do
   descritor não visível"; hoje a reprovação em massa impede o uso como trava real.
4. **Duração** (#21): medir por estágio (bruto → lip-sync → mix → montagem) e preservar a
   folga do TTS.
5. **Unificar** `screenplay_ui` com `screenplay_to_video.run_stage` (#19).
