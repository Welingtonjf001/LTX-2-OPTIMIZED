# Mapa do projeto — geração de dança com LTX 2.3

Documento de acompanhamento. Cobre os **dois repositórios**, os módulos, as
integrações externas, em que ponto cada frente está e o que fazer a seguir.

- **Última revisão:** 2026-09-22 (4ª passagem: ChoreoEngine parado há 17 dias, regeneração raposinhas_v2 interrompida em 2/73 janelas)
- **Próxima revisão:** 2026-09-25 (ciclo de 3 dias)

⚠️ **Esta revisão deveria ter acontecido em 2026-08-23.** Ficou 33 dias sem
rodar — o ciclo de 3 dias não foi mantido nesse intervalo. Os fatos abaixo
foram conferidos nesta data; onde a fonte é datada de antes, o texto diz.

---

## 1. Os dois repositórios

| | `LTX-2-OPTIMIZED` | `ChoreoEngine` |
|---|---|---|
| Papel | **Gera o vídeo.** Pipelines LTX 2.3/22b, modelos, UIs de produção | **Decide o movimento.** Extrai dança de vídeo, indexa, compõe coreografia e exporta condicionamento |
| Caminho | `E:\Users\home\Documents\LTX-2-OPTIMIZED` | `E:\Users\home\Documents\ChoreoEngine` |
| Python | `.venv` (torch 2.8+cu128) | **Python global** (é onde vive o `onnxruntime-gpu`) |
| Git | branch `update_v2_3`, 82 arquivos com mudança não commitada (normal aqui — commits acontecem em lote, sem push) | `master`, 4 commits (o último em 2026-08-31); 6 arquivos com mudança não commitada, incluindo 2 módulos novos do Kimodo |
| Tamanho | 3 pacotes + ~20 UIs/ferramentas na raiz | 77 arquivos `.py` versionados, 11.139 linhas em `choreo/` (medido nesta revisão; era 10.508 em 08-20) |
| **Atividade recente** | ativo — commit mais recente 2026-09-17, trabalho não commitado até 2026-09-20/22 (decupagem roteiro→filme, não relacionado a dança) | **parado desde 2026-09-05, 19:17** — nenhum commit, arquivo ou log novo em 17 dias |

A ligação entre os dois é **por disco, não por import**: o ChoreoEngine grava um
*bundle* (`pose.mp4` + `pose_target.npz` + manifesto + grade de beats) e o lado
LTX consome. Ambientes Python separados de propósito.

---

## 2. Módulos — ChoreoEngine

Ordem do fluxo real, de vídeo bruto até bundle:

| Módulo | O que faz |
|---|---|
| `extract/` | vídeo → keypoints 2D (DWPose ONNX), tracking, religação de fragmentos |
| `segment/` | clipe contínuo → **primitivas** (corte por beat, por período ou por novidade) |
| `corpus.py` | varredura de acervo, classificação por gênero, flags de qualidade |
| `laban.py` · `semantics.py` | **classificação semântica**: esforço Laban (peso/tempo/espaço/fluxo) e ações (giro, salto, chute alto…) |
| `library/` | índice SQLite + facetas + grafo de compatibilidade (`compatible`) |
| `sync/` | análise musical: beats, downbeats, seções, envelope de energia |
| `compose/` | **timeline** → clipe único: crossfade, IK de pé, pontes, regras de apoio |
| `procedural/` | rig FK sintético, 12 curvas de tempo, fórmulas, caminhos paramétricos |
| `volume.py` | rasterizador de cápsulas com z-buffer → clay / profundidade / normais |
| `scene.py` | **multi-dançarino**: formações linha/cunha/par, cânone, profundidade |
| `compile.py` | clipe → bundle de condicionamento + comando LTX |
| `prompt_to_params.py` | **LLM local** (Ollama): texto livre → parâmetros e plano coreográfico |
| `kimodo_retarget.py` | **novo (2026-09-04, não commitado)**: remapeia o esqueleto SOMA-77 do Kimodo (texto→movimento 3D) para o COCO-17/kp133 canônico, puro numpy, sem depender do pacote `kimodo` |
| `skeleton.py` · `types.py` · `render.py` · `render_dwpose.py` · `temporal.py` · `dataset.py` · `aist.py` · `io.py` · `cli.py` | módulos já presentes no repo mas fora da listagem anterior deste mapa — nenhuma mudança de comportamento identificada nesta revisão, só correção da lista |
| `ui/` | servidor `http.server` + editor web (porta 8730; não estava no ar no momento desta verificação) |
| `verify.py` | mede o gerado contra o alvo (PCK, MPJPE) |
| `bridge/` | 17 pontes para fora (era 15): geração, Blender, DAZ, GVHMR, auditorias, câmera, mesh preview, testes de cena/rig, e o novo `run_kimodo.py` (subprocesso para o `kimodo-env`, mesmo padrão do `run_f3.py`) |

## 3. Módulos — LTX-2-OPTIMIZED

| Módulo | O que faz |
|---|---|
| `packages/ltx-core` | transformer, atenção, VAE, quantização, ledger de memória |
| `packages/ltx-pipelines` | `ic_lora`, `distilled`, `ti2vid_one_stage`, `ti2vid_two_stages`, `keyframe_interpolation`, `music_to_video` |
| `packages/ltx-trainer` | treino/fine-tune |
| `comfy_*.py` | patches e orquestração do ComfyUI (union-control, GGUF, ingredients) |
| `gguf_backend.py` | sobe e mantém o servidor ComfyUI com modelo residente |
| UIs | `film_maker_ui_v4`, `music_maker_ui_v2`, `web_ui_v4`, `character3d_webui`, `krea2_test_ui` |
| `daz_integration/` | composição de personagem Genesis 9 por LLM local |

**Modelos instalados:** LTX 2.3 22b (dev, distilled-fp8, 2 GGUF), upscaler
espacial x2, Gemma 3 (text encoder), 3 IC-LoRAs (union-control, ingredients,
distilled-384), SD 3.5 Medium, Flux (krea/klein/dev).

---

## 4. Integrações externas

| Integração | Estado | Observação |
|---|---|---|
| **ComfyUI** | ✅ em uso | É o caminho **real** de geração (`bridge/generate.py`). Medido melhor E mais rápido que o pipeline nativo: 768×512 em 265 s (PCK@0.5 0,224) contra 384×256 em 742 s (0,160) |
| **Ollama** | ✅ em uso | `qwen3:8b` para o coreógrafo por texto. Modelos em `G:\ollama\models`. Precisa de `ollama serve` no ar |
| **GVHMR** | ⚠️ parcial | Extrai pose 3D + câmera (SMPL-X). Usado por `bridge/export_3d.py`, mas `_lift_3d()` do pipeline principal continua um stub |
| **Godot 4.7.1** | ⚠️ protótipo | `godot_preview/` renderiza cápsulas a partir do JSON 3D. Fora do editor, roda por linha de comando |
| **Blender 5.2** | ⚠️ pronto, sem uso | Agora no PATH. `bridge/daz_retarget.py` escrito e testado até onde dá |
| **DAZ Studio / Genesis 9** | 🔴 **bloqueado num passo manual — sem mudança** | Esqueleto confirmado (316 ossos, mapeia ~1:1 no COCO-17). Export FBX por script falha (código 101, 6 tentativas). Conferido nesta revisão: **nenhum `.fbx` em `daz_integration/` nem em `daz_integration/runs/`**, e nenhuma pasta de corrida nova desde a verificação anterior. Ver §7 |
| **SAM 3D Body + MHR** | ✅ **funcionando** | Foto → corpo rigado (127 juntas) → **parâmetros SMPL-X** (`betas`, `body_pose`), erro de ajuste 0,0086. Ponte em `mhr_to_smplx.py`, ambiente conda `mhr` |
| **Hunyuan3D-2** | ⚠️ instalado | Malha PBR texturizada a partir de foto, **sem rig** — é aparência, não animação |
| **MoCapAnything V2** | ⚠️ avaliado, **não recomendado agora** | MIT. Mocap de **animais**. O `pose2rot` é agnóstico a esqueleto de verdade — condiciona em embeddings T5 dos *nomes* das juntas + grafo da hierarquia — mas o dicionário de espécies tem **73 animais e 935 objetos, zero humanos**, e os embeddings são pré-computados. Testar com humano exige criar a entrada e baixar um T5. Útil de imediato: `preprocess/extract_character_from_fbx.py` e a saída **BVH** |
| **OpenHands** | ⚠️ instalado, não configurado | `G:\openhands` (SDK 1.34). Conferido nesta revisão: continua sem `config.toml` próprio (só o `pyproject.toml` do pacote). Servidor sobe; falta apontar o LLM. Sem Docker na máquina |
| **Kimodo** (NVIDIA, texto→movimento 3D) | ⚠️ **novo, integrado até o passo 4 de 5** | `E:\Users\home\Documents\kimodo-env` (venv própria) + `kimodo-hf-cache` (17 GB). Testado ponta a ponta em 2026-09-04: prompt → geração 3D → retarget → `segment_clip` → `Library.add`, 1 primitiva real inserida sem caminho especial. **Falta o passo 5**: compor com outras primitivas, gerar pela Union-Control e olhar o vídeo — só os NÚMEROS do retargeting foram conferidos, não se o gerador de vídeo lê o esqueleto sintético como lê um extraído de vídeo real. Ver MEMORIAL §33 (04/09) |
| **LatentSync / Demucs / Whisper / XTTS** | — | Ferramentas de áudio/lipsync na órbita do projeto, fora deste fluxo |

---

## 5. Fase atual

**O ciclo fecha ponta a ponta:** música → coreografia → composição → bundle →
**vídeo gerado na 3090**. Isso não mudou desde 08-20 — mas não houve trabalho
novo no ChoreoEngine desde 2026-09-05, então nada disto foi reexercitado nesta
revisão.

Última geração medida no bundle de referência (curto, `data/bundle_ui/`):
410 s, `PCK@0.2 = 0,76`, `PCK@0.5 = 0,92`, veredito automático "aderência
alta" — **conferido nesta revisão, o arquivo não recebeu entrada nova desde
2026-08-20**, é o mesmo número.

**Teste de música inteira (145,5 s, "Apanhai-me as raposinhas"), 2026-09-05,
ficou pendurado a meio caminho.** Resumo do que o MEMORIAL §33 (05/09)
registra e que este mapa não tinha ainda:

- Duas correções reais em `bridge/`: `ensure_server()` matava o próprio
  ComfyUI que tinha acabado de subir (subprocesso descartável disparava
  `atexit`; corrigido com `os._exit(0)`), e faltava `--disable-dynamic-vram`
  no `gguf_backend.py` (mesma classe de bug já vista duas vezes no
  LTX-2-OPTIMIZED).
- Primeira geração completa (73 janelas, ~4h20 de GPU) saiu com
  `pck@0.2 = 0,015` — mas a causa **não era o gerador**: um bug de
  comparação (`< 4` em vez de "cai dentro da janela") fazia o timeline
  inserir SEMPRE o primitivo parado (idle), nunca o gesto de braço. Corrigido.
- A regeneração corrigida (`bundle_raposinhas_v2`) foi relançada e **parou em
  2 das 73 janelas** (`windows_full2/w00`, `w01`; log
  `outputs_raposinhas_gen_v2.log` termina no início da janela 1/73, sem
  erro visível — parece interrupção manual ou da máquina, não crash
  registrado). **Nenhum resultado novo de aderência existe para o bundle
  v2.** Verificado nesta revisão: os arquivos têm timestamp de 2026-09-05 e
  não avançaram desde então.

O editor web (`choreo/ui/static/index.html`, servido em `/`) tem hoje:

- transporte de música em beats, órbita de câmera que funciona em movimento capturado
- biblioteca de **6.262 primitivas** com filtro semântico (Laban + ações + corpo + faixas)
- coreógrafo automático com dois knobs distintos: *segue a música* (contraste de energia) e *variedade* (tamanho do sorteio)
- **coreógrafo por texto** com LLM local — o modelo planeja intenção por trecho, o acervo resolve
- transição, sincronização ao beat, elenco de 2–6 corpos com formações e cânone
- exportação de bundle e disparo da geração com acompanhamento de progresso

Há um **segundo editor** em `/editor/` (`Interface para Editor de Danças/*.dc.html`)
com prévia em three.js — trilha paralela, não é a principal.

---

## 6. Riscos — situação em 2026-09-22

### Risco novo e principal desta revisão: ChoreoEngine parado

**17 dias sem nenhuma atividade** (último commit 2026-08-31, último arquivo
tocado 2026-09-05 19:17 — `git -C ChoreoEngine log`/`find -newer` conferidos
nesta revisão). Não é um bug: é ausência de trabalho. Duas coisas ficaram
penduradas exatamente no ponto em que pararam — a regeneração `raposinhas_v2`
(§5, 2 de 73 janelas) e a integração Kimodo (§4, passo 5 de 5, "olhar o
vídeo"). Enquanto isso, o LTX-2-OPTIMIZED seguiu ativo, mas numa frente
**não relacionada a dança** (decupagem roteiro→filme, Voo 702) — não há
sinal de que o projeto de dança tenha sido abandonado por decisão, só que a
atenção foi para outro lugar. Vale confirmar com o usuário se é intencional.

### Riscos da revisão de 2026-08-20 — situação confirmada nesta revisão

Os quatro riscos daquela revisão foram atacados; sobra um, e ele é operacional.
Reconferido em 2026-09-22 onde deu: `onnxruntime` segue só com o pacote GPU
instalado (CUDA+TensorRT nos providers) — nenhuma regressão.

| Risco | Situação |
|---|---|
| ChoreoEngine sem versionamento | **resolvido.** 108 arquivos, 103.971 linhas em 2 commits. O `.gitignore` foi corrigido antes: as regras antigas deixavam passar 36 GB de `models/` e 3,1 GB de `packs/`, e um `models/` sem barra inicial teria excluído em silêncio o módulo `choreo/models/` |
| `onnxruntime` CPU sobrepondo o GPU | **resolvido.** Removido o pacote CPU 1.29.0; `onnxruntime-gpu` 1.19.2 reinstalado. Providers voltaram a incluir CUDA e TensorRT — a extração de pose saiu da CPU |
| Editor nunca testado com áudio real | **resolvido, e revelou um bug.** Com `music_input.mp3`: 102,5 BPM e 1.219 onsets. O bug: a grade enviada ao servidor mandava `onsets_sec` vazio sempre, então o coreógrafo caía para energia constante e o slider "segue a música" era **inerte**. Corrigido — agora responde "energia da música" |
| Reinício obrigatório após mexer em pacotes | **mitigado, não eliminado.** Continua sendo a natureza da coisa; agora está documentado com o sintoma exato no `CLAUDE.md` do LTX |

### Risco descoberto em 2026-08-20 — ainda vale, parcialmente reconferido

**Qual instância do Ollama está no ar importa.** Se o aplicativo desktop subir,
ele lê outro diretório de modelos e não enxerga `G:\ollama\models` — pedir
`qwen3:8b` devolve 404 em `/api/generate` mesmo com o servidor respondendo em
`/api/tags`, e o coreógrafo por texto para de funcionar. O código detecta esse
caso e diz quais modelos aquela instância realmente serve; a correção de fato é
rodar `ollama serve` com `OLLAMA_MODELS` apontando para o G:. Nesta revisão,
`/api/tags` respondeu e listou `qwen3-vl:30b` (modelo do lado LTX, não do
ChoreoEngine) — a instância no ar está lendo `G:\ollama\models` corretamente
agora, mas isso não foi testado especificamente com `qwen3:8b`/o coreógrafo por
texto, que não rodou nesta revisão.

## 7. Próximas etapas, por ordem de retorno

### 1. Terminar o que já foi pago — *retorno mais barato, dois itens já em andamento*
Dois trabalhos de 2026-09-05 pararam no meio, não por bloqueio técnico, e o
custo de GPU já entrou:

- **Retomar (ou relançar) a regeneração `bundle_raposinhas_v2`.** Só 2 das 73
  janelas foram renderizadas antes de parar; o bug que invalidou a primeira
  tentativa (timeline 100% parado) já está corrigido. Faltam ~4h de GPU.
- **Fechar o passo 5 do Kimodo**: compor a primitiva sintética (já
  retargeada e testada numericamente) com outras, gerar pela Union-Control e
  olhar o vídeo. É o único jeito de saber se o gerador de vídeo lê o
  esqueleto sintético do Kimodo como lê um extraído de vídeo real — sem isso,
  a integração é matematicamente correta mas não comprovadamente útil.

### 2. Ligar o corpo da foto ao movimento — *a frente que destravou em agosto*
O caminho SAM 3D Body → MHR → SMPL-X **já funciona** e entrega `betas` + `body_pose`
no mesmo espaço que o GVHMR usa. Falta aplicar o movimento do ChoreoEngine sobre
esse corpo e renderizar. Isto tornou a rota do DAZ opcional, não obrigatória.
Sem progresso desde 08-20 (nenhuma atividade no ChoreoEngine para verificar).

Nota de licença: o rig **MHR cru** é Apache 2.0 🟢; só a conversão *para SMPL-X*
herda a restrição acadêmica. Dirigir pelo MHR direto evita isso.

### 3. Personagem via DAZ — *alternativa, não mais o caminho crítico*
Abrir o DAZ Studio na mão, carregar `People/Genesis 9/Genesis 9.duf`,
**File → Export → Autodesk FBX**, salvar. Uma vez só — o personagem não muda por
dança. O `bridge/daz_retarget.py` já está escrito e testado, e o Blender 5.2 já
está no PATH. O script falha hoje exatamente onde deve: no FBX inexistente
(reconferido nesta revisão — continua ausente).

### 4. Investigar a qualidade da geração — *PCK 0,76 contra 0,94 de referência*
O 0,94 citado na interface foi medido "com alvo limpo". Vale entender a diferença:
material de origem, enquadramento ou parâmetros. Sem progresso desde 08-20.

### 5. Terminar o OpenHands — *pendência aberta, sem bloquear nada*
O `agent-server` sobe; falta configurar o LLM (Ollama) via API — esta versão não
tem `config.toml` (reconferido nesta revisão — ainda não).

---

## 8. Como retomar

```bash
ChoreoEngine.bat
```

Sobe o servidor e abre o editor. Interface principal em `http://127.0.0.1:8730/`,
editor alternativo em `/editor/`. Para o coreógrafo por texto, `ollama serve`
precisa estar rodando.

Documentos vivos: `ChoreoEngine/CLAUDE.md` (configuração verificada),
`ChoreoEngine/MEMORIAL.md` (33 seções de experimentos datados, com os números
medidos — a última, §33 de 2026-09-05, é onde o trabalho parou) e
`LTX-2-OPTIMIZED/CLAUDE.md` (hardware, modelos, pipelines).

---

## Mudanças nesta revisão

### 2026-09-22 (esta revisão, atrasada — devida em 08-23)

- **ChoreoEngine parado há 17 dias** (último commit 2026-08-31, último arquivo
  tocado 2026-09-05). Achado principal desta passagem — ver §6.
- Documentado pela primeira vez neste mapa o que o MEMORIAL §33 (2 entradas,
  04/09 e 05/09) já registrava e que ainda não tinha entrado aqui: a
  integração do **Kimodo** (texto→movimento 3D) até o passo 4 de 5, e o teste
  de música inteira ("raposinhas") que achou e corrigiu dois bugs reais em
  `bridge/` mas cuja regeneração corrigida **parou em 2 de 73 janelas** — sem
  resultado de aderência para o bundle v2.
- DAZ/FBX: reconferido, continua bloqueado no mesmo passo manual, nenhuma
  mudança.
- `onnxruntime` CPU+GPU: reconferido, continua resolvido (só o pacote GPU
  instalado).
- OpenHands: reconferido, continua sem `config.toml`.
- Módulos do repositório corrigidos na listagem (choreo/ tinha mais arquivos
  do que o mapa listava; nenhuma mudança de comportamento identificada além
  do Kimodo).
- Reordenadas as próximas etapas: terminar os dois trabalhos já pagos
  (regeneração v2, validação em vídeo do Kimodo) passou a ser o item 1, à
  frente do corpo-via-foto (que não teve progresso porque o repo ficou
  parado).

### 2026-08-20 (revisão anterior)

3ª passagem. SAM 3D Body → MHR → SMPL-X funcionando (fecha a frente de
personagem real sem depender do DAZ). `onnxruntime` CPU/GPU resolvido, editor
testado com áudio real (revelou e corrigiu bug do slider "segue a música").
ChoreoEngine versionado (2 commits, 103.971 linhas). Risco novo: instância
errada do Ollama serve catálogo diferente.

### Revisões anteriores a 08-20

Não preservadas neste documento — a primeira entrada desta seção foi criada
nesta revisão (2026-09-22).
