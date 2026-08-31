# Mapa do projeto — geração de dança com LTX 2.3

Documento de acompanhamento. Cobre os **dois repositórios**, os módulos, as
integrações externas, em que ponto cada frente está e o que fazer a seguir.

- **Última revisão:** 2026-08-20 (3ª passagem: SAM 3D Body → SMPL-X funcionando)
- **Próxima revisão:** 2026-08-23 (ciclo de 3 dias)

---

## 1. Os dois repositórios

| | `LTX-2-OPTIMIZED` | `ChoreoEngine` |
|---|---|---|
| Papel | **Gera o vídeo.** Pipelines LTX 2.3/22b, modelos, UIs de produção | **Decide o movimento.** Extrai dança de vídeo, indexa, compõe coreografia e exporta condicionamento |
| Caminho | `E:\Users\home\Documents\LTX-2-OPTIMIZED` | `E:\Users\home\Documents\ChoreoEngine` |
| Python | `.venv` (torch 2.8+cu128) | **Python global** (é onde vive o `onnxruntime-gpu`) |
| Git | branch `update_v2_3`, com histórico | `master`, 2 commits (import inicial em 2026-08-20) |
| Tamanho | 3 pacotes + ~20 UIs/ferramentas na raiz | 66 arquivos `.py`, 10.508 linhas em `choreo/` |

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
| `ui/` | servidor `http.server` + editor web |
| `verify.py` | mede o gerado contra o alvo (PCK, MPJPE) |
| `bridge/` | 15 pontes para fora: geração, Blender, DAZ, GVHMR, auditorias |

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
| **DAZ Studio / Genesis 9** | 🔴 **bloqueado num passo manual** | Esqueleto confirmado (316 ossos, mapeia ~1:1 no COCO-17). Export FBX por script falha (código 101, 6 tentativas). Ver §7 |
| **SAM 3D Body + MHR** | ✅ **funcionando** | Foto → corpo rigado (127 juntas) → **parâmetros SMPL-X** (`betas`, `body_pose`), erro de ajuste 0,0086. Ponte em `mhr_to_smplx.py`, ambiente conda `mhr` |
| **Hunyuan3D-2** | ⚠️ instalado | Malha PBR texturizada a partir de foto, **sem rig** — é aparência, não animação |
| **MoCapAnything V2** | ⚠️ avaliado, **não recomendado agora** | MIT. Mocap de **animais**. O `pose2rot` é agnóstico a esqueleto de verdade — condiciona em embeddings T5 dos *nomes* das juntas + grafo da hierarquia — mas o dicionário de espécies tem **73 animais e 935 objetos, zero humanos**, e os embeddings são pré-computados. Testar com humano exige criar a entrada e baixar um T5. Útil de imediato: `preprocess/extract_character_from_fbx.py` e a saída **BVH** |
| **OpenHands** | ⚠️ instalado, não configurado | `G:\openhands` (SDK 1.34). Servidor sobe; falta apontar o LLM. Sem Docker na máquina |
| **LatentSync / Demucs / Whisper / XTTS** | — | Ferramentas de áudio/lipsync na órbita do projeto, fora deste fluxo |

---

## 5. Fase atual

**O ciclo fecha ponta a ponta:** música → coreografia → composição → bundle →
**vídeo gerado na 3090**.

Última geração medida: 410 s, `PCK@0.2 = 0,76`, `PCK@0.5 = 0,92`, veredito
automático "aderência alta".

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

## 6. Riscos — situação em 2026-08-20

Os quatro riscos da revisão anterior foram atacados; sobra um, e ele é operacional.

| Risco | Situação |
|---|---|
| ChoreoEngine sem versionamento | **resolvido.** 108 arquivos, 103.971 linhas em 2 commits. O `.gitignore` foi corrigido antes: as regras antigas deixavam passar 36 GB de `models/` e 3,1 GB de `packs/`, e um `models/` sem barra inicial teria excluído em silêncio o módulo `choreo/models/` |
| `onnxruntime` CPU sobrepondo o GPU | **resolvido.** Removido o pacote CPU 1.29.0; `onnxruntime-gpu` 1.19.2 reinstalado. Providers voltaram a incluir CUDA e TensorRT — a extração de pose saiu da CPU |
| Editor nunca testado com áudio real | **resolvido, e revelou um bug.** Com `music_input.mp3`: 102,5 BPM e 1.219 onsets. O bug: a grade enviada ao servidor mandava `onsets_sec` vazio sempre, então o coreógrafo caía para energia constante e o slider "segue a música" era **inerte**. Corrigido — agora responde "energia da música" |
| Reinício obrigatório após mexer em pacotes | **mitigado, não eliminado.** Continua sendo a natureza da coisa; agora está documentado com o sintoma exato no `CLAUDE.md` do LTX |

### Risco novo, descoberto nesta revisão

**Qual instância do Ollama está no ar importa.** Se o aplicativo desktop subir,
ele lê outro diretório de modelos e não enxerga `G:\ollama\models` — pedir
`qwen3:8b` devolve 404 em `/api/generate` mesmo com o servidor respondendo em
`/api/tags`, e o coreógrafo por texto para de funcionar. O código agora detecta
esse caso e diz quais modelos aquela instância realmente serve, mas a correção
de fato é rodar `ollama serve` com `OLLAMA_MODELS` apontando para o G:.

## 7. Próximas etapas, por ordem de retorno

### 1. Ligar o corpo da foto ao movimento — *a frente que destravou*
O caminho SAM 3D Body → MHR → SMPL-X **já funciona** e entrega `betas` + `body_pose`
no mesmo espaço que o GVHMR usa. Falta aplicar o movimento do ChoreoEngine sobre
esse corpo e renderizar. Isto tornou a rota do DAZ opcional, não obrigatória.

Nota de licença: o rig **MHR cru** é Apache 2.0 🟢; só a conversão *para SMPL-X*
herda a restrição acadêmica. Dirigir pelo MHR direto evita isso.

### 2. Personagem via DAZ — *alternativa, não mais o caminho crítico*
Abrir o DAZ Studio na mão, carregar `People/Genesis 9/Genesis 9.duf`,
**File → Export → Autodesk FBX**, salvar. Uma vez só — o personagem não muda por
dança. O `bridge/daz_retarget.py` já está escrito e testado, e o Blender 5.2 já
está no PATH. O script falha hoje exatamente onde deve: no FBX inexistente.

### 3. Investigar a qualidade da geração — *PCK 0,76 contra 0,94 de referência*
O 0,94 citado na interface foi medido "com alvo limpo". Vale entender a diferença:
material de origem, enquadramento ou parâmetros.

### 4. Terminar o OpenHands — *pendência aberta, sem bloquear nada*
O `agent-server` sobe; falta configurar o LLM (Ollama) via API — esta versão não
tem `config.toml`.

---

## 8. Como retomar

```bash
ChoreoEngine.bat
```

Sobe o servidor e abre o editor. Interface principal em `http://127.0.0.1:8730/`,
editor alternativo em `/editor/`. Para o coreógrafo por texto, `ollama serve`
precisa estar rodando.

Documentos vivos: `ChoreoEngine/CLAUDE.md` (configuração verificada),
`ChoreoEngine/MEMORIAL.md` (26 seções de experimentos datados, com os números
medidos) e `LTX-2-OPTIMIZED/CLAUDE.md` (hardware, modelos, pipelines).
