# Sistema de vídeo — roteiro → filme (`script_pipeline`)

Transforma um roteiro em texto puro num filme com falas sincronizadas, som ambiente e
montagem final. Roda inteiramente local, em duas GPUs (RTX 3090 24 GB + RTX 4070 12 GB),
sem nenhuma chamada a serviço externo.

Este é o pipeline **irmão** do sistema música → vídeo já existente
(`music_maker_ui_v2/v3.py`, `tensorxx_ge/webui_v2/v3.py`). Os dois compartilham a camada
pesada — geração LTX, lip-sync, reverb, montagem ffmpeg — e diferem no que dispara a
geração: lá é uma faixa de áudio fatiada em cenas, aqui é um roteiro decomposto em planos.

---

## 1. Visão geral

```
roteiro.txt
    │
    ▼
[1] parse      ── estrutura + enriquecimento LLM  → parse/scenes_enriched.json
    ▼
[2] cast       ── elenco, descritores, vozes      → characters/cast.json
    ▼
[3] storyboard ── uma imagem POR PLANO (FLUX)     → storyboard/*.png
    ▼
[4] dialogue   ── uma voz por fala (TTS)          → dialogue/*.wav
    ▼
[5] render     ── um clipe por plano (LTX 2.3)    → scenes/*.mp4
    ▼
[6] lipsync    ── boca sincronizada com a fala    → lipsync/*_synced.mp4
    ▼
[7] mix        ── reverb/posição + som ambiente   → intermediate/*.mp4
    ▼
[8] assemble   ── concat + normalização de áudio  → final/movie.mp4
    ▼
[9] verify     ── auto-revisão pós-render         → verification.json
```

Cada estágio roda como **subprocesso próprio**. Isso não é organização estética: é a
disciplina de VRAM do projeto. O LTX carrega transformer e upsampler que juntos não cabem
com o FLUX ou o LatentSync residentes, então cada modelo pesado precisa ser liberado antes
do próximo começar. Processos separados garantem isso; threads não garantiriam.

Cada estágio marca sua conclusão em `generation_manifest.json`, então `--resume-from`
retoma sem refazer o que já ficou pronto — o que permite editar `cast.json`, um prompt de
plano ou um WAV de fala à mão no meio do caminho.

### Contrato de entrega (modo final)

`run_decupagem --ate final` é fail-closed: não aceita `--no-visual-audit` nem
`--visual-unresolved continue`. Esses modos continuam disponíveis para
diagnóstico/previews, mas não para declarar um filme pronto. Os gates de still e
vídeo precisam aprovar todos os planos; depois de lipsync/mix,
`clip_identity_audit --strict` roda antes da montagem; por fim
`verify_output --strict` bloqueia o sucesso se houver erro técnico. Um MP4
existir não significa aprovação.

O modo espacial MANUAL não é inferido automaticamente do texto livre. Ele exige um
blocking/spec de locação, entidades, câmeras e eventos; quando usado, deve ser
gerado e auditado antes dos stills/vídeos conforme `SPATIAL_PIPELINE.md`. A
ausência de spec não pode ser descrita como “mapa 3D executado”.

Desde 2026-09-27 existe um segundo modo, AUTOMÁTICO (`previs_spec.py`, MEMORIAL §3.135, só no
caminho da decupagem — `run_decupagem`/`decupagem_ui`, não neste pipeline de 9 estágios): gera o
spec sozinho a partir de `shot_plan.json` + `cast.json` + `motion_plan.json` +
`shots/complexity_report.json` e renderiza manequins/câmera em baixa resolução no Blender (CPU,
sem difusão) para os planos que a etapa `[M] complexidade` marcou como complexos —
`shots/previs_3d.png`/`.mp4`. É um rascunho de blocking para REVISÃO, não o estado persistente
entre cenas do modo manual; não confundir os dois. `--previs3d {complexas,medias,todos,off}`.

---

## 2. Os estágios em detalhe

### [1] parse — `parse_screenplay.py`

Extrai a estrutura do roteiro em duas camadas, deliberadamente separadas:

**Camada determinística (regex/heurística):** cabeçalho `INT./EXT.`, nome de personagem em
maiúsculas, texto da fala, rubrica entre parênteses. Também aceita **roteiro corrido** —
parágrafo único sem formatação, do tipo "cena - no ponto de ônibus a jovem Park Min aguarda
- ela exclama: ..." — através de `_extract_freeform_scene()`.

**Camada LLM (enriquecimento):** produz o `visual_prompt` da cena, a emoção de cada fala
quando não há rubrica, e a **shot list** — a decupagem ordenada que define quantos clipes a
cena vai gerar e em que ordem:

```json
"shot_list": [
  {"type": "action",   "visual": "Park Min waits at the bus stop in the rain."},
  {"type": "dialogue", "line_index": 0},
  {"type": "action",   "visual": "Park Min stands in front of the bus door."},
  {"type": "dialogue", "line_index": 1}
]
```

A estrutura nunca é decidida pelo LLM sozinho — ele só enriquece. Isso é proposital: quando
o enriquecimento falha, o pipeline ainda tem uma estrutura válida para seguir.

> **Motor de enriquecimento: use `gemma4`.** É o padrão desde 2026-08-11. O Gemma 3 12B foi
> medido devolvendo token-salad multilíngue em vez de JSON (`'reso ... repart... Yn
> ...cnicas...'`). A falha é silenciosa e cascateia: sem `visual_prompt` e sem `shot_list`,
> o storyboard cai para uma imagem por cena e o render usa **o roteiro cru em português**
> como prompt do LTX — inclusive as falas entre aspas, que o modelo desenha como balão de
> fala sobre o rosto do ator. O estágio `verify` hoje detecta essa condição
> (`ENRICHMENT_FAILED`). Gemma 4 E2B roda em venv isolado (`gemma4_env`).

### [2] cast — `cast_characters.py`

Consolida os personagens: nome canônico, descritor visual fixo (repetido em todo prompt para
manter consistência) e escolha de voz. `cast.json` é **editável à mão** antes de continuar.

**Voz (desde 2026-09-27).** Gênero, em ordem de confiança: foto de referência (insightface
genderage) > LLM lendo o roteiro inteiro > pista textual do descritor > alternância marcada
`gender_guessed: true`. Idade: o **roteiro** manda (`_parse_age`: "mid-30s", "7 anos",
"idoso"); a foto só vale quando o texto cala — ela estima a idade do ATOR (Min-jun "mid-30s"
saiu 23 pela foto). A voz é escolhida pela faixa etária (infantil/jovem/adulto/maduro), nunca
infantil para adulto, sem repetir voz de outro personagem. `import_reference` reavalia as
vozes ao importar uma foto; `cast_characters --revoice` faz só isso; `"voice_locked": true`
protege uma voz escolhida à mão. MEDIDO no CERCO EM SEUL: Ha-eun (sem pronome no descritor)
tinha voz masculina e Seo-yeon (35 anos) a voz infantil — os dois corrigidos pela foto.

**Figurantes recorrentes sem nome** (o Presidente, o motociclista, o garçom): entram no
elenco com `"extra": true` e `"aliases"` (todas as formas do texto, nos dois idiomas —
"Presidente" e "the President"). Com `--engine`, o LLM junta sinônimos que são a MESMA pessoa
("the motorcycle rider" = "the terrorist" — decisão de enredo, nunca de tabela) e escreve a
aparência, que passa pela mesma auditoria de completude dos demais. Sem LLM, uma lista de
papéis com artigo definido (≥ 2 frases) cobre o básico, juntando só tradução PT↔EN. Sem
figurante cadastrado, cada still inventava outra pessoa para o mesmo papel.

Personagens aceitam **foto de referência** (`reference_image`), roteada para o workflow
`character_flux_reference.json`. O nó `ReferenceLatent` do FLUX.2 honra a imagem — confirmado
em teste controlado com imagem de controle sem referência.

### [2b] lacunas — `screenplay_gaps.py` (desde 2026-09-27)

Relatório do que o roteiro **não diz** e o vídeo vai inventar, antes de gastar GPU:
`parse/lacunas.md` + `.json`. Determinístico: pessoa que age sem cadastro, ação sem agente,
ação de contato/tiro sem alvo, fala cujo beat repete a ação vizinha (o evento recomeça na
tela), fala sobre ação física (um close não a mostra), aparência incompleta, voz por palpite
e a lista de **mudanças de estado** a carregar (explodiu, fechou, algemado). Com `--engine`,
o LLM acrescenta sugestões por ação, marcadas como inferidas. Nunca reescreve o roteiro.
Roda nos dois caminhos (decupagem e screenplay); só na decupagem pode bloquear
(`--lacunas bloquear`). O que o roteiro precisa indicar: [`GUIA_ROTEIRO.md`](GUIA_ROTEIRO.md).

### [3] storyboard — `generate_storyboards.py`

Uma imagem **por plano**, não por cena (`--per-shot`, padrão). A versão por cena existia e
foi medida como uma das duas causas de "todas as cenas iguais": com uma única imagem por
cena, todo clipe daquela cena partia do mesmo frame e convergia visualmente.

Checkpoint padrão **FLUX.2 Klein 9B fp8**, com encoder Qwen3-8B e `flux2-vae`. O prompt leva
âncora de estilo positiva ("Photorealistic cinematic film still, live action footage") e
negativa reforçada contra balões de fala, letreiro e desenho.

> FLUX.2 **dev** está no disco mas **não é utilizável**: exige o encoder Mistral-Small-3.2-24B
> (hidden 5120 → 15360) e o projeto tem o Qwen3-8B (4096 → 12288). Falha com
> `mat1 and mat2 shapes cannot be multiplied (512x12288 and 15360x6144)`.

### [4] dialogue — `synthesize_dialogue.py` + `dialogue_tts.py`

Uma voz seca por fala, por personagem. Dois motores atrás da mesma interface
(`engine="auto"|"xtts"|"qwen"`), cada um em venv isolado:

| Motor | Estado | Característica |
|---|---|---|
| **XTTS-v2.0.2** | padrão, modelo já baixado | 4 vozes de referência prontas, sem controle de emoção |
| **Qwen3-TTS** | alternativa | 9 timbres, emoção por instrução textual, voz por descrição |

O idioma das vozes é selecionável (padrão pt-BR). Quando a tradução de cenas para inglês está
ligada — o LTX responde melhor a prompt em inglês — **o texto das falas nunca é traduzido**:
só a descrição visual da cena vai para o inglês.

### [5] render — `render_scenes.py`

O coração. Um clipe LTX 2.3 por plano da shot list.

**Composição do prompt** (`compose_prompt()`) segue a ordem do guia de prompt do LTX 2.3:
ação principal → detalhe de movimento → personagem/ambiente → **câmera e iluminação por
último**; parágrafo único no presente, abaixo de 200 palavras, fala entre aspas com direção
de atuação, emoção como pista física visível, e áudio declarado ao final em camadas
(ambiência / voz / efeitos).

**Regras compartilhadas com a decupagem (desde 2026-09-27):** clipe de ação leva a aparência
de até dois personagens citados (inclusive figurantes) — antes ia sem descrição de ninguém;
fala cujo beat é ação física (proteger, derrubar, atirar) vira **reação** ("reacts urgently to
the action happening just outside the frame") — o modelo transferia a ação para quem fala; e
a emoção vira expressão visível (`shot_plan.emocao_para_video`), sem sorriso em cena de ação.

**Restrições do LTX 2.3** aplicadas automaticamente:
- resolução múltipla de **64**
- grade de frames **8k+1**
- duração do clipe derivada da duração do áudio da fala, com teto `--max-clip-seconds`
  (clipes longos travavam o upsampler perto do limite de VRAM)

**Condicionamento por imagem:** `--image CAMINHO FRAME_IDX FORÇA`, repetível.
`frame_idx == 0` usa `VideoConditionByLatentIndex`; qualquer outro índice usa
`VideoConditionByKeyframeIndex` — é o recurso de condicionar em qualquer frame latente.

**Continuidade encadeada** (`--chain-continuity`): o último frame de um clipe vira imagem
inicial do próximo.

> **Cuidado com `--end-keyframe-strength`.** Ancorar o fim de todo clipe no mesmo storyboard
> cria um ponto fixo que colapsa a cadeia inteira numa imagem só — foi a segunda causa medida
> de "todas as cenas iguais". Hoje o keyframe final aponta para o storyboard do **próximo**
> plano. O padrão é `0.0` (desligado).

**Som ambiente** (`--ambient-audio`): clipes sem fala recebem áudio ambiente gerado pelo LTX
(`LTX_GENERATE_AMBIENT_AUDIO=1`), em vez de ficarem mudos e puxarem o volume do filme para
baixo.

**Motor alternativo Wan 2.2** (`--engine wan`): duas arquiteturas distintas, roteadas
corretamente — 5B TI2V usa `Wan22ImageToVideoLatent` (48 canais latentes, /16 espacial,
`wan2.2_vae`, só imagem inicial); 14B usa `WanImageToVideo`/`WanFirstLastFrameToVideo`
(16 canais, /8 espacial, `wan_2.1_vae`, dois arquivos MoE). Grade de frames **4k+1**, passo de
resolução **16**. Código escrito e modelos baixados; **ainda não exercitado ponta a ponta.**

### [6] lipsync — `lipsync_scenes.py`

Reusa `tensorxx_ge/lipsync.py` (`apply_lipsync`, `engine="auto"`): LatentSync primeiro,
Wav2Lip como fallback. Um rosto e um áudio seco por chamada.

> **Limitação ativa:** o Wav2Lip está barrado por conflito de ambiente
> (`Numba needs NumPy 2.2 or less. Got NumPy 2.5`). Quando o LatentSync não encontra rosto
> ("Face not detected"), o clipe segue **sem sincronia labial** — o pipeline registra e
> continua, em vez de falhar.

### [7] mix — `mix_audio.py`

Reverb e posicionamento espacial via `audio_fx.py`, mais som ambiente opcional. Sempre
**depois** do lip-sync, nunca antes — o lip-sync precisa da voz seca.

### [8] assemble — `assemble_final.py`

Concat na ordem do roteiro. Crucial: `_normalize_audio_for_concat()` reencoda **todos** os
clipes para **AAC / 48 kHz / estéreo** antes de concatenar.

> Isso não é zelo excessivo. O demuxer concat com `-c copy` exige parâmetros idênticos e
> adota os do **primeiro** arquivo. Um filme já foi entregue com faixa AAC aparentemente
> saudável e **−91 dB** — silêncio digital puro — porque os clipes de fala eram PCM 16 kHz e
> o concat descartou todos.

### [9] verify — `verify_output.py`

Auto-revisão pós-render. Roda automático no fim de `--stage all` e atrás do botão de
montagem na interface. Não-estrita por padrão: reporta sem invalidar a execução;
`--strict-verify` faz falhar. Grava `verification.json`.

| Código | Detecta |
|---|---|
| `NO_FILM` | arquivo final ausente |
| `SILENT_FILM` | sem faixa de áudio **ou** faixa praticamente muda (< −60 dB) |
| `MUTE_DIALOGUE` | clipe com fala mas sem áudio audível |
| `BLACK_CLIP` | clipe totalmente preto |
| `SHORT_FILM` | filme mais curto que a soma dos clipes (concat descartou material) |
| `ENRICHMENT_FAILED` | cenas sem `visual_prompt` nem `shot_list` (ver estágio 1) |

A medição de áudio usa `volumedetect`, **não** presença de stream. `ffprobe` mostrava faixa
AAC saudável nos dois filmes mudos que passaram batido; só medir o volume revelou os −91 dB.

> **Um check que foi tentado e descartado.** `IDENTICAL_SHOTS`, para detectar a cadeia
> colapsada, não funciona e por isso não emite veredito. Medido contra um run ruim conhecido
> e um bom: dHash de frame deu máx **0.734** (ruim) contra **0.766** (bom); histograma de cor
> deu **0.814** contra **0.828**. O run **bom pontua mais alto nas duas métricas** — nenhum
> limiar separa os dois. Estatística de frame não enxerga esse defeito porque ele é semântico
> (mesmo arranjo de sujeitos, mesmo blocking) enquanto todos os planos de uma cena
> legitimamente dividem local, paleta e luz. Os números ficam em `verification.json` como
> medição informativa. O caminho honesto para resolver é embedding CLIP, já presente em
> `models/clip_interrogator`.

---

## 3. Como rodar

### Interface Gradio

```bash
python -u screenplay_ui.py --port 7810
```

Abas: **Cenas** (visão do parse), **Decupagem** (edição plano a plano antes de renderizar),
**Elenco** (descritores, vozes, foto de referência), **Storyboards** (pré-visualização,
regerar ou substituir por imagem própria), **Vídeo final** (laudo do `verify` acima do
player). Log ao vivo por `gr.Timer`.

### CLI

```bash
python screenplay_to_video.py --script roteiro.txt --stage all --ambient-audio
```

Retomar uma execução a partir de um estágio:

```bash
python screenplay_to_video.py --resume-from outputs/screenplay/<run> --stage render
```

Flags que mais importam:

| Flag | Padrão | Efeito |
|---|---|---|
| `--enrich-engine` | `gemma4` | motor de enriquecimento (não use `gemma3`) |
| `--width` / `--height` | 896 × 512 | múltiplos de 64 obrigatórios |
| `--max-clip-seconds` | 5.0 | teto de duração por clipe |
| `--ambient-audio` | desligado | som ambiente nos planos sem fala |
| `--chain-continuity` | ligado | último frame alimenta o clipe seguinte |
| `--end-keyframe-strength` | 0.0 | ver alerta no estágio 5 |
| `--engine` | `ltx` | `ltx` ou `wan` |
| `--strict-verify` | desligado | faz a execução falhar em caso de defeito |
| `--no-verify` | — | pula a auto-revisão |

---

## 4. Pasta de execução

```
outputs/screenplay/<timestamp>_<slug>/
├── input/                     roteiro original copiado
├── parse/scenes_enriched.json estrutura + shot list  [editável]
├── characters/cast.json       elenco e vozes         [editável]
├── storyboard/*.png           uma imagem por plano
├── dialogue/*.wav             falas secas
├── scenes/*.mp4               clipes brutos do LTX
├── lipsync/*_synced.mp4       clipes com boca sincronizada
├── intermediate/              normalização de áudio, frames de cadeia
├── final/movie.mp4            filme montado
├── verification.json          laudo da auto-revisão
├── logs/                      log completo por execução
└── generation_manifest.json   estágios concluídos (base do --resume-from)
```

---

## 5. Modelos usados

| Papel | Arquivo | Onde |
|---|---|---|
| Geração de vídeo | `ltx-2.3-22b-distilled-fp8.safetensors` | `models/` |
| Upscaler espacial | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `models/` |
| Encoder de texto do LTX | Gemma 3 12B | `models/gemma3` |
| Storyboard | `flux-2-klein-9b-fp8.safetensors` + Qwen3-8B + `flux2-vae` | ComfyUI |
| Storyboard alternativo | Qwen-Image-2.1 (texto e até 10 referências) | `E:\Users\home\Documents\Qwen-Image-2.1` |
| Enriquecimento de roteiro | Gemma 4 E2B | `models/gemma4-e2b` (venv `gemma4_env`) |
| Vídeo alternativo | Wan 2.2 TI2V 5B | `models/` |
| TTS | XTTS-v2.0.2 / Qwen3-TTS | venvs externos |
| Lip-sync | LatentSync / Wav2Lip | venvs externos |

> O encoder de texto do LTX **não pode ser trocado**. O conector treinado tem formato
> `[4096 × 188160]`, onde 188160 = 3840 × 49 (hidden size do Gemma 3 × saídas de camada).
> A arquitetura está travada nesse encoder; o Gemma 4 entra no pipeline apenas como LLM de
> enriquecimento, num processo separado.

---

## 6. Pontos abertos

- **Wav2Lip inoperante** — conflito numpy 2.5 / numba. Sem fallback quando o LatentSync não
  acha rosto.
- **Wan 2.2 não testado ponta a ponta** — código escrito, modelos baixados, nunca exercitado.
- **Detecção de planos repetidos sem solução** — ver o alerta no estágio 9; o caminho é CLIP.
- **Encoder de texto como gargalo** — medido em ~44% do tempo de render, limitado a 4 GiB
  enquanto transformer e upsampler recebem 18 GiB. Hipótese de realocação nunca testada.
