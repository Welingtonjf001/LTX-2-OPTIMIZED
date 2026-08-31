# Memorial do LTX-2-OPTIMIZED

Documento de continuidade. Primeira seção escrita em 2026-08-21, ao fim da
sessão que investigou e integrou o LTX-2.5. Quem pegar o projeto daqui deve
conseguir retomar sem reconstruir raciocínio nem repetir erro já pago.

Os outros documentos têm papéis distintos: `README.md` explica **como usar**
(upstream); `CLAUDE.md` é a **configuração operacional** verificada nesta
máquina (hardware, ambientes, comandos, bugs conhecidos). Este aqui registra
**por que o sistema é assim** e **onde cada frente está** — a parte que some
se só o `CLAUDE.md` sobreviver.

---

## 1. O que o sistema é

Fork local do `Lightricks/LTX-2`, rodando a geração **2.3/22b** em produção
(RTX 3090) através de UIs próprias (`music_maker_ui*.py`, `film_maker_ui_v4.py`,
`web_ui_v4.py`) que chamam `ltx_pipelines` via subprocesso. `packages/ltx-core`
e `packages/ltx-pipelines` são o motor; tudo na raiz é orquestração/UI em cima
dele. Ver `CLAUDE.md` para a config verificada da máquina (índices de GPU
trocados, ambientes Python, modelos instalados).

---

## 2. Estado por versão

*(atualizado em 2026-08-25; as seções §3.x abaixo estão em ordem cronológica e
as mais recentes ficam DEPOIS de §6 — o histórico foi crescendo por acréscimo.)*

| Versão | Estado | Evidência |
|---|---|---|
| **2.3/22b** | **produção** | UIs rodando, pipelines `distilled`/`ic_lora`/`ti2vid_*` validados, modelos em `models/` |
| **2.5/22b** | **produção**, rota ComfyUI | clipes reais de 19,3s a 1280x704 com áudio condicionado; UIs `*_25` completas; §3.4 em diante |

O 2.5 **não** usa `ltx_pipelines`: usa `ltx25_backend.py` (ComfyUI via HTTP),
com `ltx_pipelines_25.py` como shim de CLI para as UIs não precisarem saber
disso. Motivo em §3.6.

### O que está ligado no 2.5, e por padrão ou não

| recurso | padrão | chave |
|---|---|---|
| condicionamento por imagem (`strength`) | **ligado** | corrigido em §3.16 |
| condicionamento por trilha de áudio | **ligado** | `LTX25_AUDIO_COND=0` desliga (§3.15) |
| two-stage (upscale latente + refino) | desligado | `LTX25_TWO_STAGE=1` (§3.18) |
| variante dev (CFG real) | desligado | `LTX25_VARIANT=dev` (§3.8) |
| keyframes / quadros intermediários | por chamada | `keyframes=[...]` (§3.12) |

---

## 3. LTX-2.5 — sessão de 2026-08-21

### Por que isso começou

Pedido do usuário: verificar o que o LTX-2.5 exige (quantizações, encoders) e
o impacto nos scripts daqui. A resposta inicial (pesquisa web) apontou Gemma 4
como encoder novo — mas a pesquisa web por si só não é confiável o bastante
para decidir arquitetura (ver §6, "erro evitado #1"), então tudo abaixo foi
**verificado no código e nos pesos reais**, não só em documentação da Lightricks.

### O que se descobriu, em ordem

1. **O checkpoint 2.5 vem fatiado por componente** (`diffusion_models/`,
   `text_encoders/`, `vae/`, `latent_upscale_models/` — convenção ComfyUI),
   diferente do monolito único que o 2.3 usa. Baixados os componentes
   essenciais para T2V/I2V em `models/2.5/` (75 GB): transformer distilled
   bf16, text encoder Gemma4-12B bf16, VAEs, upscalers, duration head, LoRA
   distilled-450. Não baixado: variante `dev` (42 GB, treinável, não
   necessária pra inferência), `nvfp4` (exige Blackwell, não temos), variantes
   `comfy-int8-convrot` (formato específico do ComfyUI, redundante com o
   caminho que seguimos).

2. **O encoder é "Gemma 4 12B unified"** (`gemma4_unified`, encoder-free
   multimodal — patches de imagem/áudio crus projetados direto no espaço do
   LLM), não um Gemma 3 "maior". Não existe implementação disso em `ltx_core`
   nem no repo oficial da Lightricks (confirmado direto no GitHub
   `Lightricks/LTX-2`, branch `main`, em 2026-08-21 — só tem `text_encoders/gemma/`).
   **Mas existe pronta e testada dentro do ComfyUI vendorizado neste repo**
   (`ComfyUI/comfy/text_encoders/gemma4.py` — `Gemma4_12B`/`Gemma4UnifiedBase`),
   porque o ComfyUI upstream já adicionou suporte a Gemma 4 antes da Lightricks
   publicar código próprio. `scripts_2.5/gemma4_text_encoder.py` reaproveita
   isso via `comfy.sd.load_text_encoder_state_dicts` (que detecta a arquitetura
   pelos próprios tensores do checkpoint — não precisei hardcodar a config).

3. **Bug real encontrado no `gemma4.py` vendorizado**: um `Gemma4Model.process_tokens`
   sobrescrito que não existe no ComfyUI upstream e quebrava o contrato de
   `sd1_clip.py::forward` (esperava 4 valores, o override só devolvia 1).
   Comparado contra `comfyanonymous/ComfyUI` no GitHub pra confirmar que era
   mesmo um bug local, não uma diferença de versão legítima. Removido.

4. **O DiT 2.5 (`AVTransformer3DModel`) usa `use_embeddings_connector: true`**
   — um sub-transformer que processa a saída do encoder antes da cross-attention
   principal. Achado inicialmente como "risco alto, sem referência nenhuma" —
   **errado**: `ltx_core/text_encoders/gemma/embeddings_connector.py::Embeddings1DConnector`
   já existe, já é genérico/config-driven, e já lê exatamente os campos de
   config do 2.5 (`connector_num_layers`, `connector_attention_head_dim`,
   `connector_apply_gated_attention`). Batia com os tensores do checkpoint
   tensor por tensor. O mesmo vale pro "aggregator" que funde os 49 hidden
   states do Gemma em uma única projeção (`FeatureExtractorV2`, também já
   existente, também já genérico). **A auditoria de state-dict do embeddings
   processor 2.5 fechou em 0 missing / 0 unexpected** usando essas classes
   sem nenhuma modificação — só uma configuração nova
   (`Gemma25EmbeddingsProcessorConfigurator` em `scripts_2.5/dit_bridge.py`)
   dimensionada pro Gemma 4 (hidden_size 3840, 49 camadas) em vez do Gemma 3.

5. **Bug real de arquitetura no `LTXModel`**: `FeedForward` sempre criava
   `bias=True`, mas o checkpoint 2.5 tem `ff_bias: false` para o `ff` de vídeo
   dos blocos principais (**não** para `audio_ff`, que mantém bias sempre —
   nuance só percebida comparando os tensores reais do checkpoint, a
   suposição inicial de "flag global" estava errada). Sem o fix, 96 tensores
   (`transformer_blocks.N.ff.net.{0,2}.bias`) ficavam sem inicializar
   (`meta` device), o que quebraria qualquer forward pass sem erro claro no
   load — só explodiria depois, dentro do forward. Corrigido de forma aditiva
   em `feed_forward.py`/`gelu_approx.py`/`transformer.py`/`model.py`/
   `model_configurator.py` (novo parâmetro `ff_bias: bool = True`, default
   preserva o comportamento do 2.3 exatamente). Depois do fix: **auditoria do
   transformer fechou em 0 missing**; os "unexpected" restantes (259) são só
   os connectors (contados à parte, 0/0) e `keyframes_abs_pos_embedding`
   (gap conhecido, não crítico pra T2V simples).

6. **Validação de ponta a ponta real**: `scripts_2.5/dit_forward_test.py` —
   prompt de texto → `Gemma25TextEncoder` → embeddings processor → transformer
   de 40 GB (offloaded GPU+CPU via `accelerate`, sem precisar de quantização
   fp8) → um passo de denoising real. `FORWARD_TEST_OK`, sem NaN/Inf. Usa os
   helpers de produção do `ltx_pipelines` (`PipelineComponents`,
   `noise_video_state`, `modality_from_latent_state`) sem modificação nenhuma
   — evitou ter que reinventar a convenção de grid de posição/patchify.

### O que ainda falta pra gerar vídeo de verdade

> **TUDO DESTA LISTA FOI RESOLVIDO.** Scheduler em §3.2, VAE decode em §3.4
> (via ComfyUI atualizado, não por engenharia reversa), pipeline completo em
> §3.5, CLI/UIs em §3.6. Mantida como registro do que parecia difícil na
> véspera. O único item que sobrou é o `keyframes_abs_pos_embedding`, e ele
> se mostrou irrelevante para tudo que se usa hoje — ver item 4 de §7.

Não é mais risco de arquitetura desconhecida — é plumbing que já existe pro
2.3 e precisa ser conectado:

1. Scheduler multi-passo (`RectifiedFlowScheduler`/`LinearQuadratic` — a
   config também vem embutida no checkpoint, em `scheduler.*`).
2. Decode do VAE 2.5 (vídeo e áudio) — nunca testado ainda.
3. `keyframes_abs_pos_embedding` — provavelmente só importa para multishot
   nativo; T2V/I2V simples deve funcionar sem, mas isso é suposição, não
   verificado.
4. Um script de entrada real (equivalente a `ltx_pipelines/distilled.py`).
   `scripts_2.5/` hoje são blocos de validação, não uma CLI de geração.

---

## 4. Como reproduzir a validação

```bash
# encoder isolado
E:/Users/home/Documents/LTX-2-OPTIMIZED/.venv/Scripts/python.exe scripts_2.5/gemma4_text_encoder.py

# auditoria de state-dict (transformer + embeddings processor vs checkpoints reais)
E:/Users/home/Documents/LTX-2-OPTIMIZED/.venv-2.5/Scripts/python.exe scripts_2.5/dit_bridge.py

# ponta a ponta: encoder -> connector -> transformer -> um passo de denoising
CUDA_VISIBLE_DEVICES=1 E:/Users/home/Documents/LTX-2-OPTIMIZED/.venv/Scripts/python.exe scripts_2.5/dit_forward_test.py
```

Nota: `dit_bridge.py`/`dit_forward_test.py` importam o pacote `comfy` (via
`gemma4_text_encoder.py`), que só está instalado no `.venv` **principal**
(torchvision etc.). O `.venv-2.5` isolado serve pra testar dependências novas
(`transformers`/`torch` mais recentes), não é onde os scripts de produção
rodam — só usei ele pra `dit_bridge.py` porque naquele momento eu ainda não
tinha confirmado que o `comfy` bastava; hoje os três scripts rodam do `.venv`
principal sem problema (confirmado no forward test).

---

## 5. Mapa do código novo

*(atualizado 2026-08-25)*

### Motor 2.5 — a rota que realmente gera

| Arquivo | Papel |
|---|---|
| `ltx25_backend.py` | **O motor.** Sobe/reaproveita o ComfyUI, converte o workflow oficial 2.5, injeta os parâmetros, submete e devolve o mp4. Todos os recursos do 2.5 moram aqui: variantes, keyframes, áudio condicionado, two-stage |
| `ltx_pipelines_25.py` | Shim de CLI com a mesma argv de `ltx_pipelines.distilled`/`music_to_video`. É o que deixa as UIs trocarem 2.3↔2.5 com diff de duas linhas |
| `comfy_workflow_tool.py` | Converte workflow de UI (com subgrafos) para formato API. **Fonte de dois bugs sérios** — ver §3.6 e §3.16 |

### UIs e launchers 2.5

`music_maker_ui_v2_25.py` (7903), `music_maker_ui_v3_25.py` (7904),
`web_ui_v4_25.py` (7960), `film_maker_ui_v4_25.py` (7961),
`storyplay25.py` (7911, storyboard com quadros intermediários),
`script_pipeline/` com `LTX_PIPELINE_MODULE=ltx_pipelines_25` (7910).
Cada um tem seu `start_*.bat`, todos com liberação de porta antes e depois.

### Pós-produção (serve 2.3 E 2.5)

| Arquivo | Papel |
|---|---|
| `video_doctor.py` | Diagnóstico temporal + reparo. CLI: `analyze` / `repair` / `doctor` |
| `video_doctor_ui.py` | Aba Gradio reutilizável; embutida nas 4 UIs de music maker e avulsa em `start_video_doctor.bat` (7912) |
| `tools/rife/` | RIFE ncnn-vulkan (interpolação aprendida) |
| `tools/realesrgan/` | Real-ESRGAN (upscale 2D final) |
| `upscale_video.ps1` | Chamado pelas UIs; modelo/escala escolhidos na UI desde §3.18 |

### Extração de roteiro

`script_pipeline/prose_to_screenplay.py`, `parse_screenplay.py`,
`render_scenes.py`. Motor de LLM padrão: Ollama (§3.13).

### Blocos de validação 2.5 (histórico, não produção)

`scripts_2.5/gemma4_text_encoder.py`, `dit_bridge.py`, `dit_forward_test.py`,
`generate_latent_test.py`, `vae_decode_official.py`.
`na_diffusion_decoder.py` é registro de tentativa fracassada (§3.3).
`.venv-2.5/` foi útil para isolar a instalação; **os scripts de produção rodam
do `.venv` principal**.

### Mudanças em código de terceiros

`packages/ltx-core`: `feed_forward.py`, `gelu_approx.py`, `transformer.py`,
`model.py`, `model_configurator.py` — parâmetro `ff_bias`, aditivo, não afeta 2.3.
`ComfyUI/comfy/text_encoders/gemma4.py`: removido `process_tokens` quebrado.
Fora isso o ComfyUI vendorizado está **intocado** — os bugs encontrados nele
(§3.14) foram contornados do lado de fora, de propósito, para não conflitar com
atualizações futuras.

---

## 6. Erros evitados / armadilhas encontradas

1. **Pesquisa web sozinha não decide arquitetura.** A primeira rodada de
   `WebFetch`/`WebSearch` sobre o LTX-2.5 trouxe nomes de arquivo e specs
   plausíveis demais — só virou confiável depois de cruzar com a API real do
   HuggingFace (`?blobs=true`, tamanhos exatos) e com os headers reais dos
   `.safetensors` baixados. Pra qualquer decisão de arquitetura, checar o
   tensor de verdade, não o texto sobre o tensor.
2. **Download duplicado trava em vez de falhar.** Dois processos baixando o
   mesmo arquivo pro mesmo destino via `huggingface_hub` ficaram presos em
   lock mútuo por quase 1h40 sem erro nenhum — só percebido comparando o
   tamanho do `.incomplete` em dois instantes. Sintoma: tamanho parado, sem
   exceção, sem timeout.
3. **`uv pip install torch` sem `--index-strategy unsafe-best-match` ignora
   o índice CUDA configurado no `pyproject.toml`** quando rodado fora de
   `uv sync` — a estratégia default "first index wins" pega a versão mais
   nova do PyPI genérico (CPU-only) antes de considerar o índice cu129.
4. **`nvidia-smi` e `torch.cuda` numeram as GPUs diferente nesta máquina**
   (já documentado no `CLAUDE.md`, reconfirmado aqui): `CUDA_VISIBLE_DEVICES=1`
   é sempre a 3090, independente do que `nvidia-smi` mostra como índice 0.
5. **"Sem referência" não significa "não existe no repo".** O
   `embeddings_connector.py` que eu achava que precisaria escrever do zero já
   estava lá, genérico, só não tinha sido usado com essa config ainda. Vale
   sempre grepar o repo local a fundo antes de assumir que uma peça precisa
   ser reimplementada.

### 6.1 A armadilha que mais custou: métrica plausível que mede a coisa errada

*(acrescentado 2026-08-25, depois de a mesma falha aparecer QUATRO vezes em
um único dia)*

| # | métrica usada | o que ela dizia | a verdade |
|---|---|---|---|
| 1 | correlação de **forma de onda** entre áudio gerado e trilha (§3.15) | "condicionamento não funciona" (+0,03) | Funcionava. O VAE de áudio é mel+vocoder e **não preserva fase** — forma de onda lê ~0 num casamento perfeito. Envelope dava +0,97 |
| 2 | **variância de Laplaciano** como "nitidez" (§3.18) | o upscaler de anime era o melhor (245 contra 149) | Era o pior. Ele pontuava alto **por achatar** gradiente em borda dura. Menos informação, mais contraste |
| 3 | **z-score contra o clipe inteiro** (§3.19) | 17 defeitos num clipe sadio | Nenhum. Oscilação real de 0,5/255 — invisível. Faltava piso absoluto |
| 4 | **redução do pico de diferença** como "reparo funcionou" (§3.19) | reparo melhorou 38% | Piorou. Ele melhorou a métrica **apagando o conteúdo** que causava a diferença |

Os quatro têm a mesma forma: a métrica correlaciona com o que interessa em
casos normais e **descorrelaciona exatamente no caso que se quer medir**.

Três defesas, nesta ordem:

1. **A métrica sobrevive ao pipeline?** Round-trip de VAE, vocoder, reencode e
   requantização destroem identidade de sinal sem destruir conteúdo. Antes de
   declarar que algo generativo não funciona, verificar se a medida atravessa
   o processamento.
2. **Rodar o controle cruzado.** Um número alto sozinho não prova nada. Em
   §3.15 um par deliberadamente TROCADO marcou +0,885; só a matriz cruzada
   (cada cena casando com a PRÓPRIA fatia e não com a da vizinha) separou.
3. **Olhar.** Nos casos 2 e 4 nenhuma estatística salvaria — foi preciso ver a
   imagem. Em §3.19 medir por caminho independente (`phaseCorrelate` +
   diferença direta de pixel) derrubou um diagnóstico meu que estava errado.

Corolário de produto: quando o detector não consegue ser confiável, **o
desenho certo é revisão humana no meio**, não um limiar mais esperto. Foi por
isso que o `video_doctor` gera tiras de revisão em vez de reparar sozinho.

### 6.2 Armadilhas de ambiente que se repetiram

- **Processo longo com código velho em memória.** Editar `ltx25_backend.py` não
  muda nada numa UI já aberta. Vale para módulos do repo, não só para pacotes.
- **`ensure_server()` reaproveita servidor no ar** — correto, mas isso põe
  teste e produção na MESMA fila e na mesma VRAM. Derrubei um run do usuário
  duas vezes assim. Antes de gerar para teste: conferir `/queue`, `nvidia-smi`
  E se há cena recente em `outputs/`. **GPU ociosa não significa ausência de
  run** — pode ser o intervalo entre cenas.
- **Locale do sistema afeta binários de terceiros.** O `rife-ncnn-vulkan` lê
  `-s 0.2` como 0 em pt-BR e recusa; `0,2` funciona (§3.20). O sinal estava à
  vista: o Real-ESRGAN imprimia "75,00%".
- **Injeção automatizada de código pode virar código morto que COMPILA.** A
  aba do doctor caiu depois de um `return` e a UI subia sem ela (§3.20).
  Verificar que o efeito existe, não que o arquivo compila.

---

## 3.1 Continuação de 2026-08-21 (mesma sessão): scheduler pronto, VAE é a parede

Segui pela lista de §7 (versão anterior). Achados:

- **Scheduler multi-passo: já existe, pronto.** `ltx_core/components/schedulers.py::LinearQuadraticScheduler`
  bate exatamente com `"sampler": "LinearQuadratic"` da config embutida no
  transformer 2.5. `EulerDiffusionStep` (stepper) e `euler_denoising_loop`
  (`ltx_pipelines/utils/samplers.py`) também são genéricos, sem nada
  específico do 2.3 — dá pra rodar N passos reais de denoising na latente sem
  escrever nada novo. O único hiperparâmetro sem valor oficial confirmado é o
  número de passos/`threshold_noise` ideal pro checkpoint distilled do 2.5
  (2.3 usa `DISTILLED_SIGMA_VALUES`, uma lista fixa afinada à mão — não temos
  o equivalente pro 2.5, teria que estimar ou testar).

- **VAE decode: é a peça mais dura encontrada até agora.** O decoder de
  vídeo 2.5 é uma arquitetura **nova**, `NADiffusionDecoder` (config com
  `stage_channels`, `stage_depths`, `det_stages` com atenção) — não é o
  `VideoDecoder` do 2.3 (`decoder_blocks` do tipo `compress_space`/`compress_time`).
  Auditoria de state-dict: praticamente nada bate com o `VideoDecoderConfigurator`
  atual. Diferente do encoder Gemma4 (que eu achava sem referência e estava
  pronto no ComfyUI) e do embeddings connector do DiT (que eu achava sem
  referência e já existia genérico no `ltx_core`) — **aqui não tem atalho em
  lugar nenhum**: grep no repo inteiro, incluindo `ComfyUI/`, não achou
  `NADiffusionDecoder`/`det_stages`. O VAE de áudio também usa um schema
  diferente (`audio_vae.model.params.ddconfig`, estilo VQGAN de
  mel-spectrogram) do `AudioDecoder` atual. Isso muda a ordem de prioridade
  de §7 abaixo — decode virou o item de maior risco/esforço, não mais
  "só testar".

## 3.2 Continuação: latente completa gerada (8 passos reais)

`scripts_2.5/generate_latent_test.py`: loop de denoising real (não mais um
passo) rodou de ponta a ponta — `LinearQuadraticScheduler(steps=8)` +
`EulerDiffusionStep` + `euler_denoising_loop`, tudo reaproveitado sem
modificação. **`LATENT_GENERATION_OK`**: latente final `(1, 320, 128)`, sem
NaN/Inf, média ≈0 / desvio ≈0,9 (saudável pra uma latente de difusão
normalizada). 320×512, 9 frames, 103s pros 8 passos (offload pesado).

Achado colateral importante: os sigmas gerados por
`LinearQuadraticScheduler().execute(steps=8)` batem **exatamente** com
`DISTILLED_SIGMA_VALUES` do 2.3 (`[1.0, 0.99375, 0.9875, 0.98125, 0.975,
0.909375, 0.725, 0.421875, 0.0]`) — não é coincidência, confirma que aquela
constante hardcoded do 2.3 É a saída dessa fórmula genérica. Isso remove a
incerteza que havia sobre "step count certo pro 2.5 distilled": 8 passos com
os defaults do `LinearQuadraticScheduler` é a escolha correta, não um chute.

Latente salva em `scripts_2.5/last_latent.pt` (não decodificável em pixels
ainda — falta o `NADiffusionDecoder`, ver §3.1).

## 3.3 Tentativa do NADiffusionDecoder (2026-08-22) — mecânica certa, semântica errada

A pedido do usuário ("tente o nad"), implementei `scripts_2.5/na_diffusion_decoder.py`
por engenharia reversa pura dos tensores/shapes/config do checkpoint (ver
docstring do arquivo pra reconstrução completa do raciocínio). Resultado:

**O que bateu (verificável, sem ambiguidade):**
- Auditoria de state-dict: **0 missing, 0 unexpected** contra os 396 tensores
  reais do checkpoint (`decoder.*` + `per_channel_statistics.*`) — a
  decomposição em `det_stages` (4 grupos de blocos ViT simples, sem AdaLN) +
  `upsamples` (upsample "depth-to-space" via Linear, `resampler_kind: "linear"`
  bate com a config) + `diff_blocks` (8 blocos com AdaLN + atenção conjunta) +
  embedders está estruturalmente certa.
- Forward roda sem crash, shape de saída coerente `(1, 3, 16, 320, 512)`
  (16 frames em vez dos 9 pedidos — diferença esperada, não implementei a
  correção causal "-1/+1" que o `VideoDecoder` do 2.3 usa), sem NaN/Inf.

**O que não bateu (achado por evidência direta, não suposição):**
- A saída final não muda **nada** — nem um bit — se a des-normalização por
  canal (`per_channel_statistics.un_normalize`) é aplicada ou não no latente
  de entrada. Isso só é possível se o `diff_blocks` (o estágio que de fato
  refina o conteúdo condicionado no latente) estiver produzindo uma
  contribuição próxima de zero — ou seja, meu palpite pra ordem dos 7 valores
  do `scale_shift_table`/`shared_adaln` (chute: `[shift1,scale1,gate1,
  gate_ctx,shift2,scale2,gate2]`) provavelmente está errado de um jeito que
  zera os *gates*.
- **Confirmado visualmente**: o frame decodificado é um padrão repetitivo
  raso, sem nenhuma relação com o prompt — não é vídeo, é ruído estruturado
  (provavelmente dominado pelo rearranjo do upsample "depth-to-space", cuja
  ordem dos eixos também é só um palpite fundamentado, não verificado).

**Conclusão**: sem uma implementação de referência, a parte mecânica dá pra
acertar por aritmética de shape (e deu), mas as fórmulas de combinação
(ordem dos slots de AdaLN, eixo do rearrange, ponto de combinação
timestep+contexto) têm espaço grande demais de variações plausíveis pra
acertar por tentativa isolada. Mais uma rodada de ajuste às cegas tem
retorno decrescente — cada parâmetro errado pode mascarar o efeito de outro
já corrigido (como aconteceu aqui: a des-normalização está certa, mas o
sintoma dela ficou invisível por causa do bug nos gates).

## 3.4 Resolvido via ComfyUI atualizado (2026-08-22) — pipeline completo funcionando

Em vez de continuar ajustando a implementação reversa às cegas, atualizei o
ComfyUI vendorizado (usuário pediu "como roda no comfyui?" → "sim" para
atualizar). Isso resolveu o bloqueio do decode de vez:

- **`ComfyUI/custom_nodes/ComfyUI-LTXVideo`**: estava em `aceeae9` (30/06),
  atualizado via `git merge origin/master --ff-only` para `15d09ab` (20/08) —
  trouxe `example_workflows/2.5/` (10 workflows oficiais prontos, incluindo
  T2V/I2V) e node code atualizado (`embeddings_connector.py`, `gemma_encoder.py`).
  Modificação local (`pyramid_blending.py`, fix `kornia.pad` → `F.pad`) foi
  stashed antes e reaplicada depois sem conflito.
- **`ComfyUI/` (core, `comfyanonymous/ComfyUI`)**: estava em `2e47082`
  (22/07), atualizado para `783545f` (22/08) — **aqui que apareceu a peça que
  faltava**: `comfy/ldm/lightricks/vae/na_diffusion_decoder.py`, arquivo novo
  com a implementação real do `NADiffusionDecoder`. `comfy/sd.py` também
  ganhou a detecção certa (`elif "decoder.conv_in_x_t.weight" in sd`).
  Modificação local (`comfy/text_encoders/gemma4.py`, o fix do
  `process_tokens` de 2026-08-21) **não precisou ser reaplicada** — o upstream
  já resolveu o mesmo bug de forma diferente (arquivo reescrito).
- Precisou atualizar `comfy-kitchen` de 0.2.22 para 0.2.31 (`pip install -r
  ComfyUI/requirements.txt`) — o core novo usa símbolos que a versão antiga
  do pacote não tinha (`AsymW4A8Int8Layout`, `int8_attention_is_available`).
- `ComfyUI/extra_model_paths.yaml`: adicionada seção `ltx_25` apontando pros
  subdiretórios já baixados em `models/2.5/` (mesma convenção de nomes que o
  ComfyUI já usa), sem precisar re-baixar os 75 GB.

**Diferenças entre a implementação real e o meu palpite (`na_diffusion_decoder.py`,
§3.3) — pra quem for comparar depois:**
- Atenção é **neighborhood attention** (janela por estágio, ex. `(3,7,7)`) com
  **RoPE absoluto por eixo** — eu tinha usado atenção global sem nenhuma
  codificação posicional. Essa é provavelmente a causa raiz do ruído.
- O contexto (saída dos `det_stages`) é **somado** a `x` antes da atenção do
  `diff_blocks`, não concatenado como tokens extras (sem atenção conjunta).
- Os *gates* do AdaLN de 7 posições **não são usados** — só scale/shift dos
  dois primeiros pares importam; os 3 gates são vestigiais ("folded at
  export"). Eu multiplicava cada residual por um gate — errado.
- `x_t` (a estimativa inicial ruidosa) é **ruído gaussiano de verdade** (seed
  fixa = 0), não zeros.
- O segundo número de cada entrada de `upsamples` (`[[1,2,2], 2]`) é um
  **divisor de canais de saída** (`out_channels_reduction_factor`), não um
  contador de repetição.

**`scripts_2.5/vae_decode_official.py`**: carrega `CausalDiffusionVAE` real
direto do `comfy` (0 missing, 1 unexpected — `decoder.type_emb`, tensor do
checkpoint que a classe real nem usa), decodifica a MESMA latente salva em
`generate_latent_test.py`. Resultado: **`VAE_DECODE_OFFICIAL_OK`**, shape
`(1, 3, 9, 320, 512)` — 9 frames exatos (bateu com o pedido, ao contrário da
minha versão que dava 16), sem NaN/Inf, range de pixel saudável (-0.93 a 1.08).
**Confirmado visualmente**: paisagem nevada com pôr do sol, condizente com o
prompt ("red fox running through fresh snow at sunset, warm rim light") — não
é ruído, é uma imagem real e coerente.

**Conclusão prática**: o pipeline nativo (`scripts_2.5/gemma4_text_encoder.py`
→ `dit_bridge.py` → `generate_latent_test.py`) combinado com o decoder oficial
do ComfyUI atualizado gera vídeo real do LTX-2.5. `na_diffusion_decoder.py`
(a versão própria) fica mantida no repo só como registro do processo de
engenharia reversa — não usar em produção, usar `vae_decode_official.py`.

## 3.5 Screenplay real via ComfyUI (2026-08-22) — primeira geração AV completa

Usuário pediu teste com prompts de screenplay reais (formato do guia oficial
LTX-2.3: parágrafo contínuo, falas entre aspas, áudio ambiental explícito,
diálogo em PT-BR com lip sync). Rota escolhida: workflow oficial
`LTX-2.5_T2V_I2V_Single_Stage_Distilled.json` via ComfyUI (não o pipeline
nativo — esses workflows fazem T2V **com áudio/diálogo**, que o pipeline
nativo nunca testou).

### `comfy_workflow_tool.py` ganhou suporte a subgraphs

Os workflows oficiais do 2.5 usam a feature de **subgraph** do ComfyUI
(nós colapsáveis que escondem um grafo interno — `definitions.subgraphs` no
JSON, com nós sentinela `-10`/`-20` marcando as bordas de entrada/saída). O
conversor antigo (`convert()`) não sabia disso e reduzia o workflow de 46 nós
pra 1 ao podar. Reescrito pra expandir subgraphs recursivamente, com
memoização de redirecionamento de saída (`resolved`) e casamento por **nome**
(não índice posicional) entre a porta exposta na instância e a entrada
declarada do subgraph — os dois nem sempre têm a mesma contagem (portas nunca
conectadas ficam de fora da instância). Também trata `Reroute` como
passthrough transparente (sem schema no backend desta versão do ComfyUI).
Resultado: 45 nós corretos, zero referência quebrada.

### Erros de validação/execução corrigidos, em ordem

1. `clip_name` do enhancer (`gemma4_e2b_it_bf16.safetensors`) não baixado —
   não usado de fato (prompt enhancement desligado), mas a validação estática
   do ComfyUI exige um valor válido mesmo assim. Trocado por um checkpoint
   que já temos.
2. `scale_method: 'scale longer dimension'` não existe nas opções do
   `ResizeImageMaskNode` desta versão — trocado por `'lanczos'`.
3. `LoadImage` com `image=''` tenta abrir a PASTA `ComfyUI/input` como
   arquivo de vídeo (erro de permissão) — mesmo sem I2V, o nó roda porque o
   grafo é estático (switches escolhem o valor, não podam a execução). Fix:
   apontar pra uma imagem placeholder qualquer já existente.
4. `ResizeImageMaskNode.resize_type` é um **DYNAMICCOMBO_V3** (tipo de input
   novo do ComfyUI, uma combo cujo valor selecionado expõe campos aninhados
   diferentes). Formato de API não documentado — descoberto por tentativa:
   `resize_type: "<key>"` mais os campos do sub-schema **com prefixo
   `"resize_type."`** (ex.: `"resize_type.width"`), não aninhados nem soltos.
5. **A causa raiz real**: depois de tudo isso, `VAELoader` ainda carregava a
   arquitetura errada (`AutoencodingEngine`, não `CausalDiffusionVAE`) e
   dava erro de shape. Não era bug de código — o **servidor ComfyUI estava
   rodando desde antes da atualização do core** (processo Python já tinha o
   `comfy/sd.py` antigo importado em memória; editar o arquivo em disco não
   afeta um processo já rodando). Reiniciar o servidor resolveu na hora.

### Resultado

Prompt #2 do usuário (Velho Oeste, Clara vs. Cole, 15s/361 frames, 768×512,
diálogo em PT-BR + áudio ambiente) — **completo em 377s**. Vídeo real com
faixa de vídeo (h264) e áudio (aac), duração exata (15.04s = 361/24).
Confirmado visualmente: os dois personagens batem com a descrição (chapéu
bege/lenço vermelho vs. chapéu preto/barba, rua vazia, luz dourada). Um
artefato notado: texto ilegível tipo legenda aparece em alguns frames, apesar
de "no subtitles" estar no negative prompt — não investigado a fundo (pode
ser força do negative prompt, pode ser característica do modelo).
Lip-sync/qualidade do diálogo em si não verificados por mim (sem forma de
ouvir) — avaliação de áudio/fala fica para o usuário.

### Correção: os bugs do lote seguinte NÃO eram cache (2026-08-22, mesmo dia)

Ao rodar os outros 5 prompts em lote, batendo a MESMA config que funcionou
pro Velho Oeste, todos falharam com `Sizes of tensors must match... Expected
size 1 but got size 25`. Diagnóstico inicial (errado): cache de execução do
ComfyUI ficando poluído entre submissões — reiniciei o servidor com
`--cache-none` (zero cache), rodou 49 minutos de computação genuinamente
fresca e **caiu no mesmo erro**, provando que a causa nunca foi cache.

**Causa raiz real, achada por diff direto** entre o JSON que funcionou
(Velho Oeste) e o gerado de novo pelo conversor: `LTXVEmptyLatentAudio.batch_size`
saía `25` em vez de `1`. O motivo: `widget_input_names()` (em
`comfy_workflow_tool.py`) só reconhecia tipos primitivos exatos
(`"INT"`/`"FLOAT"`/`"STRING"`/`"BOOLEAN"`), mas o schema atual do ComfyUI
declara `frame_rate` como `"FLOAT,INT"` (união). Isso fazia `frame_rate` ser
excluído da lista de nomes, desalinhando o zip posicional contra
`widgets_values` — `batch_size` acabava pegando o valor que era de
`frame_rate` (`25`, o default de fps daquele node). Corrigido: checar se
QUALQUER parte separada por vírgula do tipo está no conjunto de primitivos.

Segundo bug, mesma família, achado logo depois (o primeiro mascarava este):
`SaveVideo.format`/`.codec` viraram `COMFY_DYNAMICCOMBO_V3` na atualização do
core do ComfyUI puxada no meio da sessão — a conversão do Velho Oeste tinha
sido feita ANTES dessa atualização, quando o schema ainda era `STRING`/`COMBO`
simples, por isso funcionou sem ajuste. Resolvido com o mesmo padrão de chave
com ponto que já tínhamos usado pro `resize_type` (`format="auto"` +
`format.codec="auto"`).

**Lição**: quando o mesmo workflow, com os mesmos valores, funciona uma vez e
falha na repetição, suspeitar de **schema do node mudando entre a conversão
original e a nova** (o core do ComfyUI é atualizado no meio da sessão) antes
de suspeitar de cache/estado do servidor — é mais fácil de confirmar (diff
direto dos JSONs) e, neste caso, foi a causa de verdade duas vezes seguidas.
`--cache-none` ficou nos launchers mesmo assim (não custa, e protege contra
cache genuinamente stale em uso interativo longo), mas não foi o que resolveu
aqui.

**Resultado final do lote**: com os dois bugs corrigidos, os 6 prompts do
screenplay (Velho Oeste, Fantasia, Sci-fi, Anime, Alice, Conto de fadas)
rodaram **6/6 com sucesso**, 351-507s cada, todos com vídeo+áudio reais e
duração exata. `_run_screenplay_batch.py`/`_screenplay_prompts.py` ficam como
ferramenta reutilizável pra rodar screenplays LTX-2.3-style no 2.5 via
ComfyUI.

## 3.6 UIs 2.5 e launchers (2026-08-22)

Pedido: `.bat` equivalentes (v2, v3, webui, screenplay…) para 2.5. Um `.bat`
sozinho **não** resolveria: as 6 UIs têm 2.3 hardcoded (`ltx-2.3-22b-distilled-fp8`,
`gemma3`, upscaler 2.3) e 5 delas chamam `ltx_pipelines.distilled` /
`.music_to_video`, que não suporta 2.5. Um launcher com variáveis de 2.5
apontando para elas rodaria 2.3 em silêncio, com nome de 2.5.

### Arquitetura escolhida: shim CLI, não reescrita

Todas as UIs geram via **subprocesso com lista de argv** e streamam o stdout
para o log ao vivo. Então `ltx_pipelines_25.py` implementa exatamente esse
contrato de argumentos e roteia para `ltx25_backend.py` (rota ComfyUI 2.5).
Consequência: cada UI 2.5 difere da 2.3 por ~4 linhas, em vez de um fork de
700-1900 linhas de lógica de cena/áudio/lipsync que nada tem a ver com a
versão do modelo.

- **`ltx25_backend.py`** — mantém o ComfyUI vivo, patcha o workflow oficial
  2.5, submete pela API, devolve o mp4. Papel análogo ao `gguf_backend.py` do
  caminho 2.3/GGUF. Validado: 25 frames exatos a 512×320 em 197s.
- **`ltx_pipelines_25.py`** — shim drop-in. Aceita e **ignora com aviso** o que
  não se aplica à rota 2.5 (`--quantization`, `--lora`, `--torch-compile`,
  `--teacache-threshold`, `--num-inference-steps`, `--audio-input-path`,
  `--enhance-prompt`), corrige `num_frames` para 1+múltiplo de 8, e usa só a
  primeira imagem de condicionamento. Validado com o argv exato que as UIs
  passam: exit 0, 25 frames, 512×320.
- **`_make_25_uis.py`** — gera as 4 UIs por substituição **verificada** (cada
  âncora precisa casar exatamente 1x, senão falha em vez de escrever um arquivo
  que parece portado e roda 2.3).

### Arquivos novos

| Arquivo | Porta | Origem |
|---|---|---|
| `web_ui_v4_25.py` + `start_webui_25.bat` | 7960 | `web_ui_v4.py` |
| `film_maker_ui_v4_25.py` + `start_cinema_25.bat` | 7961 | `film_maker_ui_v4.py` |
| `music_maker_ui_v2_25.py` + `start_music_video_v2_25.bat` | 7903 | `music_maker_ui_v2.py` |
| `music_maker_ui_v3_25.py` + `start_music_video_v3_25.bat` | 7904 | `music_maker_ui_v3.py` |
| `start_screenplay_25.bat` | 7910 | reusa `screenplay_ui.py` |

Portas na faixa 79xx para as 2.5 rodarem lado a lado com as 2.3.

### Screenplay: chave aditiva em vez de fork

A cadeia é `screenplay_ui.py` → `screenplay_to_video.py` → `script_pipeline/render_scenes.py`,
e só a última spawna o renderer. Duplicar as três seria ~1900 linhas. Em vez
disso, `render_scenes.py` ganhou
`LTX_PIPELINE_MODULE = os.environ.get("LTX_PIPELINE_MODULE", "ltx_pipelines.music_to_video")`
— aditivo, default idêntico ao comportamento anterior (verificado: sem env var
resolve para o módulo 2.3; com env var, para o shim). `start_screenplay_25.bat`
só seta a variável e a porta. Nenhum arquivo Python novo para o screenplay.
Nota: só o estágio **render** vira 2.5; storyboard (Flux), TTS, lip-sync e mix
seguem com seus próprios modelos.

### Limites conhecidos desta rota (herdados do grafo single-stage 2.5)

Sem two-stage (upscale 2x + refine), sem LoRA, sem áudio condicionado pela
trilha (o áudio é T2A do próprio grafo), e progresso reportado como tempo
decorrido — a API HTTP do ComfyUI só expõe o job pronto via `/history`;
progresso por passo exigiria o canal websocket, não ligado aqui. As UIs
herdadas ainda exibem esses controles; o shim avisa no log que foram ignorados.

## 3.7 Legendas queimadas: causa confirmada, sem solução ainda

> **SUPERADA em §3.11 (2026-08-23).** A conclusão desta seção — "não há como
> tirar as legendas no distilled, porque CFG 1 anula o negative prompt" —
> estava certa sobre o CFG e **errada sobre a causa das legendas**. Elas são
> artefato de **duração curta**: medido, 17s dá legenda em 11 de 12 frames,
> 30s e 45s dão zero. Todas as tentativas registradas abaixo (negative prompt,
> NAG, dev com CFG real) foram testadas a 15s — atacando a variável errada.
> Leia esta seção pelo raciocínio sobre CFG, não pela conclusão.

O artefato de legenda notado em §3.5 foi investigado a fundo.

**Causa confirmada**: o grafo distilled amostra em **CFG = 1**. Em CFG 1 a
fórmula do classifier-free guidance vira `uncond + 1·(cond − uncond) = cond` —
o ramo negativo cancela **exatamente**. Ou seja o negative prompt era um no-op
matemático o tempo todo, escrevesse o que escrevesse. Confirmado no nosso
próprio grafo (`CFGGuider.cfg == 1`) e corroborado pela resposta da Lightricks
nas discussões do HF do LTX-2 (recomendam NAG para o modelo distilled).

**O que foi testado** (A/B controlado, mesma seed 42, mesmo prompt, único
fator variando):

| Braço | Legendas |
|---|---|
| baseline (negative prompt comum) | 8/12 frames |
| NAG `nag_scale=11` (default) | 8/12 frames |
| NAG `nag_scale=11` + negativo focado só em texto (15 termos) | persiste |
| NAG `nag_scale=35` + negativo focado | 8/12 frames |

NAG **está ativo** — a composição muda visivelmente entre baseline e NAG com a
mesma seed — mas não elimina as legendas em nenhuma escala testada.

**Hipótese que sobra, não testada**: as legendas transcrevem literalmente o
diálogo do prompt ("Largue o revólver...", "Não vim pel cidde. Vim ouro do
trem."). O gatilho provável é o **diálogo entre aspas no prompt positivo** — o
modelo aprendeu de vídeo web onde fala transcrita anda junto de legenda
queimada. Teste decisivo seria descrever a fala sem aspas e ver se some; o
custo é possivelmente perder lip-sync, que é justamente o que o formato de
prompt do guia LTX busca. Alternativa independente do modelo: recortar/inpaint
a faixa inferior em pós-produção.

## 3.8 Variante dev (não destilada) e escolha de versão (2026-08-22)

Pergunta do usuário: "e se usar a versão não destilada?" — e ela é a resposta
certa para §3.7. O problema das legendas não é o negative prompt: é o CFG 1 do
distilled. O **dev** roda CFG real, então o negative prompt volta a existir.

### Onde estava a receita

Nenhum dos 10 workflows oficiais 2.5 usa dev (todos são "Distilled"). Mas o
workflow **2.3** `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json` tem os dois
ramos, e o ramo dev dá a configuração exata (nós 4966/4808/4963/4964/4802):

| | distilled | dev |
|---|---|---|
| sigmas | `ManualSigmas` (8 fixos) | `LTXVScheduler` steps=15, max_shift 2.05, base_shift 0.95, stretch, terminal 0.1 |
| guider | `CFGGuider` cfg=1 | `MultimodalGuider` + `GuiderParameters` VIDEO cfg=3 / AUDIO cfg=7, skip_blocks=28 |
| negative prompt | inerte | **ativo** |

Os widgets posicionais foram mapeados para nomes de input pelo `/object_info` de
cada nó (mesma armadilha de §3.5: widget list ≠ ordem do schema). Nós
necessários já instalados, exceto `ClownSampler_Beta` (RES4LYF, terceiro) — não
é bloqueante: quem faz o negative prompt valer é o *guider*, não o sampler, então
o `KSamplerSelect` existente serve.

### Implementação

`ltx25_backend.VARIANTS` = {distilled, dev}. `_apply_dev_variant()` remove
`CFGGuider`/`ManualSigmas` e insere a cadeia dev, religando o
`SamplerCustomAdvanced`. Validado estruturalmente antes do checkpoint chegar:
4 nós adicionados, 2 removidos, sampler religado, zero referência pendente.

Seleção em três níveis, todos convergindo no mesmo backend:
- **CLI**: `--variant dev [--steps N --video-cfg X --audio-cfg Y]`
- **Env var**: `LTX25_VARIANT=dev` (é o que as UIs usam, já que elas montam um
  argv fixo e não conhecem a flag)
- **Launchers**: bloco comentado no topo dos 5 `.bat` `_25`, uma linha para editar

Guarda: se o checkpoint da variante não existir, erro explícito dizendo qual
arquivo baixar — em vez de o ComfyUI falhar lá adiante com erro de shape.

Default = `distilled` (preserva o comportamento validado em §3.5/§3.6).

## 3.9 Dev testado: CFG real NÃO resolve as legendas (2026-08-22)

Teste decisivo com o checkpoint dev baixado (42GB, 40min): mesmo prompt do
Velho Oeste, mesma seed 42, mesmo negative prompt — única variável, a variante.

| Variante | CFG | Negative prompt | Legendas |
|---|---|---|---|
| distilled | 1 | inerte | 8/12 frames |
| distilled + NAG 11/35 | 1 | via atenção | 8/12 frames |
| **dev** | **3 vídeo / 7 áudio** | **ativo** | **9/12 frames** |

O dev funciona (22min de geração, estilo visual claramente distinto, mais
filmico) e o negative prompt está ativo de verdade nele — mas as legendas
persistem, agora até em formato de roteiro (com travessão). **CFG real não é a
solução.**

Isso deixa a hipótese do prompt positivo como a explicação mais provável: as
legendas transcrevem literalmente as falas entre aspas do prompt. O modelo
parece ter aprendido a associação "fala transcrita ⇒ legenda queimada" forte o
bastante para sobreviver a guidance negativa de CFG 3 contra "subtitles,
captions, burned-in text...". Teste em andamento: mesma cena com as falas
descritas indiretamente (sem aspas).

Nota de valor colateral: mesmo sem resolver legendas, a variante dev agora está
integrada e disponível — vale por qualidade/aderência, não como cura do
artefato.

### ~~PENDENTE~~: teste sem aspas — não vale mais retomar

> **PERDEU O SENTIDO com §3.11.** A hipótese era que as legendas viessem do
> diálogo entre aspas no prompt. A causa real é **duração curta**, e este
> teste rodaria a 15s, dentro da faixa em que a legenda aparece de qualquer
> jeito — ele não distinguiria nada. Registro mantido abaixo só para quem
> encontrar o `_test_noquotes.py` na raiz e se perguntar o que era.

`_test_noquotes.py` (mesma cena, falas descritas indiretamente em vez de entre
aspas, distilled, seed 42) foi disparado mas **morreu no meio** com
`ConnectionResetError` — o servidor ComfyUI caiu durante a geração. Não há
vídeo nem resultado; a hipótese continua **não testada**.

Para retomar:
```bash
.venv/Scripts/python.exe -u _test_noquotes.py distilled
```
e comparar o contact sheet com o baseline de 8/12 frames. Se as legendas
sumirem, a causa é o diálogo entre aspas no prompt positivo — e aí o
trade-off vira "falas literais em PT-BR" contra "sem legendas", o que torna a
rota de pós-produção (recorte/inpaint da faixa inferior) mais atraente, já que
preserva as duas coisas.

## 3.10 Reestruturação automática prosa → roteiro (2026-08-23)

Sintoma: o estágio `parse` falhou com um prompt LTX colado como entrada
("Nenhuma cena detectada -- confira se o roteiro tem cabecalhos INT./EXT.").
Não era bug: o parser é determinístico e espera formato de roteiro; a entrada
era um parágrafo corrido em inglês com falas entre aspas.

`script_pipeline/prose_to_screenplay.py` (novo) converte prosa/prompt-LTX em
formato de roteiro, e `parse_screenplay.main()` o aciona **apenas** quando a
passagem determinística não encontrou nenhuma cena.

O ponto delicado é que o valor do parser é justamente **não inventar estrutura**
("never hallucinates a scene or a line of dialogue that isn't in the source").
Um LLM escrevendo o roteiro quebraria essa garantia. As proteções:

- Só roda quando a via determinística achou **zero** cenas; nunca substitui.
- O roteiro gerado é gravado em `parse/screenplay_auto.txt` para leitura/edição
  — nada acontece invisivelmente.
- **Verificação verbatim**: `extract_quoted_dialogue()` (regex, sem LLM) lista
  toda fala entre aspas na origem e confere se cada uma sobreviveu literalmente
  à conversão. Falas alteradas são listadas em alerta explícito. Isso importa
  porque diálogo não é decoração aqui: vai para TTS e depois lip-sync — uma
  fala parafraseada seria *falada errada* em silêncio, pior que falha de
  formato.
- Depois da conversão, quem extrai a estrutura continua sendo o parser regex.
- `--no-auto-structure` desliga e restaura o comportamento anterior.

Suporta os dois motores do projeto (`gemma4` E2B em venv isolado, padrão;
`gemma3` 12B in-process), reusando o padrão de worker em subprocesso de
`enrich_with_llm_gemma4`.

### Três defeitos achados rodando de verdade (e o padrão que eles revelam)

A primeira conversão real expôs, em sequência, o mesmo tipo de problema: **o LLM
acerta o conteúdo e erra o formato, de maneira variável entre execuções**. Pedir
formatação melhor no prompt não resolve isso; impor os invariantes
deterministicamente depois, sim. Os três:

1. **Falso alarme no verificador.** O texto de origem tinha um typo
   ("presença.Desde"); o modelo emitiu "presença. Desde" e o check reportou a
   fala inteira como perdida. Um verificador que grita por correção cosmética
   é ignorado — vira ruído. `_normalize()` passou a normalizar espaço após
   pontuação.
2. **Fala silenciosamente descartada.** O modelo escreveu o cue como `Lyra` em
   vez de `LYRA`; a `CUE_RE` exige maiúsculas, então aquela fala saiu do parse
   (3 de 4). `enforce_screenplay_format()` promove a maiúsculas apenas nomes
   que já aparecem como cue válido em outro ponto do documento — conservador,
   não confunde início de frase com personagem.
3. **Diálogo enterrado em prosa.** Em outra execução, 2 das 4 falas vieram como
   `Lyra olha para cima ... e diz com a respiração curta, "A lua está...` — as
   palavras estão lá (passa em busca textual) mas o parser lê como AÇÃO, então a
   fala **nunca chega ao TTS**. Isso motivou duas mudanças:
   `_promote_embedded_dialogue()` reconstrói cue+fala dessas linhas, e
   `check_dialogue_preserved()` deixou de fazer busca textual: agora roda o
   parser de verdade e só conta como preservada a fala que saiu **como diálogo
   parseado**. É a diferença entre "as palavras existem" e "a fala será falada".

Validado de ponta a ponta na entrada real: parse deixou de falhar, 1 cena,
**4/4 falas** (incluindo a estendida de ~20s), decupagem com 6 planos. A mesma
saída de LLM que antes rendia 2 falas agora rende 4 — a correção determinística
absorve a variabilidade do modelo entre execuções.

## 3.11 Tomada contínua: 45s funciona (2026-08-23)

Pergunta: fala longa / vídeo maior que 15-30s **sem cortes** é possível? O teto
de 15s do slider `max_clip_seconds` no screenplay UI vinha com a nota "clipes
longos travam o upsampler" — limite herdado do caminho nativo 2.3, nunca
verificado na rota ComfyUI 2.5.

Medido (`_test_longtake.py`, prompt de fantasia com a fala estendida do usuário,
distilled, 768×512, seed 42):

| Duração | Frames | Tempo | Resultado |
|---|---|---|---|
| 30s | 721 | ~19 min | OK, duração exata, com áudio |
| 45s | 1081 | ~20 min | OK, duração exata, com áudio |

**Ambos geraram numa passagem só, sem corte.** Identidade dos dois personagens,
cenário, sombra do dragão e runas permanecem estáveis do primeiro ao último
frame. O movimento de câmera pedido no prompt ("medium two-shot, slowly pushes
closer while circling") executa ao longo de toda a duração.

Ou seja: **o teto de 15s do screenplay é conservador, não é limite do modelo**.
Para fala longa numa tomada só, o caminho é gerar direto pelo backend (ou subir
esse teto), não fragmentar em clipes.

Ressalva de enquadramento: o push-in composto ao longo de 45s termina bem mais
fechado do que terminaria em 17s — a mesma instrução de câmera "rende" mais
quanto maior a tomada, então vale calibrar a linguagem de câmera pela duração.

### RESOLVIDO: as legendas são um artefato de DURAÇÃO CURTA

Teste controlado, **texto idêntico**, mesma seed 42, distilled, 768×512 — só a
duração varia:

| Duração | Frames | Frames com legenda |
|---|---|---|
| 17s | 409 | **11/12** |
| 30s | 721 | **0/12** |
| 45s | 1081 | **0/16** |

**Duração é a causa.** O limiar está entre 17s e 30s.

Isso fecha a investigação de §3.7/§3.9 e explica por que tudo antes falhou:
negative prompt reforçado, NAG (4 configurações) e CFG real com o dev foram
todos testados **em 15s**, dentro do regime onde o artefato aparece. Nenhum
deles estava errado como técnica — estavam atacando a variável errada.

Solução prática: **gerar em 30s ou mais**. Não custa nada além de tempo de
geração, e é compatível com o que o usuário queria de qualquer forma (fala
longa em tomada contínua). Para clipes curtos que precisem ficar curtos, a
alternativa continua sendo recorte/inpaint da faixa inferior em pós.

Não investigado (e não necessário para a solução): por que a duração afeta
isso. Hipótese plausível, não testada, é regime de treino — clipes curtos com
fala transcrita são exatamente o material web onde legenda queimada é comum.

### Observação colateral (superada pela medição acima)

Comparando a MESMA cena, mesma seed, mesmo modelo, mesmo tamanho:

| Versão | Frames | Legendas |
|---|---|---|
| fantasia 17s (lote de ontem) | 409 | 11/12 frames |
| fantasia 45s (hoje) | 1081 | **0/16 frames** |

O negative prompt não explica (é inerte em CFG 1 nas duas). As variáveis que
diferem são duração e o texto do prompt (a versão de 45s inclui a fala
estendida). **Causa não isolada** — uma amostra de cada não separa duração de
redação. Mas é a primeira condição observada em que o artefato simplesmente não
apareceu, depois de negative prompt reforçado, NAG (4 configs) e CFG real terem
todos falhado. Vale um teste controlado: mesma cena em 17s e 45s com o texto
idêntico.

## 3.12 storyplay25: storyboard com quadros intermediarios (2026-08-23)

`storyplay25.py` (novo, porta 7911): decompoe cada cena nos seus planos (o
`shot_list` que o enriquecimento do parse ja produz) e gera **uma imagem por
plano**, visivel na galeria — e essas imagens entram na geracao como
**keyframes** via `LTXVAddGuideAdvanced` (`frame_idx` + `strength` por guia,
encadeaveis). O storyboard passa a dirigir o clipe inteiro, nao so seu primeiro
frame. `ltx25_backend.generate(..., keyframes=[(img, frame_idx, strength)])`.

### Tres erros meus, achados so ao olhar o resultado

O ciclo passou (`E2E_OK`) na terceira tentativa mas produziu um *slideshow* de
imagens sem relacao. O mecanismo estava certo; a entrada, nao:

1. **Descartei `build_prompt()`.** Passei `prompt_override` so com a frase curta
   do plano, jogando fora os descritores de personagem E a ancora de estilo
   fotorrealista. O comentario do proprio `generate_storyboards.py` descreve a
   falha que eu reproduzi: "scene text alone left FLUX free to answer in
   illustration style, which then set the look of the whole clip".
2. **Nunca rodei o estagio `cast`.** Sem `cast.json` nao ha descritor, e repetir
   o descritor em cada plano e o **unico** mecanismo de continuidade de
   identidade deste pipeline (sem IP-Adapter/face-lock).
3. **Bug de parenteses:** `round(x-1)/8` em vez de `round((x-1)/8)` — 30s@24fps
   virou 713 frames em vez de 721. Perto o bastante para passar num log.

Corrigidos: `_beat_prompt()` monta contexto completo da cena + descritores +
acao do plano, mantendo a ancora de estilo por ultimo; `do_parse()` roda o
estagio `cast` quando falta `cast.json` (e avisa se ficar sem descritores).

Depois: os 5 quadros saem consistentes (mesma Lyra de manto azul e cajado,
mesmo Thoren de armadura, mesma camara com runas, estilo unico) e o video de
30s segue os planos mantendo identidade e cenario — sem legendas, por estar
acima do limiar de §3.11.

### Comportamento medido: cada keyframe alonga o clipe em 8 frames

| Run | keyframes | pedido | saida | delta |
|---|---|---|---|---|
| 1 | 8 | 713 | 777 | 64 = 8x8 |
| 2 | 5 | 721 | 761 | 40 = 5x8 |

`LTXVAddGuide.append_keyframe` acrescenta 1 frame latente (=8 de video) por
guia. Documentado em vez de compensado automaticamente: subtrair
`len(keyframes)*8` do pedido silenciaria uma diferenca que pode mudar com a
versao do no.

### Incompatibilidade estrutural que custou uma rodada

As guias tem de entrar **antes** de `LTXVConcatAVLatent`, nao depois: o latente
que chega ao sampler no grafo 2.5 e audio+video combinado (um `NestedTensor`), e
`LTXVAddGuide` faz `torch.cat` nele -> "expected Tensor as element 0 in argument
0, but got NestedTensor". Guiar o latente de video puro e deixar o concat
depois tambem e conceitualmente correto: um quadro de storyboard nao diz nada
sobre a trilha. Nota de metodo: a validacao contra `/object_info` pegou o campo
`crop` faltando, mas **nao** pegaria este — schema valido, semantica
incompativel. Sao camadas de erro diferentes.

## 3.13 Motor de LLM da extracao: medido, nao escolhido por gosto (2026-08-23)

Pergunta: "gemma4 seria a melhor opcao para extracao? temos outros modelos".
Havia como responder por medicao -- o verificador de §3.10 ja da um placar
objetivo (falas preservadas COMO DIALOGO PARSEADO, planos recuperados, tempo).

Mesma entrada, mesmos criterios:

| Motor | Tempo | Falas verbatim | Planos | beat_visual |
|---|---|---|---|---|
| gemma4-e2b (transformers, era o padrao) | 91s | **2/4** | 5 (1 acao) | 2/4 |
| qwen3.6-35b-a3b (Ollama) | **6s** | **4/4** | 9 (5 acao) | **4/4** |

15x mais rapido e melhor em todos os criterios. "E2B" = ~2B de parametros
efetivos: o gemma4-e2b e pequeno demais para a tarefa, e era ele a origem dos
defeitos de formato que exigiram as reparacoes deterministicas de §3.10.
Padrao trocado para o Ollama em `storyplay25` (dropdown, com `allow_custom_value`
para outras tags) e suportado em `parse_screenplay --enrich-engine <tag>`.

### Tres armadilhas encontradas ao ligar o Ollama

1. **O catalogo servido nao e o da pasta.** `G:\ollama\models` listava gemma4,
   phi4, qwq, qwen3..., mas a instancia no ar servia SO `qwen3.6-35b-a3b` --
   exatamente a armadilha ja documentada no CLAUDE.md (o app desktop le outro
   diretorio). `_run_ollama` agora lista o que a instancia realmente serve
   quando um modelo 404.
2. **Modelo de raciocinio falha em SILENCIO.** qwen3.6 escreve o
   chain-of-thought num campo `thinking` separado e gastou todo o orcamento de
   tokens nele: `done_reason="length"` e `content` **vazio** -- parece recusa do
   modelo. `think: False` resolve (reformatar nao precisa de raciocinio). Vale
   para qwq e deepseek-r1 tambem.
3. **Modelo melhor erra DIFERENTE, nao menos.** O qwen preservou 4/4 falas mas
   escreveu uma rubrica sob um cue:
   `LYRA / "Toca uma runa... e diz com urgencia." / LYRA / "Thoren, as runas..."`.
   O parser le como fala, entao o **TTS falaria a rubrica**. Corrigido em
   `_demote_stage_direction_blocks()`: o sinal e o MESMO cue repetido em
   seguida com verbo de fala na primeira linha -- roteiro real nunca repete cue
   assim, entao a regra nao dispara em dialogo legitimo.

Licao geral desta sessao, repetida em §3.12 e aqui: trocar de modelo melhora as
estatisticas mas nao dispensa as reparacoes deterministicas -- so muda quais
erros aparecem. As reparacoes ficam.

## 3.14 dev + keyframes: STG e as guias sao incompativeis (2026-08-24)

`storyplay25` na variante **dev** com quadros intermediarios morria no
`SamplerCustomAdvanced`:

```
The expanded size of the tensor (29184) must match the existing size (30336)
at non-singleton dimension 1.  Target sizes: [1, 29184, 4096].
```

**Nao era memoria** -- OOM diz "CUDA out of memory". Era shape. A aritmetica
entrega o caso na hora: a 768x512 cada frame latente tem `(768/32)*(512/32) =
384` tokens, entao `29184/384 = 76` e `30336/384 = 79` frames latentes. A
diferenca e **exatamente 3 = o numero de keyframes** (cada `LTXVAddGuide`
acrescenta 1 frame latente, ver §3.12).

Isolamento em tres corridas curtas (97 frames, 2 passos):

| combinacao | resultado |
|---|---|
| dev + 3 keyframes | **FALHA** (4992 vs 6144 = os mesmos 13 vs 16 frames) |
| dev sem keyframes | OK |
| distilled + 3 keyframes | OK |

Ou seja: a combinacao. Nao a variante, nao as guias.

**Causa.** O traceback aponta `comfy/ldm/lightricks/model.py:407`,
`_attention_with_guide_mask`, que particiona Q em grupo "ruidoso" e grupo
"guia" para nao materializar a mascara densa (1,1,T,T), e devolve cada
resultado numa **fatia** de `out`. Instrumentando a funcao (uma corrida,
`LTX_DIAG_GUIDE=1`) as quatro primeiras chamadas usam `attention_xformers` e
devolvem 4992 certinho; a quinta usa `stg_attention` e devolve 6144:

```
attn_fn=stg_attention  q_slice=(1, 4992, 4096) -> ret=(1, 6144, 4096)
```

`custom_nodes/ComfyUI-LTXVideo/stg.py:154` troca o `optimized_attention` global
por um passthrough para o bloco marcado:

```python
def stg_attention(self, q, k, v, heads, *args, **kwargs):
    self.current_idx += 1
    if self.current_idx in self.attn_idx:
        return v          # v INTEIRO -- ignora o q fatiado
```

O node custom foi escrito quando attention era uma chamada por camada. O core
agora faz ate tres, com Q fatiado, e o `return v` cai num slot mais curto. Ha
um segundo defeito latente na mesma linha: `current_idx` incrementa a cada
chamada, entao o **indice do bloco STG se desalinha** sempre que ha guias,
mesmo que os shapes casassem.

**Correcao** (`ltx25_backend.py`, `DEV_STG` / `DEV_STG_WITH_KEYFRAMES`): com
keyframes presentes, `GuiderParameters.stg = 0`. `do_perturbed()` so testa esse
valor (`parameters.py:61`), entao zero desliga o passe perturbado inteiro e o
monkeypatch nunca entra. `perturb_attn` sozinho **nao** basta -- ele nao e
consultado por `do_perturbed()`.

Custo: perde-se o ganho de qualidade do STG nessas corridas. Mantem-se o CFG
real, que e a razao de usar dev. Efeito colateral bem-vindo: um forward a menos
por passo.

Verificado: `dev + 3 keyframes` gera, e sai com **121 frames** (97 pedidos +
3x8), identico ao `distilled + 3 keyframes` -- o crescimento por guia de §3.12
vale igual nas duas variantes.

**Lado ferramenta.** O erro original chegou ilegivel porque
`submit_and_wait()` fazia `json.dumps(messages)[:3000]`, e a parte util --
o no que falhou e os quadros MAIS FUNDOS do traceback -- fica no **fim** do
payload. Trocado por `_format_exec_error()`, que extrai os campos e, quando
precisa cortar, **corta o topo**. Sem isso o diagnostico comeca as cegas.

Armadilha de processo, de novo (ver §6): o ComfyUI que rodou a falha do usuario
tinha sido iniciado por outro launcher em 23/08 17:49, sem `--cache-none` --
por isso o log dele nao estava em `logs/comfyui_ltx25.log` e a resposta trazia
30 nos `execution_cached`. Antes de culpar cache, confira **qual** servidor
esta no ar: `Get-CimInstance Win32_Process` mostra a linha de comando.

## 3.15 Condicionamento por trilha real no 2.5 (2026-08-24) -- FUNCIONA, ligado por padrão

O `music_maker_ui_v2_25` gerava cenas com trilha **inventada pelo modelo**: o
shim aceitava `--audio-input-path` e o descartava. Medido por correlacao entre
o audio da cena e a fatia da musica de entrada: **-0.005** (nenhuma relacao).
O remux final (`music_maker_ui_v2_25.py:1247`, `-map 0:v:0 -map 1:a:0`) repunha
a musica do usuario, entao o MP4 final saia certo -- mas o **movimento nao
seguia a batida**, porque a geracao nunca viu a faixa. No 2.3
(`music_to_video.py:176`) a faixa virava latente de audio e condicionava.

**Duas pecas, e uma sozinha nao serve:**

1. `LTXVAudioVAEEncode` (core, `nodes_lt_audio.py:37`) codifica o wav e
   substitui o `LTXVEmptyLatentAudio` no `LTXVConcatAVLatent`.
2. `LTXVSetAudioVideoMaskByTime` (custom node, `latents.py:636`) prende esse
   latente com um `noise_mask` por modalidade. **Sem ele o encode e desperdicio
   puro**: o sampler ruidifica tudo a denoise=1.0 e a musica morre antes do
   primeiro passo.

Convencao da mascara, de `comfy/samplers.py:642`
(`out = out*mask + latent_image*(1-mask)`): **1 = gerar, 0 = preservar**. Logo
`mask_video=True/init 1.0` (gera todo o video) e `mask_audio=False/init 0.0`
(preserva toda a faixa).

`LTXVReferenceAudio` **nao** serve aqui -- e ID-LoRA para identidade de voz.

Implementado em `ltx25_backend._apply_audio_conditioning()`, exposto como
`generate(audio_conditioning=...)`, e o shim voltou a honrar
`--audio-input-path` (so ignora se o arquivo nao existir). Ordem importa: as
guias de keyframe trabalham no latente de VIDEO puro, antes do concat; o
condicionamento de audio embrulha a saida do concat -- entao ele vem **depois**.

**Estado: FUNCIONA, validado 2026-08-24** num run real de 2 cenas. Ligado por
padrao; `LTX25_AUDIO_COND=0` desliga.

O casamento de duracao, que eu apontava como o risco, nao e problema (rodou com
wav de 8,0s contra video de 9,7s).

### A licao que quase matou a feature: medir com o instrumento errado

Eu declarei "nao funciona" com base em correlacao de **forma de onda**, que deu
+0,03. Estava errado, e o usuario percebeu ouvindo antes de eu perceber medindo.

O VAE de audio do 2.5 e **mel-spectrogram + vocoder** (ja anotado no CLAUDE.md).
Resintese por vocoder **nao preserva fase**: a forma de onda le ~0 mesmo num
casamento perfeito. O que se compara e o **envelope de intensidade**, com
cross-correlacao tolerante a deslocamento (a saida vem com ~30-70 ms de lag).

E um numero alto sozinho nao prova nada -- este material e uniforme, e um par
deliberadamente TROCADO chegou a +0,885. A prova e a **matriz cruzada**: cada
cena tem que casar com a PROPRIA fatia e nao com a da vizinha.

| pareamento | condicionado | inventado (antes) |
|---|---|---|
| cena1 x fatia1 (certo) | **+0,966** | +0,501 |
| cena1 x fatia2 (errado) | +0,549 | +0,299 |
| cena2 x fatia2 (certo) | **+0,973** | +0,315 |
| cena2 x fatia1 (errado) | +0,573 | +0,524 |

Separacao nitida depois; nenhuma antes -- no run antigo a cena 2 casava MELHOR
com a fatia errada (+0,524) do que com a propria (+0,315), que e exatamente o
que se espera de trilha inventada.

Generalizando: **antes de declarar que um recurso generativo nao funciona,
confira se a metrica sobrevive ao pipeline.** Round-trip de VAE, vocoder,
requantizacao e reencode destroem identidade de sinal sem destruir conteudo. E
sempre rode o controle cruzado, nos dois sentidos.

**Erro de processo meu, repetido duas vezes.** Rodei validacao na GPU enquanto o
usuario tinha run ativo -- na segunda vez derrubei o ComfyUI e **parei o run dele
na cena 2**. `ensure_server()` reaproveita servidor no ar, entao teste e producao
caem na MESMA fila e disputam a MESMA VRAM. Antes de qualquer geracao de teste:
`ls -dt outputs/music_video_v2/*/ | head -1` e ver se ha cena recente, alem de
`/queue` e `nvidia-smi`. GPU ociosa NAO significa que nao ha run em andamento --
pode estar entre cenas.

**Armadilha de convivencia.** `ensure_server()` reaproveita servidor ja no ar
-- correto -- mas isso significa que um teste meu entra na MESMA fila do run do
usuario e disputa a mesma VRAM. O servidor reiniciou as 11:31 com os dois
juntos. Antes de rodar geracao de teste, confira a fila (`/queue`) e o
`nvidia-smi`.

## 3.16 Condicionamento de imagem estava DESLIGADO no 2.5 (2026-08-24) -- corrigido e validado

Sintoma relatado: "a personagem mantem a aderencia, mas nao consistencia, muda
as roupas de cena em cena".

Havia DUAS causas. A primeira e um bug, e era a dominante.

**O bug.** `LTXVImgToVideoInplace.strength` e `FLOAT` (default 1.0), mas o grafo
achatado carregava `"strength": false`. `float(False) == 0.0` = condicionamento
com peso ZERO: a imagem era carregada, redimensionada, pre-processada e
codificada no VAE -- e descartada na hora de aplicar.

Origem: no workflow oficial, `strength` e `bypass` sao AMBOS entradas de
subgrafo (sentinel `-10`, slots 9 e 5 do no pai 5514). O `bypass` resolveu; o
`strength` nao, e o fallback pegou o indice errado de
`widgets_values = [121, 0, false, 960, 544, 18, 0.7]`, entregando o booleano no
lugar do 0.7. **Mesma classe do bug do `batch_size`** (§3.6): o conversor nao e
confiavel para valores de widget quando ha entradas de subgrafo linkadas.

Auditoria do grafo inteiro contra `/object_info` achou 7 divergencias de tipo;
6 estao no ramo do prompt enhancer, que esta desligado e nunca executa. So a do
`strength` afetava a saida.

**Medicao (mesma imagem, mesmo prompt, mesmo seed 1234, 97 frames):**

| | dist. do frame 0 a imagem de condicionamento (0-255) |
|---|---|
| `strength=1.0` (corrigido) | **5,24** |
| `strength=0.0` (o bug) | **63,10** |
| run real de 16 cenas, antes | 65,12 |

5,24 e nivel de round-trip de VAE/compressao -- o frame 0 gerado **e** a imagem.
O controle a 0.0 reproduziu o run real, entao nenhuma outra variavel explica.

Corrigido fixando `api[N_IMG2VID]["inputs"]["strength"]` explicitamente, no
mesmo bloco dos outros consertos de schema drift, e o shim passou a honrar a
forca do `--image PATH IDX STRENGTH` (que as UIs ja mandavam como 1.0 e ele
descartava -- o que importava pouco, porque o grafo zerava de qualquer jeito).

**Decaimento, medido no clipe corrigido:**

| frame | 0 | 24 (1s) | 48 | 72 | 96 (4s) |
|---|---|---|---|---|---|
| distancia | 5,24 | 38,14 | 42,36 | 45,43 | 48,38 |

O condicionamento ancora o inicio e o prompt conduz o resto. Isso importa na
cadeia de cenas: e o ULTIMO frame que alimenta a proxima, e em cenas de 233
frames ele ja esta longe do que entrou. **Ressalva:** diferenca media de pixel
nao separa "mudou a personagem" de "a camera girou"; quantificar identidade
exigiria embedding facial, e nenhum esta instalado aqui (insightface /
facexlib / deepface / face_recognition, todos ausentes).

**A segunda causa, que continua valendo.** Os prompts padrao do
`music_maker_ui_v2_25` (linhas 150-161) descrevem a personagem como
*"A consistent female singer"* / *"The same singer"*. Isso e o PEDIDO de
consistencia, nao uma ancora: sem cabelo, roupa, idade ou cor, o modelo
preenche o vazio de novo a cada cena. E o que o `storyplay25` ja resolve com os
descritores concretos do `cast.json`. Com o bug corrigido o frame inicial passa
a puxar; descritores concretos fazem o prompt puxar na MESMA direcao em vez de
competir.

## 3.17 Qualidade de rosto: resolucao, enquadramento e two-stage (2026-08-24)

Sintoma: rosto e olhar distorcidos nas cenas do music maker.

### A causa e aritmetica, nao artistica

O LTX comprime 32x no espaco. A 768x512 com plano aberto, o rosto media 52x58 px
= **1,6 x 1,8 celulas latentes**, 0,77% do quadro. Cada olho tinha uma FRACAO de
celula. Nao adianta passo, prompt ou upscaler.

| config | rosto em celulas latentes |
|---|---|
| 768x512, plano aberto | 1,6 x 1,8 |
| 1280x704, plano medio | **3,8 x 3,9** (5x a area) |

Corrigido com as duas coisas juntas: resolucao (~1,7x) e enquadramento (o resto).
Rosto limpo, olhos alinhados, colar e brincos legiveis. Sem legendas queimadas a
19,3s (perto da zona de risco dos 17s, mas passou).

### Custo real: cuidado com sondagem fria

| | tempo |
|---|---|
| sondagem 121 frames 1280x704 (modelo FRIO) | 474s |
| final 465 frames 1280x704 (modelo quente) | 887s |

Quase 4x os frames em menos de 2x o tempo. Extrapolar da sondagem me fez
estimar 16 min/cena quando o real e ~7-8 min. **Descontar o carregamento do
modelo de 40 GB antes de extrapolar.**

## 3.18 Two-stage ligado e medido -- e o upscaler do pipeline esta errado (2026-08-24)

`LTX-2.5_T2V_I2V_Two_Stage_Distilled.json` ligado em `ltx25_backend`
(`two_stage=True`, `--two-stage`, `LTX25_TWO_STAGE=1`). Opt-in de proposito:
**nao** derivado de `--spatial-upsampler-path`, que as UIs 2.3 mandam em todo
run -- keyar nisso mudaria o custo de todos em silencio.

40 dos 45 nos sao compartilhados com o single-stage; so os loaders mudam de
instancia (`5004:*` vs `5575:*`), resolvido em `stage_ids()` em vez de forkar
`build_workflow`. **O bug do `strength` aparece de novo no estagio 2**
(`5517:4970`) -- o defeito esta no subgrafo compartilhado, entao ha uma
ocorrencia por instancia. O `RandomNoise` do refino tambem vinha com
`noise_seed` fixo em 42.

**ATENCAO: o two-stage DOBRA a saida.** Estagio 1 a 640x352 -> 1280x704.

### Medicoes (mesmo prompt, seed, musica)

| | tempo | saida |
|---|---|---|
| single-stage 1280x704 | 887s | 1280x704 |
| two-stage (640x352 -> refino) | **793s** | 1280x704 |
| single-stage 640x352 (baseline) | 433s | 640x352 |

Two-stage saiu MAIS RAPIDO na mesma resolucao final: o estagio 1 e barato e o
refino faz 3 passos contra 8. Custo do refino isolado: 793-433 = 360s.
O condicionamento de audio sobrevive ao refino (+0,969, igual ao single-stage) --
apesar de o estagio 2 re-amostrar sem mascara a partir de sigma 0,85.

### A metrica de nitidez MENTE -- olhe as imagens

Com o estagio 1 identico (mesma cena, diff 9,73/255), rosto ampliado a 1280x704
por quatro caminhos:

| caminho | nitidez (Laplacian) | veredito visual |
|---|---|---|
| lanczos | 80,8 | mole |
| **realesr-animevideov3-x2** (padrao do `upscale_video.ps1`) | **245,6** | **pior de todos** |
| realesrgan-x4plus | 201,0 | ok, mantem volume 3D, leve artefato |
| refino latente | 149,6 | **melhor** |

O anime-v3 pontua MAIS **porque** achata: converte gradiente em bloco de cor
chapada, olhos viram trace preto duro, boca vira mancha, o brinco de esmeralda
some. Variancia de Laplaciano mede contraste de borda, nao correcao -- afiar
borda errada pontua igual. **Segunda vez nesta sessao que uma metrica plausivel
deu a resposta errada** (a primeira foi correlacao de forma de onda no audio,
§3.15). Regra: metrica de qualidade generativa exige inspecao visual de
confirmacao.

### Resolvido: seletor na UI (2026-08-24)

O modelo do upscale era fixo em `realesr-animevideov3` -- **line art de anime**
aplicado a render 3D estilo Pixar. Virou seletor em `music_maker_ui_v2_25`
(`UPSCALE_PRESETS`), com o padrao ANTIGO preservado: a 1280x704 a diferenca
deixa de ser gritante, entao trocar o default silenciosamente mudaria a saida
de quem ja calibrou com ele.

Presets amarram modelo E escala, porque o executavel resolve
`<nome>-x<escala>.bin` e o `x4plus` so existe em 4x:

| preset | modelo | escala |
|---|---|---|
| Anime 3x (padrao) | realesr-animevideov3 | 3 |
| 3D/Foto 4x | realesrgan-x4plus | 4 |
| Anime 4x | realesrgan-x4plus-anime | 4 |
| Sem upscale | -- | 0 |

"Sem upscale" e util agora que o two-stage ja entrega o dobro da resolucao.

Encanamento pelo padrao que o arquivo ja usava para `torch_compile`/`ambient`:
variavel de ambiente (`LTX_UPSCALE_MODEL`/`LTX_UPSCALE_SCALE`) setada em
`start_generation_thread`, em vez de engrossar a assinatura de
`process_chain_generation`. O parametro novo entra ANTES de `start_index` --
os botoes por cena passam `start_index`/`mode` por palavra-chave, entao a
insercao posicional e segura; acrescentar no fim teria caido em `start_index`.

`music_maker_ui_v3_25` tem a mesma estrutura (com params de latentsync a mais) e
NAO recebeu o seletor -- fica igual ao que era.

## 3.19 video_doctor: deteccao e reparo temporal pos-geracao (2026-08-24)

`video_doctor.py` -- mede um clipe, classifica os defeitos e aplica o reparo
mais barato que resolve cada um. Offline: DIS optical flow (OpenCV 5), sem
baixar nada. RAFT existe no torchvision mas os pesos NAO estao em cache aqui.
Nao ha RIFE nem SAM instalados.

    analyze  -> mede e classifica, nao altera o video
    repair   -> aplica um plano JSON
    doctor   -> as duas coisas numa passada

### A medida central

    residuo = | warp(frame[t-1] -> t, fluxo) - frame[t] |

Movimento normal e EXPLICADO pelo fluxo, entao o residuo fica baixo por mais
rapida que seja a camera. O que sobra e materia aparecendo, sumindo ou trocando
de forma. Isso separa "a camera girou" de "o rosto derreteu" -- coisa que
diferenca de pixel simples nao faz.

Duas medidas auxiliares classificam: delta de luminancia (flicker) e delta de
alta frequencia (textura pulsando).

### Tres erros de metrica, em sequencia, no mesmo dia

Custou tres tentativas erradas chegar no detector certo. Vale registrar porque
o padrao se repete em todo este MEMORIAL:

1. **z-score contra o clipe inteiro** -> 17 falsos positivos num clipe SADIO.
   Um por-do-sol com camera orbitando muda de brilho de verdade; z-score sem
   ancora absoluta promove fotografia legitima a defeito. Corrigido com
   (a) exigir REVERSAO de sinal para chamar de flicker -- flicker sobe e volta,
   iluminacao real deriva -- e (b) PISOS ABSOLUTOS: os "flickers" restantes
   oscilavam 0,5/255, meio por cento de brilho, invisivel.

2. **normalizar residuo por movimento** -> matou os falsos positivos E os
   verdadeiros. Eu tinha diagnosticado "pan rapido engana o fluxo". **Estava
   errado**: medindo com `cv2.phaseCorrelate`, a camera andava a 0,8 px/frame
   CONSTANTES, e o frame acusado tinha diferenca direta de 6,50 contra 4,05 de
   linha de base -- anomalia real. O fluxo erratico ali era CONSEQUENCIA do
   defeito, nao causa. Inverti causa e efeito.

3. **escolher a pior celula descontando o movimento dela** -> a mao derretendo
   perdia a selecao para uma celula de fundo parado, justamente por se mover.

O que funciona: **contraste temporal local**. Compara-se cada ponto com a
MEDIANA DA PROPRIA VIZINHANCA (janela 31), nao com o clipe todo. O que separa
defeito de movimento legitimo nao e magnitude, e DURACAO: falha de fluxo por
camera sobe e desce suavemente ao longo de dezenas de frames e a mediana da
vizinhanca sobe junto; defeito de poucos frames destoa por definicao.

**Licao, agora pela terceira vez nesta sessao** (audio §3.15, nitidez §3.18):
metrica plausivel mente. Antes de aceitar uma leitura, medir por um caminho
INDEPENDENTE -- aqui foi diferenca direta de pixel + `phaseCorrelate`, que
derrubaram meu diagnostico de "pan rapido".

### Deteccao local: a media do quadro nao enxerga

Um rosto ocupa 0,77% do quadro a 768x512 (§3.17). Derreter nao move a media.
O detector divide em grade 8x8 e guarda a PIOR celula -- e assim achou uma mao
deformando nos frames 18-21 de um clipe que parecia limpo, apontando a celula
certa (y=616, x=640). Confirmado por medida independente: a regiao saltou de
3,27 para 28,50 (3x a mediana) enquanto o quadro inteiro ficou plano em ~3,2.

### Reparos

| defeito | acao |
|---|---|
| 1-3 frames | interpolacao por warp bidirecional |
| luminancia oscilando | deflicker por MEDIANA da vizinhanca |
| 4-24 frames | regeneracao parcial via LTX, ancorada nos frames bons |
| > 24 frames | reporta e nao mexe |

Deflicker usa MEDIANA, nao media: com media a curva-alvo e puxada pelos proprios
frames defeituosos e o reparo sai pela metade.

A regeneracao parcial usa `ltx25_backend` com keyframes nas duas pontas -- e a
mesma maquina do storyplay25 -- com margem de 2 frames bons de cada lado.
Consertar SO o frame defeituoso cria duas descontinuidades no lugar de uma.
Exige `--prompt`; sem ele cai para interpolacao, avisando.

### O RESULTADO MAIS IMPORTANTE: o reparo automatico PIOROU o video

Instalei RIFE e reexecutei o reparo daquele segmento 18-21. Olhando os quadros
lado a lado -- original / warp / RIFE:

- original: a mao esta **bem formada e nitida**
- warp: a mao virou borrao de azul e pele
- RIFE: a mao foi praticamente **apagada** dentro do vestido

Aquilo nunca foi defeito. Era a mao **ENTRANDO EM QUADRO** rapidamente --
movimento legitimo. Interpolar entre os frames 17 e 22 apagou a trajetoria dela.

E os "38% de reducao do pico" que eu tinha comemorado mediram a coisa errada:
o reparo melhorou a metrica **destruindo o conteudo** que causava a diferenca.
**Quarta falha de metrica da sessao**, e a mais instrutiva -- as outras me
fizeram tirar conclusao errada; esta teria me feito estragar video bom em lote.

**Nenhuma medida temporal separa "materia surgindo porque o modelo errou" de
"materia surgindo porque entrou em cena".** Quem separa e quem olha.

Por isso o desenho mudou: **revisao humana no meio, nao auto-reparo**.
`--previews` gera uma tira por segmento (frames marcados com borda vermelha,
vizinhos sadios em verde, numerados) e `--only 0,2` repara so os escolhidos.

## 3.20 RIFE e RAFT instalados; doctor embutido nas UIs (2026-08-24)

**RAFT**: pesos oficiais do torchvision (3,8 MB, download.pytorch.org). Fluxo
melhor que o DIS, ~45x mais lento (4,5s contra 0,1s por par). `--flow raft`;
o DIS segue padrao.

**RIFE**: `rife-ncnn-vulkan` 20221029, release oficial nihui, 431 MB com os
modelos rife-v2 a v4.6, em `tools/rife/` -- mesmo padrao do realesrgan-ncnn-vulkan
que ja estava la. O pacote PyPI `rife-ncnn-vulkan-python` NAO serve: exige
compilar (sem wheel para cp312) e o build falha.

**ARMADILHA DE LOCALE, e custou uma rodada de depuracao**: o binario converte o
argumento `-s` (timestep) com o **locale do sistema**. Nesta maquina, em pt-BR,
`-s 0.2` e recusado com "invalid timestep, must be 0~1" -- o `atof` para no
ponto e le 0 -- enquanto `-s 0,2` funciona. Fixar a virgula quebraria numa
maquina em ingles, entao `repair_interpolate_rife()` tenta os dois e memoriza
o que funcionou. O sinal estava a vista desde antes: o Real-ESRGAN imprimia
"75,00%" com virgula.

Com o timestep certo, o RIFE alisa melhor que o warp DENTRO do vao (media das
transicoes reparadas 7,75 contra 10,9) -- mas isso so importa quando o
segmento e defeito de verdade, ver acima.

### Integracao

`video_doctor_ui.build_doctor_tab()` -- um modulo, quatro UIs. Duplicar a
interface em cada arquivo daria seis copias divergindo.

Embutido em: `music_maker_ui_v2_25`, `v3_25` (2.5), `music_maker_ui_v2`, `v3`
(2.3). Avulso: `start_video_doctor.bat`, porta 7912.

Fluxo da aba: Analisar -> galeria de tiras -> marcar segmentos -> Reparar.
Fica FORA do fluxo de geracao de proposito: e pos-producao, aplicada ao video
final ja concatenado e upscalado, e so quando o usuario quiser.

**Armadilha de insercao automatizada**: a primeira tentativa injetou a chamada
logo apos um `return`, dentro de `collect_prompts_and_start`. **Compilou** --
sintaxe valida, codigo morto -- e a aba simplesmente nao aparecia. Ancorar por
regex em arquivo de 1900 linhas exige conferir que a ancora esta no escopo
certo, e depois VERIFICAR que o componente existe na arvore do Blocks, nao so
que o arquivo compila.

## 3.21 O LTX obedece a câmera -- mas como DESTINO, não como estado (2026-08-26)

Pergunta que sustentava a ideia de dar direção às cenas: o modelo responde a
vocabulário de enquadramento? Teste controlado -- mesmo sujeito, mesmo seed
(99), 640x352, 97 frames, só a especificação de câmera muda.

| especificação | obedeceu? |
|---|---|
| (nenhuma) | plano médio-aberto, altura dos olhos -- **é o default** |
| plano geral extremo | **sim, forte** -- personagem minúscula, castelo e paisagem inteiros |
| close-up no rosto | **sim, mas só no fim** -- ver abaixo |
| contra-plongée | sim, moderado |
| plongée | **sim, forte** -- vista de cima direto no pátio |

### O achado que importa

Extraindo os frames 0, 48 e 96 do mesmo clipe de close-up:

- **frame 0**: plano geral, a personagem pequena no quadro
- **frame 48**: plano médio
- **frame 96**: close-up de verdade, rosto preenchendo o quadro

O modelo **não começa** no enquadramento pedido: ele parte perto do próprio
default e **deriva** até a especificação ao longo do clipe. O contra-plongée faz
o mesmo.

Três consequências, e elas redesenham a proposta de decupagem:

1. **Explica a sensação de "sem direção" por completo.** Todo plano é um push
   lento do genérico até o que o prompt pede. Todos os planos têm o mesmo arco
   de movimento. Isso É um estilo -- acidental e monótono.
2. **Não dá para cortar entre ângulos estáticos só por prompt.** Um close-up de
   2 segundos nunca chegaria a close-up. E ritmo de montagem depende de planos
   curtos.
3. **O enquadramento tem de vir da IMAGEM de condicionamento, não do prompt.**
   O prompt deve descrever o MOVIMENTO; o quadro inicial define o
   enquadramento. É exatamente a máquina já validada em §3.16 (`strength`) e
   §3.12 (keyframes) -- o que faltava era saber que ela é obrigatória aqui, não
   opcional.

Isso também caracteriza direito uma observação solta no docstring de
`render_scenes.py`, que atribuía a deriva a um descasamento entre imagem-âncora
e prompt. Não é: é comportamento padrão do modelo, com âncora ou sem.

**Arquiteturalmente, o `storyplay25` já estava certo** -- storyboard por beat,
com quadro condicionando cada trecho. O que faltava era o quadro do storyboard
CARREGAR o enquadramento, em vez de ser sempre um plano genérico.

## 3.22 story_structure: a camada entre cenas que não existia (2026-08-26)

Diagnóstico do usuário: as sequências não têm direção, posicionamento nem
montagem. Investigando de quem seria o papel, a resposta foi: de ninguém.

**A LLM do parse não pode fazer isso, e não é questão de modelo.**
`_build_scene_user_prompt(scene)` monta o prompt com dados de UMA cena --
cabeçalho, local, período, ação, personagens, falas -- e o laço de
enriquecimento é `for scene in scenes:`, uma chamada independente por cena.
Nenhuma referência à anterior ou à seguinte. Pedir arco a quem lê uma cena por
vez é pedir o impossível; trocar o qwen por algo maior não mudaria nada.

São **três trabalhos distintos** que estavam conflados:

| trabalho | quando | quem faz hoje |
|---|---|---|
| estrutura dramática (arco, função de cada cena) | antes do parse | ninguém |
| continuidade (figurino, local, tempo, quem entra/sai) | atravessa tudo | só `cast.json`, parcial |
| decupagem (planos por cena) | depois do parse | ninguém |

`script_pipeline/story_structure.py` cobre os dois primeiros. Lê o roteiro
INTEIRO numa chamada e emite metadados por cena. **Não reescreve nada** -- a
saída é aditiva, um JSON ao lado. Isso preserva a garantia verbatim de
`prose_to_screenplay`: se alucinar, perde-se anotação, nunca uma fala.

Validado num roteiro de 5 cenas escrito para o teste (**todos os runs reais
anteriores têm UMA cena só** -- o problema entre cenas nunca tinha sido
exercitado): curva de tensão 0,30 → 0,50 → 0,60 → **0,90 na virada** → 0,40 no
desfecho. Sem Ollama, degrada para a camada determinística sozinha.

### Dois bugs no parser, achados pelo teste

Afetam **2.3 e 2.5** (o `script_pipeline` é compartilhado):

1. **Cabeçalho de TRÊS partes.** `INT. TORRE DO RELÓGIO - ESCADA - NOITE`
   produzia `time_of_day="ESCADA - NOITE"`, porque o regex divide no PRIMEIRO
   hífen. Não é cosmético: esse campo vai para os prompts de storyboard, então
   o modelo recebia "escada" como período do dia -- e o sub-local sumia,
   fundindo cenas espacialmente distintas num lugar só.
2. **`INT/EXT.` colocava "EXT." dentro do nome do local.** A alternância
   `(INT|EXT|INT/EXT|I/E)` tenta `INT` primeiro, casa, e o `[./]` engole a
   barra. **Defeito pré-existente**, herdado, não introduzido. Corrigido
   ordenando do mais longo para o mais curto.

Corrigido trabalhando no texto CRU do cabeçalho, não nos grupos do regex:
remontar a partir deles reformatava nome próprio com hífen interno
(`PONTE RIO-NITEROI` virava `PONTE RIO - NITEROI`). Verificado contra 8
formatos.

### Decisão de projeto que o teste forçou

Eu pedia `link_to_previous` à LLM. Ela devolveu **"continuo" para as cinco
cenas** -- inclusive na que vai de meia-noite para amanhecer. Mas isso é
calculável: mudou o período? mudou o local? o local já apareceu? Movido para a
camada determinística e removido do pedido. É a mesma lição de §3.13 e §6.1:
não pergunte o que dá para contar.

## 3.23 shot_plan: a decupagem determinística (2026-08-26)

`script_pipeline/shot_plan.py` -- cenas + estrutura dramática viram uma lista de
PLANOS com câmera concreta. **Sem LLM neste arquivo**: padrão de cobertura vem
da função dramática, duração vem do perfil de estilo e da tensão, lado de tela
vem da regra dos 180 graus. Tudo conta.

### A decisão que a medição de §3.21 forçou: DOIS prompts por plano

Como o LTX trata câmera como destino e não como estado, enquadramento não pode
vir do prompt de vídeo -- um plano de 3s nunca chega lá. Então cada plano sai
com:

- `storyboard_prompt` -> o STILL. Carrega enquadramento, ângulo e lado de tela.
- `video_prompt` -> o LTX. Carrega MOVIMENTO e ação, nunca enquadramento.

Separar os dois é o ponto inteiro do módulo. O still fixa o frame 0; o vídeo
descreve a trajetória, que é o que o modelo controla bem.

### Estilo é JSON, não nome de diretor

Quatro perfis escritos à mão (`classico`, `tenso`, `nervoso`, `contemplativo`),
seis parâmetros cada. Nome de diretor num prompt de difusão é condicionamento
fraco e irreprodutível; "câmera travada, plano médio dominante, corte no fim da
fala" é executável. Para meia dúzia de perfis, RAG devolveria prosa SOBRE
estilo -- o que a decupagem consome é parâmetro.

Mesmo roteiro de 5 cenas, estilos diferentes:

| estilo | planos | total |
|---|---|---|
| tenso | 8 | 23,7s |
| nervoso | 8 | 18,1s |
| classico | 8 | 28,4s |
| contemplativo | 8 | 63,3s |

A cena 4 (virada, tensão 0,90) recebe os planos mais curtos de cada estilo e o
único close em contra-plongée -- estrutura virando direção, que era o objetivo.

### Legenda queimada: o gatilho é a FALA CITADA, não a duração

Refina §3.11. `render_scenes.py:466` injeta `saying: "<texto da fala>"` no
prompt. O clipe de 19,3s da princesa, SEM texto citado, saiu limpo -- enquanto
17s COM texto deu legenda em 11 de 12 frames.

Isso destrava o ritmo de corte: como o pipeline faz TTS e lip-sync depois, a
fala citada só melhora a articulação da boca ANTES do Wav2Lip, e em plano curto
o preço não compensa. `include_quotes` existe e vem DESLIGADO.

### Dois defeitos achados olhando a saída

1. **Lado de tela invertia entre cenas.** Atribuindo pela ordem de aparição
   DENTRO da cena, LYRA saía à esquerda na cena 1 e à direita na cena 5, porque
   na 5 quem fala primeiro é ORION. Violação de eixo justamente no retorno ao
   mesmo cenário. Corrigido: atribuição GLOBAL, uma vez para o filme.
2. **Plano estático sem `beat_visual` gerava prompt só com o look** -- nada
   sobre quem está em cena. Movimento vazio é correto para câmera travada; ação
   vazia não é. Cadeia de fallback: `beat_visual` -> `action_text` -> frase
   neutra com o nome do sujeito.

### Limite honesto

A regra dos 180 é uma DICA no prompt, não garantia: o LTX não tem cena 3D, e
não há como forçar geografia entre cortes. Melhora a chance; não assegura.
O mesmo vale para continuidade espacial em geral -- é a parede conhecida.

### Acrescentado no mesmo dia: intimista, extreme close-up e troca por segmento

**`extreme_close`** ("só os olhos e a sobrancelha preenchem o quadro") passou a
existir SEPARADO de `close` ("o rosto preenche a maior parte"). São gestos
diferentes: um mostra a expressão inteira, o outro isola o olhar. Dura 0,8x um
close normal -- olhar isolado cansa antes que o rosto.

**Estilo `intimista`**: close e olhar dominam, luz quente, profundidade rasa,
câmera quase parada. Foi o primeiro estilo a precisar de `coverage` própria,
porque intimismo não é só iluminação -- é a decisão de ficar perto **mesmo
quando a estrutura pediria plano aberto**. Daí perfil poder sobrepor a
cobertura por função, não só o look e o movimento.

**Troca de estilo por segmento** (`--style-changes "1:nervoso,2:contemplativo,
4:tenso,5:nervoso"`): a marca vigora a partir daquela cena **até a próxima
marca**, como troca de bobina. Cena não marcada herda a anterior -- no teste, a
cena 3 não foi marcada e seguiu contemplativa. Cada plano carrega o campo
`style`, e o plano final traz `segments` com a divisão resultante.

Defeito achado ao testar: o ramo de plano de AÇÃO fixava `wide` na marra,
ignorando o estilo -- então a cena sem diálogo escapava do intimista e abria o
quadro justamente onde o estilo pede proximidade. Passou a usar a cobertura do
estilo também.

## 3.24 render_shots: decupagem virando vídeo, de ponta a ponta (2026-08-26)

`script_pipeline/render_shots.py` executa um `shot_plan`: para cada plano, gera
o STILL (FLUX, com `storyboard_prompt`) e depois o VÍDEO (LTX, com
`video_prompt`, condicionado por aquele still).

**Primeira cena decupada de verdade do projeto**: 10 planos, 49,8s, cortes em
3,4 / 12,4 / 15,8 / 20,2 / 23,5 / 30,9 / 34,3 / 43,3 / 46,7s. Roteiro de teste:
duas personagens numa câmara imperial chinesa, 5 falas.

### A hipótese central, confirmada

O still vira o frame 0 do vídeo: distância medida de **4,24/255** entre eles --
nível de round-trip de codec. Ou seja, o enquadramento **não depende mais** do
LTX derivar até ele (§3.21), o que é o que viabiliza plano curto.

E o movimento continua vindo do prompt: no mesmo clipe, o frame 80 está a 59,84
do still, porque o `push` aconteceu. A divisão projetada em §3.23 -- still manda
no enquadramento, prompt manda no movimento -- se comporta como previsto.

Ressalva: 3,47s de `push` fecharam de plano médio para quase close. O movimento
do LTX é mais agressivo do que "lento" sugere; o parâmetro dos perfis precisa
de calibração.

### Cast ligado: o que muda

Sem `cast.json`, o FLUX inventou duas mulheres do mesmo tipo, ambas ~35 anos, e
a referência as ancorou nesse erro. Com o cast, cada uma sai com o figurino e a
idade do roteiro (hanfu azul bordado / verde-menta simples), distinguíveis.

Precisou de `appearance_only()`: o descritor do cast vem do texto de AÇÃO e
mistura aparência com o que a pessoa faz. Colado inteiro, o still do close
tentava também "atravessar o arco da lua-cheia". Só o parêntese de apresentação
e a oração de figurino entram.

### ERRO DE ARQUITETURA MEU: agrupar por plano, não por modelo

O laço original fazia still, vídeo, still, vídeo. Cada volta forçava carregar
FLUX (Qwen3-8B + DiT + VAE), descarregar, carregar LTX (Gemma4-12B + DiT de
40 GB + dois VAEs), descarregar: **~90 GB de leitura de disco por plano**, com
`--cache-none`.

Os tempos denunciaram, e é o sintoma a reconhecer: **321s → 477s → 1275s** para
clipes de tamanho parecido. Degradação crescente = fragmentação, não conteúdo.

Corrigido com duas passadas (`--stills-only`, depois `--videos-only`) e
retomada: still ou clipe que já existe é reaproveitado.

### O custo que sobra, com número

Mesmo em duas passadas: **0,19 frames/s a 960x544**, contra **0,52 frames/s a
1280x704** no clipe único da princesa (§3.17). Quatro vezes mais lento com
resolução menor.

Causa: `--cache-none` faz o ComfyUI re-executar o grafo INTEIRO a cada chamada,
incluindo o encoder Gemma4 de 25 GB. Custo fixo de ~350s por chamada, que num
clipe de 3s domina tudo. Dez clipes pagam dez vezes o que um clipe longo paga
uma.

**Vale reavaliar o `--cache-none`**: ele entrou por precaução contra os bugs de
§3.5, que a própria §3.5 depois registra como **NÃO sendo de cache**. Se for
mesmo desnecessário, recupera-se boa parte de 1h20.

### Defeito de especificação no over-the-shoulder

O `ots` de 9s -- a fala mais importante da cena -- saiu como uma NUCA. O prompt
diz "the listener's shoulder in the foreground" mas nunca diz **de quem é o
rosto**. Over-the-shoulder enquadra quem ESCUTA, por cima do ombro de quem
fala; do jeito que está, o FLUX escolhe. Precisa nomear os dois papéis.

### Falta o estabelecimento

A câmara imperial descrita em detalhe no roteiro nunca aparece inteira, porque
a cobertura do intimista abre em `medium`. Coerente com o estilo, errado para
uma cena que apresenta um lugar novo. Sugere que a cobertura precise de uma
regra "primeira cena neste local ganha um wide", independente do estilo.

### Armadilha de ferramenta: `\b` virando BACKSPACE

Escrevendo código por heredoc, `\b` de regex virou o byte **`\x08`** em vez das
duas letras. A regex passou a exigir um backspace literal e nunca casava -- e
como o caractere é invisível, o arquivo PARECIA correto ao ser lido. Só
apareceu no `repr()` da linha. Custou três tentativas.

Saída: construir a string com `chr(92)`, e usar edição direta em vez de heredoc
para qualquer trecho com escape.

### Ainda: reiniciar o ComfyUI ao mudar o cenário de memória

Derrubar o ACE-Step (16 GB) **quebrou** o render: com mais VRAM livre, o
ComfyUI escolheu uma estratégia de alocação mais agressiva, foi a 23,9 GB de
24,5 e travou a 100% de uso sem progredir, estourando o timeout de 600s no que
antes levava 90. Reiniciar o servidor limpo resolveu. Ele decide a estratégia
na carga do modelo, não continuamente.

## 3.25 Voz: rascunho, emoção dirigida e lip-sync (2026-08-26)

### Modo rascunho: animatic, sem GPU

`render_shots --animatic` monta os stills nas durações planejadas com as vozes
por cima. **Segundos, zero GPU.** É o que uma produção faz antes de filmar,
pela mesma razão: descobrir erro de montagem enquanto ele é barato.

`--draft` (512x288, máx. 25 frames/plano) existe para quando se quer ver
movimento, mas o animatic pega mais defeito por muito menos.

### O TTS deixou de ser etapa final e virou ENTRADA da decupagem

O proxy de `chars/14` que eu usava para dimensionar planos de fala estava
sistematicamente curto:

| fala | TTS real | estimado | erro |
|---|---|---|---|
| "A senhora não está apenas bonita..." | **14,80s** | 9,07s | **−6,3s** |
| "Princesa! O Príncipe..." | 11,01s | 8,93s | −2,7s |
| "Eu sabia!" | 1,11s | 2,93s | +1,2s |

O plano da fala mais importante da cena tinha **metade** do tempo necessário --
o vídeo cortava a fala no meio. E isso **não aparece em inspeção de still**:
só a voz revela. `shot_plan --dialogue lines.json` passa a usar
`duration_sec` real + `FALA_FOLGA_S` (0,6s), e o mux centra a fala nessa folga
(0,3s de cada lado) para o corte não cair na última sílaba.

### emotion_director: o casamento por palavra-chave não alcança roteiro

`voice_library` já trazia 12 vozes emotivas com 17 takes cada, e
`synthesize_dialogue` já escolhia take POR FALA. O mecanismo estava certo; o
casamento é que falhava. Numa cena com cinco direções distintas:

    (em tom de alerta sussurrado)         -> nenhuma -> 16_neutra
    (sobressaltada, arregalando os olhos) -> nenhuma -> 16_neutra
    (com urgência cômica)                 -> nenhuma -> 16_neutra
    (com reverência e um sorriso doce)    -> nenhuma -> 16_neutra
    (empolgada, dando uma piscadinha)     -> nenhuma -> 16_neutra

Cinco de cinco em neutro. `match_emotion` compara por PALAVRA-CHAVE, e rubrica
de roteiro não usa as palavras da lista. Ampliar a lista de sinônimos seria
enxugar gelo -- a variedade da linguagem é o problema.

`script_pipeline/emotion_director.py` faz o casamento SEMÂNTICO (LLM lê a
rubrica, escolhe um dos 17 slugs), com palavra-chave como fallback e neutro
como último recurso -- emoção errada é pior que emoção ausente. Resultado:
`com_medo`, `surpresa`, `excitada`, `apaixonada`, `alegre`.

`--recast` também reescolhe a VOZ pelo descritor (o nome codifica faixa e
timbre: `F05_madura_nobre_ruth`). Gênero é regra dura -- voz masculina em
personagem feminina é erro audível imediato e o modelo escorrega nisso.

**Efeito colateral que define a ordem do pipeline**: mudar a emoção MUDA A
DURAÇÃO da fala (uma fala foi de 8,85s para 10,19s ao virar "excitada"), o que
muda a decupagem. Então:

    emoção -> TTS -> shot_plan -> render

não é preferência, é dependência.

### Lip-sync ligado

`apply_voice()` em `render_shots`: lip-sync (LatentSync, fallback Wav2Lip) nos
planos com fala, depois mux da voz. Plano de ação recebe faixa MUDA de
propósito -- o `concat` exige que todos os trechos tenham áudio, ou os que têm
somem na junção. Lip-sync que falha não derruba o plano: fica o clipe original
com a voz por cima. 5/5 aplicados no teste.

### BUG: reaproveitamento sem validação

A retomada perguntava se o clipe EXISTE, não se ele CORRESPONDE ao plano atual.
Quando o TTS trocou as durações estimadas pelas reais, cinco clipes ficaram com
a contagem de frames antiga (o pior: plano pedindo 345f, clipe com 217f) e
foram reusados assim mesmo. O mux tentou encaixar a voz numa duração que o
vídeo não tinha, o ffmpeg errou, e a montagem saiu com **47,6s de imagem contra
40s de áudio**.

**Clipe obsoleto é pior que clipe ausente: passa em silêncio.** Corrigido
comparando `nb_frames` com `shot["frames"]` antes de reusar.

Generalizando: cache de etapa cara precisa de chave que inclua o que a etapa
consome, não só o nome do arquivo. Vale para os stills também -- hoje eles são
reusados por nome, e um `storyboard_prompt` alterado não invalida o still.

## 3.26 Caminho paralelo DECLARADO, e não por acidente (2026-08-26)

Pergunta do usuário que forçou a arrumação: *"esse script seria versão de qual
ou quais scripts?"* -- e a resposta honesta era que eu tinha criado uma
duplicação sem nomeá-la.

`render_shots` sobrepunha três estágios validados de uma vez: **[5]
render_scenes**, **[6] lipsync_scenes** e **[8] assemble_final**. E o
`render_scenes` já fazia mais do que eu vinha sugerindo -- um clipe por fala,
duração vinda do TTS, storyboard como condicionamento. A diferença real é
menor e mais específica:

| | render_scenes [5] | decupagem [5-D] |
|---|---|---|
| unidade | uma FALA | um PLANO |
| still | um por CENA, igual para todos os clipes | um por PLANO, fixando o enquadramento |
| enquadramento | mesma nota de "close-up" em todo clipe (linha 466) | por plano, do `shot_plan` |
| ações intercaladas | não viram plano | viram |
| texto da fala no prompt | sim (gatilho de legenda, §3.23) | não |

### A solução: contrato de manifesto, não fork

`render_shots_stage.py` escreve `scenes/clips.json` no **mesmo formato** que
`render_scenes` escreve -- verificados os sete campos que o estágio 6 lê. O que
liga os estágios é o ARTEFATO, não o nome do módulo que o produziu.

Consequência: **[6] lipsync, [7] mix_audio, [8] assemble_final e [9]
verify_output rodam depois dele sem uma linha alterada.** Confirmado por
`git diff`: os cinco arquivos validados estão intocados.

Por isso `render_shots_stage` NÃO faz lip-sync nem concat, embora
`render_shots.py` faça para uso avulso. Duas implementações da mesma coisa é
como elas divergem.

`run_decupagem.py` orquestra, e `start_decupagem.bat` chama -- este último
difere dos outros `start_*.bat` por ser CLI, não UI numa porta: aceita o
roteiro arrastado sobre o ícone. Padrão de parada: `animatic`.

### A ordem do pipeline é dependência, não gosto

    emoção -> TTS -> story_structure -> shot_plan -> render

Trocar a emoção MUDA a duração da fala (8,85s -> 10,19s ao virar "excitada"),
que muda a decupagem. Se a decupagem viesse antes, ficaria vencida -- e foi
exatamente o que aconteceu uma vez, invalidando cinco clipes já renderizados.

## 3.27 Vozes novas: o que serve e o que não serve (2026-08-26)

Pacote trazido do notebook do usuário para `E:\Users\home\Documents\xtts_speakers`.

| pacote | veredito |
|---|---|
| **EMNS** (1205 gravações, 1 falante feminina, 20s, inglês, Apache-2.0) | **importado** como `F07_jovem_ingles_emns`, 7 takes |
| **Thorsten** (alemão, CC0, 8 arquivos) | **rejeitado** -- todos entre 1,6 e 3,1s; XTTS quer 6s+ |
| **Free Vocal Samples.zip** (50 wav) | **rejeitado** -- é sample de PRODUÇÃO MUSICAL (`Vocal Chop Loop_keyGmin_100bpm`, `pitchvox_Eb`); áudio cantado, processado e em loop não é referência de fala |

Seleção do EMNS: melhor take por emoção cruzando **intensidade** (campo `level`,
0-10) com **duração útil** (5-13s), entre 1181 gravações -- não as oito já
separadas na pasta `audio/`.

`Disgust` ficou SEM slug: o mais próximo é `desdem` ("desprezo, ironia,
superioridade"), que `Sarcastic` preenche melhor. Forçar os dois no mesmo slug
faria um sobrescrever o outro em silêncio. Está em `_nao_classificados/`.

**Medido, e é o que decide o uso:** mesma frase em português, mesma emoção, só a
referência muda --

| | duração | sílabas/s | RMS |
|---|---|---|---|
| voz existente (pt) | 8,19s | 2,44 | 0,126 |
| nova (inglês) | **11,48s** | 2,70 | 0,073 |

40% mais longa e quase metade da energia. Importa concretamente: a duração do
plano vem da duração do TTS, então essa voz alonga a montagem, e o volume baixo
destoa na mistura. O ID carrega `ingles` de propósito, porque
`emotion_director.cast_voices` lê os IDs para escolher.

`script_pipeline/import_voices.py` normaliza qualquer entrada para a convenção
(`<ID>/<NN>_<slug>.wav`), classifica por nome com sinônimos pt/en, e **não
adivinha gênero**: sem marca explícita o ID sai com `?` e para, pedindo
`--voice-id`. Voz masculina rotulada como feminina é erro que só aparece
ouvindo, e tarde.

## 3.28 Lip-sync: o conflito entre duas decisões minhas (2026-08-26)

Com o log religado (eu o havia silenciado -- ver abaixo), a falha de dois planos
ficou legível:

    Number of frames available for inference: 121
    Length of mel chunks: 107
    | Wav2Lip failed. |

O vídeo tem 121 frames e o áudio dá 107 chunks. O Wav2Lip exige que o áudio
CUBRA o vídeo -- e a fala é mais curta que o plano justamente por causa da
`FALA_FOLGA_S` de 0,6s que eu adiciono para o corte não cair na última sílaba.

**Duas decisões minhas em conflito direto.** A folga protege o corte e quebra o
lip-sync. A correção certa é sincronizar sobre o trecho falado e recolar as
pontas, em vez de mandar o plano inteiro -- muda o desenho de `apply_voice`,
não foi feito.

Estado: 8 de 10 planos com boca sincronizada; os planos 3 e 5 ficam com o clipe
original e a voz por cima (fallback previsto, melhor que clipe mudo).

### Outros três defeitos da mesma rodada

**fps.** O lip-sync devolve **25 fps**, não 24. Sem `-r` forçado, o corte por
segundos virava contagem errada: 10,375s a 25 fps são 260 frames onde o plano
pede 249.

**`tpad=stop_duration` ADICIONA a duração, não completa até ela.** Eu escrevi
como se fosse alvo. Quem define o tamanho final é o `-t`.

**Silenciei o log de FALHA.** Passei `log=lambda m: None` para `apply_lipsync`
reduzindo ruído -- e com isso "falhou" não dizia nada. Silenciar o caminho de
sucesso é razoável; o de falha não. Custou uma rodada inteira de diagnóstico às
cegas.

### Reaproveitamento: a TERCEIRA etapa cara a ganhar

Depois de stills (§3.24) e clipes (§3.25), o lip-sync também estava sendo
refeito a cada execução. O padrão que se repetiu a sessão inteira: **toda etapa
cara precisa de chave que inclua o que ela CONSOME**, ou refaz à toa, ou reusa
material vencido. Chave aqui: `(frames do clipe, tamanho do wav)`.

## 3.29 A cadeia de decupagem rodada inteira pela primeira vez (2026-08-27)

O item 0 do §7 dizia que `run_decupagem` nunca tinha rodado de ponta a ponta.
Rodou. Fecha até o animatic -- e **tudo abaixo estava no caminho**. Só o
primeiro defeito impedia a corrida de terminar; todos os outros a deixavam
terminar com sucesso aparente e o conteúdo errado, que é a falha mais cara que
existe aqui. Nenhum apareceria em teste de módulo: cada módulo isolado
funcionava, o que estava quebrado era a junção -- e três deles são o MESMO erro
em lugares diferentes (placeholder de exemplo copiado ao pé da letra, e palpite
de modelo sobrescrevendo dado lido).

### O defeito que anulava o próprio ponto de revisão

`PARADAS` listava `"animatic"` ANTES de `"stills"`, e `ate()` compara ÍNDICES.
Como `main()` executa stills antes do animatic, `--ate animatic` -- o ponto de
revisão barato que a própria documentação manda usar primeiro -- **pulava a
geração dos stills**. O animatic não achava imagem nenhuma e saía vazio.

Pior que o defeito: `run_decupagem` devolvia **0** e imprimia "parado no
rascunho: .../animatic.mp4" apontando para um arquivo que não existia. O estágio
do rascunho era `obrigatorio=False`, o que faz sentido no meio de uma corrida
até o vídeo e não faz nenhum quando o rascunho É o destino pedido. Mesma lição
da §3.28: **silenciar o caminho de sucesso é razoável; o de falha não.**

Corrigido nos dois pontos, e a lista ganhou um comentário dizendo que a ordem
dela é contrato com `main()`, não estética.

### Roteiro em prosa perdia o cenário inteiro

Prosa corrida não tem cabeçalho de cena, então `_extract_freeform_scene` grava
`location=""` e `time_of_day=""` -- não há de onde tirar. **Ninguém preenchia
depois.** O plano de ESTABELECIMENTO da cena do ponto de ônibus saiu assim:

    extreme wide establishing shot, the figure small within a vast frame.
    natural cinematic lighting, balanced composition.

Sem ponto de ônibus, sem chuva, sem dia -- e como o still vira o frame 0 do
clipe (§3.24), o filme inteiro nascia no lugar errado. O `story_structure`
sofria do mesmo: local `local_sem_nome`, sem identidade para comparar entre
cenas. Corrigido em `parse_screenplay._apply_setting`: o enriquecimento passou a
devolver `location`/`time_of_day`, aplicados **só quando o parse não os achou**
-- cabeçalho real ganha do palpite do modelo.

Medido depois do conserto, o mesmo roteiro: `location='bus stop'`,
`time_of_day='day, raining'`.

### Descritor de personagem: enredo em vez de aparência

O descritor determinístico é um recorte do texto de AÇÃO, e só devolve
aparência quando o roteiro apresenta o personagem entre parênteses. Em prosa
corrida devolve enredo puro:

    "cena - no ponto de onibus a jovem - Park Min - personagem feminina -
     aguarda o onibus - dia chuvoso - o onibus para no ponto - ela"

Isso ia colado em TODO `storyboard_prompt` e TODO `video_prompt`. E
`appearance_only()` não salva: sem parêntese e sem "wearing" ele devolve o
descritor inteiro **de propósito** -- a troca que ele assume ("perder aparência
é pior que carregar um pouco de ação") vale para uma linha de apresentação e se
inverte aqui, onde o texto é 100% ação e 0% aparência.

`cast_characters` ganhou caminho Ollama (`--engine`), o mesmo motor do resto da
cadeia, e `run_decupagem` passa a chave. O `--llm` com Gemma 3 em processo
continua existindo; ligar um segundo motor pesado para isso seria pagar de novo
o que a §3.13 já decidiu.

### Duas falhas SILENCIOSAS de LLM na mesma etapa

Ligado o caminho novo, ele devolveu **0/2 descritores** sem erro nenhum. Duas
causas empilhadas, e as duas só apareceram porque o log conta quantos achou:

1. **O modelo copiou o placeholder.** O formato anunciado era
   `{"descriptors": {"NOME": "..."}}` e o qwen3.6 devolveu literalmente a chave
   `"NOME"`, com os dois personagens espremidos numa string só. Placeholder
   dentro de exemplo de formato é ambíguo: o modelo não tem como saber que era
   para substituir. As chaves exigidas passaram a ir **explícitas na mensagem do
   usuário** -- dizer no system prompt não bastou.
2. **Caixa alta.** Corrigido o item 1, as chaves voltaram como `"PARK MIN"` --
   convenção de roteiro. `descriptors.get("Park Min")` não acha. O casamento
   agora normaliza caixa e espaço (`_match_descriptor`), nos dois motores.

A lição operacional: **etapa opcional que cai em fallback precisa dizer QUANTOS
itens ela resolveu**, ou "funcionou" e "não achou nada" ficam indistinguíveis.

### A aparência era descartada antes de chegar ao casting

Achado ao retestar com os prompts audiovisuais do usuário, que **fixam
identidade de propósito** ("Keep exactly two samurai heroes with fixed
identities. Akemi ... black ponytail, red lacquered armor, a white scarf...").
Depois de `prose_to_screenplay`, o `action_text` só tinha "Akemi desvia de dois
ataques controlados." -- rabo de cavalo, armadura e cachecol tinham sumido na
conversão. O estágio 1 jogava fora a informação mais valiosa do prompt, e
**nenhum estágio seguinte tem como recuperá-la.**

O `SYSTEM_PROMPT` do `prose_to_screenplay` ganhou a regra 4: aparência é
conteúdo, e vai no parêntese de apresentação -- que é exatamente a convenção
que `appearance_only()` minera. Resultado medido:

    AKEMI (jovem samurai, rabo de cavalo preto, armadura vermelha laqueada,
    cachecol branco, katana curta brilhante) bloqueia um golpe das sombras.

e o cast passou a sair com "Young female samurai with black ponytail, wearing
lacquered red armor and a white scarf, holding a short shiny katana."

### `appearance_only()` cortava a âncora de identidade

Com descritor bom na mão, o prompt ainda recebia só metade dele: sem parêntese,
a função devolvia **apenas** a oração do "wearing", descartando tudo que vinha
antes -- idade, gênero e cabelo. "Young female samurai with black ponytail,
wearing..." virava "wearing...". Justamente as âncoras que a §3.24 mostrou serem
o que impede a personagem de virar outra a cada plano.

O corte só faz sentido quando existe um parêntese, porque aí o que vem antes é
nome + ação. Sem ele, o que vem antes é o resto da aparência. Corrigido: o
prefixo entra quando não há parêntese, e o comportamento com parêntese -- o caso
que motivou a função -- fica idêntico.

### §7 item 3 fechado: over-the-shoulder e estabelecimento

**`ots`** dizia "the listener's shoulder in the foreground": nomeava o OMBRO e
nunca o ROSTO, então o FLUX escolhia -- e escolheu a nuca. Os dois papéis agora
estão nomeados: o texto do enquadramento descreve quem está **de costas** em
primeiro plano, e `SUBJECT_HINTS["ots"]` diz que o rosto visível e em foco é o do
`subject`. Um `ots` sem sujeito cai para `medium` -- sem rosto para enquadrar,
sobraria só a nuca de novo.

**Estabelecimento** virou regra própria em `plan_all`: a primeira cena de cada
LOCAL ganha um wide do lugar, **independente do estilo**, e só quando a cobertura
ainda não abriu o quadro sozinha. É uma vez por local, não por cena -- medido
com três cenas em dois locais, o intimista ganhou wide nas cenas 1 e 3 e não na
2, que repete o local da 1. Depende do `location` estar preenchido, que é o que
o conserto do cenário lá em cima destravou.

### O MEIO da obra não tinha canal -- agora tem

Os dois prompts do reteste começam dizendo o MEIO ("polished hand-drawn cel
animation with sharp ink lines" / "whimsical family-friendly 3D storybook
animation"), e a cadeia não tinha onde guardar isso: o `look` do prompt vinha só
do perfil de estilo (`classico` = "natural cinematic lighting, balanced
composition"), e o meio se perdia na conversão junto com a aparência.

O preço era maior do que "não obedeceu": **os planos divergiam entre si.** No
mesmo roteiro e na mesma passada, o plano 1 saiu fotorreal e o plano 2 saiu
ilustrado. A referência por personagem da §3.24 ancora a IDENTIDADE, não o
acabamento -- então duas imagens do mesmo filme não pareciam do mesmo filme. E o
caso mais duro foi o infantil: "3D storybook animation inspirada em Alice"
produziu **a fotografia de uma mulher adulta de avental num parque real**.

O canal tem três peças, e a ordem entre elas é o ponto:

1. `prose_to_screenplay` escreve uma linha `ESTILO VISUAL: <termos em inglês>`
   antes do cabeçalho, quando o texto de origem diz o meio.
2. `parse_screenplay.extract_art_direction()` **tira essa linha do texto** --
   deixada lá viraria linha de ação, e o acabamento entraria no prompt de VÍDEO
   de algum plano -- e a grava em `Scene.art_direction`, em todas as cenas.
3. `shot_plan._look_e_interior()` põe a direção de arte **antes** do `look` do
   perfil: o meio manda, o perfil dá só luz e composição.

O enriquecimento também sabe devolver `art_direction`, mas só preenche o que
estiver vazio. **Isso importou na prática**: o texto dizia "polished hand-drawn
cel animation, sharp ink lines" e o modelo palpitou "3D cinematic animation,
high contrast lighting". Desenho à mão e render 3D não são a mesma obra -- e sem
a guarda o palpite ganhava, porque o enriquecimento roda depois. Mesma regra do
cabeçalho de cena: **dado lido ganha de palpite.**

Medido depois do conserto: o plano de estabelecimento do templo saiu em cel
animation com linha de tinta e paleta chapada, não mais em fotografia.

### Mais dois placeholders copiados ao pé da letra

O mesmo erro do `"NOME"` do casting apareceu em outro lugar: o molde de saída do
`prose_to_screenplay` dizia `INT. LOCAL - MOMENTO`, e o cabeçalho gerado saiu
`INT. COLORFUL CURVED HALLWAY - MOMENTO`. A palavra "MOMENTO" atravessou o parse
e chegou ao prompt do still como se fosse o período do dia.

O padrão vale como regra: **placeholder dentro de exemplo de formato é ambíguo
para o modelo.** Ou o exemplo é concreto e a instrução diz que aqueles são
valores de exemplo, ou o campo é descrito por extenso e a palavra proibida é
dita como proibida. Os dois casos desta sessão foram corrigidos do segundo jeito.

### INT./EXT. não chegava ao prompt

"CORREDOR CURVO COLORIDO" sem o `INT.` do cabeçalho virou uma estrutura pintada
**ao ar livre** no still. O marcador de interior/exterior estava no cabeçalho e o
prompt não o carregava -- e "corredor" sozinho não obriga o modelo a ficar dentro
de nada. `_storyboard_prompt` passou a receber `interior`/`exterior`, derivado do
cabeçalho da cena.

### Custo medido, e a parede de VRAM de novo

Still de FLUX a 960x544 (gerado em 1920x1088): **~1 min por plano** com o
ComfyUI já quente, mais ~4 min de carga na primeira chamada. Nove planos = ~10
min. O animatic em si é montado em segundos -- a frase "custa segundos" do §7
valia para a montagem, não para os stills que ela consome.

E a armadilha da §3.24 reapareceu inteira: num dos lotes o ComfyUI subiu a
24,0 GB de 24,5 e ficou **100% de uso sem progredir** por mais de oito minutos
num still que antes levava um. Reiniciar o servidor limpo resolveu. A regra
operacional que sai daqui: **rode primeiro tudo que usa o Ollama, descarregue o
modelo (`keep_alive: 0`), e só então deixe o ComfyUI subir.** Foi o que fez o
lote seguinte passar sem travar.

## 3.30 SD 3.5 como motor de still, e o `--cache-none` que nunca esteve ligado (2026-08-27)

### O motor de imagem virou escolha, com padroes proprios

`--image-engine {flux,sd35,sdxl}` em `run_decupagem`, `render_shots_stage` e
`render_shots`. O que estava espalhado -- nome de checkpoint, encoders, passos,
CFG, guidance -- virou uma tabela so, `IMAGE_ENGINES` em `generate_storyboards`.

Isso consertou um defeito latente no caminho: **`render_shots` fixava `steps=8`
no codigo**. Oito passos e numero de modelo DESTILADO por guidance; apontar esse
render para qualquer modelo com CFG real devolveria imagem crua, sem aviso. Agora
os passos vem do motor, e `--steps`/`--cfg` explicitos ganham.

`_still_key` passou a incluir o CHECKPOINT. Sem isso, trocar de motor reusaria em
silencio o still do outro -- a mesma classe de defeito que a chave existe para
impedir. O preco e que a troca refaz os stills, que e o comportamento certo.

### Medido na 3090, still 1920x1088, mesmo prompt

| | FLUX.2 Klein 9B fp8 | SD 3.5 Medium |
|---|---|---|
| pesos + encoder | 9,4 GB + Qwen3-8B | 5,1 GB + clip_g/clip_l/t5 |
| pico de VRAM | **~24 GB** (teto da placa) | **~12 GB** |
| carga fria | ~4 min | **~1 min** |
| still com modelo quente | ~60 s (8 passos) | **34-36 s** (28 passos) |
| encalha depois de ~6 stills | **sim** (§3.29) | nao observado |

O SD 3.5 ganha nos quatro numeros, inclusive na amostragem -- apesar de precisar
de 28 passos contra 8, porque nao e destilado.

### E perde no que o still existe para fazer

O still nao e ilustracao: ele fixa o FRAME 0 do clipe, e o que ele carrega e
enquadramento, angulo e lado de tela (§3.21, §3.23). Nisso o SD 3.5 e
visivelmente pior. No mesmo prompt de "medium shot, framed from the waist up (...)
positioned on the LEFT of the frame", ele devolveu a personagem em retrato
**a direita**, contra ceu vazio, com o templo em ruinas reduzido a um detalhe ao
fundo -- ou seja, errou o enquadramento, o lado de tela e o cenario, tres dos
quatro campos que a decupagem decide. O meio pedido (cel animation) ele acertou.

E ele **nao tem imagem de referencia**: o `character_flux_reference.json` e um
grafo do FLUX, e a ancora de identidade entre planos (§3.24) so existe la.

Por isso o padrao continua `flux`. O `sd35` serve para: iterar decupagem sem
esperar 4 min de carga por lote, rodar em maquina com menos VRAM, e como saida
quando o FLUX encalhar.

### O prompt negativo brigava com a direcao de arte

`NEGATIVE_PROMPT_DEFAULT` proibia "cartoon, anime, manga, illustration, drawing,
painting, 3d render" -- escrito em 2026-08-10 para impedir que o storyboard
saisse como ilustracao chapada em vez de material filmado. Depois da §3.29 a
cadeia passou a pedir explicitamente "polished hand-drawn cel animation", e as
duas metades do prompt passaram a se contradizer.

O positivo venceu no teste, mas manter isso e pedir resultado instavel de graca.
Agora o negativo tem duas partes: a base (texto, marca d'agua, membro a mais,
balao de fala) vale sempre; o reforco de fotorrealismo entra **so quando o
roteiro nao declara meio proprio** (`negative_prompt_for(art_directed=...)`).

### O `--cache-none` não estava ligado -- e é ele que faz o vídeo caber

Havia DOIS lançadores do ComfyUI com flags diferentes:

- `ltx25_backend.ensure_server` -> **com** `--cache-none`
- `generate_storyboards.ensure_comfyui_running` -> **sem**

E os dois voltam cedo quando a porta já responde. Na cadeia de decupagem quem
sobe o servidor é SEMPRE o estágio dos stills, que roda antes do de vídeo --
então o estágio de vídeo herdava um servidor **sem** `--cache-none`, e a flag que
o `ltx25_backend` acha que passa nunca chegava a valer ali. A política de todo
mundo era decidida por quem chegou primeiro na porta.

O §7 item 1 queria "testar sem `--cache-none`" algo que já rodava sem. Testado o
contrário, a flag mostrou-se **necessária**, não custosa. MEDIDO 2026-08-27,
mesmo clipe de 145 frames a 960x544, servidor limpo nos dois casos:

| | resultado |
|---|---|
| **sem** `--cache-none` | encalha a **24,2 GB** de 24,5, 100% de uso, sem sair da carga do encoder |
| **com** `--cache-none` | passa da carga com **15,9 GB** e chega a amostrar |

O encoder do LTX 2.5 tem 25 GB e o transformer 40 GB, numa placa de 24,5. Sem a
flag o ComfyUI ainda segura resultados intermediários e não sobra espaço para a
troca entre os dois.

Corrigido em três níveis:

1. `comfy_launch_args()` monta o comando num lugar só e o escreve no log, para a
   divergência ser escolhida em vez de sorteada.
2. `render_shots.render()` liga `LTX_COMFY_CACHE_NONE=1` sempre que a passada
   NÃO é de stills. Não é preferência: é o que faz a passada de vídeo caber.
3. `LTX_COMFY_CACHE_NONE` no ambiente decide quando ninguém declara (padrão 0,
   que é o histórico do storyboard).

**Limite honesto**: com a flag, o clipe de 145 frames passou da carga mas depois
encalhou no decode tiled do VAE, também a 24 GB. Ou seja, a flag move a parede
mas não a remove -- a 960x544 esta placa aguenta o clipe de 73 frames (511 s) e
não aguenta o de 145. Isso, e não o cache, é o que explica os 0,19 frames/s da
§3.24.


### O MAIOR ganho de carregamento da sessao: reiniciar entre as passadas

A §3.24 resolveu o recarregamento POR PLANO com duas passadas -- todos os stills
com uma carga do FLUX, todos os vídeos com uma carga do LTX. O que ela **não**
resolveu é a troca de cenário ENTRE as duas passadas, que é exatamente o caso que
a própria §3.24 registra como fatal: *"ele decide a estratégia na carga do
modelo, não continuamente"*.

Na cadeia de decupagem o servidor é subido pelo estágio dos stills e fica com o
alocador moldado para o FLUX (9 GB de pesos + encoder Qwen3-8B). O estágio de
vídeo herda esse mesmo processo e pede o LTX 2.5 -- 40 GB de transformer + 25 GB
de encoder num cartão de 24 GB. Ele não se recupera.

**MEDIDO 2026-08-27, o MESMO clipe de 73 frames, mesmo prompt, mesmo still:**

| servidor | tempo |
|---|---|
| herdado do estágio de stills | **55 min sem terminar** (interrompido) |
| reiniciado limpo antes do vídeo | **511 s (8,5 min)** |

Um boot custa ~1 min. Não dar o boot custou quase uma hora, e o timeout de 2400 s
do `render_shots` nem chegou a disparar dentro da janela observada.

`stop_comfyui()` em `generate_storyboards` derruba quem estiver na porta
(`netstat`+`taskkill`, o mesmo que o `start_comfyui_ltx.bat` já fazia -- sem
dependência nova), e `run_decupagem` o chama **antes do estágio de vídeo**. O
`render()` sobe o servidor de novo sozinho.

### O custo fixo por clipe: o encoder é recarregado sempre

Com o log do servidor de volta, dá para ver o que a §3.24 atribuiu ao
`--cache-none`. Ela estimava ~350 s fixos por chamada e culpava a flag. **A flag
não estava ligada** (ver acima), e mesmo assim o custo fixo existe -- por outro
motivo, visível no log:

    Requested to load LTXAVTEModel_
    Model LTXAVTEModel_ prepared for dynamic VRAM loading. 24998MB Staged.

O encoder de texto do LTX 2.5 tem **25 GB** e o transformer tem **40 GB**, numa
placa de **24,5 GB**. Não cabem juntos, então o ComfyUI despeja um para carregar
o outro e **recarrega o encoder a cada clipe** -- medido: 2 clipes, 2
`Requested to load LTXAVTEModel_`. O `--cache-none` ajuda a caber, mas não
elimina a recarga: os dois modelos não coexistem em 24,5 GB de jeito nenhum.

Isso muda o que vale a pena tentar. Desligar flag não resolve. O que resolveria é
**codificar todos os prompts de uma vez e só então amostrar todos os clipes** --
uma terceira passada, do mesmo tipo que a §3.24 introduziu para separar FLUX de
LTX, agora separando encoder de transformer dentro do próprio LTX. Exige mexer no
grafo em `ltx25_backend`, não é mudança de flag.

### O log do servidor estava sendo jogado fora

`ensure_comfyui_running` abria o ComfyUI com `stdout=DEVNULL`. Como na cadeia de
decupagem é sempre ele quem sobe o servidor, isso apagava também o log do
**estágio de vídeo**, que roda no mesmo processo. Fui ler
`logs/comfyui_ltx25.log` para separar carga de amostragem e o arquivo estava
parado numa sessão antiga -- o servidor em uso não escrevia em lugar nenhum.

Sem esse log não dá para medir carregamento, que é justamente o que esta sessão
precisava medir. Agora vai para `logs/comfyui_storyboard.log`, e foi por ele que
os números acima puderam ser separados: **~3 min de "Model Initializing" + ~28 s
de amostragem (8 passos)** por chamada fria.

### Lip-sync: o diagnóstico da §3.28 estava ERRADO

A §3.28 leu isto no log --

    Number of frames available for inference: 121
    Length of mel chunks: 107
    | Wav2Lip failed. |

-- e concluiu que o Wav2Lip **falha** quando o áudio não cobre o vídeo, culpando
a `FALA_FOLGA_S` de 0,6 s. Testado de verdade em 2026-08-27, com um clipe de 73
frames (3,04 s) e uma fala de 2,65 s: **o Wav2Lip termina com código 0.** Ele não
falha por áudio curto.

O que ele faz é pior, porque é silencioso. `inference.py:250`:

    full_frames = full_frames[:len(mel_chunks)]

Ele **apara o vídeo** ao número de chunks de mel. Medido no mesmo clipe:

| áudio | saída |
|---|---|
| sem preencher (2,65 s) | **61 frames / 2,64 s** — 12 frames a menos |
| preenchido +0,05 s | 71 frames |
| preenchido +0,25 s | 76 frames — **3 a MAIS** |

Doze frames a menos que o plano pede é exatamente o defeito de clipe vencido da
§3.24: passa adiante sem avisar, e o mux depois encaixa a voz numa duração que o
vídeo não tem.

E o excesso também é defeito: sobrando chunk, o `datagen` volta ao frame 0 e a
saída repete o começo do plano.

**A causa real do "Wav2Lip failed"** apareceu ao rodar o mesmo teste num plano
GERAL: `ValueError: Face not detected! Ensure the video contains a face in all
the frames.` Num extreme wide a personagem não tem rosto detectável. É a
explicação mais provável para os 2 de 5 planos que a §3.28 registrou — e ela
mesma diz que o log de falha estava silenciado na época, então a mensagem nunca
foi lida.

### O conserto: preencher com folga e aparar por CONTAGEM DE FRAMES

Duas funções em `tensorxx_ge/lipsync.py`, dentro do `run_wav2lip`, valendo para
os dois chamadores (estágio [6] `lipsync_scenes` e `apply_voice`):

- `pad_audio_to_cover()` acrescenta silêncio ao fim. A margem sai do próprio
  Wav2Lip: `mel_step_size = 16` sobre mel de 80 Hz, ou seja o último chunk pede
  0,2 s além do último frame. `apad` sozinho preenche para sempre; quem define o
  tamanho é o `-t`.
- `trim_to_frames()` devolve a saída ao número de frames da ENTRADA. Por
  contagem e não por duração: `-t 3.042` num vídeo de 24 fps devolveu 74 frames
  em vez de 73, porque o corte por tempo inclui o frame que começa no instante do
  corte.

Não dá para acertar o número de chunks na bala — sai de uma divisão inteira lá
dentro. Então a margem é folgada de propósito e quem garante o tamanho é o corte.
Verificado: **73 frames entram, 73 frames saem**, com a boca sincronizada.

O que **não** foi verificado: se isso recupera os 2 planos da §3.28. Se a causa
deles era rosto não detectado, nenhum ajuste de áudio resolve — o caminho ali é
não pedir lip-sync em plano sem rosto, o que a decupagem já sabe (`framing`).


## 3.31 Por que o 2.3 renderiza e o 2.5 encalha (2026-08-27)

Pergunta do usuário, e ela tem resposta objetiva no código. Não é o modelo ser
maior; é que **uma rota declara o orçamento de memória e a outra não**.

| | 2.3 (`ltx_pipelines`) | 2.5 (ComfyUI) |
|---|---|---|
| checkpoint | `ltx-2.3-22b-distilled-**fp8**`, 31,7 GB | `ltx-2.5-22b-distilled-transformer-**bf16**`, 40 GB |
| quantização | `--quantization fp8-cast` | `UNETLoader[..., **'default'**]` — nenhuma |
| encoder | Gemma 3, roda antes e **é liberado** | Gemma4 25 GB, **recarregado a cada clipe** |
| orçamento | explícito, via accelerate: `LTX_TRANSFORMER_GPU_MEMORY=12GiB`, `CPU=34GiB` | automático; o ComfyUI decide na carga e não revisa |

O `CLAUDE.md` já registrava a metade que importa do lado 2.3: **`--quantization
fp8-cast` é obrigatório para haver offload** -- `ModelLedger.transformer()` só
passa `max_memory` quando há quantização, e sem ela faz `.to(device)` com os
31 GB inteiros. Ou seja, o 2.3 não renderiza por ser leve. Renderiza porque
alguém escreveu quanto ele pode usar em cada lugar.

O 2.5 entrega 40 GB de transformer mais 25 GB de encoder, ambos bf16, numa placa
de 24,5 GB, e confia no gerenciador automático -- que, como a §3.24 já
documentava, **decide a estratégia na carga do modelo e não se recupera depois**.

### Os dois botões que o workflow 2.5 nunca apertou

Lendo os nós de carga do
`ComfyUI-LTXVideo/example_workflows/2.5/LTX-2.5_T2V_I2V_Single_Stage_Distilled.json`:

    UNETLoader  ['ltx-2.5-22b-distilled-transformer-bf16.safetensors', 'default']
    CLIPLoader  ['gemma4-12b-with-proj-ltx-2.5-bf16.safetensors', 'ltxv', 'default']

E o próprio ComfyUI oferece os dois (`nodes.py`):

- `UNETLoader.weight_dtype` aceita `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2`
  -- é o mesmo botão que o workflow de storyboard do FLUX já usa.
- `CLIPLoader.device` aceita `cpu` -- tiraria os 25 GB do encoder da disputa.

Nenhum dos dois estava sendo tocado pelo `ltx25_backend`, que só escreve
`unet_name` e `clip_name`. Agora existem como `LTX25_UNET_DTYPE` e
`LTX25_CLIP_DEVICE`, **ambos com padrão `default`** -- a rota validada não muda
sem alguém pedir.

### Medido: ajuda, mas NÃO derruba a parede dos 145 frames

Com `LTX25_UNET_DTYPE=fp8_e4m3fn` e `LTX25_CLIP_DEVICE=cpu`, no clipe de 145
frames a 960x544 que não fechava:

- durante a codificação de texto a VRAM ficou em **10,9 GB** (contra ~24 GB), com
  a GPU a 3% -- o encoder saiu mesmo da placa;
- mas na amostragem ela voltou a **23,7 GB / 100%** e o clipe seguiu sem fechar
  por 13 min, quando o de 73 frames fecha em 511 s.

Duas conclusões honestas:

1. O gargalo dos 145 frames **não é o encoder nem a precisão dos pesos** -- é o
   tamanho do latente na amostragem e no decode tiled do VAE. Baixar peso ajuda a
   caber, não a processar mais frames.
2. `fp8_e4m3fn` sobre um arquivo **bf16** não reduz leitura de disco: o ComfyUI
   lê os 40 GB e converte na carga. Só o residente encolhe. Para ganhar na carga
   seria preciso um checkpoint fp8 de verdade, que não existe instalado (o
   `nvfp4` exige Blackwell e o `comfy-int8-convrot` não foi baixado -- §3, item 1).

O caminho que sobra para clipe longo continua sendo dividir o plano, baixar a
resolução da passada de vídeo, ou `LTX25_TWO_STAGE=1`. Nenhum testado.

### Curva de capacidade, atualizada com a cena da câmara imperial (2026-08-27)

| frames | 960x544 | resultado |
|---|---|---|
| 73 | Alice, corredor | **fecha em 511 s** |
| 81 | Mei-Li, câmara imperial | **não fecha** -- 13 min parado em "Model Initializing" |
| 145 | Alice | não fecha |

O 81 caindo depois do 73 ter passado mostra que **o limite não é só contagem de
frames**: o still de condicionamento, o tamanho do prompt e o estado do processo
entram na conta. Ou seja, não há um número seguro para prometer -- há uma faixa
onde às vezes passa. Para produção isso equivale a dizer que a rota 2.5 **não é
confiável para clipe de diálogo nesta placa**, e a decisão de rota tem de ser
tomada antes de gerar, não descoberta no meio.

O mesmo lote mostrou o custo do lado dos STILLS: 10 planos exigiram **3
reinícios** do ComfyUI (a cadência de ~6 stills por processo da §3.29 se
confirmou), a ~82 s por still no servidor limpo.

## 3.32 O int8 do 2.5: existe, carrega, e NÃO resolve (2026-08-27)

### Não existe fp8 do LTX 2.5

Verificado na listagem do repositório oficial. `diffusion_models/` publica cinco
arquivos e **nenhum** é fp8:

| arquivo | tamanho | serve na 3090? |
|---|---|---|
| `ltx-2.5-22b-dev-transformer-bf16` | 42 GB | sim |
| `ltx-2.5-22b-distilled-transformer-bf16` | 42 GB | é o padrão |
| `ltx-2.5-22b-dev-transformer-comfy-int8-convrot` | 21,5 GB | sim |
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | 21,5 GB | **baixado** |
| `ltx-2.5-22b-distilled-transformer-nvfp4` | 18,7 GB | **não** -- NVFP4 é Blackwell, a 3090 é Ampere |

O fp8 é do 2.3 (`ltx-2.3-22b-distilled-fp8`, 31,7 GB). A §3 desta memória
descartou o `comfy-int8-convrot` como *"formato específico do ComfyUI, redundante
com o caminho que seguimos"* -- escrito em 2026-08-21, **antes** de se decidir
que o 2.5 iria pelo ComfyUI. Foi. Então o formato não era redundante: era o
certo, descartado por um motivo que deixou de valer no dia seguinte.

### Carrega nativamente, sem nó especial

O suporte é do CORE do ComfyUI, não do custom node: `comfy/ops.py` trata
`int8_tensorwise` + `convrot` como `QuantizedTensor` e chama `int8_linear` a cada
forward, com a config de quantização viajando dentro do próprio safetensors. O
`UNETLoader` comum carrega. No log:

    Found quantization metadata version 1
    Detected mixed precision quantization
    Native ops: asym_w4a8_int8, convrot_w4a4, int8_tensorwise ,
    emulated ops: mxfp8, nvfp4, float8_e4m3fn, float8_e5m2

`int8_tensorwise` e `convrot_w4a4` saem como **nativas**, e isso apesar de o
backend CUDA do `comfy-kitchen` estar DESLIGADO nesta máquina: `quant_ops.py`
exige torch cu130+ e temos cu128, e o triton fica desligado por padrão
(`--enable-triton-backend` liga, e o triton 3.4.0 está instalado). Sobra o
`eager`, que tem `dequantize_int8_convrot_weight` e `int8_linear` nativos.

Adicionado como variante `distilled-int8` em `ltx25_backend.VARIANTS`.

### E não derruba a parede

Testado no MESMO plano de 81 frames a 960x544, mesmo still, mesma seed, mudando
uma variável de cada vez:

| transformer | encoder | resultado |
|---|---|---|
| bf16 (39,1 GiB) | GPU | 13 min, não fecha |
| bf16 + `LTX25_UNET_DTYPE=fp8_e4m3fn` | CPU | 13 min, não fecha (era o de 145f) |
| **int8-convrot (20,0 GiB)** | GPU | 12 min, não fecha |
| **int8-convrot** | **CPU** | 17 min, não fecha |

Quatro configurações, quatro paredes, todas travando ANTES de qualquer passo de
amostragem -- a última linha do log é sempre o staging do encoder de 25 GB.

**Conclusão: peso não é a variável.** Quantizar o transformer de 42 para 20 GB
não muda o resultado, porque a parada acontece na fase que vem antes do
transformer. Isso encerra a linha de investigação "achar um formato menor".

### O que a medição derrubou junto

Levantei a hipótese de que a RAM do sistema fosse o limite (20 GB de transformer
+ 25 GB de encoder contra os 47,9 GB que o `CLAUDE.md` registrava). Medido: a
máquina tem **79,9 GB**, com 35,5 GB livres e o pagefile ocioso durante a
travada. A hipótese morre, e de quebra a documentação estava desatualizada --
corrigida, com a observação de que `LTX_TRANSFORMER_CPU_MEMORY=34GiB` da rota 2.3
foi dimensionado para a máquina antiga e tem folga para crescer.

### Defeito colateral consertado

`render_shots` chamava `ltx25_backend.generate(..., log_cb=lambda m: None)` --
**todo o diagnóstico do estágio de vídeo ia para o lixo**: qual variante
carregou, se o encoder foi para a CPU, tempo por fase. É a mesma armadilha da
§3.28, e foi ela que tornou esta investigação mais lenta do que precisava. O log
agora sai prefixado com `[ltx25]`.

## 3.33 `--disable-dynamic-vram`: a cena inteira, do roteiro ao filme (2026-08-27)

**A parede não era capacidade. Era o mecanismo de carga.**

Todas as travadas das §3.30-3.32 pararam na MESMA linha do log:

    Model LTXAVTEModel_ prepared for dynamic VRAM loading. 24998MB Staged.

O ComfyUI liga *dynamic VRAM* por padrão (`enables_dynamic_vram()`, ligado a
menos que `--disable-dynamic-vram`/`--highvram`/`--novram`/`--cpu`), e esse
mecanismo engasgava ao encenar o encoder de 25 GB nesta placa. Trocado pelo
offload clássico -- que usa a RAM do sistema, onde havia 35 GB livres o tempo
todo -- o mesmo plano passa.

### A/B no plano de 81 frames, mesmo still, mesma seed

| configuração | resultado |
|---|---|
| bf16, encoder na GPU | 13 min, não fecha |
| bf16 + `LTX25_UNET_DTYPE=fp8_e4m3fn`, encoder na CPU | 13 min, não fecha |
| int8-convrot, encoder na GPU | 12 min, não fecha |
| int8-convrot, encoder na CPU | 17 min, não fecha |
| **int8-convrot + `--disable-dynamic-vram`** | **151 s, fecha** |

Contra a única execução que já tinha fechado antes (73 frames em 511 s): **3,4x
mais rápido num clipe maior**.

### E a cena inteira fechou

`run_decupagem --ate final`, 10 planos, incluindo os que a §3.31 declarou
impossíveis:

| plano | frames | tempo |
|---|---|---|
| shot000 wide | 81 | 151 s |
| shot001 medium | **287** | 562 s |
| shot002 wide | 81 | 168 s |
| shot003 medium | 120 | 283 s |
| shot004 wide | 81 | 256 s |
| shot005 medium | 251 | 352 s |
| shot006 wide | 81 | 216 s |
| shot007 medium | **337** | 394 s |
| shot008 wide | 81 | 213 s |
| shot009 medium | 41 | 271 s |

10/10 clipes, lipsync 5/5 nas falas (os 5 planos abertos passam direto, como
devem), mix 10/10, montagem OK, `verify_output`: **0 erros, 0 avisos**. Filme
final de 60,2 s a 960x544.

**§7 item 0 fechado de verdade**: a cadeia roda do roteiro ao filme.

### O que isso corrige nas seções anteriores

- A "curva de capacidade" da §3.31 (73 fecha, 81 não, 145 não) **não media
  capacidade** -- media o mecanismo de carga falhando de forma não determinística.
  O 337 frames fecha em 394 s.
- A investigação do int8 (§3.32) concluiu "peso não é a variável", e isso
  continua verdade -- mas a conclusão implícita de que nada resolveria estava
  errada. A variável era outra flag.
- A §3.30 registrou `--cache-none` como necessário. Continua ligado nesta
  execução, então não foi isolado; o que mudou aqui foi só o dynamic VRAM.

### Lip-sync: 5 de 5, sem fallback

A §3.28 tinha 2 de 5 planos de diálogo caindo no fallback. Nesta corrida os 5
saíram com lip-sync aplicado. Os consertos de §3.30 (preenchimento de áudio e
corte por contagem de frames) estavam no caminho, mas o que mudou aqui foi haver
clipes de verdade para sincronizar -- antes nem chegava.

### Descartado com medição: a RTX 4070 não ajuda

Pedido do usuário, verificado antes de tentar:

- `can_device_access_peer` é **False** nos dois sentidos -- sem P2P.
- O caminho de compute do ComfyUI é de UM dispositivo: `get_torch_device()`
  devolve `torch.cuda.current_device()`.
- `comfy/multigpu.py` faz *deep clones* -- cópia inteira do modelo por GPU,
  paralelismo de dados. Um clone de 20 GB não entra nos 12 GB da 4070.
- `SelectModelDevice` existe mas aceita só `MODEL`, e o transformer não cabe.
  Os VAEs caberiam (1,4 + 0,3 GiB) e não há nó para colocá-los lá.

### Ferramenta que destravou o diagnóstico

`LTX_COMFY_EXTRA_ARGS` passa flags de memória ao ComfyUI sem editar código
(`--disable-dynamic-vram`, `--reserve-vram`, `--vram-headroom`, `--cuda-device`).
Foi assim que esta flag foi medida.

E o conserto do `log_cb=lambda m: None` do `render_shots` (§3.32) pagou na hora:
foi por ver as linhas `[ltx25]` aparecerem e a linha `Staged` **sumir** que deu
para saber POR QUE tinha funcionado, em vez de só registrar que funcionou.

## 3.34 UI da decupagem, estilo por tomada e o que vale para os caminhos antigos (2026-08-27)

### Estilo por TOMADA, não só por cena

`--style-changes` aceitava só `CENA:estilo`. Uma cena de 20 tomadas não tem um
único registro — pode abrir convencional e fechar intimista a partir da décima —
e a única saída era quebrar a cena em duas, quebrando junto a continuidade que a
cena é. Agora aceita `CENA.TOMADA:estilo` também:

    "1:classico,1.10:intimista,2:tenso"

A semântica é a mesma dos dois lados, de mudança de bobina: vale dali para a
frente, **atravessando o fim da cena**, até aparecer outra marca.

`plan_scene` passou a resolver o estilo POR PLANO (cobertura, movimento, duração,
`look` e ângulo saem do perfil daquele plano). O `style` de cada plano vem do
resolvedor, não mais de uma atribuição única por cena.

**Um erro de contagem no caminho, que valeu a pena caçar**: a marca usava o
índice do ITEM, e o plano de estabelecimento é inserido na frente *depois* do
laço, deslocando tudo em um. Quem escreve "intimista da 10" conta pela tabela do
plano, então a marca tem de casar com o número de lá. Dá para saber o
deslocamento antes do laço: em todos os ramos o item 0 recebe `cobertura[0]`,
então a decisão do estabelecimento é conhecida de antemão.

### `decupagem_ui.py` (porta 7913)

A cadeia só existia como CLI: vinte minutos olhando console. A UI mostra o log ao
vivo, os stills conforme saem, os dois prompts do plano clicado, o `cast.json`
editável, o plano em tabela e o vídeo. Ela **não reimplementa estágio nenhum** —
monta a mesma argv do `.bat` e transmite o stdout.

Três coisas que ela conserta além de mostrar imagem:

- **Retomada.** O `.bat` cria um diretório novo com timestamp a cada execução,
  então uma corrida interrompida recomeça do zero — mesmo o `run_decupagem`
  sabendo pular etapa cujo artefato já existe. Aconteceu de verdade: uma corrida
  morreu em 3 de 6 stills e não havia como continuar sem montar a linha de
  comando à mão.
- **Refazer imagens** (`apagar_stills`): apaga o still E a entrada dele no
  manifesto. Apagar só o arquivo não basta — `render_shots` guarda ali a chave
  (prompt + enquadramento + referência + motor) e, com o arquivo ausente mas a
  entrada viva, o glob de retomada pode achar um PNG antigo de outra versão do
  prompt e reusá-lo em silêncio.
- **Motor de imagem escolhível** na tela (`flux`/`sd35`/`sdxl`).

Duas armadilhas de Gradio encontradas montando: `allowed_paths` é obrigatório
para servir arquivo de `outputs/` (sem ele a galeria fica vazia sem dizer por
quê), e o `change` de um Dropdown não dispara quando o valor chega por outro
caminho — daí o botão **Carregar** explícito ao lado.

### Gradio 6: três incompatibilidades, e só uma era óbvia

A primeira aparece lendo o código; as outras duas só aparecem SUBINDO a UI --
que é por isso que o teste foi subir as onze, uma a uma, em vez de procurar
padrão no fonte.

1. **`show_api=False`** em `demo.launch()`, removido na versão 5.
   `video_doctor_ui` e `character3d_webui` nem subiam.
2. **`gr.Slider(minimum=8, maximum=8)`** em `web_ui_v2.py`. O Gradio 6 rejeita
   min == max e a UI inteira morre. Era um slider degenerado usado só para
   MOSTRAR um valor fixo ("Fixed to 8 for distilled model"); virou
   `gr.Number(interactive=False)`, que diz a mesma coisa sem fingir que existe
   uma faixa para arrastar.
3. **`theme=` / `css=` no construtor do `gr.Blocks`**, movidos para `launch()`.
   Este é o mais traiçoeiro: **não é fatal**. Vira aviso, a UI sobe, e o tema é
   descartado em silêncio. Seis arquivos estavam nessa situação -- todos
   "funcionando", todos sem o tema que alguém escolheu.

**Todas as nove UIs principais foram subidas uma a uma para verificar**, não
lidas. Todas sobem. Nenhuma outra API removida está em uso (`gr.update` continua
válido). E o teste corrigiu a tabela de portas do `CLAUDE.md`: o screenplay é
**7810**, não 7910, e `web_ui_v4`/`film_maker_ui_v4` não definem porta — caem no
7860 padrão do Gradio, e por isso não sobem as duas ao mesmo tempo.

### O doctor em todas as UIs de vídeo

Ele estava em 5 de 13. Para plugá-lo nas outras foi preciso mexer no próprio
módulo: `build_doctor_tab()` criava sempre um `gr.Tab`, e metade das UIs deste
repo é layout PLANO -- `web_ui_*`, `film_maker_*`, a decupagem. Uma aba solta ali
cria um container de aba única embaixo de todo o resto, que fica ilegível. Agora
aceita `container="accordion"`; o conteúdo é o mesmo, muda a casca.

Três ficaram de fora de propósito -- `character3d_webui`,
`character_sheet_flux_webui`, `krea2_test_ui` -- porque produzem imagem ou 3D, e
o doctor mede defeito TEMPORAL entre quadros.

**Duas armadilhas de inserção automática, ambas do mesmo tipo:** achar que a
âncora é geral quando é específica.

1. Ancorar antes de `if __name__` funciona para as UIs que montam o `Blocks` em
   nível de módulo. O `screenplay_ui` monta dentro de `main()`, e a chamada caiu
   depois do `return 0` -- fora de qualquer função, arquivo quebrado.
2. A guarda escrita CONTRA isso errou duas vezes seguidas: primeiro contou
   comentário como "código desindentado", depois contou a continuação da própria
   linha `with gr.Blocks(` multilinha. Ancorar em `as demo:` resolveu.

A guarda, ainda assim, fez o trabalho dela: **recusou em vez de estragar**.

### O que da decupagem já valia para os caminhos antigos, e o que faltava ligar

Boa parte das melhorias de hoje mora em módulos COMPARTILHADOS e já valia para o
caminho screenplay sem ninguém fazer nada:

| melhoria | onde mora | valia antes? |
|---|---|---|
| `location`/`time_of_day` em prosa | `parse_screenplay` | **sim** |
| aparência, idade e `ESTILO VISUAL` na conversão | `prose_to_screenplay` | **sim** |
| lip-sync: preenche áudio e apara por frames | `tensorxx_ge/lipsync` | **sim**, e também para as UIs de música |
| negativo que não briga com a direção de arte | `generate_storyboards` | **sim** |
| workflow SD 3.5 | `generate_storyboards` | sim, mas sem como escolher |
| `--disable-dynamic-vram`, `distilled-int8` | `ltx25_backend` | sim, mas nenhum lançador pedia |
| descritor de cast via Ollama | `cast_characters` | **não** — faltava passar `--engine` |
| direção de arte no `look` | `shot_plan` | não — só a decupagem |
| estilo por tomada, estabelecimento, `ots` | `shot_plan` | não — só a decupagem |
| reinício do ComfyUI entre passadas | `run_decupagem` | não |

Ligados agora em `screenplay_to_video`, **os dois como opção, padrão inalterado**:
`--cast-engine` (descritores via Ollama, muito mais barato que o `--llm` que
carrega o Gemma 3 em processo) e `--image-engine` (motor de imagem, que com a
flag passa a preencher checkpoint/encoders/passos/CFG de uma vez).

O que fica de fora é o que é intrinsecamente da decupagem: `shot_plan` não existe
no caminho por FALA, então estilo por tomada, estabelecimento por local e o
conserto do `ots` não têm onde se aplicar lá.

## 3.35 O int8 ATRAPALHA quando cabe: a inversão (2026-08-28)

**Recomendei `distilled-int8` como padrão ontem. Estava errado, e a medição de
hoje mostra por quê.**

Na cena Lyra/Thoren, o plano de 129 frames a 960x544 travou **duas vezes** com
`LTX25_VARIANT=distilled-int8` -- 23 min e 9,5 min presos em `0/8`, sem sair da
inicialização do amostrador, com servidor limpo nas duas. Trocado SÓ o
checkpoint para `distilled` (bf16), o mesmo plano fechou em **1087 s**.

### O log diz o mecanismo, palavra por palavra

    int8 : loaded completely;  20487.25 MB loaded, full load: True
    bf16 : loaded partially;   20713.81 MB loaded, 19337.66 MB offloaded

Os dois colocam ~20,7 GB na placa. A diferença é o que o ComfyUI faz **depois**:

- O **bf16** tem 39 GB e não cabe. O gerenciador sabe disso, declara
  `loaded partially` e mantém 19,3 GB fora da VRAM -- sobrando espaço para o
  latente, as ativações e o VAE.
- O **int8** tem 20 GB e cabe. O gerenciador declara `full load: True`, ocupa a
  placa inteira, e **não sobra espaço para o resto do grafo**. O plano curto
  (73 frames) ainda passa; o de 129 não.

Ou seja: **caber inteiro na VRAM é o problema, não a solução.** O que salva o
bf16 é exatamente a limitação dele.

### O que isso corrige nas seções anteriores

- A §3.32 concluiu "peso não é a variável". Continua verdade para **caber** --
  quantizar não removeu a parede daquele lote. Mas é a variável para **rodar**,
  e no sentido inverso do intuitivo.
- A §3.33 mediu a cena Mei-Li inteira com int8 + `--disable-dynamic-vram`, e ela
  fechou 10 planos incluindo um de 337 frames. Isso NÃO contradiz o de cima: ali
  o que destravou foi o `--disable-dynamic-vram`, e o int8 veio junto por
  acidente de configuração. A Mei-Li provavelmente teria fechado igual, ou
  melhor, em bf16 -- não foi testado.
- O `start_decupagem.bat`, a `decupagem_ui` e o `CLAUDE.md` receberam
  `distilled-int8` como padrão ontem. **Revertido para `distilled`.**

### O que fica valendo

`--disable-dynamic-vram` continua obrigatório e não está em questão: ele é o que
tirou a cena Mei-Li de "não fecha" para "fecha em 151 s".

O `distilled-int8` continua instalado e disponível. Ele serve para o caso oposto
-- placa menor, onde o bf16 nem começa. Nesta 3090 de 24,5 GB, com planos de
diálogo de 100+ frames, ele estorva.

**A regra que sai daqui, e que vale além do LTX**: num gerenciador de memória que
decide a estratégia pelo que cabe, um modelo MENOR pode ser mais lento ou
inviável, porque muda a decisão dele. Medir o modelo isolado não responde; só
medir o grafo inteiro responde.

## 3.36 Quatro defeitos que o filme montado revelou (2026-08-28)

O usuário assistiu ao filme da cena Lyra/Thoren e apontou quatro coisas. Os
quatro tinham causa em código, e **três eram a mesma classe de erro**: um
estágio calculando algo bom e o estágio seguinte jogando fora.

### 3.36.1 O doctor apontava os CORTES como defeito

Dos 10 segmentos que ele acusou no filme montado, **quatro eram as quatro
fronteiras de corte** — frames 73, 202, 305, 410, exatamente as contagens
acumuladas dos clipes. E eram os de maior pontuação: `z_glob` de 190 a 224
contra uma mediana de resíduo de 5,93.

Faz sentido que fossem: um corte é a maior mudança possível entre dois quadros.
É a definição de corte. Para três deles o veredito era `INTERPOLATE`, que numa
fronteira significa fabricar um quadro misturando o fim de um plano com o começo
do outro. Não conserta nada e destrói a montagem.

**O que separa corte de defeito não é a magnitude — é se a mudança PERSISTE.**
Defeito de um quadro volta ao normal no seguinte, então o anterior e o posterior
se parecem. Corte não volta, porque o plano novo continua. Duas distâncias
resolvem:

    d_passo = hist(t-1) -> hist(t)      grande nos dois casos
    d_salto = hist(t-1) -> hist(t+1)    grande SÓ no corte

**Uma tentativa anterior falhou, e vale registrar por quê.** O primeiro
`detect_cuts` chamava o detector de cena do ffmpeg. Ele pontuou os quatro cortes
entre **0,28 e 0,32** — abaixo do limiar usual de 0,35 — porque os planos são no
mesmo cenário, com as mesmas duas pessoas. E o `pts` que ele devolve nem cai na
grade de frames do arquivo (os frames são `258 + n*512`; ele reportou 107991),
então converter para índice errava por até 9 quadros. Fazer dentro do doctor usa
os mesmos índices do resto da análise e a mesma passada de decodificação.

Existe também `--cuts 73,202,305,410`: **quando o pipeline sabe as fronteiras,
não há o que inferir.** Ele sabe — é a contagem de frames por clipe.

### 3.36.2 A trilha sonora sumia em cada fala

Medido:

| | grave | médio | agudo | mean |
|---|---|---|---|---|
| `shot000` cru (plano de AÇÃO) | -21,8 | -42,5 | -73,3 | -21,0 |
| `shot002` cru (plano de FALA) | -16,9 | **-25,0** | **-41,1** | -14,8 |
| `shot002` saída do lip-sync | | | | -18,2 |

Três fatos encadeados:

1. O LTX 2.5 gera trilha junto com o vídeo. No plano de ação é só zumbido grave.
   No plano de fala tem médios e agudos de **voz** — o modelo gerou fala própria.
2. O Wav2Lip entrega, **por construção e não por opção**, como faixa de áudio
   exatamente o wav que dirigiu a boca. A trilha é substituída pela voz nua.
3. O plano de ação não passa pelo lip-sync, então mantém a trilha.

Resultado: o som existe, some em cada fala, volta depois.

**A causa raiz é uma divergência entre os dois caminhos.** O `render_scenes.py`
(por FALA) passa `--audio-input-path` com o wav do TTS na linha 550, e o LTX
constrói o som *em volta* da fala real. O `render_shots.py` (por PLANO, a
decupagem) não passa nada — o modelo inventa tudo, inclusive a voz. **A
decupagem perdeu um recurso que o caminho antigo tem**, e isso responde parte da
pergunta "o que da decupagem vale para os anteriores" na direção contrária: aqui
era o antigo que tinha o que faltava.

Consertado nos dois lados: `render_shots` passa `audio_conditioning` (o
`ltx25_backend.generate()` já aceitava — está descrito lá como "o equivalente
2.5 do `--audio-input-path`"), e o `mix_audio` devolve a trilha do clipe cru sob
a fala com `sidechaincompress`. Não um `amix` simples: somar em nível fixo
deixaria a fala disputando com a trilha. O sidechain faz a trilha **abaixar
enquanto há fala e voltar no silêncio**, sozinho.

Medido depois, `shot002`, janelas de 0,5 s:

| janela | só a fala | com a cama |
|---|---|---|
| 0,0 s (fala forte) | -16,0 | -15,9 |
| 1,0 s (fala forte) | -16,6 | -16,4 |
| 4,0 s (caindo) | -17,9 | -16,6 |
| 5,0 s (fala acabou) | -18,2 | -16,6 |

Sob fala forte a mistura é a fala — a trilha sai do caminho. Onde a fala cai, a
trilha sustenta.

**Um erro meu no meio disso, que a medição pegou.** A primeira versão deixou a
fala **2,5 a 2,9 dB mais BAIXA** do que entrou — o oposto do que somar uma faixa
deveria fazer. A saída do Wav2Lip é MONO a 24 kHz e a cama é estéreo a 48 kHz, e
`aformat=channel_layouts=stereo` converte pela matriz que preserva POTÊNCIA
TOTAL: espalhar um canal em dois dá 1/√2 em cada, que são exatos −3,01 dB. Certo
para ambiência, **errado para diálogo** — quer-se o mesmo sinal nos dois canais,
não a mesma potência repartida. `pan=stereo|c0=c0|c1=c0` copia com ganho 1.

### 3.36.3 A emoção era calculada e descartada

`cast.json` saía com `emotive_voice: null` nos dois personagens, e as quatro
falas — com emoções distintas **já decididas** (`com_medo`, `calma`, `com_medo`,
`espontanea_entusiasmada`) — foram todas sintetizadas com o mesmo clipe de
referência neutro.

Duas causas independentes:

**(a) O ramo do voice_map curado zerava o campo.** `cast_characters.py` tinha
`"emotive_voice": None` literal no ramo do CSV, e o ramo heurístico logo abaixo
preenchia. Ou seja: **escolher a voz à mão fazia o personagem perder as 17
tomadas emotivas** — o caminho curado, que existe para ser melhor, era
estritamente pior em atuação.

O dado para ligar os dois já estava no CSV: `voz_origem` nomeia o intérprete
("Kristin Hughes") e a biblioteca emotiva nomeia a pasta pelo mesmo primeiro
nome (`F02_jovem_expressiva_kristin`). Nove dos doze do mapa casam.

**O casamento é exato, nunca aproximado, e o motivo está no próprio CSV**: a
Clara tem `voz_origem` "Andi", que é mulher, e existe `M01_jovem_leve_andy`, que
é homem. Prefixo ou distância de edição trocariam o gênero dela em silêncio.

**(b) O XTTS não recebia prosódia nenhuma.** O `instruct` só chega ao qwen — o
XTTS não tem controle textual de emoção. Mas tem dois parâmetros que carregam
prosódia, e os dois estavam fixos: `speed=1.0` para tudo, e uma pausa entre
frases **fixa em 0,35 s**.

Essa pausa fixa é também a resposta ao quarto ponto do usuário, "o texto falado
dita em voz audível a pontuação": o worker quebra a fala em frases (para fugir
de um bug do `xtts_funcs`, ver 3.28) e emendava com o mesmo intervalo sempre.
Todo ponto, toda reticência, mesmo silêncio. **Intervalo idêntico repetido é o
que o ouvido lê como pontuação sendo anunciada.** Agora a pausa vem do sinal —
reticência 0,55 s, ponto 0,32 s, exclamação 0,18 s — escalada por um fator de
emoção.

Medido, as mesmas quatro falas:

| fala | emoção | antes | agora |
|---|---|---|---|
| line00 LYRA | com_medo | 24,02 s | **19,09 s** (−21%) |
| line01 THOREN | calma | 4,61 s | 5,49 s (+19%) |
| line02 LYRA | com_medo | 4,05 s | 3,50 s (−14%) |
| line03 THOREN | espontanea_entusiasmada | 3,86 s | **2,87 s** (−26%) |

Medo e entusiasmo encurtam, calma alonga. É prosódia, não corte.

### 3.36.4 Plano de FALA em quadro aberto

As duas falas do Thoren saíram `wide` + `static`, com o storyboard dizendo
literalmente *"extreme wide establishing shot, the figure small within a vast
frame"* — **para um plano de diálogo**. A 3.17 já mede a física: em quadro
aberto o rosto ocupa 1,6 × 1,8 células latentes e derrete. Não há prompt que
resolva; é falta de pixel.

A causa é que o ciclo de cobertura é indexado pela POSIÇÃO do plano na cena, e
em `estabelecimento` ele é `["wide", "medium"]`. As falas caem em posições de
paridade fixa, então o ciclo **nunca gira para elas** — um falante em cada dois
caía no aberto, sempre o mesmo.

**A primeira correção tirou o defeito grave e criou outro.** Trocar `wide` por
`medium` deu quatro planos médios idênticos: decupagem plana, que é a outra
metade da queixa. O conserto certo é indexar pelo **ordinal da fala**, não pela
posição, e restringir a escada aos enquadramentos que sustentam rosto — os
abertos saem, e o `insert` também, que por definição não tem rosto no quadro.

| # | tipo | antes | agora |
|---|---|---|---|
| 0 | ação | wide / static | wide / static |
| 1 | fala LYRA | medium / push | medium / push |
| 2 | fala THOREN | **wide / static** | **close / static** |
| 3 | fala LYRA | medium / push | medium / push |
| 4 | fala THOREN | **wide / static** | **close / static** |
| 5 | ação | insert | insert |

Vira plano-contraplano de verdade. E o movimento veio junto de graça: o perfil
de estilo mapeia movimento POR enquadramento, então consertar o enquadramento
consertou os planos parados.

**Quinto achado, no mesmo lugar:** a emoção já decidida por fala não chegava ao
`video_prompt`. O modelo animava uma pessoa neutra dizendo uma fala de pânico.
Agora chega, traduzida para o que se VÊ — `com_medo` não diz nada ao LTX, "eyes
wide and darting, body tensing, breathing fast" diz.

### O que muda no plano por causa disso

As falas mais curtas encolhem os planos: 1046 → **902 frames** (43,6 s → 37,6 s)
na mesma cena. O plano 1, que ficou por fazer com 593 frames, passa a **473**.

### A lição comum

Três dos cinco achados são o mesmo erro: **um estágio decide algo bom e o
seguinte descarta**. A emoção calculada e não usada, a trilha gerada e
substituída, o enquadramento escolhido por um índice que nunca girava. Nenhum
deles falha com erro — todos produzem saída plausível. É o modo de falha mais
caro que existe, porque só aparece quando alguém assiste ao filme.

## 3.37 O prompt como checklist: medir, reescrever, validar (2026-08-29)

O usuário escreveu à mão um prompt para a cena Lyra/Thoren e perguntou: o que o
dele tem que o nosso não tem, dá para medir, e dá para um script ou LLM chegar
lá sozinho? As três respostas viraram `script_pipeline/prompt_polish.py`.

### As 9 perguntas

Um prompt completo responde: (1) quem está em quadro e onde na tela; (2) o que
cada um FAZ, em cadeia conectada; (3) o que DIZ e como; (4) quando cada boca
fecha; (5) onde o clipe termina (beat final — mesmo princípio do keyframe
final); (6) onde estamos e com que luz; (7) qual o meio da obra; (8) o que a
câmera faz; (9) o que se OUVE além das falas. Medido: o prompt do pipeline
pontuava **5/8** (sem beat final, sem ambiente, sem áudio) e o do usuário
**10/11** (só a ordem difere do guia LTX: ação primeiro, áudio por último).

### A divisão de trabalho, e por que ela é assim

O SCRIPT mede (9 testes de presença, determinísticos) e valida; a LLM reescreve
(conectores, âncoras, beat final); o script decide se aceita. A LLM nunca piora
em silêncio: perdeu o sujeito, perdeu fala, score caiu → fica o original.

**A fala NÃO viaja pela LLM.** Primeira versão pedia a citação verbatim:
funcionou para fala curta e falhou 4/4 vezes para a fala longa de 4 frases — o
qwen3.6 não parafraseava, **OMITIA** a citação inteira ("...speaks to Thoren,
then closes her mouth"), mesmo com instrução estrita na retentativa. A solução é
um placeholder: a LLM posiciona o token `<<FALA>>` e o código substitui pelo
texto real. Fidelidade é trabalho de script, não de modelo. Com isso o plano 1
foi de 5/11 a 9/11 com a fala intacta.

Dois defeitos meus que o primeiro teste real (cena Mei-Li) expôs:
`<delivery verb + adverb>` do MEU template vazou impresso no prompt (agora
qualquer colchete angular remanescente reprova e retenta), e o descritor físico
sumia ("young elegant woman in mint-green silk robes" virava "is a handmaid")
porque a ficha tinha o campo e a CLI nunca o preenchia do cast.json. Corrigidos;
a segunda passada melhorou os 10 planos (4→8, 4→9, 3→6...) sem perder nada.

### O teste do prompt inteiro: T2V puro, diálogo nativo

O prompt do usuário foi gerado VERBATIM como clipe único — sem still, sem TTS,
sem lip-sync: o próprio LTX 2.5 fala as falas em pt-BR. 457 frames a 960x544,
distilled, **1355 s**. Resultado: os dois personagens com identidade estável e
lados mantidos, a sombra do dragão no arco, e a câmera fez sozinha o que a
decupagem faz com cortes — abriu em plano geral e EMPURROU até um dois-em-close
com rosto nítido. **Sem legenda queimada.** Isso refina a 3.22: a fala citada
queimou legenda em clipe CURTO; num clipe de 19 s com o modo de diálogo nativo,
não queimou. As duas observações coexistem — o custo da citação depende da
duração.

O que o clipe único NÃO dá: plano-contraplano, controle de duração por fala,
voz consistente entre cenas (cada geração inventa um timbre). A decupagem
continua sendo o caminho para filme com elenco; o T2V com diálogo nativo é o
caminho para clipe-vitrine de uma cena.
## 3.38 "O melhor de todos até agora" — por quê, e o experimento de atribuição (2026-08-29)

O usuário julgou o clipe T2V do prompt dele (3.37) melhor que tudo que a cadeia
de decupagem produziu. O problema com esse veredito não é ele estar errado — é
que o clipe muda CINCO fatores ao mesmo tempo, e sem separá-los a conclusão
"T2V contínuo ganha" viraria dogma sem causa conhecida.

### Os cinco fatores, na ordem de peso suspeitado

1. **Uma geração contínua, não cinco clipes costurados.** Identidade, luz,
   paleta e timbre consistentes POR CONSTRUÇÃO — não há fronteira onde o modelo
   possa trocar o rosto. A decupagem renasce do zero a cada clipe e a
   consistência é pedida (still + referência + descritor), não garantida.
2. **A câmera substituiu os cortes.** O push-in de geral a close resolve a
   resolução de rosto (3.17) DINAMICAMENTE — o rosto ganha pixels ao longo do
   plano, sem precisar de corte para close.
3. **Sem still a força 1.0.** O still FLUX ancora enquadramento e figurino, mas
   congela a composição inicial e briga com movimento amplo. O T2V compôs
   livre — daí a ação corporal mais elaborada.
4. **Áudio nativo.** Voz, vento, música e a acústica da câmara saem da mesma
   geração; a voz soa DENTRO do espaço. TTS+Wav2Lip cola voz seca de estúdio
   sobre o vídeo — mesmo com a cama restaurada (3.36.2), é colagem.
5. **O prompt: 10/11 contra 5/8** no checklist de 3.37.

O que o clipe único NÃO dá (e por que a decupagem continua): voz consistente
entre cenas (cada geração inventa um timbre), duração controlada por fala,
plano-contraplano.

### O experimento

Três braços, seed 77, 960x544, distilled, mesmas flags; UM fator alterado por
braço, contra o clipe de referência:

| braço | o que muda | o que mede |
|---|---|---|
| A | + still a força 1.0 — **o próprio frame 0 do clipe de referência** | só o custo da âncora dura de imagem. Usar o frame 0 do próprio clipe é o ponto do desenho: a composição inicial é IDÊNTICA, então qualquer diferença é o mecanismo de ancoragem, não a imagem |
| B | mesmo conteúdo em 2 clipes (265+193 frames) + concat, corte na troca de falante | o custo da continuidade quebrada: identidade, luz e áudio através do corte |
| C | prompt estilo pipeline antigo (fragmentos, sem falas, sem som, sem beat, 5/8) em clipe contínuo | só a qualidade do prompt |

**PARADO em 2026-08-29 a pedido do usuário. CORREÇÃO: eu disse ao usuário que
"nada foi concluído" -- errado.** O log da tarefa morta mostra o braço B1 a
813s de amostragem no momento da interrupção, o que só é possível se o braço A
já tivesse terminado antes. Conferido: `atrib_A_still.mp4` existe, 19,04s
completos, gravado às 00:32 -- o braço A **terminou e foi analisado**. B e C
não chegaram a gravar saída.

### Braço A: resultado real

Doctor: mediana de resíduo **1,87** (mais liso que a referência, 2,83, e que o
filme montado, 5,93) e só 2 segmentos pequenos (5f e 3f, ambos defeito local,
não corte). Espectro de áudio comparável ao da referência.

**Mas o estilo visual DERIVA ao longo do clipe, apesar da âncora.** O quadro
em t=1s repete fielmente a composição e o acabamento fotorreal-3D do frame de
ancoragem (idêntico ao t=1 da referência). Em t=12s e t=18s, Lyra e Thoren
aparecem num acabamento visivelmente mais "desenho animado" -- olhos maiores,
proporções mais cartunescas -- que NÃO é o estilo do frame 0. A âncora de
still a força 1.0 trava o PRIMEIRO quadro, não o clipe inteiro; o modelo pode
derivar de estilo depois, e o resíduo de fluxo (que mede movimento, não
identidade de estilo) não acusa isso -- é um eixo diferente do que o doctor
mede.

**Isso muda a leitura do fator 3 em 3.38.** Eu tinha suspeitado que a âncora
de still ajudaria consistência; o dado real é o oposto na dimensão de
ESTILO: o clipe SEM âncora (a referência) manteve um acabamento mais
uniforme do início ao fim do que o clipe COM âncora. A âncora garante
enquadramento/composição inicial, não estilo sustentado.

### O que falta

Braços B (continuidade quebrada por corte) e C (prompt fragmentado) não
rodaram. Retomar com `.venv/Scripts/python.exe -u _atribuicao_338.py` na
raiz do repo -- o script tenta regerar os TRÊS braços do zero (não há cache),
então ou aceita re-gerar A também (~15 min a mais, resultado já conhecido) ou
edita o script para pular a chamada de A antes de rodar.

## 3.39 Auditoria externa: dez pontos, dez confirmados, seis corrigidos (2026-08-29)

Um revisor externo (outro agente, sem acesso a este arquivo) leu o repositório
do zero e devolveu dez pontos de risco/divida tecnica. Conferido cada um contra
o codigo antes de agir -- nenhum foi tomado de graca:

| # | apontou | confirmado? | acao |
|---|---|---|---|
| 1 | UIs em `0.0.0.0`, sem autenticacao | **sim**, 15 arquivos vivos | **corrigido**: `127.0.0.1` por padrao, `LTX_UI_HOST` sobrescreve |
| 2 | duas UIs podem subir o ComfyUI ao mesmo tempo | **sim**, `ensure_server` so via `server_is_up()` | **corrigido**: lock entre processos (`_boot_lock`), ver abaixo |
| 3 | staging em `ComfyUI/input` nunca e limpo | **sim** | **corrigido**: `finally` remove os arquivos copiados |
| 4 | `files[0]` e fragil p/ multiplas saidas | **sim**, varredura por TODOS os nos, ordem de dict | **corrigido**: `expect_node=N_SAVE` primeiro, varredura como fallback |
| 5 | familia de UIs duplicada (`music_maker_ui*`, `web_ui*`) | **sim**, 17 arquivos vivos + backups | **avaliado, adiado** -- ver nota |
| 6 | estado global (`CURRENT_*`) em vez de objeto de job | **sim** | **avaliado, adiado** -- mesma nota |
| 7 | cancelamento inconsistente (`terminate()` vs `taskkill /T`) | **sim** | **avaliado, adiado** -- mesma nota |
| 8 | `.gitignore` nao cobre `outputs/`, `models/`, `ComfyUI/`... | **sim, e pior do que apontado** -- ver abaixo | **corrigido** |
| 9 | sem `pytest` instalado, sem suite | **sim** | **corrigido**: instalado + `tests/test_pipeline_smoke.py`, 30 casos |
| 10 | ~354 `except Exception` sem contexto | **sim** (323 medido, fora de vendors) | **avaliado, adiado** -- mesma nota |

Achado a mais que a auditoria nao citou: `daz_integration/app.py:108` tinha o
`0.0.0.0` **e** o `show_api=False` obsoleto (Gradio 6) na mesma linha -- os
dois corrigidos junto.

### #8 era maior do que "faltam diretorios no gitignore"

`git ls-files | wc -l` devolve **171**. E so o esqueleto upstream --
`packages/`, um punhado de UIs antigas, `README.md`, `pyproject.toml`. **Todo
o trabalho desta sessao e das anteriores nunca foi commitado**:
`script_pipeline/` inteiro (a cadeia de decupagem), `ltx25_backend.py`,
`video_doctor.py`, `video_doctor_ui.py`, `MEMORIAL.md`, `CLAUDE.md` -- nada
disso tem historico de git. `git status -uall` contava **20.944** entradas nao
rastreadas antes do ajuste.

A causa maior nao era a falta de regras de diretorio -- era `*.json` como
bloqueio GERAL, que escondia workflows autorais (`comfyui_workflows/*.json`,
escritos a mao nesta sessao) junto com os manifestos de execucao que
DEVERIAM ficar de fora. `.gitignore` ganhou regras por DIRETORIO
(`/outputs/`, `/logs/`, `/ComfyUI/`, `/Hunyuan3D-2/`, `/MHR/`,
`/MiniMax-H3/`, `/gemma4_env/`, `/hf-cache/`, backups `*.bak*.py`/`*.old.py`)
e uma negacao pontual para `comfyui_workflows/**/*.json` -- os manifestos
continuam de fora porque `/outputs/` ja os cobre por pasta, nao por extensao.
`git status -uall` caiu para **380**.

**Nao commitei nada.** Editar `.gitignore` e seguro e reversivel; decidir O QUE
entra no primeiro commit de um corpo de trabalho deste tamanho e uma decisao
do usuario, nao uma acao de rotina.

### #2: o lock entre processos

`ensure_server()` so perguntava "a porta responde?" -- sem exclusao, duas UIs
abertas ao mesmo tempo podem concluir juntas que nao, e cada uma tentar subir
o ComfyUI. `_boot_lock()` usa `os.open(..., O_CREAT|O_EXCL)`, atomico tambem
no Windows: so um `open()` concorrente ganha o arquivo. Double-checked
locking -- quem espera a vez reconfere `server_is_up()` antes de agir, porque
quem segurou o lock antes pode ja ter resolvido o boot. Testado com 3 threads
concorrentes: serializou sem sobreposicao, lockfile nao sobrou. Timeout de
espera de 300s: se travar, segue sem o lock -- nunca pior que o comportamento
de antes de ele existir.

### #4: qual no e a resposta

`base_api()` ja fixa `SAVE_NODE_ID` como constante; a unica mudanca foi usar
esse ID para procurar a saida ANTES de varrer todos os nos. `submit_and_wait`
ganhou `expect_node`, e a varredura antiga vira fallback com aviso no log, nao
comportamento silencioso.

### #9: a suite pegou um bug de verdade

Escrever teste para `detect_cuts` (o detector de corte da 3.36.1) achou uma
assimetria real: uma anomalia de UM QUADRO produz duas transicoes de mesma
magnitude -- entrada e saida. A checagem de persistencia (`d_salto`) rejeita a
de ENTRADA corretamente, mas a de SAIDA usa o **proprio quadro anomalo** como
ancora "antes", e comparado dali pra frente sempre parece que a mudanca
persistiu. E exatamente o perfil dos defeitos "1-2 frames, vizinhos sadios"
que o doctor ja cataloga -- um alvo real, nao hipotetico.

Corrigido com uma terceira distancia, `d_lead = hist(t-2) -> hist(t-1)`:
exige que o quadro ANTES da transicao ja fosse ele mesmo estavel, nao a cauda
de uma anomalia. Validado sem regressao contra os 4 cortes reais da cena Lyra
e contra os defeitos internos do clipe T2V do usuario (228-240 sobrevive
intacto). 30 casos, todos verdes.

Nota a parte, achada ao validar a correcao: o filme montado da Lyra teve as
posicoes dos 4 cortes **deslocarem** entre duas remontagens de hoje (73/202/
305/410 -> 73/198/297/401) e isso comeu um defeito real (189-196) que ficou
perto demais do novo corte em 198. A causa nao foi identificada -- e
candidata a investigacao, nao a correcao as pressas.

### O que foi avaliado e adiado, e por que (#5, #6, #7, #10)

Consolidar 17 UIs, trocar estado global por `Job` dataclass, e centralizar
cancelamento sao refatoracoes de VARIOS arquivos com estado (filas, threads,
componentes Gradio) e **nenhuma suite de UI para pegar regressao** -- a
suite nova cobre funcao pura, nao fluxo de Gradio. Fazer isso "de passagem"
dentro de uma sessao que ja mexeu em quatro frentes e o tipo de risco que
vale uma sessao propria, com o usuario testando cada UI manualmente depois.
Reduzir os 323 `except Exception` e a mesma categoria: uns escondem causa real,
outros sao fallback deliberado (audio, GPU), e tria-los um a um sem quebrar
comportamento existente e trabalho de auditoria, nao de correcao mecanica.

## 3.40 MiniMax H3: segunda rota de video, integrada e testada (2026-08-29)

Terceiro motor de geracao no repositorio, ao lado do LTX 2.3 (`ltx_pipelines`)
e do LTX 2.5 (`ltx25_backend.py` -> ComfyUI na porta 8188): o **MiniMax H3**,
instalacao PROPRIA de ComfyUI em `MiniMax-H3/` (venv, porta 8189,
`--cuda-device 1 --lowvram --cpu-vae`), com um workflow oficial de
referencia-para-video (`workflows/video_minimax_h3_r2v.json`).

### O que o modelo e

`MiniMaxH3ReferenceToVideo`: recebe ate 2 imagens de referencia (identidade),
um prompt de texto, e devolve video COM audio (VAE de video + VAE de audio
proprios, igual ao LTX 2.5). `length` (frames) so aceita `5 + 17k` a 24fps
-- 124 = ~5s, faixa treinada ~124-362 (~5-15s). Duas variantes de amostragem:
20 passos padrao, ou 4 passos com a LoRA `..._turbo_4step...` -- um switch
booleano no grafo liga as duas.

**O modelo espera a referencia CITADA NO TEXTO.** O proprio exemplo oficial do
workflow ensina isso: "Use <Picture 2> and <Picture 1> as reference frames
and <Audio 1> exactly as it is." -- a imagem entra tanto por input dedicado
quanto por mencao explicita no prompt. `build_prompt_with_refs()` injeta essa
citacao automaticamente.

### `minimax_h3_backend.py` (novo)

Mesmo desenho do `ltx25_backend.py` -- e as MESMAS correcoes da 3.39 desde o
primeiro commit, nao como divida futura: `_boot_lock()` proprio (path
separado, mesmo mecanismo), `expect_node=N_SAVE` na selecao de saida, limpeza
das referencias staged em `finally`. Os IDs de no (`N_PROMPT="138"`,
`N_RESOLUTION="115"`, `N_DURATION_S="132"`, `N_REF_IMAGE_0/1="137"/"139"`,
`N_SEED="129"`, `N_TURBO_SWITCH="146"`, `N_R2V="136"`, `N_SAVE="92"`) foram
lidos direto do `/object_info` do servidor rodando -- nao adivinhados dos
`widgets_values` do JSON de UI, que e exatamente a armadilha que o CLAUDE.md
ja registra para `comfy_workflow_tool.convert()` (secao 3.16).

A duracao nao e recalculada em Python: o proprio grafo tem um
`ComfyMathExpression` que arredonda segundos para o multiplo valido mais
proximo -- `build_workflow()` so alimenta `duration_seconds` no no certo e
deixa o ComfyUI fazer a conta, em vez de reimplementar a formula e arriscar
divergir dela.

### Teste real: prompt polido da decupagem, still da propria decupagem

Sem adaptar nada no roteiro -- o objetivo era testar se o prompt e a
referencia que a decupagem JA produz servem a este motor sem traducao. Usado
o `video_prompt` polido do plano 1 da cena Lyra (secao 3.37, fala de medo,
"Thoren, as runas despertaram...") e `shots/stills/shot001_medium.png` (o
still FLUX que a propria decupagem gerou) como referencia de identidade.
10s pedidos, seed 77, turbo (4 passos).

@@RESULTADO@@

### O que falta para plugar na decupagem de verdade

Isto e um BACKEND que fala com o servidor certo, testado com um caso real --
nao e ainda uma opcao `--engine minimax_h3` no `render_shots.py`. Para isso:
decidir se ele SUBSTITUI o TTS+lip-sync (o modelo ja fala a fala, como o
proprio LTX 2.5 T2V da secao 3.37) ou se recebe audio externo como o LTX faz;
mapear `framing`/`movement` do shot_plan para algo que este modelo entenda
(ele nao tem conceito de camera explicito testado -- precisa medir se ele
obedece instrucao de camera em texto, como o LTX); e decidir a convencao de 2
referencias (hoje `render_shots` so produz 1 still por plano) quando uma cena
tem 2 personagens. Nenhuma das tres foi resolvida aqui -- sao a proxima etapa,
nao um debito escondido.

## 3.41 MiniMax H3: por que travava, e o encoder na CPU (2026-08-29)

O teste da 3.40 nunca terminou -- passou de 30 minutos para um clipe de 10s em
modo turbo (4 passos), com a 3090 presa a 24,2 GB de 24,5 e 100% de uso o
tempo todo. Perguntado sobre trocar para as versões FP8/INT8 do unsloth
(https://huggingface.co/unsloth/MiniMax-H3-FP8), investiguei antes de
recomendar.

### Por que NÃO trocar de quantização

O que está instalado já é W4A8 (4 bits) via `comfy-kitchen` -- mais agressivo
que qualquer coisa naquele link. Medido pela página do HF: o encoder de texto
sozinho salta de 15,7 GB (W4A8, instalado) para **27,1 GB** em INT8 -- maior
que a placa inteira. O próprio README do unsloth admite pico fim-a-fim de
**36,97 GB**, 12,5 GB acima dos 24,5 GB desta 3090. Não existe variante de 4
bits lá. Trocar pioraria o encaixe, não melhoraria -- descartado com dado, não
com opinião.

### A causa real: dois pesos grandes disputando a mesma VRAM

`minimax_h3_ref2va_pruned-w4a8_convrot_pruned.safetensors` (UNET, 12,5 GB) e
`qwen3vl_32b_minimax_h3-w4a8_convrot.safetensors` (encoder de texto, Qwen3-VL
32B, 15,7 GB) juntos somam 28,2 GB -- acima da placa antes mesmo do latente.
`--lowvram` cobria isso trocando camada por camada entre CPU e GPU, e é
exatamente esse tipo de operação que fica ordens de magnitude mais lenta.

**A ideia que resolve: os dois pesos não precisam estar na GPU ao MESMO
TEMPO.** O encoder de texto roda UMA VEZ por geração, antes da amostragem. O
UNET roda em CADA passo (4x no modo turbo). Manter o maior peso (o encoder)
fora do caminho que se repete é a otimização óbvia.

### O conserto: `CLIPLoader(device="cpu")`

O próprio nó já expunha essa opção (visto no `/object_info`, nunca usado). Com
o encoder de texto forçado à CPU, o UNET + LoRA turbo sozinhos (~14,5 GB)
cabem folgados nos 24,5 GB -- `--lowvram` deixou de fazer sentido e foi
removido dos argumentos de boot; `--cpu-vae` ficou (decodificar é passo único,
custo baixo de manter fora da GPU, margem de segurança).

`minimax_h3_backend.py` ganhou `clip_on_cpu: bool = True` em `build_workflow`
e `generate` -- default LIGADO, porque MEDIDO ser a diferença entre "não
termina" e terminar, não uma escolha de estilo.

### Reteste

Mesmo prompt, mesma seed (77), mesma duração pedida (10s) do teste da 3.40.
Medido nos primeiros 60s: RAM subiu ~15,5 GB (bate com os 15,7 GB do encoder
indo para a CPU) enquanto a GPU ficou em ~1 GB / <20% de uso -- confirma que a
carga migrou de fato.

**Resultado: dois consertos a mais foram necessários além do CLIP na CPU, os
três juntos fecharam.**

1. **`--cpu-vae` tinha que sair também.** Com o CLIP fora da GPU, sobra
   espaço de sobra para o VAE também (~9,9 GB carregado) -- mas mantê-lo
   forçado à CPU criou um erro NOVO: `expected m1 and m2 to have the same
   dtype, but got: float != struct c10::Half`. O VAE em CPU roda em
   float32; o latente que chega da GPU está em meia-precisão. Tirado
   `--cpu-vae` dos argumentos de boot -- o VAE fica no mesmo dtype do resto
   do grafo, sem conversão.
2. **`SaveVideo` faltava o campo `format`.** `comfy_workflow_tool.convert()`
   não preenche esse widget (`COMFY_DYNAMICCOMBO_V3`) -- é a MESMA classe de
   bug que o CLAUDE.md já registra para o workflow do LTX 2.5 (§3.16).
   `build_workflow()` agora fixa `format = "auto"` explicitamente, como o
   `ltx25_backend.py` já fazia.

Com os três (`clip_on_cpu`, sem `--lowvram`, sem `--cpu-vae`, `format` fixo):
**641s total** (server boot 8,6s + ~10 min de encoder na CPU + amostragem e
decodificação na GPU), clipe de 10,1s com áudio, mesmo prompt polido e still
de referência da decupagem que o teste original usava. Contra o "não termina
em 30 min" de antes.

### Resumo da cadeia de causas, do sintoma ao conserto

| sintoma | causa | conserto |
|---|---|---|
| travado em `0/4`, 30+ min | kernel rápido do comfy-kitchen desligado (torch < cu130) | atualizar torch para 2.9.0+cu130 (pareado: torchvision 0.24.0, torchaudio 2.9.0 -- desencontro de versão entre os três quebrou o import do torchaudio na primeira tentativa) |
| VRAM em 24,2 GB mesmo com UNET sozinho cabendo | CLIP de 32B disputando GPU com o UNET | `CLIPLoader(device="cpu")`, `--lowvram` removido |
| `float != Half` no VAEDecode | VAE forçado à CPU em float32, latente da GPU em meia-precisão | remover `--cpu-vae` também -- já havia VRAM de sobra |
| `SaveVideo.execute() missing... 'format'` | bug conhecido do conversor de workflow com `COMFY_DYNAMICCOMBO_V3` | fixar `format = "auto"` explicitamente |

Nenhum destes seria óbvio isolado; cada um só apareceu depois do anterior ser
corrigido -- a mensagem de erro de cada estágio escondia a do próximo.

### De brinde: um arquivo órfão de 4,6 GB

`minimax_h3_audio_vae_fp32.safetensors.wrong-video-partial` era resto de um
download que falhou às 07:40 de 21/08 (2 min antes do VAE de áudio real
terminar certo, e 15 min antes do VAE de vídeo ser baixado de novo com
sucesso). Não referenciado em lugar nenhum -- confirmado antes de remover, a
pedido do usuário.

## 3.42 comfy-kitchen já está no LTX -- e pode explicar a §3.35 (2026-08-29)

Perguntado se o `comfy-kitchen` (o mecanismo por trás da lentidão do MiniMax
H3, §3.41) poderia se aplicar ao LTX, e se precisaria de um checkpoint W4A8
novo. As duas respostas vieram direto do disco, sem precisar baixar nada:

- **`comfy-kitchen` 0.2.31 já está instalado no `.venv` principal do LTX**
  (mesma versão do MiniMax H3, veio como dependência do ComfyUI vendorizado).
- **Já existe um checkpoint no formato certo**:
  `models/2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-
  comfy-int8-convrot.safetensors` (21,5 GB) -- é o próprio `distilled-int8`
  que a §3.35 mediu e descartou. Não precisa de W4A8 novo; o INT8-convrot que
  já está aqui é um dos formatos que o `comfy-kitchen` acelera (a lista de
  capacidades do backend `cuda` inclui `dequantize_int8_convrot_weight` e
  `w4a8_int8_linear`).
- **O mesmo bloqueio do MiniMax se aplica**: o `.venv` do LTX tem torch
  **2.8.0+cu128** -- abaixo do cu130 que os kernels rápidos exigem.

### Isso pode reabrir a conclusão da §3.35

A §3.35 mediu `distilled-int8` **travando em `0/8`, sem sair da inicialização
do amostrador** -- 23 min e depois 9,5 min, duas vezes. É a MESMA assinatura
exata do que travou o MiniMax H3 (§3.41): passos parados em `0/N` por muito
tempo, não devagar-mas-progredindo. A explicação de então ("cabe inteiro,
zero folga para o resto do grafo") é plausível e pode ainda ser parte da
causa -- mas agora há uma segunda explicação testável, e as duas produzem o
MESMO sintoma: kernel rápido desligado por falta de cu130 força o backend
`eager` (dequantiza cada peso na hora, sem kernel fundido), que é lento por
natureza, independente de a VRAM caber ou não.

**Não retestei isto ainda.** Atualizar o torch do `.venv` PRINCIPAL do LTX é
uma mudança de risco maior que a do MiniMax H3 -- aquele venv é isolado e serve
só ao MiniMax; este é compartilhado por TODAS as rotas 2.3/2.5 e tem
compatibilidade documentada com numpy 1.26.4 (CLAUDE.md). Pedir confirmação
antes de mexer.

Se confirmado que o cu130 também destrava o `distilled-int8` no LTX, a
implicação é grande: um checkpoint 21,5 GB (contra 39 GB do bf16) rodando em
velocidade normal cabe com folga real na 3090, sobrando espaço para planos mais
longos sem precisar do jogo de encaixe que a §3.35 documentou.

## 3.43 O torch novo do LTX principal, e um gap real nos lançadores (2026-08-29)

Atualizado o `.venv` principal para torch 2.9.0+cu130 (mesmo par validado no
MiniMax H3, §3.41-3.42: torchvision 0.24.0+cu130, torchaudio 2.9.0+cu130,
xformers 0.0.33 -- os quatro pareados, senão quebra igual ao torchaudio
quebrou na primeira tentativa do MiniMax). `torchcodec` ficou quebrado e foi
deixado assim -- `Required-by:` vazio, nenhum script deste repo o importa,
só aparece em código do `gemma4_env` (venv separado) como acelerador
OPCIONAL do torchvision/transformers.

Todos os módulos próprios (`ltx25_backend`, `minimax_h3_backend`,
`video_doctor`, `script_pipeline.*`, `ltx_core`, `ltx_pipelines`) importam
limpos com o torch novo.

### O smoke test achou um gap real, não uma regressão

Um teste mínimo (25 frames, 768x512) ficou **20 minutos travado**, timeout.
Susto -- parecia o torch quebrando alguma coisa. Não era: o log do ComfyUI
mostrava exatamente `"Model LTXAVTEModel_ prepared for dynamic VRAM
loading. 24998MB Staged."` e parava aí -- a MESMA assinatura que a §3.33 já
tinha medido e resolvido com `--disable-dynamic-vram`. Só que **esse smoke
test não estava passando a flag.**

Conferido então se os lançadores de produção passam: **nenhum dos 4 `.bat`
da rota LTX 2.5** (`start_music_video_v2_25.bat`, `start_music_video_v3_25.bat`,
`start_storyplay25.bat`, `start_screenplay_25.bat`) definia
`LTX_COMFY_EXTRA_ARGS=--disable-dynamic-vram`. Não há default embutido no
código -- `generate_storyboards.py:343` só lê da variável de ambiente, nunca
fixa um valor. Ou seja: **qualquer geração de vídeo real por essas 4 UIs,
até agora, pode ter engasgado no mesmo lugar**, a menos que alguém já
estivesse setando essa variável manualmente fora do `.bat`.

Corrigido nos 4 lançadores -- a linha entra logo depois de
`LTX25_VARIANT=distilled`, com o comentário explicando o porquê e a medição.
Reteste do smoke test com a flag: progride normalmente (sem travar em
"Staged"), confirmando o diagnóstico.

**Isto muda a leitura de "poucos itens de §7 foram testados de verdade"**:
não era só falta de tempo -- pode ter sido este gap silenciando toda
geração de vídeo real nessas 4 rotas.

## 3.44 As 7 UIs de produção, geração representativa por rota (2026-08-29)

Pedido do usuário: rodar geração REPRESENTATIVA (música/roteiro inteiro, não
um clipe mínimo) nas 7 rotas atrás dos lançadores `start_*.bat`. Sem Claude
in Chrome conectado nesta sessão para clicar na UI de verdade, as funções de
geração foram chamadas direto em Python, com os MESMOS valores padrão que
cada UI usa (lidos componente por componente do código-fonte de cada uma) --
mesmo pipeline de geração, só sem a casca do navegador.

| # | rota | música/roteiro | resultado | tempo |
|---|---|---|---|---|
| 1 | `music_maker_ui_v2_25` (LTX 2.5) | jackson5.mp3, 15s | **OK** -- 2 cenas, 3840x2112, 14,7s final | 2180s (36,3 min) |
| 2 | `music_maker_ui_v3_25` (LTX 2.5 + LatentSync) | dhoom2.mp3, 30s | **OK** -- 4 cenas, 3840x2112, 29,8s final | 4640s (77,3 min) |
| 3 | `storyplay25` (LTX 2.5) | roteiro Lyra/Thoren | **OK** -- parse+4 quadros+video c/ keyframes, 11,4s | 1195s (~20 min) |
| 4 | `screenplay_25` (caminho validado, render 2.5) | roteiro Lyra/Thoren | **OK** -- 9 estagios, 9 clipes, verify 0 erros, 38,2s | ~62,5 min |
| 5 | `screenplay_ui` (caminho validado, render 2.3 nativo) | roteiro Lyra/Thoren | em andamento/resultado abaixo | -- |
| 6 | `music_maker_ui_v2` (2.3 nativo) | -- | pendente | -- |
| 7 | `music_maker_ui_v3` (2.3 nativo + LatentSync) | -- | pendente | -- |

Achado no meio do caminho: rodar `screenplay_25` com `CUDA_VISIBLE_DEVICES=1`
setado quebrou o `gemma4_worker.py` -- exatamente o aviso que o proprio
`start_storyplay25.bat` ja documenta em comentario (o worker faz
`torch.cuda.set_device(1)` esperando as DUAS placas visiveis; fixar a
variavel deixa a 3090 como unica, ela vira indice 0, e "invalid device
ordinal"). Erro meu, nao do codigo -- corrigido removendo a variavel, como
os `.bat` desta cadeia ja fazem de proposito.

As duas primeiras (LTX 2.5) fecharam sem nenhum ajuste além do que já tinha
sido corrigido em 3.43 (`--disable-dynamic-vram`). Confirma que o gap dos
lançadores era real e que, corrigido, as rotas 2.5 funcionam ponta a ponta:
fatiamento de áudio (com Demucs + ASR para timing de letra), geração LTX por
cena, concatenação, recolocação da música original sem cortes, upscale 3x --
e no caso da v3, LatentSync por cima.

## 3.45 O caminho validado tinha os MESMOS três defeitos da decupagem -- só em outro lugar (2026-08-29)

O usuário assistiu ao `movie.mp4` do teste `screenplay_25` (§3.44) e reportou:
"ficou muito bom num estilo cinematográfico realista, porém perde o
sincronismo, truncam-se as falas, vocalizam a pontuação". Os mesmos três
sintomas da §3.36 (decupagem) -- mas o `screenplay_ui`/`screenplay_to_video`
é um caminho de código DIFERENTE (`script_pipeline/render_scenes.py` +
`lipsync_scenes.py`, não `render_shots.py`), então os consertos de ontem não
chegaram aqui automaticamente. Investigado achado por achado, sem tocar a
GPU (havia um teste rodando).

### 1. `_split_audio_into_segments` corta em ponto ARITMÉTICO, no meio da palavra

`render_scenes.py` existe para partir uma fala longa em clipes curtos (evita o
travamento do upsampler em clipe longo, §3.30) -- mas o corte era
`duração_total / n_segmentos`, sem nenhuma noção de onde a fala PARA. Numa
fala de 22,3s (a mesma "Thoren, as runas despertaram..." de sempre), virou 5
segmentos de 4,52s cada, cortados em pontos aritméticos -- e cada segmento
vira um clipe de vídeo + um lip-sync **independentes**. Corte no meio de uma
sílaba, em toda fronteira interna, é exatamente "truncam-se as falas, perde o
sincronismo".

**Conserto**: `_boundaries_from_gaps()` usa `librosa.effects.split` para achar
os silêncios reais do áudio, e percorre-os escolhendo sempre o ÚLTIMO
silêncio que ainda cabe dentro do teto `max_clip_seconds` -- o teto continua
sendo respeitado à risca (é ele que evita o travamento), só que agora cada
corte cai numa pausa de verdade em vez de no meio de uma palavra. Testado
contra a fala real de 22,3s: a versão antiga cortava nos 4 pontos aritméticos
sem olhar o áudio; a nova acha 17 silêncios reais e produz 6 segmentos
(3,34s a 4,55s, todos ≤ 5,0s) com toda fronteira interna caindo num gap
detectado. Sem silêncio algum dentro do teto (fala corrida), cai no corte
aritmético de antes -- nunca pior, só melhor quando há onde cortar certo.

### 2. Emoção em vocabulário livre não batia com os 17 slugs -- prosódia ficava sempre neutra

A §3.36.3 corrigiu a prosódia do XTTS (pausa e velocidade por emoção) contra
o vocabulário do Ollama/qwen3.6 (`com_medo`, `calma`, os 17 slugs de
`speakers/01_vozes_emotivas`). Este caminho usa Gemma4 para o enriquecimento,
que escreve emoção como **texto livre**: `"urgente"`, `"com respiração
curta"`, `"com força crescente"`, `"determined, resolute"` -- nenhum bate por
substring com um slug, e `prosodia_de()` caía sempre no neutro (1.0, 1.0) em
silêncio. Conferido nas 4 falas reais desta rodada: `emotion` era `None` em 3
de 4 (o Gemma4 escreve a pista em `parenthetical`, não em `emotion`).

**Conserto**: `SINONIMOS_DE_EMOCAO` em `xtts_worker.py` -- um classificador
por PALAVRA-CHAVE (não por frase), PT/EN, mapeando termos comuns
("urgente"/"urgent", "determined"/"resolute", "respiração curta"/"breathless"
etc.) para o slug mais próximo dos 17. Testado: as 4 frases reais desta
rodada agora resolvem para valores não-neutros (`prosodia_de("urgente")` →
`(1.10, 0.65)`, antes caía em `(1.0, 1.0)`).

### 3. A reticência era passada LITERALMENTE ao XTTS -- e é o tipo de sinal que TTS mais erra

A §3.36.3 já tinha adicionado pausa proporcional ao sinal de pontuação
(`PAUSA_POR_SINAL`), incluindo 0,55s para reticência -- mas isso só decide a
DURAÇÃO do silêncio inserido por código; o texto "A lua está
desaparecendo**...**" continuava indo, com os três pontos literais, para
`process_tts_to_file()`. Reticência (três caracteres seguidos, sem
equivalente comum em fala corrida) é exatamente o tipo de pontuação que
tokenizers de TTS mais erram -- é a explicação mais direta para "vocalizam a
pontuação".

**Conserto**: `_texto_para_modelo()` remove `"..."`/`"…"` do texto que
efetivamente chega ao modelo, enquanto `_pausa_apos()` continua lendo a
frase ORIGINAL (com a reticência) para escolher a pausa certa -- a pausa já
está garantida por silêncio inserido por código; pedir para o XTTS também
"ler" o sinal é pedir a mesma coisa duas vezes, e é a segunda pedida que
falha.

### Verificação

Suite de testes (`pytest tests/ -q`) segue 30/30 depois dos três consertos.
`render_scenes.py` é o MESMO módulo nos dois caminhos (2.3 nativo e 2.5) --
`LTX_PIPELINE_MODULE` só troca o backend de vídeo, não o de áudio -- então os
três consertos valem para as duas rotas. Ainda não visto num filme novo
depois do conserto; o teste nativo 2.3 em andamento no momento (§3.44 item 5)
usa o MESMO roteiro Lyra/Thoren com a mesma fala longa e vai exercitar o
conserto #1 -- conferir o resultado quando terminar.

## 3.46 ChoreoEngine + LTX: primeira cena de dança de ponta a ponta (2026-08-30)

Pedido novo, fora do fluxo screenplay/decupagem: uma cena curta gerada a
partir do editor de danças (`E:\Users\home\Documents\ChoreoEngine`) integrado
a este repositório -- Thoren chega ao palácio, grita "Vencemos!", Lyra
responde com uma dança estilo Bollywood. Script novo do lado do
ChoreoEngine, `bridge/dance_scene_test.py`; **detalhe completo, achados e
números em `ChoreoEngine/MEMORIAL.md` §31 e §31.1** -- aqui só o que
interessa para quem mexe neste repositório.

**Confirma uma peça que faltava documentar: `ic_lora.py` aceita `--image`
(identidade) e `--video-conditioning` (pose) ao mesmo tempo.**
`ChoreoEngine/bridge/run_f3.py`, a ponte que já existia, nunca usava os dois
juntos -- só pose, com a identidade inteira dependendo do texto do prompt. O
script novo passa a MESMA imagem de referência que a decupagem já gera
(`outputs/decupagem/<cena>/shots/stills/*.png`) junto com o `pose.mp4`, e é
isso que resolve a deriva de identidade entre frames que o ChoreoEngine já
tinha como problema aberto.

**Pose condicionada só existe para o checkpoint 2.3 aqui** -- o IC-LoRA
Union-Control (`models/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`)
não tem equivalente para 2.5. Qualquer integração de dança/pose real cai
sempre no caminho 2.3 nativo, mesmo quando o pedido original citava uma UI
2.5.

**Bug de concatenação relevante para qualquer script que junte clipes de
áudio de origens diferentes.** O ffmpeg com `-f concat` (demuxer) exige
parâmetros de stream idênticos entre os arquivos; o script juntava um clipe
com áudio do XTTS (16kHz) e um clipe com áudio gerado pelo LTX (48kHz), e o
resultado era uma faixa de áudio **3x mais longa que o vídeo**, sem erro
nenhum do ffmpeg. `ffprobe -show_entries format=duration` reporta a duração
da faixa MAIS LONGA do contêiner, o que mascarava o sintoma. Corrigido
trocando para `-filter_complex` com `aresample` explícito por entrada antes
do `concat`. Se algum script daqui vier a concatenar clipes de fontes de
áudio diferentes (TTS + geração LTX é o caso óbvio), use filtro, não
demuxer.

**Um hang de 20+ min, mesmo sintoma do `screenplay_ui` (§7 item pendente
"investigar hang"), teve causa identificada desta vez: contenção de GPU.**
O clipe 1 travou em "Stage 1: Initial low resolution video generation" com
CPU e as duas GPUs ociosas -- o usuário identificou que outro gerador de
imagem estava ocupando a RTX 4070 no momento (não a 3090, que é onde este
pipeline roda). Com a 4070 liberada, a mesma chamada rodou sem travar. Isso
não prova a causa do hang do `screenplay_ui` (roda só na 3090, nunca teve a
raiz confirmada), mas é a primeira evidência real de que "outro processo
segurando GPU" é uma hipótese a testar antes de qualquer outra.

## 7. Próximas etapas, por ordem de retorno

*(reescrita em 2026-08-29, depois da auditoria externa, dos quatro defeitos do
filme montado, do `prompt_polish` e do MiniMax H3. Itens fechados desde
2026-08-26 foram removidos daqui -- procure pelo número da seção no changelog
acima se precisar do histórico completo. Item 15 acrescentado em 2026-08-30.)*

### P1 -- maior retorno, prontos para executar

**1. Regerar os clipes da Lyra com TODOS os consertos de hoje combinados.**
Hoje cada conserto (enquadramento de fala, `prompt_polish`, condicionamento de
áudio, cama restaurada, prosódia/voz emotiva) foi validado separadamente ou
contra o plano antigo. Nenhum filme ainda combina os cinco. É o passo que
transforma tudo isso em resultado, e é o material que falta para o item 8 do
pedido original (comparar qualidade entre vídeos) ter algo real para comparar.
Ordem: `prompt_polish --apply` já rodou (plano 1 tem `video_prompt` polido);
falta rodar `render_shots --videos-only` com os enquadramentos novos + `--
no-audio-conditioning` DESLIGADO (condicionar pela fala) e depois `mix_audio`
com a cama restaurada (já é o padrão).

**2. Terminar o experimento de atribuição.** Braço A JÁ RODOU e tem resultado
real (§3.38): a âncora de still trava o primeiro quadro mas o estilo deriva
depois -- achado que muda a leitura do fator 3. Braços B (corte) e C (prompt
fragmentado) faltam. Retomar com `.venv/Scripts/python.exe -u
_atribuicao_338.py` (~70 min de GPU total, ou ~55 min pulando A de novo;
script na raiz do repo).

**3. O primeiro commit.** `git status -uall` caía de 20.944 para 380 depois do
`.gitignore` corrigido (§3.39) -- mas **nada foi commitado**. Praticamente todo
o corpo de trabalho de várias sessões (`script_pipeline/` inteiro,
`ltx25_backend.py`, `minimax_h3_backend.py`, `video_doctor.py`, os dois
`.md` vivos) só existe em disco. Decisão do usuário, não ação de rotina --
mas quanto mais tempo passa, maior o commit inicial e maior o risco de perda.

### P2 -- valor real, precisa de decisão de design antes de codar

**4. Plugar o MiniMax H3 na decupagem** (§3.40). O backend existe e foi
testado com um caso real, mas três perguntas seguem abertas: ele SUBSTITUI
TTS+lip-sync (o modelo já fala a fala) ou recebe áudio externo como o LTX?
Ele obedece instrução de câmera em texto (não testado)? Como usar as 2
referências quando uma cena tem 2 personagens, já que `render_shots` hoje só
produz 1 still por plano? Resolver as três antes de escrever `--engine
minimax_h3` no `render_shots.py`.

**5. Investigar o deslocamento de posição de corte entre remontagens**
(achado em §3.39, ao validar o conserto do doctor). O mesmo `movie.mp4` da
Lyra teve os 4 cortes reais mudarem de índice (73/202/305/410 ->
73/198/297/401) entre duas rodadas de `assemble_final` no mesmo dia, e isso
comeu um defeito real que ficou perto demais do novo corte. Hipótese não
testada: reencode de áudio no `assemble_final`/`mix_audio` desalinhando o
vídeo por alguns frames. Preciso de causa antes de confiar em posição de
corte entre remontagens.

**6. Calibrar o `video_doctor` contra defeitos REAIS.** Os limiares foram
ajustados contra poucos clipes, e a maioria das falhas confirmadas até agora
era falso positivo (mão entrando em quadro, §3.19; cortes, §3.36.1). Sem um
conjunto de casos com defeito de verdade marcado à mão, o reparo automático
continua desaconselhado e a revisão humana é obrigatória, não opcional.

### P3 -- baixo risco, baixa urgência

**7. Verificação visual do `ots`/estabelecimento** (§3.29). Corrigido no
código, nunca visto numa imagem -- nenhum roteiro de teste desde então
produziu um `ots`. Rode um roteiro de `desenvolvimento` e olhe o still.

**8. Colisão de vocabulário entre direção de arte e perfil de estilo**
(§3.29). Um roteiro em cel animation ainda recebe "natural cinematic
lighting" colado no fim do prompt. Não estragou nenhum resultado medido; é
inconsistência escrita, não defeito visível.

**9. Calibrar o movimento dos perfis de estilo** (§3.24). 3,47s de `push`
fecharam de plano médio a quase close -- o LTX é mais agressivo que "lento"
sugere.

**10. Identidade entre cenas, sem ferramenta de medida** (§3.24). A
referência do FLUX resolve identidade no still; não há como MEDIR deriva
(sem insightface/facexlib/deepface instalados). Qualquer melhoria aqui hoje é
opinião.

**11. `--cache-none` de vídeo em plano longo** (§3.30). 960x544 fecha 73
frames em 511s e NÃO fecha 145 -- encalha no decode tiled do VAE. Caminhos
não testados: partir o plano em pedaços e concatenar, baixar resolução e
recuperar no upscale, ou `LTX25_TWO_STAGE=1`.

**12. Não pedir lip-sync em plano sem rosto** (§3.30). `wide`/`insert`/
`establishing` não têm rosto pra sincronizar. Regra de uma linha, ainda não
escrita.

**13. Aplicar o seletor de upscale ao `music_maker_ui_v3_25`.** Mesma
estrutura do v2_25; ficou de fora.

**14. Trocar o modelo padrão de upscale** -- decisão do usuário, medida já
feita (§3.18): `realesrgan-x4plus` mede melhor que `realesr-animevideov3` em
render 3D. Escolhível na UI, padrão mantido de propósito.

**15. Testar a hipótese de contenção de GPU no hang do `screenplay_ui`**
(§3.46). Nunca reproduzido de propósito -- confira `nvidia-smi` e a fila de
CADA ComfyUI (§6.2) ANTES de rodar de novo, e se travar, confira se algum
outro processo (mesmo em outra placa) está segurando GPU no momento antes de
declarar bug no pipeline. Duas vezes anteriores o usuário dispensou a
investigação; se pedir de novo, comece por aqui.

### Adiado de propósito, com nota própria em §3.39

Consolidar as 17 UIs duplicadas, trocar estado global por `Job` dataclass,
centralizar cancelamento de processo, e triar os 323 `except Exception` --
avaliados, confirmados como reais, e deliberadamente NÃO tocados numa sessão
que já mexeu em quatro frentes sem suíte de UI para pegar regressão. Cada um
é candidato a sessão própria.

## 8. Como retomar

*(atualizado em 2026-08-30.)*

**Sessão de 2026-08-30 em uma frase:** integração ChoreoEngine+LTX para cena
de dança fechou de ponta a ponta (§3.46; detalhe em
`ChoreoEngine/MEMORIAL.md` §31/§31.1) -- vídeo de ~12s entregue, bug de
concatenação de áudio corrigido. Nada dos itens P1-P3 abaixo foi tocado hoje;
a lista de 2026-08-29 continua valendo integralmente, com o item 15 novo.

**Se for continuar a integração de dança:** script em
`ChoreoEngine/bridge/dance_scene_test.py`, roda sozinho (não depende de UI
nenhuma daqui). Próximos passos, se virar produto, estão listados no fim de
`ChoreoEngine/MEMORIAL.md` §31 -- aba real em vez de CLI, referência de
Bollywood de verdade, métrica de identidade separada da de pose.

Quatro fumaças, da mais barata para a mais cara. Rode na ordem e pare na
primeira que falhar -- cada uma pressupõe a anterior.

```bash
# 1. a suite de testes passa? (CPU, segundos)
.venv\Scripts\python.exe -m pytest tests/ -q

# 2. o diagnóstico roda? (só CPU, segundos)
.venv\Scripts\python.exe video_doctor.py analyze outputs/_diag/smoke.mp4 --previews outputs/_diag/tiras

# 3. a cadeia de decupagem fecha? (~2 min, sem GPU de difusão; precisa do Ollama no ar)
.venv\Scripts\python.exe -m script_pipeline.run_decupagem ^
  --run-dir outputs/decupagem/_smoke --script <roteiro.txt> --ate animatic

# 4. a rota 2.5 gera? (sobe o ComfyUI sozinho, ~15 min para 19s a 1280x704)
.venv\Scripts\python.exe -m ltx_pipelines_25 --prompt "a woman singing at sunset" ^
  --output-path outputs/_diag/smoke.mp4 --width 1280 --height 704 ^
  --num-frames 121 --frame-rate 24 --seed 7
```

Todos os quatro já rodaram com sucesso nesta máquina -- não são fumaça de
"nunca testado", são fumaça de "confirme que nada regrediu desde então".

**Se for continuar o MiniMax H3** (§3.40): o servidor sobe sozinho
(`minimax_h3_backend.ensure_server()`), mas se já estiver no ar em
`http://127.0.0.1:8189` de uma sessão anterior, não suba de novo -- confira
com `curl -s http://127.0.0.1:8189/system_stats`.

**Se for continuar o experimento de atribuição** (§3.38): script pronto em
`_atribuicao_338.py`, ~70 min de GPU, três braços em sequência.

**Antes de qualquer geração de teste**, confira que não há run do usuário
ativo: `/queue` de CADA ComfyUI (8188 do LTX 2.5, 8189 do MiniMax H3),
`nvidia-smi`, e a cena mais recente em `outputs/music_video_v2/*/scenes/`.
Ver §6.2. E lembre de §3.11: derrubar outro processo para "liberar VRAM" já
quebrou uma geração em curso -- mais VRAM livre faz o ComfyUI escolher
estratégia mais agressiva, e ele bate no teto.

Documentos vivos: `CLAUDE.md` (configuração operacional verificada da máquina) e
este `MEMORIAL.md` (por quê, histórico e o que vem depois). Quando divergirem,
este é o mais recente.
