# CLAUDE.md — LTX-2-OPTIMIZED

Configuração desta máquina e do que já foi verificado nela. O `README.md` do
repositório descreve o projeto upstream (e ainda documenta a geração **19b**);
este arquivo cobre **o que existe aqui**: as gerações **2.3/22b** e
**2.5/22b**, ambas em produção, por caminhos de código diferentes.

Divisão de papéis: aqui fica a **configuração operacional verificada**
(hardware, ambientes, comandos, armadilhas). O `MEMORIAL.md` registra **por que
o sistema é assim** e o histórico das decisões — quando os dois divergirem, o
MEMORIAL é o mais detalhado e o mais recente.

Última verificação: 2026-09-07.

---

## Hardware

| | |
|---|---|
| GPU 0 (nvidia-smi) | **RTX 3090, 24,5 GB** — é a GPU de geração |
| GPU 1 (nvidia-smi) | RTX 4070, 12,9 GB — display/desktop |
| RAM | **79,9 GB** (medido 2026-08-27; ~35 GB livres com o LTX 2.5 carregado) |
| Pagefile | C: 38 GB + E: 31,2 GB — praticamente ocioso (0,9 + 0,1 GB em uso) |

⚠️ **A RAM foi ampliada de 48 para 80 GB e a documentação não acompanhou.**
Isso importa além do registro: `LTX_TRANSFORMER_CPU_MEMORY=34GiB`, que a seção
da ICLoraPipeline mais abaixo usa, foi dimensionado para a máquina de 48 GB.
Com 80 GB dá para dar bem mais folga ao offload da rota 2.3 — não testado.
| Driver | 610.74 |

**Atenção à ordem dos índices.** O `nvidia-smi` enumera a 3090 como `0`; o torch
enumera como `cuda:1`. A convenção do repositório é `CUDA_VISIBLE_DEVICES=1` =
3090 (ver `gguf_backend.py:121`, `generate_upscale.ps1:13`, `music_maker_ui_v*`),
e ela **bate com o torch**, não com o nvidia-smi. Não assuma que os dois
coincidem — escolha a GPU pela VRAM total, não pelo índice.

## Ambientes Python

| Ambiente | numpy | torch | `torch.from_numpy` | onnxruntime |
|---|---|---|---|---|
| `.venv` — **onde o LTX roda** | 2.4.4 | 2.8.0+cu128 | ok | 1.27 (só CPU) |
| `gemma4_env` | 2.5.2 | 2.11.0+cu128 | ok | — |
| Python global (user site) | **1.26.4** | 2.4.0+cu124 | **ok** | gpu 1.19.2 (CUDA ok) |

**Medido em 2026-08-20.** O global foi rebaixado de numpy 2.5.1 para 1.26.4, e
isso reverteu três problemas que este arquivo documentava como abertos:

- `torch.from_numpy` / `tensor.numpy()` **voltaram a funcionar** — o torch 2.4
  foi compilado contra numpy 1.x, que é justamente o que está instalado agora.
- `librosa` (0.11.0) **importa e roda** no global. O `analyze_music.py` da raiz
  deixou de estar bloqueado. O backend em numpy puro do ChoreoEngine
  (`choreo/sync/audio.py`) continua válido, só não é mais obrigatório.
- `onnxruntime` volta a expor **CUDA e TensorRT**. Estavam instalados
  `onnxruntime` (CPU, 1.29.0) **e** `onnxruntime-gpu` (1.19.2) ao mesmo tempo;
  os dois ocupam o mesmo namespace e o CPU, por ser mais novo, vencia — a
  extração de pose caía para CPU em silêncio. Removido o pacote CPU.

⚠️ **Ao remover um dos dois, o outro quebra junto** (compartilham arquivos): o
módulo passa a importar sem `__version__` e sem providers. A saída é reinstalar
o que deve ficar — `pip install --force-reinstall --no-deps onnxruntime-gpu==1.19.2`.

⚠️ **Reinicie processos longos depois de mexer em pacotes.** Processo antigo com
pacote novo em disco produz erro que parece bug de código: o `np.median` chegou
a estourar `AttributeError: 'numpy.dtypes.BoolDType' object has no attribute
'bits'` porque o servidor carregara módulos do numpy 2.x e importou o resto do
1.26.4 já em execução.

## Ferramentas externas (`tools/`)

| pasta | o quê | usado por |
|---|---|---|
| `realesrgan/` | Real-ESRGAN ncnn-vulkan (upscale 2D final) | `upscale_video.ps1`, UIs de music maker |
| `rife/` | RIFE ncnn-vulkan 20221029, modelos v2 a v4.6 | `video_doctor.py` (interpolação) |
| `Wav2Lip/` | sincronia labial por cena | `music_maker_ui_v*` |

RAFT (fluxo óptico) não fica aqui: são pesos do torchvision, em
`C:/Users/user/.cache/torch/hub/checkpoints/`. Opcional no `video_doctor`
(`--flow raft`); ~45x mais lento que o DIS do OpenCV, que é o padrão.

**Escolha do modelo de upscale importa mais do que parece.** O padrão
histórico é `realesr-animevideov3`, treinado em *line art de anime*. Aplicado a
render 3D ele **achata** gradiente em bloco de cor chapada — medido pior que
`realesrgan-x4plus`, apesar de pontuar MAIS em métricas de nitidez, porque
troca degradê por borda dura. Já é escolhível na UI do `music_maker_ui_v2_25`.
Ver `MEMORIAL.md` §3.18.

⚠️ **`realesrgan-x4plus` e `realesrgan-x4plus-anime` (família RRDBNet) estão
QUEBRADOS nesta instalação — não use.** MEDIDO 2026-09-03: `realesrgan-
ncnn-vulkan.exe -n realesrgan-x4plus` devolve saída determinística sem
relação com a entrada (grid quadriculado ou perda total do sujeito),
**independente de resolução, tile size (`-t`) ou GPU** — reproduzido em
frame isolado, um tile só (`-t` maior que a imagem), nas duas GPUs. Só
`realesr-animevideov3` (arquitetura mais rasa, SRVGG) sai correto. Isso
inverte a recomendação do parágrafo acima até alguém reinstalar/validar os
arquivos `.bin`/`.param` do x4plus — fique no padrão `animevideov3` em
`upscale_video.ps1`. Ver `MEMORIAL.md` §3.56.1 (a investigação começou com um
falso-positivo no RIFE, que foi corrigido junto).

## Ollama: qual instância está no ar importa

Os modelos vivem em `G:\ollama\models`. Se quem subir for o **aplicativo
desktop**, ele lê outro diretório e serve um catálogo diferente — pedir
`qwen3:8b` devolve **404 em `/api/generate`** mesmo com o servidor respondendo
em `/api/tags`. Rode `ollama serve` com `OLLAMA_MODELS=G:\ollama\models`, ou
encerre o app desktop antes. O `choreo/prompt_to_params.py` detecta esse caso e
diz quais modelos a instância realmente serve.

**O Ollama é o motor padrão da extração de roteiro** desde 2026-08-23
(`script_pipeline/`, `storyplay25`), medido contra o gemma4-e2b local: 6s
contra 91s, 4/4 falas preservadas verbatim contra 2/4, 9 planos contra 5.

Duas armadilhas ao trocar de modelo:
- **Modelo de raciocínio falha em silêncio.** O qwen3.6 gastava todo o
  orçamento de tokens no campo `thinking` e devolvia `content` **vazio** —
  parece recusa do modelo. `think: False` resolve. Vale para `qwq` e
  `deepseek-r1` também.
- **A instância no ar em 2026-08-25 serve só `qwen3.6-35b-a3b`**, apesar da
  pasta ter vários. É o caso do parágrafo acima.

⚠️ **`qwen3.6-35b-a3b:latest` (MoE) crasha o backend CUDA do Ollama em prompt
LONGO** -- MEDIDO 2026-09-07: `an error was encountered while running the
model: CUDA error: an illegal memory access was encountered` (HTTP 500),
reproduzido de forma determinística com o prompt real de enriquecimento do
`parse_screenplay` (~8k caracteres) e isolado por bisseção (prompt curto/
médio ok, prompt longo falha SEMPRE, em qualquer contagem de tentativas).
Não é corrupção de GPU nem de processo: sobreviveu a reiniciar o Ollama e a
`nvidia-smi --gpu-reset -i 0`. Isso explica os dois colapsos de decupagem
consecutivos (20260907_ltx_distilled e a primeira tentativa do
20260907_ltx_gguf) -- o HTTP 500 na etapa de enriquecimento nunca era
retentado, e o `shot_plan.py` caía no fallback de scene completa para
TODOS os planos. `qwen2.5:32b-instruct-q4_K_M` roda o MESMO prompt sem erro
(~41s, JSON valido) -- use-o para `--engine` em roteiros que gerem prompt de
enriquecimento grande, ate alguem investigar se e bug do llama.cpp com MoE
em contexto longo ou do proprio GGUF. Ver `MEMORIAL.md` §3.65.

## Modelos 2.3 instalados (`models/`)

*(os do 2.5 ficam em `models/2.5/` — seção própria mais abaixo)*

```
ltx-2.3-22b-distilled-fp8.safetensors            31,7 GB   pipelines nativos
ltx-2.3-22b-dev.safetensors                      46,1 GB
ltx-2.3-22b-dev-Q6_K.gguf                        17,8 GB   ComfyUI
ltx-2.3-22b-dev-UD-Q5_K_S.gguf                   16,3 GB   ComfyUI
ltx-2.3-spatial-upscaler-x2-1.0.safetensors       1,0 GB   obrigatório em 2 estágios
gemma3/                                          24,4 GB   text encoder (bf16)
loras/ltx-2.3-22b-distilled-lora-384-1.1          7,6 GB
loras/ltx-2.3-22b-ic-lora-ingredients-0.9         1,3 GB
loras/ltx-2.3-22b-ic-lora-union-control-ref0.5    0,65 GB  Canny+Depth+POSE
```

### IC-LoRA de controle: use a Union-Control

O `README.md` aponta `LTX-2-19b-IC-LoRA-Pose-Control`. **Ela não serve aqui** —
é 19b, arquitetura incompatível com o checkpoint 2.3/22b. Não existe
Pose-Control para 2.3/22b.

A `LTX-2.3-22b-IC-LoRA-Union-Control` cobre Canny + Depth + **Pose** e traz no
metadata do safetensors `model_version=2.3.0` e `reference_downscale_factor=2`.
O `ic_lora.py:454` lê esse fator sozinho e carrega o vídeo de condicionamento em
`height//2 × width//2` — passe o controle em **resolução cheia**.

## Rodando a ICLoraPipeline na 3090 com offload

```bash
CUDA_VISIBLE_DEVICES=1 \
LTX_TRANSFORMER_GPU_MEMORY=12GiB \
LTX_TEXT_ENCODER_GPU_MEMORY=12GiB \
LTX_TRANSFORMER_CPU_MEMORY=34GiB \
.venv/Scripts/python.exe -m ltx_pipelines.ic_lora \
  --distilled-checkpoint-path models/ltx-2.3-22b-distilled-fp8.safetensors \
  --spatial-upsampler-path models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root models/gemma3 \
  --prompt "..." --video-conditioning pose.mp4 1.0 \
  --height 512 --width 768 --num-frames 25 --frame-rate 24 \
  --lora models/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors 1.0 \
  --quantization fp8-cast --skip-stage-2 --output-path out.mp4
```

Três coisas que **têm** que estar certas, ou não gera nada:

1. **`--quantization fp8-cast` é obrigatório para haver offload.**
   `ModelLedger.transformer()` só aplica `max_memory` quando há quantização
   (`model_ledger.py:220`). Sem ela, tenta carregar 31 GB direto na GPU.
2. **`LTX_TEXT_ENCODER_GPU_MEMORY` precisa ser aumentado.** O default do ledger
   é `2GiB` e o Gemma 3 12B ocupa ~24 GB em bf16. Se GPU+CPU não cobrirem o
   modelo, o accelerate exige um `offload_dir` que o ltx-core não passa, e a
   execução morre com `ValueError: We need an offload_dir`. O text encoder roda
   antes do transformer e é liberado depois — pode usar a VRAM livre.
3. **`LTX_TRANSFORMER_CPU_MEMORY` serve aos dois** (transformer e text encoder).

Medido: 25 frames a 768×512, 8 passos distilled, `--skip-stage-2` → **~6,5 min**
(carga dos modelos domina; a difusão em si foi 40 s).

### Liberação de memória

Rode a geração em **subprocesso**. Ao sair, o SO reclama VRAM e RAM do offload
com ou sem erro; liberar tensores no próprio processo deixa dezenas de GB presas
no working set. É o que `ChoreoEngine/bridge/run_f3.py` faz, incluindo medir
VRAM/RAM antes e depois.

## Bugs conhecidos neste checkout

- `ic_lora.py` tinha `--conditioning-attention-mask` e `--skip-stage-2` usados em
  `main()` mas nunca registrados no parser (corrigido em 2026-08-10; comentários
  `BUGFIX` no arquivo).
- `analyze_music.py` deriva downbeats com `beats[::4]`, o que erra com anacruse.
- Chamar a `ICLoraPipeline` diretamente em processo (em vez da CLI) falha no VAE
  com `RuntimeError: Inference tensors cannot be saved for backward`. A CLI é o
  caminho validado; não duplique a chamada.
- `ComfyUI/comfy/text_encoders/gemma4.py` tinha um `Gemma4Model.process_tokens`
  sobrescrito que não existe no ComfyUI upstream (`comfyanonymous/ComfyUI`) e
  quebrava o contrato de `sd1_clip.py::forward` (esperava 4 valores, o override
  só devolvia `embeds`). Corrigido em 2026-08-21 removendo o override (o método
  herdado de `SDClipModel` já está correto). Sem esse fix, `encode_from_tokens_scheduled`
  falha com `ValueError: not enough values to unpack (expected 4, got 1)` para
  qualquer checkpoint Gemma4 (E2B/E4B/31B/12B), não só o do LTX-2.5.
- **`encode_prompts([prompt, negative_prompt], ...)` com dois prompts numa
  chamada só derruba com `torch.OutOfMemoryError`** reportando uma quantidade
  alocada fisicamente impossível (ex.: "60.73 GiB" numa placa de 24GB).
  Parece um leak nos hooks de offload do `accelerate` que só aparece
  codificando um segundo prompt no mesmo processo sem um `cleanup_memory()`
  no meio — `music_to_video.py`/`distilled.py`/`ic_lora.py` (os caminhos
  validados) sempre chamaram com UM prompt só, por isso nunca bateram nisso.
  Corrigido em 2026-09-02 em `keyframe_interpolation.py`,
  `music_to_video_v2.py`, `ti2vid_one_stage.py` e `ti2vid_two_stages.py`
  (nenhum desses é chamado por UI ativa neste checkout — bug dormente, não
  estava afetando produção): cada um agora faz duas chamadas de
  `encode_prompts()`, uma por prompt. De quebra, `keyframe_interpolation.py`
  e `ti2vid_one_stage.py` tinham um bug **pré-existente e não relacionado**:
  chamavam `encode_prompts([prompt], ...)` com um prompt só mas
  desempacotavam em duas variáveis (`context_p, context_n`) — `ValueError`
  garantido em qualquer invocação real, e `negative_prompt` nunca era
  codificado. Investigação completa (incluindo o bug irmão do lado ComfyUI:
  `av_model.py` exige o conditioning de vídeo+áudio concatenado, 4096+2048
  canais, pra este checkpoint) documentada em `lora_storyboard_encode.py`.

### Bugs do ComfyUI vendorizado — contornados de fora, NÃO corrigidos lá

Deliberado: mexer no `ComfyUI/` conflitaria com atualizações futuras. Os
contornos vivem no `ltx25_backend.py`.

- **STG é incompatível com keyframes** (variante `dev`). O passe perturbado
  troca `optimized_attention` por um passthrough que devolve `v` inteiro
  (`custom_nodes/ComfyUI-LTXVideo/stg.py:154`), mas o core agora fatia o Q
  (`comfy/ldm/lightricks/model.py`, `_attention_with_guide_mask`), e o `v`
  cheio não cabe na fatia. Sintoma: `The expanded size of the tensor (N) must
  match the existing size (M)`, onde a diferença M−N é exatamente
  `n_keyframes × tokens_por_frame_latente`. Contorno: `stg=0` quando há
  keyframes. `perturb_attn=False` **não** basta — `do_perturbed()` só olha
  `stg_scale`. Ver `MEMORIAL.md` §3.14.
- **`comfy_workflow_tool.convert()` erra valores de widget** quando o workflow
  tem subgrafos. Já mordeu duas vezes: `batch_size` recebeu `25` em vez de `1`
  (tipo união `"FLOAT,INT"` não reconhecido), e `LTXVImgToVideoInplace.strength`
  recebeu `false` — que vira `0.0` e **desliga o condicionamento de imagem
  inteiro**, em silêncio. Os workflows 2.3 são imunes porque não têm subgrafo.
  Contorno: fixar o valor explicitamente em `build_workflow()`. Se aparecer
  comportamento estranho num nó novo, auditar tipos contra `/object_info`
  antes de investigar qualquer outra coisa. Ver `MEMORIAL.md` §3.16.

### Locale afeta binários de terceiros

`tools/rife/rife-ncnn-vulkan.exe` converte o argumento `-s` (timestep) com o
locale do sistema. Nesta máquina (pt-BR) `-s 0.2` é recusado com
`invalid timestep, must be 0~1`, e `-s 0,2` funciona. O `video_doctor` tenta os
dois e memoriza. O mesmo vale para qualquer ncnn-vulkan aqui — o Real-ESRGAN já
imprimia `75,00%` com vírgula.

## LTX-2.5 — em produção

Status em 2026-08-25: **rota completa em produção**, com UIs próprias. Clipes
reais de 19,3s a 1280x704, com áudio condicionado pela trilha do usuário.

**O 2.5 NÃO usa `ltx_pipelines`.** Usa `ltx25_backend.py`, que dirige o ComfyUI
por HTTP. O motivo: `ltx_core` não tem encoder Gemma4 nem o `NADiffusionDecoder`
do VAE 2.5, e o ComfyUI vendorizado tem os dois. `ltx_pipelines_25.py` é um shim
que aceita a mesma argv de `ltx_pipelines.distilled`/`music_to_video`, então as
UIs trocam 2.3↔2.5 mudando o nome do módulo. Ver `MEMORIAL.md` §3.6.

### Modelos (`models/2.5/`, 114 GB)

```
diffusion_models/ltx-2.5-22b-distilled-transformer-bf16   40G   padrão
diffusion_models/ltx-2.5-22b-dev-transformer-bf16         40G   CFG real
text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16           25G
loras/ltx-2.5-22b-distilled-lora-450-bf16                  9G
vae/ltx-2.5-video-vae-bf16  +  ltx-2.5-audio-vae-bf16      3G
latent_upscale_models/  spatial-x2 + temporal-x2           2G
model_patches/ltx-2.5-duration-head-bf16                   1G
```

Resolvidos por `ComfyUI/extra_model_paths.yaml` (seção `ltx_25`), não por
caminho passado na linha de comando.

### Chaves de ambiente

| variável | padrão | efeito |
|---|---|---|
| `LTX25_VARIANT` | `distilled` | `dev` = CFG real, negative prompt funciona, várias vezes mais lento. `gguf-q6k` = ver abaixo |
| `LTX25_AUDIO_COND` | `1` | condiciona a geração pela trilha; `0` desliga |
| `LTX25_TWO_STAGE` | `0` | `1` = upscale latente x2 + refino. **Dobra a resolução de saída** |
| `LTX25_RIFE_MODEL` | `rife-v4.6` | modelo do RIFE no `video_doctor` |

### Variante GGUF do transformer (`gguf-q6k`) — mais rápida, escopo ainda parcial

MEDIDO 2026-09-06: `LTX25_VARIANT=gguf-q6k` troca o `UNETLoader` pelo
`UnetLoaderGGUF` (node já usado pro 2.3), apontando pro
`ltx-2.5-22b-distilled-transformer-Q6_K.gguf` (baixado de
`realrebelai/LTX-2.5_GGUFs`, ~18,7 GB, em `models/2.5/diffusion_models/
gguf_test/`). **37% mais rápido que o `distilled` bf16 na mesma cena**
(440s contra 698s numa cena simples; 265s contra 797s numa cena de diálogo
de 10s) — o GGUF cabe inteiro na VRAM sem o offload parcial que o bf16 de
40 GB precisa, mesmo custando mais por passo de amostragem (dequantização).

`audio_conditioning` **testado e confirmado compatível** (430s, sem erro) —
é a feature que sustenta o lip-sync da decupagem, então isso importava. O
código de `_apply_audio_conditioning` já foi escrito de propósito
variant-agnostic (pega o `model` do guider, não do loader). **Ainda não
testado**: variante `dev`, `LTX25_TWO_STAGE=1`, keyframes — não trocar pra
`gguf-q6k` num fluxo que usa algum desses sem testar primeiro. O padrão
de produção continua `distilled`; a opção está documentada nos `.bat`
(`start_webui_25.bat` e os outros 6 que tocam 2.5) e disponível como
dropdown "Model variant" na UI que `start_webui_25.bat` sobe
(`web_ui_v4_25.py`, seletor gerado via `_make_25_uis.py`).

⚠️ **`_make_25_uis.py` regenera as 4 UIs 2.5 de uma vez** e reescreve cada
`_25.py` do zero a partir do original 2.3 + substituições do script —
qualquer edição manual feita direto num `_25.py` (como o seletor de upscale
do `music_maker_ui_v2_25.py`, ver `MEMORIAL.md` §3.18) **desaparece sem
aviso** se o gerador não souber recriá-la. Depois de rodar, sempre
`git status --short *_25.py` + `git diff` de cada arquivo modificado antes
de confiar.

### UIs e portas

| script | porta |
|---|---|
| `music_maker_ui_v2.py` / `v3.py` (2.3) | 7803 / 7804 |
| `music_maker_ui_v2_25.py` | 7903 |
| `music_maker_ui_v3_25.py` | 7904 |
| `storyplay25.py` (storyboard c/ quadros intermediários) | 7911 |
| `screenplay_ui.py` (caminho validado, um clipe por FALA) | **7810** |
| `web_ui_v4_25.py` / `film_maker_ui_v4_25.py` | 7960 / 7961 |
| `video_doctor_ui.py` (pós-produção, serve 2.3 e 2.5) | 7912 |
| `decupagem_ui.py` (roteiro → filme, com log e retomada) | 7913 |
| `web_ui_v4.py` / `film_maker_ui_v4.py` (2.3) | **7860** (Gradio padrão) |

Portas **medidas em 2026-08-27** subindo cada UI, não lidas do código: as três
últimas linhas estavam erradas aqui. O screenplay é 7810 (o `.bat` sempre disse
isso), e as duas UIs 2.3 não definem porta — caem no padrão 7860 do Gradio, o que
significa que **não sobem as duas ao mesmo tempo** (`start_cinema.bat` já libera
7860 e 7861 por causa disso).

**Todas as 11 UIs sobem no Gradio 6.20**, verificado subindo uma a uma. Três
incompatibilidades, todas corrigidas em 2026-08-27:

1. `show_api=False` em `demo.launch()` — removido na versão 5. Atingia
   `video_doctor_ui` e `character3d_webui`: as duas nem subiam.
2. `gr.Slider(minimum=8, maximum=8)` em `web_ui_v2.py` — o Gradio 6 rejeita
   min == max e a UI inteira morria. Era um slider degenerado usado só para
   MOSTRAR um valor fixo; virou `gr.Number(interactive=False)`.
3. `theme=` / `css=` no construtor do `gr.Blocks` — moveram para `launch()` na
   versão 6. Não é fatal: vira aviso e o tema é **silenciosamente ignorado**.
   Atingia seis arquivos, que subiam sem tema sem ninguém notar.

`gr.update` continua válido e não precisou de mudança em lugar nenhum.

Cada um tem seu `start_*.bat`. **Reinicie a UI depois de editar
`ltx25_backend.py`** — o processo carrega o módulo em memória e continua com o
código velho.

`start_decupagem.bat` **não sobe UI nem porta**: é CLI, arraste o roteiro sobre
o ícone. Desde 2026-08-27 existe `start_decupagem_ui.bat` (porta 7913) para a
mesma cadeia, com o log à vista, os stills aparecendo conforme saem, o
`cast.json` editável e — o que o `.bat` não faz — **retomada**: o `.bat` cria um
diretório novo a cada execução, então uma corrida interrompida recomeça do zero,
enquanto a UI lista as existentes e continua de onde parou.

⚠️ **As UIs antigas passam `show_api=False` para `demo.launch()`, que o Gradio 6
removeu** (temos 6.20) — `video_doctor_ui.standalone()` e `character3d_webui`
levantam `TypeError` ao subir. Não corrigido; a `decupagem_ui` já nasceu sem.

### Custo medido (3090, distilled, modelo já carregado)

| config | tempo |
|---|---|
| 768x512, 233 frames (9,7s) | ~5,5 min |
| 1280x704, 465 frames (19,3s) | ~15 min |
| two-stage 640x352 → 1280x704, 465 frames | ~13 min |

O two-stage sai **mais rápido** que gerar direto em 1280x704: o estágio 1 é
barato e o refino faz 3 passos contra 8. Cuidado ao extrapolar de teste curto —
a primeira geração inclui ~6 min carregando o transformer de 40 GB.

### Qualidade: resolução e enquadramento mandam

O LTX comprime 32x no espaço. A 768x512 com plano aberto, um rosto ocupa
**1,6 x 1,8 células latentes** — cada olho fica com uma fração de célula, e é
por isso que ele derrete. A 1280x704 com plano médio vira 3,8 x 3,9 (5x a área).
Nem passo nem prompt resolvem isso; é falta de pixel. Ver `MEMORIAL.md` §3.17.

Descritores concretos de figurino repetidos em todas as cenas importam tanto
quanto: "a consistent female singer" não é âncora, é desejo.

### Pós-produção

**A aba do doctor está em TODAS as 13 UIs que geram vídeo** (desde 2026-08-28).
Onde a UI tem abas ela entra como aba; onde o layout é plano, como painel
retrátil — `build_doctor_tab(container="accordion")`. Ficam de fora só
`character3d_webui`, `character_sheet_flux_webui` e `krea2_test_ui`, que
produzem imagem ou 3D: o doctor mede defeito TEMPORAL entre quadros e não teria
o que analisar ali.

**CORTE NÃO É DEFEITO, e o doctor precisa saber disso.** Num filme montado toda
fronteira entre planos é o maior pico de mudança do arquivo — MEDIDO 2026-08-28
na cena Lyra: 4 dos 10 segmentos apontados eram os 4 cortes, com `z_glob` de 190
a 224 contra mediana 5,93, e o veredito para três era INTERPOLATE, que ali
significa fabricar um quadro misturando dois enquadramentos. Ele agora detecta e
descarta sozinho (histograma: corte PERSISTE, defeito VOLTA). Quando você souber
as fronteiras, passe `--cuts 73,202,305,410` e não há o que inferir — o pipeline
sabe, é a contagem de frames por clipe. `--no-cut-detection` desliga, e só faz
sentido em clipe único. Ver `MEMORIAL.md` §3.36.1.

`video_doctor.py` acha frames em que o conteúdo muda de um jeito que o
movimento não explica, e repara. Usa RIFE (`tools/rife/`) e, opcionalmente,
RAFT (pesos em cache do torch). **Revise as tiras antes de reparar**: o
detector acha *mudança*, e movimento rápido legítimo aparece igual a defeito —
reparar isso apaga o movimento. Ver `MEMORIAL.md` §3.19.

## MiniMax H3 — terceiro motor de vídeo

Instalação de ComfyUI SEPARADA, em `E:\Users\home\Documents\MiniMax-H3`
(venv próprio, porta 8189). `minimax_h3_backend.py` dirige por HTTP, mesmo
padrão do `ltx25_backend.py`. Disputa a MESMA 3090 física do LTX — nunca
rode os dois ao mesmo tempo. Ver `MEMORIAL.md` §3.40-3.42, §3.52.

### Checkpoint: `MINIMAX_H3_VARIANT` (padrão `fp8int8` desde 2026-09-06/07)

| variante | unet | tempo medido | veredito |
|---|---|---|---|
| `fp8int8` (padrão) | FP8 pruned + INT8 text encoder | 397-666s | mais rápido E maior qualidade nominal que o w4a8 |
| `w4a8` | o quantizado mais agressivo (o que o workflow oficial traz fixo) | 554-704s | mais lento, mantido só pra comparação/rollback |
| `gguf-q4km` | GGUF via `UnetLoaderGGUF` | 707-986s | **mais lento que os dois** — ao contrário do LTX-2.5, GGUF NÃO ganha aqui (o gargalo é o encoder de texto na CPU, fixo independente do formato do transformer) |

Trocar: `MINIMAX_H3_VARIANT=w4a8` (ou `gguf-q4km`). `MINIMAX_H3_UNET`/
`MINIMAX_H3_CLIP` sozinhos continuam funcionando por cima pra apontar um
arquivo específico. Ver memória de projeto "MiniMax H3 watchdog bug and
quality tests" pro detalhe completo dos números.

### VAE TensorRT — compilada, disponível, NÃO recomendada

`MINIMAX_H3_TRT_VAE=1` troca a VAE de vídeo (só a de vídeo; a de áudio não
tem engine) pelos engines TensorRT compilados (`ComfyUI-H3VAE_TRT`,
`models/vae/minimax_h3_vae_{decoder,encoder}.engine`, 4,85 GB + 346 MB).
MEDIDO 2026-09-06: **piora o tempo em vez de melhorar** — o node reserva o
tamanho do arquivo como orçamento de VRAM antes de carregar de verdade,
empurrando o transformer pro modo lowvram (mais lento, ou trava de vez
combinado com `fp8int8`). Deixar desligado.

### Watchdog: `stall_seconds` precisa de folga (1800s, não 300s)

`gpu_watchdog.StallWatch` mata o servidor (`taskkill /F`, sem traceback —
parece crash) se o mesmo job ficar "rodando" por mais que `stall_seconds`,
mesmo com a GPU ociosa por design (encoder de texto na CPU). MEDIDO
2026-09-06: 300s matava jobs saudáveis de 5-16 min; corrigido pra 1800s
tanto aqui (`minimax_h3_backend.py`) quanto no watchdog compartilhado do
LTX (`script_pipeline/generate_storyboards.py`, mesmo bug, mesmo fix). Se o
MiniMax H3 "morrer sem traceback" de novo num ponto específico e
reproduzível, suspeitar do watchdog ANTES de suspeitar de crash nativo.

## Roteiro → filme (`script_pipeline/`)

Duas variantes do mesmo pipeline, escolhidas pelo comando, não por acaso. Elas
compartilham parse, cast, TTS, lip-sync, mistura e montagem; divergem só na
**unidade de renderização**.

| | UI screenplay (porta 7810) | decupagem (`start_decupagem.bat`) |
|---|---|---|
| unidade | uma FALA | um PLANO |
| still | um por CENA | um por PLANO, fixando o enquadramento |
| ações sem fala | não viram clipe | viram |
| estado | **validado** | **até o animatic**, exercitado em 2026-08-27; do vídeo em diante, não |

    [1] parse   [2] cast   [E] emoção   [4] TTS   [S] estrutura   [P] decupagem
      -> [5] render_scenes  OU  [5-D] render_shots_stage
      -> [6] lipsync  [7] mix  [8] assemble  [9] verify

**O que liga os estágios é `scenes/clips.json`, não o nome do módulo.** O
`render_shots_stage` escreve esse manifesto no mesmo formato que o
`render_scenes`, e por isso os estágios 6–9 rodam depois dele **sem alteração**.
Ao mexer em qualquer um dos dois, preserve o formato — é o contrato.

**A ordem é dependência, não gosto.** Emoção muda a duração da fala; a duração
da fala define a duração do plano. Decupar antes do TTS produz plano vencido.

**Motor dos stills: `--image-engine {flux,sd35,sdxl}`.** O padrão `flux` é o que
obedece enquadramento e lado de tela, e é o único com imagem de referência por
personagem. `sd35` (`models/sd3.5_medium.safetensors` + clip_g/clip_l/t5xxl)
carrega em ~1 min contra ~4, cabe em ~12 GB de VRAM contra ~24 e amostra em ~35 s
contra ~60 — mas erra o enquadramento, que é para o que o still serve. Use para
iterar decupagem, não para entregar. Medições em `MEMORIAL.md` §3.30.

**REINICIE o ComfyUI entre os stills e o vídeo.** O `run_decupagem` já faz isso
sozinho desde 2026-08-27; se você chamar os estágios à mão, faça igual. O
servidor que carregou o FLUX não consegue receber o LTX 2.5 depois: vai a 24,0 GB
de 24,5, fica a 100% e para. Medido no mesmo clipe de 73 frames: **55 min sem
terminar** no servidor herdado contra **511 s** no reiniciado. Ver §3.30.

**O lip-sync SUBSTITUI a faixa de áudio; ele não mistura.** O Wav2Lip entrega
como áudio exatamente o wav que dirigiu a boca, então o plano de diálogo perde a
trilha que o LTX 2.5 gerou — e o plano de ação, que não passa por ele, mantém.
Sintoma: o som corta em cada fala e volta depois. Dois consertos, ambos ligados
por padrão desde 2026-08-28:

- `render_shots` passa a fala ao LTX como `audio_conditioning` (o que o
  `render_scenes` já fazia com `--audio-input-path`). Sem isso o modelo inventa
  o som sozinho e **gera voz própria** — medido: médios a -25,0 dB num plano de
  fala contra -42,5 num plano de ação no mesmo cenário.
- `mix_audio` devolve a trilha do clipe cru sob a fala com `sidechaincompress`:
  ela abaixa enquanto há fala e volta no silêncio. `--no-restore-bed` desliga;
  `--bed-volume`, `--duck-ratio` e `--duck-threshold` ajustam.

⚠️ **`aformat=channel_layouts=stereo` num sinal MONO tira 3,01 dB.** A matriz
preserva potência total, então espalhar um canal em dois dá 1/√2 em cada. Certo
para ambiência, errado para diálogo. Use `pan=stereo|c0=c0|c1=c0`. A saída do
Wav2Lip aqui é mono a 24 kHz e a cama é estéreo a 48 kHz, então isso aparece.
Ver §3.36.2.

**A emoção da fala chega ao XTTS por DOIS canais.** O clipe de referência dá o
timbre; `speed` e a pausa entre frases dão o andamento. Os dois estavam fixos, e
o resultado era "nenhuma fala tem urgência" mais "dá para ouvir a pontuação":
pausa idêntica de 0,35 s em todo limite de frase é o que o ouvido lê como
pontuação anunciada. Agora a pausa vem do sinal (reticência 0,55 s, ponto
0,32 s, exclamação 0,18 s) escalada por emoção. Medido: fala de medo encurtou
21%, de entusiasmo 26%, e a calma alongou 19%.

⚠️ **Voz escolhida à mão perdia as 17 tomadas emotivas.** O ramo do
`mapa_vozes.csv` gravava `emotive_voice: None` fixo — o caminho curado era
estritamente pior em atuação que o heurístico. Corrigido casando `voz_origem`
com a pasta emotiva do mesmo intérprete (Kristin Hughes →
`F02_jovem_expressiva_kristin`), **exato e nunca aproximado**: "Andi" (Clara,
feminina) não pode casar com `M01_..._andy`. 9 dos 12 do mapa casam. Ver §3.36.3.

**Plano de FALA nunca em quadro aberto.** A escada de enquadramento do diálogo
exclui `wide`, `full` e `insert`, e gira pelo ORDINAL DA FALA — pela posição ela
não girava, porque as falas caem em paridade fixa e um falante em cada dois caía
sempre no aberto. Sem isso o rosto derrete (§3.17) e a cena vira quatro planos
iguais. Ver §3.36.4.

**O `--cache-none` é necessário para a passada de VÍDEO caber na placa.** Sem
ele o mesmo clipe de 145 frames encalha a 24,2 GB de 24,5; com ele passa da carga
com 15,9 GB. O `render_shots` já liga sozinho quando a passada não é de stills.
`LTX_COMFY_CACHE_NONE=1` força manualmente. Cuidado: os dois lançadores
(`ltx25_backend` e o do storyboard) voltam cedo se a porta já responde, então
quem sobe o servidor primeiro define a flag para todo mundo. Ver §3.30.

**Por que o 2.3 renderiza e o 2.5 encalha:** o 2.3 roda um checkpoint fp8 **com
orçamento de memória declarado** (`--quantization fp8-cast` + `LTX_TRANSFORMER_
GPU_MEMORY`); o 2.5 entrega 40 GB de transformer + 25 GB de encoder em bf16 e
confia no automático do ComfyUI. `LTX25_UNET_DTYPE=fp8_e4m3fn` e
`LTX25_CLIP_DEVICE=cpu` ligam os dois botões que o workflow 2.5 deixa em
`default` — ajudam a caber (10,9 GB durante a codificação), mas **não** resolvem
o clipe longo. Ver §3.31.

**`--disable-dynamic-vram` é OBRIGATÓRIO no estágio de vídeo.** O ComfyUI liga
*dynamic VRAM* por padrão e ele engasga ao encenar o encoder de 25 GB: o mesmo
plano de 81 frames não fechava em 13 min e passa a fechar em **151 s**. Com ele,
a cena de 10 planos fecha inteira, incluindo um plano de **337 frames em 394 s**.
Passe via `LTX_COMFY_EXTRA_ARGS="--disable-dynamic-vram"`. O antigo "teto de 73
frames" registrado aqui **não era teto** — era esse mecanismo falhando. Ver §3.33.

**NÃO use `distilled-int8` nesta placa.** Ele tem 20 GiB e CABE na 3090, então o
ComfyUI o carrega inteiro (`full load: True`) e não sobra VRAM para o latente —
um plano de 129 frames travou duas vezes. O bf16 tem 39 GiB, não cabe, e o
gerenciador é obrigado a descarregar 19 GiB (`loaded partially`): o mesmo plano
fecha em 1087 s. **Caber inteiro é o problema, não a solução.** Em compensação o
int8 amostra mais rápido (5,3 s/passo contra 8,0) — ele serve para placa menor,
onde o bf16 nem começa. Não existe fp8 do 2.5. Ver §3.35.

**O `--ate animatic` NÃO custa segundos.** O animatic em si é montado em
segundos, mas os stills que ele consome são FLUX: **~1 min por plano** (medido
em 2026-08-27, 960x544, 9 e 6 planos). Uma cena de 9 planos leva ~10 min de GPU.
Continua sendo o ponto de revisão certo — o vídeo custa várias vezes mais —, só
não é de graça. E ele sobe o ComfyUI sozinho: confira a fila antes.

**Roteiro em prosa/prompt audiovisual passa por `prose_to_screenplay` primeiro**,
e o que ele descarta não volta. Ele preserva aparência (parêntese de
apresentação), idade e a linha `ESTILO VISUAL:` — as três coisas que os estágios
seguintes não têm como recuperar. Se um personagem sair com a aparência errada,
olhe `parse/screenplay_auto.txt` antes de olhar qualquer outra coisa: é lá que a
informação se perde ou sobrevive. Ver `MEMORIAL.md` §3.29.

### Paradas do `run_decupagem`

`--ate animatic` é o **ponto de revisão barato**: tudo já foi decidido (emoção,
voz, estrutura, enquadramento) e monta com os stills e as vozes reais, em
segundos, **sem GPU de difusão**. Revise ali antes de gastar ~1h numa cena de
1 minuto. Depois rode de novo com `--ate final`.

Estilos: `classico`, `tenso`, `nervoso`, `intimista`, `contemplativo`. Trocam
por segmento via `--style-changes "1:nervoso,3:contemplativo,5:tenso"` — vigora
a partir da cena marcada até a próxima marca.

**Não defina `CUDA_VISIBLE_DEVICES` para esta cadeia.** O `gemma4_worker.py`
faz `torch.cuda.set_device(1)` esperando as duas placas visíveis; quem precisa
da 3090 define a variável no subprocesso que inicia.

### Vozes XTTS

`speakers/01_vozes_emotivas/<ID>/<NN>_<emoção>.wav`, 17 slugs. O prefixo do ID
codifica gênero (`F`/`M`) e o `emotion_director` lê o ID para escolher a voz —
gênero é regra dura, não preferência. `script_pipeline/import_voices.py`
normaliza pacotes novos para essa convenção; o que ele não classifica vai para
`_nao_classificados/` em vez de virar palpite.

Duas coisas que decidem se um pacote serve: **6s+ por amostra** (Thorsten foi
rejeitado com 1,6–3,1s) e **fala, não canto** (samples de produção musical não
servem). Referência em inglês falando português mede **40% mais longa e metade
da energia** — o que alonga a montagem inteira. Ver `MEMORIAL.md` §3.27.

### Fish Speech — motor de TTS padrão da decupagem (desde 2026-09-03)

`E:\Users\home\Documents\fish-speech`, instalação própria (venv `uv`, `checkpoints\s2-pro`).
Já lê a MESMA biblioteca de vozes do XTTS acima — `dialogue_tts.py` clona a
partir do `xtts_speaker_wav` que `emotion_director`/`assign_voices` já escolhem,
sem precisar de biblioteca de referência própria.

**Precisa estar rodando ANTES de qualquer corrida de decupagem** — ao
contrário de XTTS/Qwen (carregam o modelo por chamada), o Fish é servidor
HTTP persistente:

```powershell
cd E:\Users\home\Documents\fish-speech
.\START_API.ps1
```

~1 min de carga, ~22 GB de VRAM (3090 — `CUDA_VISIBLE_DEVICES=1` já vem
setado no script), fica de pé em `http://127.0.0.1:8080`. Sem isso no ar, a
etapa 4 (TTS) da decupagem falha fala por fala com mensagem clara — não trava,
não tenta subir sozinho. `--engine auto`/`xtts` no `synthesize_dialogue.py`
volta ao caminho antigo sem precisar do servidor.

⚠️ **Sem `PYTHONUTF8=1`/`PYTHONIOENCODING=utf-8` no ambiente do servidor, texto
em coreano/chinês derruba a geração** com `UnicodeEncodeError` num `print()` de
debug do próprio fish-speech (`content_sequence.py::print_in_green`, console
Windows em cp1252) — MEDIDO 2026-09-03, os dois falharam sem a variável e
funcionaram com ela. `START_API.ps1` ainda não seta isso por padrão; se for
gerar em CJK, exporte as duas variáveis antes de chamar o script.

**Rich Emotion Library**: tags entre colchetes direto no texto —
`[whisper]`, `[laughing]`, `[angry]`, `[sad]`, `[excited]`, `[screaming]`,
etc. `EMOTION_TO_FISH_TAG` em `dialogue_tts.py` traduz sozinho o slug de
emoção que `emotion_director` já escreve por fala (os mesmos 17 do XTTS)
para a tag mais próxima — nem toda emoção tem tag equivalente exata.

**Testado nos 5 idiomas do pedido do usuário** (pt-BR, en, ko, zh, ja) + tags
de emoção em pt-BR, todos com voz clonada de um speaker XTTS — ver
`MEMORIAL.md` §3.53. Não avaliado: qualidade da clonagem comparada ao XTTS,
robustez das tags de emoção em roteiro real (só testado com frases soltas).

## Projeto relacionado

`E:\Users\home\Documents\ChoreoEngine` — biblioteca de movimento que extrai
danças de vídeo e compila faixas de condicionamento (pose) para estes pipelines.
Ver `ChoreoEngine/CLAUDE.md`.
