# CLAUDE.md — LTX-2-OPTIMIZED

Configuração desta máquina e do que já foi verificado nela. O `README.md` do
repositório descreve o projeto upstream (e ainda documenta a geração **19b**);
este arquivo cobre **o que existe aqui**: as gerações **2.3/22b** e
**2.5/22b**, ambas em produção, por caminhos de código diferentes.

Divisão de papéis: aqui fica a **configuração operacional verificada**
(hardware, ambientes, comandos, armadilhas). O `MEMORIAL.md` registra **por que
o sistema é assim** e o histórico das decisões — quando os dois divergirem, o
MEMORIAL é o mais detalhado e o mais recente.

Última verificação: 2026-09-30 (Fase 1 do plano de otimização de GPU IMPLEMENTADA E
VALIDADA: Ollama agora sobe pinado na GPU 0 via `start_ollama_gpu0.bat`, atalho de Startup
repontado -- ver seção "Ollama" abaixo. VALIDADO com GPU real: modelo de 23 GB carregou
inteiro na GPU 0, GPU 1 ficou livre. Fases 2-3 [LTX+LongCat em paralelo, despachante de
pool] seguem como plano, não implementadas. Ver `MEMORIAL.md` §3.143.
Verificação anterior: 2026-09-30 (review de código achou 2 buracos reais na validação de
texto do dia anterior -- `check_quotes`/`check_locations` procuravam fala/local em
QUALQUER lugar do roteiro-fonte, aprovando em silêncio fala/local colado na cena errada
[exatamente o bug que motivou a validação existir]; corrigido, agora checa primeiro
contra a PRÓPRIA cena do plano. `verify_output._check_gates` pulava em silêncio gate
ausente/ilegível; agora grava `status: missing/unreadable` explícito. Rodei 3 roteiros
reais pela `decupagem_ui` [webui, não só teste sintético] pra validar -- achou mais 2
bugs: relatório de correção de descritor mostrava "de X → para X" quando a diferença
ficava depois do corte de 120 chars [`_diff_excerpt` corrige]; e um prompt LTX de
parágrafo único sem nenhum meio de animação declarado saiu com "ESTILO VISUAL:
polished 3D animation" -- MESMA família de alucinação do §3.139, frase diferente do
guard de ontem, que só pegava a frase exata do exemplo; `extract_art_direction()`
ganhou `source_text` opcional pra descartar qualquer termo de animação sem base no
texto original. `minimax_h3_backend.py` ganhou 4 flags opt-in de um workflow de
terceiro [VAE int8, chunking de FF/atenção, sigma shift, block-sparse attn] --
NENHUMA validada com GPU real (o teste da VAE int8 mediu quase 2× mais rápido mas está
INVALIDADO por confusão com cache de disco entre as duas rodadas, refazer antes de
confiar). REENTRY WINDOW recriado do zero com todas as correções -- identidade dos
personagens correta pela primeira vez, mas travou no gate visual [41/46 reprovados,
causa dominante é `location_match`, NÃO investigada ainda -- reforço de enquadramento
`wide` aplicado resolve só 1 dos 41]. Ver `MEMORIAL.md` §3.142 (pendências no fim da
seção).
Verificação anterior: 2026-09-29 (nova etapa `[V validação de texto]` em `run_decupagem.py`,
`script_pipeline/text_validation.py`: compara o `shot_plan.json` INTEIRO contra o
roteiro-fonte, AINDA EM TEXTO, antes de qualquer still -- personagem/fala/local sem base no
roteiro [relata], descritor divergente entre planos [CORRIGE de verdade contra o canônico do
`cast.json`, regrava o `shot_plan.json`]. `--text-validation {relatorio,bloquear,off}`. Exige
`parse/screenplay_original.txt` [novo, run_decupagem agora salva o roteiro-fonte bruto na
primeira execução -- antes só o texto já reestruturado por LLM ficava salvo]. Testado no
REENTRY WINDOW: 0 problema, confirma que os fixes de ontem já deixaram os descritores certos.
Ver `MEMORIAL.md` §3.141. Também avaliado GLM-4.1V-9B-Thinking [venv isolado em
`W:\vision_models\glm41v_env`, não no `.venv` principal] -- multi-imagem funciona de verdade
[ao contrário do GLM-4V-9B antigo], 7,7 GB de VRAM, mas mais lento pelo raciocínio longo da
variante "Thinking". Ver `MEMORIAL.md` §3.140 (adendo).
Verificação anterior: 2026-09-29 (Ollama migrado pro W: -- achado no processo: `G:\ollama\models`
[162 GB] NÃO é o catálogo de produção, é outro catálogo não relacionado; o real
[qwen3-vl:30b, qwen2.5vl:7b, qwen2.5:32b-instruct-q4_K_M etc, 80 GB] vive em
`C:\Users\user\.ollama\models`, o default do app desktop -- ver seção "Ollama" abaixo. Fidelidade
de descritor em `cast_characters.py` ganhou checagem por ATRIBUTO [comprimento/cor de cabelo,
cor de roupa], não só sobreposição geral de palavras. GLM-4V-9B avaliado como candidato pro gate
visual: ótimo pra percepção SEM referência [mais leve, mais rápido, JSON mais confiável], mas
esse checkpoint tem limite arquitetural de UMA imagem só por conversa -- não serve pra comparação
de identidade [a maioria dos closes de personagem]. Ver `MEMORIAL.md` §3.140.
Verificação anterior: 2026-09-29 (REENTRY WINDOW, roteiro sci-fi no MiniMax: 3 fixes reais e 1
lacuna estrutural achada -- decupagem por PLANO nunca tinha âncora de fotorrealismo
[`shot_plan.py::_storyboard_prompt()`, diferente do prompt por CENA que já tinha]; causa raiz do
estilo ilustrado era um EXEMPLO de sintaxe em `prose_to_screenplay.py` ["ESTILO VISUAL: polished
hand-drawn cel animation..."] copiado literalmente pelo LLM sem o roteiro pedir estilo nenhum
-- corrigido + guard em `parse_screenplay.py::extract_art_direction()`; novo
`--gate-max-regen-fraction` em `run_decupagem.py` (limite de 40% que travava QUALQUER retry
agora é ajustável); modelo de DECISÃO do gate visual trocado de qwen3-vl:30b para
qwen2.5:32b-instruct-q4_K_M [não precisa ver imagem, só texto] -- percepção continua no VLM.
Mesmo com todos os fixes, a produção foi PAUSADA a pedido do usuário: a nave "Kestrel" não tem
referência visual fixa [só personagens têm via character-sheet], causando alucinações graves
[nave muda de modelo, vira avião, mão segurando como brinquedo, águia, personagem duplicado].
Testes novos 9 casos, suíte 399 passando, nada commitado. Ver `MEMORIAL.md` §3.139.
Verificação anterior: 2026-09-29 (folha de personagem 4-angulos+close numa geracao so via
`qwen_image21_comfy_backend.generate_character_sheet_convrot()` -- checkpoint dedicado
`qwen_image_2.1_int8_convrot.safetensors` [NAO GGUF, 7,26 GB] + LoRA Viggle Turbo 4-step,
workflow do usuario "Qwen Image 2.1 Image Edit Viggle 4-Step"; `build_workflow()` ganhou
`unet_gguf=False` [troca `UnetLoaderGGUF` por `UNETLoader`], `attention_backend` e
`model_sampling_flux`. 1-3 referencias reaproveitam o `TextEncodeQwenImage21` ja em producao no
motor GGUF Q4_K_M. VALIDADO com GPU real: 1 referencia (MIN-JUN, Cerco em Seul), sucesso de
primeira, 102,7s com servidor ja carregado, identidade e ate um prop da referencia (walkie-talkie)
preservados nas 5 vistas. Testes sem GPU 8/8, suite inteira 390 passando. Falta decidir se vira
substituto do `character_sheet.py` atual (GGUF Q4_K_M) -- so testado com 1 referencia ate agora.
Ver `MEMORIAL.md` §3.138.
Verificação anterior: 2026-09-28 (`flux-krea`/`flux-kontext` consertados -- `G:/models` nunca
estava mapeado no `extra_model_paths.yaml`, por isso os dois falhavam com arquivo não
encontrado apesar de já cadastrados há meses; `flux1-kontext-dev.safetensors` nem existia em
disco, baixado agora -- e motor novo `--image-engine flux-klein-gguf` (FLUX.2 Klein
quantizado GGUF Q8_0, `unsloth/FLUX.2-klein-9B-GGUF`). VALIDADO com GPU real os três: Klein
GGUF ~5,3s/8 passos, Krea ~17,5s/28 passos (ambos carregam inteiros na 3090, sem offload),
Kontext 253,6s. Achado importante: a lentidão de ~5 min/still vista na avaliação dos 4
roteiros com Z-Image era contenção de VRAM/RAM (Fish-Speech + Z-Image residentes), não um
problema do FLUX.2 Klein em si -- com a GPU livre ele roda em segundos. Detalhes em
`MEMORIAL.md` §3.136.
Verificação anterior: 2026-09-27 (Prévia 3D automática: `script_pipeline/previs_spec.py` gera o spec
espacial da decupagem sozinho -- antes exigia escrita manual -- e renderiza manequins/câmera no
Blender para os planos complexos, `--previs3d {complexas,medias,todos,off}` em `run_decupagem`;
corrigiu de quebra um bug de lateralidade no manequim antigo (mão "direita" no lado errado).
VALIDADO com Blender real na corrida CERCO EM SEUL v2 (25 planos, spec + 3 renders complexos) e
sondado sem erro nas 56 corridas existentes no disco; review adversarial (4 lentes) achou e corrigiu
6 defeitos reais (cabeça cortada no medium, quem sai de cena reaparecia no quadro final, pessoas
empilhadas na mesma marca, cena híbrida virando exterior por engano, vínculo velho de still
sobrevivendo a `--previs3d off`, regex de "mira alta" com falso positivo); detalhes em
`MEMORIAL.md` §3.135).
Verificação anterior: 2026-09-27 (CERCO EM SEUL v2: correções gerais de decupagem
-- insert por exclusão, figurantes sem nome no elenco com `extra`/`aliases`, descritor do
parceiro sempre presente, close de fala como reação, movimento SEMPRE aplicado (não estava
no filme entregue), fallback de fala do TTS no lip-sync, voz por gênero/idade com foto --
filme refeito no LTX 2.5 e no MiniMax long-take (take limitado a 4 planos, fallback avulso);
gate visual estava quebrado desde 26/09 pelo Ollama 0.34.4, corrigido; detalhes em
`MEMORIAL.md` §3.131-3.133).
Verificação anterior: 2026-09-25 (Licon MSR V2 INTEGRADO na decupagem --
`--ic-reference msr25` -- e VALIDADO com GPU real no caminho de PRODUÇÃO
de verdade, não mais só chamada direta ao backend: `shot_ic_spec("msr25",
...)` com um plano real (HA-EUN + JI-HO, framing wide), passado pro
`ltx25_backend.generate()` exatamente como `render_shots.py` faz. Usuário:
"confirmado, funcionando bem." Estado: motor de referência utilizável na
decupagem, não mais só mecanismo validado isolado; detalhes em
`MEMORIAL.md` §3.130).
Verificação anterior: 2026-09-25 (Licon MSR V2 -- as 5 pendências da §3.128
fechadas: 3+ referências, background, `audio_ref1`/2 (AVref) implementado e
validado com lip-sync em coreano sobre `w4a8-v10`, e I2V+wide testado sem
reproduzir o vazamento de guia do MSR 2.3 -- 5/5 corridas de GPU aprovadas
pelo usuário, zero erro técnico; falta só a integração de produto no
`render_shots`/decupagem, o mecanismo em si já funciona; detalhes em
`MEMORIAL.md` §3.129).
Verificação anterior: 2026-09-25 (Licon MSR V2 LIGADO ao pipeline 2.5 --
`ltx25_backend.generate(msr=...)` -- e VALIDADO com GPU real na primeira
tentativa: 2 personagens nomeados (HA-EUN + JI-HO), usuário confirmou
"manteve os dois personagens consistentes"; ainda faltam enquadramentos
abertos/áudio de referência/variantes quantizadas; detalhes em
`MEMORIAL.md` §3.128).
Verificação anterior: 2026-09-25 (ancoragem textual de figurino/cabelo em todo
segmento do `minimax-longtake` -- `_anchor_appearance()` -- VALIDADA com GPU
real: usuário confirmou "manteve a consistência, cabelo preso e molhado,
roupa" num take de 3 planos, resolvendo a deriva de estado do teste
anterior; e baixados o LoRA Licon MSR V2 e o custom node ComfyUI-LTX2.5-MSR
para o 2.5 [não ligados no pipeline ainda]; detalhes em `MEMORIAL.md`
§3.126/§3.127).
Verificação anterior: 2026-09-25 (auditoria geral + reforço de identidade no
`minimax-longtake`: TESTADO COM GPU REAL E REJEITADO -- trava o
comfyui-easy-media, GPU 100% sem terminar o segmento 1; achado e corrigido
bug do watchdog do MiniMax H3 no processo (1800s matava take grande
saudável, subiu pra 3600s); take de 6 planos SEM reforço VALIDADO com GPU
real, 1445s, além do máximo anterior (4); checkpoint turbo LightX2V
confirmado como LoRA já em produção (não mais "não testado"); `tensorxx_ge`
(TensorRT do LTX 2.3, engine real validada) documentado pela primeira vez
aqui; detalhes em `MEMORIAL.md` §3.126).
Verificação anterior: 2026-09-25 (Qwen-Image-2.1 ganhou referência com PAPEL
NOMEADO -- `<image1> is X.` -- em `generate_storyboards.py`/`render_shots.
py`; capacidade já existia no backend ComfyUI (10 refs), só não tinha
integração; achado e corrigido bug onde `cast_descriptors` nunca carregava
fora do motor LTX; NÃO testado com GPU real ainda; detalhes em
`MEMORIAL.md` §3.125).
Verificação anterior: 2026-09-24 (CERCO EM SEUL: filme final ENTREGUE, 33s/12
planos -- decupagem de 25 planos trocada por 12 quadros de storyboard
curados depois do gate reprovar 20/25 stills do FLUX; `minimax-longtake`
agrupou em 4 takes reais automaticamente, maior grupo testado até agora (4
planos); detalhes em `MEMORIAL.md` §3.124).
Verificação anterior: 2026-09-24 (template do long take movido pra
`comfyui_workflows/` -- estava fora do git por causa do `*.json` genérico
do `.gitignore`; e os 4 atores importados pro cast.json do CERCO EM SEUL
via `import_reference.py`; detalhes em `MEMORIAL.md` §3.122/§3.123).
Verificação anterior: 2026-09-24 (quebra de take wide/full VALIDADA com GPU
real: planos 6-9 do CERCO EM SEUL, wide no meio abriu 2 takes de 2 planos,
480s total, 4/4 clipes; detalhes em `MEMORIAL.md` §3.121).
Verificação anterior: 2026-09-24 (agrupamento do `minimax-longtake` trocado de
"cena inteira" pra "take" real -- wide/full abre take novo, pedido do
usuário depois de ver que o CERCO EM SEUL inteiro caía numa cena só; NÃO
validado com GPU real ainda; detalhes em `MEMORIAL.md` §3.120).
Verificação anterior: 2026-09-24 (`engine="minimax-longtake"` VALIDADO com GPU
real ponta a ponta numa cena curta do CERCO EM SEUL, 3/3 clipes; achado e
corrigido bug de path na cópia final `output/video/` vs `output/`; detalhes
em `MEMORIAL.md` §3.119).
Verificação anterior: 2026-09-24 (`engine="minimax-longtake"` cabeado em
`render_shots`/`render_shots_stage`/`run_decupagem`/`decupagem_ui` -- agrupa
planos da MESMA CENA num take só; NÃO validado com GPU real ainda, só
sintaxe/testes mockados; detalhes em `MEMORIAL.md` §3.118).
Verificação anterior: 2026-09-24 (workflow "long take" multitrack do MiniMax H3
VALIDADO com GPU real 2x e integrado ao `minimax_h3_backend.py` como
`generate_longtake()`; detalhes em `MEMORIAL.md` §3.117).
Verificação anterior: 2026-09-24 (checkpoints turbo do MiniMax H3 [LightX2V]
baixados para teste de velocidade — ver seção MiniMax H3 abaixo — NÃO
validados com GPU real ainda; detalhes em `MEMORIAL.md` §3.116).
Verificação anterior: 2026-09-20 (estado espacial 3D integrado à WebUI de
decupagem e animatic de cabine de 20 s do Voo 702; detalhes em `MEMORIAL.md` §3.97 e
`script_pipeline/SPATIAL_PIPELINE.md`).
Verificação de 2026-09-19: gate visual de produção com Qwen3-VL 30B entre
stills/vídeo e vídeo/montagem; detalhes e causa raiz em `MEMORIAL.md` §3.93.
Verificação de 2026-09-18: diretor de ação/movimento: InterGen/Inter-X -> vídeo de pose ->
LTX IC-LoRA Union-Control ou MiniMax H3 ControlNet, `script_pipeline/motion_director.py`; 3/3
planos reais renderizados em produção pelo caminho LTX; caminho MiniMax validado com GPU real
até o penúltimo nó do grafo -- ControlNet do H3 segue bloqueado por VRAM insuficiente para
UNET+ControlNet+VAE simultâneos. Estado atual e decisões abertas: `MEMORIAL.md` §7 P0 e
§3.87-3.91 (auditoria de scripts anterior, 2026-09-16, em §3.86, ainda não commitada).
Endurecimento do gate visual + camada de correção (regeneração só dos planos reprovados, fala longa
dividida em planos ≤ 6 s, master de áudio -16 LUFS / -1,5 dBTP, referência de locação reprovada descartada):
`MEMORIAL.md` §3.95 — flags `--visual-max-retries`, `--max-speech-seconds`, `--no-master-audio`.
Para a decupagem do Voo 702, o contrato de locação/objetos/voo/personagens e o
diagnóstico das folhas de referência estão em `script_pipeline/CONTINUIDADE_VOO702.md`.

---

## Hardware

⚠️ **MUDANÇA DE HARDWARE 2026-09-30: segunda RTX 3090 instalada, a RTX 4070 saiu.**
Isso invalida a convenção de índices documentada há meses (ver aviso abaixo) e abre uma
oportunidade real de paralelismo que antes não existia — plano completo de otimização e
distribuição em `MEMORIAL.md` §3.143. Resumo do que já mudou e do que ainda não:

| | |
|---|---|
| GPU 0 (nvidia-smi) | **RTX 3090, 24,5 GB** |
| GPU 1 (nvidia-smi) | **RTX 3090, 24,5 GB** (nova, substituiu a RTX 4070) |
| RAM | **79,9 GB** (medido 2026-08-27; ~35 GB livres com o LTX 2.5 carregado) |
| Pagefile | C: 38 GB + E: 31,2 GB — praticamente ocioso (0,9 + 0,1 GB em uso) |
| Driver | 610.74 |

⚠️ **A RAM foi ampliada de 48 para 80 GB e a documentação não acompanhou.**
Isso importa além do registro: `LTX_TRANSFORMER_CPU_MEMORY=34GiB`, que a seção
da ICLoraPipeline mais abaixo usa, foi dimensionado para a máquina de 48 GB.
Com 80 GB dá para dar bem mais folga ao offload da rota 2.3 — não testado. Com DOIS
processos pesados rodando ao mesmo tempo (um por GPU, ver §3.143), essa folga de RAM
fica mais apertada — cada offload consome RAM própria, e os dois podem competir.

⚠️ **A convenção antiga de índices ("nvidia-smi 0 = torch cuda:1") MORREU.** MEDIDO
2026-09-30 (`torch.cuda.get_device_properties(i).uuid` contra `nvidia-smi -L`): com as
duas placas IDÊNTICAS agora, **os índices do nvidia-smi e do torch batem 1:1**
(nvidia-smi 0 = torch cuda:0, nvidia-smi 1 = torch cuda:1) — o desalinhamento antigo só
existia porque a 4070 e a 3090 eram enumeradas em ordens diferentes pelos dois sistemas.
Isso **não quebra nada que já funcionava** (o código inteiro usa `CUDA_VISIBLE_DEVICES=1`,
que continua apontando pra uma 3090 de verdade — só que agora as DUAS são 3090, então
"1" não tem mais nada de especial). Mas **NÃO reintroduza a checagem antiga "escolha pela
VRAM total, não pelo índice"** em código novo sem saber por que ela existia — ela resolvia
um problema (placas diferentes) que não existe mais; o problema de hoje é o oposto,
**decidir COMO DIVIDIR duas placas iguais**, não como diferenciá-las.

**Todo o código do repositório está hardcoded para `CUDA_VISIBLE_DEVICES=1`** (LTX 2.5,
MiniMax H3, LongCat, `render_scenes.py`, `motion_director.py`, `continuous_chain.py`, mais
de 20 arquivos — catálogo completo em `MEMORIAL.md` §3.143) — herança de quando só uma das
duas placas era forte o bastante. Isso **continua funcionando sem erro**, só que agora
significa que **a GPU 0 (a segunda 3090) fica ociosa o tempo todo**, exceto pelo Ollama
(que escolhe GPU sozinho e pode estar usando as duas sem coordenação com o resto). Nenhum
código foi alterado ainda para usar a segunda placa — é trabalho de otimização em aberto,
não um bug. Ver plano de fases em `MEMORIAL.md` §3.143 antes de mexer em qualquer um dos
mais de 20 arquivos que hardcodam `"1"`.

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

⚠️ **DESDE 2026-09-30, o Ollama sobe PINADO NA GPU 0.** Fase 1 do plano de otimização de GPU
(`MEMORIAL.md` §3.143, IMPLEMENTADA E VALIDADA): o atalho de Startup (`Ollama.lnk`) foi
repontado pra `start_ollama_gpu0.bat` (raiz do repo), que seta `CUDA_VISIBLE_DEVICES=0` +
`OLLAMA_MODELS=W:\ollama\models` só pro processo do Ollama antes de subir `ollama app.exe` —
não é uma variável de ambiente global, os subprocessos deste repo (que já fixam
`CUDA_VISIBLE_DEVICES=1` no próprio ambiente) não são afetados. Atalho original preservado
como `Ollama_original_backup.lnk` na mesma pasta de Startup, pra reverter se precisar. VALIDADO
com GPU real: `qwen2.5:32b-instruct-q4_K_M` (23 GB) carregado via `/api/generate` foi inteiro
pra GPU 0, GPU 1 ficou em 56 MiB o tempo todo. Efeito: o `unload_all()` que `run_decupagem.py`
chama antes de FLUX/LTX **deixa de ser estritamente necessário** (mantido no código como
salvaguarda) — o Ollama pode ficar residente sem disputar VRAM com o motor de imagem/vídeo.
Se reiniciar a máquina e o Ollama voltar a carregar na GPU 1, confira se o `.lnk` de Startup
ainda aponta pro wrapper (pode ter sido sobrescrito por uma atualização do app Ollama).

⚠️ **CORRIGIDO 2026-09-29 — esta seção estava invertida.** Os modelos de PRODUÇÃO (qwen3-vl:30b,
qwen2.5vl:7b, qwen2.5:32b-instruct-q4_K_M, qwen3.6-35b-a3b:latest, gemma4:latest,
mistral-nemo:12b-instruct-2407-q4_K_M — 80 GB) vivem em **`W:\ollama\models`** (migrado de
`C:\Users\user\.ollama\models`, o default do app desktop quando nenhuma `OLLAMA_MODELS` é
passada — confirmado lendo `C:\Users\user\AppData\Local\Ollama\server.log`). `G:\ollama\models`
(162 GB) é um catálogo SEPARADO e NÃO RELACIONADO (qwen3:30b, deepseek-coder-v2, phi4,
gemma4-32k, qwq:32b — nenhum dos modelos usados por este projeto) — não confundir os dois.
Suba com `OLLAMA_MODELS=W:\ollama\models` explícito no ambiente do processo (o `setx` que fixa a
variável no registro do Windows só vale pra sessões/processos NOVOS, não propaga pra shells já
abertos) — se subir sem essa variável, o app desktop volta a ler `C:\Users\user\.ollama\models`
(mantido intacto como backup) e serve o catálogo certo do mesmo jeito, só que do lugar antigo.
`choreo/prompt_to_params.py` detecta o catálogo de uma instância no ar e diz quais modelos ela
realmente serve — use isso pra checar rápido de qual diretório o servidor atual está lendo. Ver
`MEMORIAL.md` §3.140.

**O Ollama é o motor padrão da extração de roteiro** desde 2026-08-23
(`script_pipeline/`, `storyplay25`), medido contra o gemma4-e2b local: 6s
contra 91s, 4/4 falas preservadas verbatim contra 2/4, 9 planos contra 5.

**Auditoria visual de produção (2026-09-19):** `qwen3-vl:30b` está instalado
no Ollama (19 GB no catálogo; ~22 GB carregado, 100% GPU na RTX 3090) e é o
padrão de `script_pipeline/visual_continuity_audit.py`. O gate roda depois dos
stills e depois dos clipes. Usa uma chamada de percepção neutra e outra de
decisão contra o contrato; não use `gemma4:latest` como substituto nesta máquina,
pois a entrada visual produziu descrições incorretas e falsos bloqueios. Ver
`MEMORIAL.md` §3.93 e `script_pipeline/CONTINUIDADE_VOO702.md`.

**Liberação de RAM/VRAM (2026-09-19):** todo encerramento de
`script_pipeline.run_decupagem` consulta `/api/ps` e descarrega os modelos
residentes com `keep_alive: 0`. A auditoria visual descarrega o VLM logo após
cada gate, antes de devolver a GPU ao FLUX/LTX. `decupagem_ui.py` repete a
limpeza ao terminar uma corrida, no botão Parar e via `atexit` quando a WebUI
fecha. O servidor Ollama continua ativo. Implementação compartilhada em
`script_pipeline/ollama_runtime.py`; falha de limpeza nunca substitui o código
de saída real da produção.

⚠️ **O Ollama se atualiza sozinho** (0.34.4 em 2026-09-26) e isso quebrou o gate em silêncio:
o `qwen3-vl:30b` passou a raciocinar mesmo com `think: false` (nem "/no_think" desliga), e com
`num_predict` 2000 o JSON saía cortado ou vazio. O relatório mostrava "bloqueado", mas era
**erro do auditor**. Corrigido com `num_ctx` 16384 / `num_predict` 8192. Se o gate voltar a
parar em 3 ou 4 planos, confira `ollama --version` e o campo `auditor_errors` do relatório antes
de acreditar nas reprovações. Ver `MEMORIAL.md` §3.133.

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

## `tensorxx_ge/` — TensorRT para o Gemma do LTX 2.3 (experimento isolado, fora dos caminhos de produção)

Não modifica os caminhos FP8/GGUF/V2/V3/WebUI existentes — único ponto de integração é o cache
nativo de prompt-embedding. Roda o **prefill do Gemma em TensorRT na RTX 4070** (`cuda:0`), mantendo
o extrator de features e os conectores do LTX em PyTorch na 3090 (`cuda:1`). `start_webui_tensorrt.bat`
sobe `tensorxx_ge.webui` na porta 7861.

**Existe engine real, construída e validada** (`tensorxx_ge/engines/gemma_hidden_states_bf16_ws.engine`
e `_v2_bf16_ws.engine`, ~23,5 GB cada) — cosine ≥ 0,9995 contra a referência PyTorch nas 49 camadas do
Gemma, 0,999997 na conditioning projetada. O `README.md` do módulo tinha ficado desatualizado dizendo
"nenhuma engine fabricada"; corrigido em 2026-09-25 (ver `MEMORIAL.md` §3.126). Diferença conhecida
contra a rota padrão (SSIM 0,85) vem do `accelerate` offload que a rota padrão usa pro Gemma
(`LTX_TEXT_ENCODER_GPU_MEMORY=2GiB`), não é bug da engine — sem offload, o LTX bate exatamente com ela.

Escopo: **só LTX 2.3**, não tem equivalente para o 2.5.

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
| `LTX25_VARIANT` | **`w4a8-v10`** (desde 2026-09-12; antes `distilled`) | `distilled` = bf16, mais lento e instável. `dev` = CFG real, negative prompt funciona, várias vezes mais lento. `gguf-q6k` = ver abaixo. **`storyplay25` continua em `distilled`**: keyframes não testados com o 4-bit |
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

### Escada de quantização medida: `w4a8-v10` é o padrão desde 2026-09-12

MEDIDO 2026-09-12, duas baterias (rosto e ação) a 1280x704 com 121 frames.
Detalhe em `MEMORIAL.md` §3.77.

| variante | rosto | ação | qualidade (usuário, ação) |
|---|---|---|---|
| `distilled` bf16 | 720s* | 1036s | pior |
| `gguf-q6k` 6-bit | 277s | 439s* | 3º |
| `w4a8-v10` 4-bit | 287s | **331s** | 2º |
| two-stage bf16 | 555s | 923s | 1º, mas **mudou a idade** da personagem no teste de rosto |

\* inclui boot do ComfyUI.

- **Mais bits não compraram qualidade visível.** O ranking com uma seed só
  não prova que 4-bit é *melhor*.
- ⚠️ **O "~3×" das baterias é majoritariamente CARGA do modelo, não
  amostragem.** Com os dois modelos já carregados (mesmo plano de 249
  frames, 960x544, I2V + fala): `w4a8-v10` **575s** contra bf16 **896s** =
  **~1,6×**. A carga do bf16 (39 GiB, com offload) é enorme e varia muito: o
  primeiro plano bf16 depois do `w4a8-v10`, com só 121 frames, levou 1990s.
  Numa decupagem que carrega o modelo uma vez e renderiza vários planos, o
  ganho real fica perto de 1,6×. Trocar de variante no meio da corrida é o que
  faz aparecer os 3×.
- **O bf16 é instável** (+44% entre baterias, contra +15% do `w4a8-v10`),
  porque não cabe na 3090 e depende de offload.
- **Não existe "mais passos" no `distilled`**: 8 sigmas fixos; `--steps` só
  vale no `dev`.
- **Two-stage não serve para personagem recorrente** (risco de identidade).
  Use em ação/ambiente.
- `w4a8-v10` validado em plano longo (337 frames, 262s, sem travar) e
  **no caminho real da decupagem** (I2V com still + `audio_conditioning` com
  fala do TTS, 2 planos, sem erro). **Ainda não testado**: keyframes.
  **Veredito do usuário (2026-09-12): `w4a8-v10` com maior qualidade nos dois
  planos**, e também na bateria de ação. São 3 de 3 comparações contra o bf16.
- **PADRÃO TROCADO para `w4a8-v10` em 2026-09-12**: backend, `run_decupagem`,
  `decupagem_ui` e os `.bat` de decupagem, webui, cinema, screenplay e music
  video. **Exceção: `storyplay25` continua `distilled`** (usa keyframes, sem
  teste). Reverter: `LTX25_VARIANT=distilled`. Ver `MEMORIAL.md` §3.77.

### LoRAs e IC-LoRAs de vídeo no 2.5 (desde 2026-09-12)

Catálogo em `ltx_loras.py`: tipo, força, gatilho e compatibilidade de cada um.
`python ltx_loras.py list`, `status --compat`, `download --set core|gated`.

- **LoRA treinado no 2.3 carrega no 2.5.** VERIFICADO de três jeitos: cabeçalho
  (1772 pesos lineares idênticos, 48 blocos, 0 shape divergente), workflows
  oficiais 2.5 vendorizados (carregam IC-LoRAs 2.3) e model card do LTX-2.5.
  Carregar não prova efeito igual — validar vendo.
- Arquivos: `models/loras/` (treino 2.3) e `models/2.5/loras/` (treino 2.5),
  pastas planas (o ComfyUI lista por nome).
- Uso: `ltx25_backend.generate(..., loras=[(arquivo, força)], ic_lora={...})`;
  decupagem `--video-lora chave[:força]` (repetível) e
  `--ic-reference {off,ingredients,msr}`; UI 7913, aba Motores. **`--lora`
  continua sendo dos STILLS** — são outros checkpoints.
- O backend insere `LTXVCropGuides`: o sampler do grafo T2V oficial não tem, e
  sem ele a guia IC do tamanho do clipe sai decodificada no fim do vídeo.
- ⚠️ **IC-LoRA com fator de referência > 1 (union-control, motion-track) +
  `audio_conditioning` é recusado com erro.** `LTXVSetAudioVideoMaskByTime` só
  preserva máscara por quadro; a guia reduzida vira máscara espacial e seria
  re-ruída em silêncio (LIDO em `ComfyUI-LTXVideo/latents.py`). Fator 1
  (ingredients, msr, cameraman, cdrama-canny) funciona com fala.
- ⚠️ **LoRA sobre `w4a8-v10`**: com o modelo inteiro na placa o ComfyUI funde o
  LoRA e requantiza o peso a 4 bits. Antes de concluir que um LoRA "não faz nada",
  compare com `gguf-q6k` ou `distilled`.
- MSR: a 2.3 V2 roda com nós nativos (sequência montada em
  `script_pipeline/ic_references.py`). **A 2.5 usa o custom node
  ComfyUI-LTX2.5-MSR** (slot embedding real, não pseudo-vídeo -- até 5
  referências + 2 áudios nativos), clonado em `ComfyUI/custom_nodes/
  ComfyUI-LTX2.5-MSR`; LoRA V2 em `models/2.5/loras/
  ltx-2.5-licon-msr-v2.safetensors` (2,22 GB). **LIGADO e VALIDADO com GPU
  real em 2026-09-25, 5/5 corridas aprovadas pelo usuário** --
  `ltx25_backend.generate(msr={"lora", "strength", "images": {"pic1".."pic4",
  "background"}, "audio": {"audio_ref1", "audio_ref2"}, "guide_strength",
  "reference_frames", "describe"})`, mutuamente exclusivo com `ic_lora`
  (mesmo ponto do grafo, antes do `LTXVConcatAVLatent`) e incompatível com
  `two_stage`. Testado e aprovado: 2 e 3+ personagens nomeados, `background`
  como cenário, `audio_ref1` (AVref) com lip-sync coerente em coreano sobre
  `variant="w4a8-v10"` (180s, bem mais rápido que `distilled`), e I2V +
  enquadramento wide **sem reproduzir** o vazamento de guia do MSR 2.3
  (linha abaixo) -- uma amostra só, não é prova estatística, mas
  contraindica portar `MSR_SEM_ENQUADRAMENTO_ABERTO`/`MSR_MAX_FRACAO` sem
  necessidade (mecanismo é arquiteturalmente diferente: slot embedding em
  posições temporais negativas, não pseudo-vídeo). **INTEGRADO na decupagem**
  (`--ic-reference msr25`, `ic_references.shot_ic_spec()` modo `"msr25"`,
  `decupagem_ui.py` dropdown) e VALIDADO com GPU real no caminho de
  produção -- não mais só chamada direta ao backend. `ltx_loras.py` já tem
  entrada de catálogo pro V2 (`msr-2.5-v2`). **Não testado ainda**: corrida
  REAL de `run_decupagem` ponta a ponta (múltiplos planos, cache, mistura
  de motores) -- só o mecanismo de montagem do spec + geração de UM plano
  foi validado. Ver MEMORIAL 3.127/3.128/3.129/3.130.
- ⚠️ **MSR em I2V vaza a guia** (MEDIDO 2026-09-13, bf16 e w4a8, força 1,0 e 0,5): com
  still aberto o vídeo larga o still no quadro 1 e vira o plano médio dos retratos; com
  guia de 65 quadros num clipe de 73 corta seco no 57. O oficial é T2V puro. Por isso
  `ic_references` só liga MSR fora de wide/full/insert/establishing e com a guia em no
  máximo ⅓ do clipe (<51 quadros = sem MSR). `script_pipeline/guide_leak_audit.py` detecta
  (corte interno + `salto_still`) e roda sozinho no `run_decupagem` com IC ligado.
  Ver `MEMORIAL.md` §3.82.
- **Cache**: o still entra na chave do clipe e o clipe na do lip-sync (antes, still novo
  seguia com clipe velho). UI 7913, aba Stills: "🧹 Limpar cache". `MEMORIAL.md` §3.81.
- Conteúdo: `motion-enhancer-n4w` é afinado para NSFW; `talking-head-av` é de UM
  personagem do autor; `cdrama-char` puxa os rostos dos atores da série.
- Oficiais com licença (aceita 2026-09-13) BAIXADOS: Ingredients, Cinemagraph,
  Pixel-Upscaler, Deblur e Clean-Plate 2.5; Relight e DubIt 2.3 — todos 0 chave
  faltando no 2.5. `--ic-reference ingredients` passa a usar o Ingredients 2.5
  sozinho. Cinemagraph é LoRA comum (`--video-lora cinemagraph-2.5`). V2V ligados
  desde 2026-09-13 (`MEMORIAL.md` §3.79): DubIt como `--lipsync-engine dubit`, Deblur e
  Pixel-Upscaler como `--post-deblur` / `--post-upscale`; clean-plate e relight seguem
  sem fluxo. MEDIDO: DubIt não superou o LatentSync em sync.
  Opcionais não baixados: `download --set gated-extra`.
- Testes sem GPU: `tests/test_ltx_loras.py`, sobre a fixture do grafo real.
  ⚠️ `tests/test_pipeline_smoke.py::test_build_workflow_nao_toca_rede` **sobe o
  ComfyUI** apesar do nome (`base_api` → `ensure_server`, lido no código): não
  rodar a suíte inteira com a GPU ocupada.

Detalhes e medições: `MEMORIAL.md` §3.78.

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
| `Qwen-Image-2.1\\start_webui.bat` (imagem por texto/referências, instalação externa) | 7862 / 8192 |
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

### Dois arquivos novos instalados de `W:\temp` (2026-09-30, NENHUM testado ainda)

Achados soltos em `W:\temp` (destino errado -- pasta de trabalho, não a de modelos), avaliados
por metadata do próprio `.safetensors` (sem precisar carregar o tensor) e movidos pro lugar
certo:

- **`models/loras/minimax_h3_lora_dianying_jingtou_cinematic-shot_ai-toolkit.safetensors`**
  (148 MB) -- LoRA de verdade (416 tensores `lora_A`/`lora_B`, `ss_base_model_version=minimax_h3`,
  treinado com `ai-toolkit`). O nome original vinha com o nome chinês do LoRA codificado em hex
  UTF-8 no próprio nome do arquivo (`...E79C9FE5AE9E...` = "电影镜头", "lente/enquadramento de
  cinema") -- por isso o rename. **Não testado.**
- **`models/diffusion_models/Minimax-h3_Singularity_int8_unpruned.safetensors`** (34 GB) --
  checkpoint completo (1035 tensores, prefixo `model.diffusion_model.*` -- ATENÇÃO, é diferente
  do prefixo `diffusion_model.*` sem `model.` do checkpoint w4a8 já em produção
  [`LONGTAKE_UNET_FILENAME`], então pode precisar de ajuste no loader antes de funcionar).
  Variante INT8 SEM pruning do mesmo finetune de terceiro "Singularity" já usado no
  `minimax-longtake` (que hoje usa a versão w4a8 PRUNED, 11,8 GB) -- maior e provavelmente mais
  lento, mas com potencial de qualidade maior por não ter perda de poda. **Não testado, não
  ligado a nenhuma variável de ambiente.**
- **`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` -- DUPLICATA EXATA
  removida** (hash MD5 idêntico ao arquivo já em produção em `models/loras/`).
- **`bunnyH3ConditioningBridge_v10.safetensors` (22 MB) -- NÃO instalado, ainda em
  `W:\temp`.** Não é LoRA nem checkpoint: é uma ponte de destilação pequena (MLP de 3 camadas,
  metadata `bridge_type=BUNNY_H3_ActionLogic_Bridge`, mapeia uma representação de um modelo
  "professor" `SenseNova_L34` pra dentro da camada 49 do MiniMax H3, aparentemente pra melhorar
  aderência a lógica de ação/movimento). **Nenhum custom node instalado neste checkout sabe
  consumir esse formato** (busca por "bunny"/"ActionLogic" nos `custom_nodes/` não achou nada)
  -- colocar num `models/` normal não faria nada, nenhum node escaneia essa pasta procurando
  por ele. Precisa achar/instalar o custom node de origem antes de ter utilidade aqui; até lá,
  fica de fora.

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

### Workflow "long take" multitrack (`comfyui-easy-media`) — VALIDADO, integrado ao backend

Segundo mecanismo de continuidade pro H3, diferente do clipe único que `generate()` faz hoje: o
custom node `comfyui-easy-media` (`easy multiTrackEditor` + `easy multitrackProject`) gera
segmentos em single-pass onde um segmento `continuity_mode="context"` herda o **latente** do
anterior, em vez de reencadear por imagem (I2V do último frame, que é o que
`render_scenes`/`render_shots` fazem hoje pra todos os motores, incluindo o H3). Clonado em
`E:\Users\home\Documents\MiniMax-H3\ComfyUI\custom_nodes\ComfyUI-Easy-Media`, checkpoint de
terceiro (`Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors`, 11,8 GB).

**VALIDADO com GPU real 2026-09-24, 2x**: T2V puro (bola rolando, 2 segmentos, 284s) e ref2v com
identidade real (HA-EUN do Voo 702, 2 segmentos, 214s) — ambos sem erro, mecanismo de contexto em
latente confirmado no log (`"Re-encoded video anchor: soft context=..."`).

**Cabeado como motor de pipeline** (`--video-engine minimax-longtake` em `run_decupagem.py`/
`decupagem_ui.py`, `--engine minimax-longtake` em `render_shots_stage.py`): agrupa os planos em
TAKES (cena nova OU plano `wide`/`full` sempre abre um take novo — `scene` no `shot_plan.json` é
cena de ROTEIRO, não de câmera; o CERCO EM SEUL inteiro tem 25 planos numa cena só, então agrupar
só por `scene` juntaria tudo num take gigante) e gera cada take com UMA chamada a
`generate_longtake()`, recortando o vídeo combinado de volta em um `shotNNN.mp4` por plano — o
resto do pipeline (cache, lipsync/mix/assemble) não muda nada. **VALIDADO com GPU real** duas vezes:
3 planos de uma sequência sem wide (3/3 clipes, 473,5s — achado e corrigido 1 bug de path na cópia
final, `filename_prefix` com subpasta `"video/"` que `submit_and_wait()` não sabia resolver de
volta) e 4 planos com um `wide` NO MEIO (planos 6-9 do CERCO EM SEUL: insert/medium/wide/close),
confirmando que a quebra de take funciona de verdade — 2 chamadas separadas a `generate_longtake()`
(262s + 218s), 4/4 clipes. **Ainda não testado**: um take de 8-9 planos (o tamanho real que o CERCO
EM SEUL completo produz, §3.120) — só grupos de até 3 foram medidos. `--minimax-ref-audio`/
`--minimax-chain-max-seconds` não têm efeito neste caminho (avisa e ignora). Ver `MEMORIAL.md`
§3.118/§3.119/§3.120/§3.121.

**Integrado ao `minimax_h3_backend.py`** como segunda função de geração:
`generate_longtake(segments, output_path, ...)` / `build_longtake_workflow()` /
`base_api_longtake()`, mesmo padrão do caminho de clipe único (r2v) já existente. `segments` é uma
lista de dicts (`prompt`, `duration_frames`, `continuity_mode` opcional, `ref_images` opcional só
no primeiro segmento). ⚠️ **NÃO usa `comfy_workflow_tool.convert()`** — o `easy multiTrackEditor`
declara widgets em tipos (`COMFY_DYNAMICCOMBO_V3`, `TRACK_DATA`) que o conversor genérico deste
repo não reconhece, e o zip posicional corromperia o grafo em silêncio. O template API
(`comfyui_workflows/minimax_h3_longtake_api_template.json`) foi capturado direto de
`window.app.graphToPrompt()` no navegador com o workflow já validado — é o conversor de verdade do
frontend, que mapeia por nome, não por posição. Testes sem GPU: `tests/test_minimax_h3_longtake.py`
(8 casos, `ensure_server()` mockado). **Não incluído nesta integração**: escolher automaticamente
quando usar long take em vez do encadeamento por imagem dentro de `render_shots`/`run_decupagem`
— isso é decisão de produto (que planos formam um "take" contíguo) não especificada ainda. Ver
`MEMORIAL.md` §3.115/§3.117.

### Checkpoints turbo (LightX2V) — CONFIRMADO: são LoRAs, e o 4-step JÁ está em produção

⚠️ **Esta seção estava desatualizada — corrigida em 2026-09-25.** O "não
verificado" do parágrafo antigo já tinha sido resolvido em algum commit
posterior sem a nota ser atualizada: os arquivos ficam em
`E:\Users\home\Documents\MiniMax-H3\ComfyUI\models\loras\` (não em
`diffusion_models\`), confirmando a suspeita de que são LoRAs, não
checkpoint completo. Inspecionado o `.safetensors` (2026-09-25):
`metadata.source_format = "Diffusers PEFT LoRA"`, rank 128, alpha 8,
`base_model = "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors"`.

**Só o `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` existe em
disco** (1,96 GB) — o `8step_v1.0_768p` citado abaixo nunca foi baixado, ou
foi removido; não confiar nessa linha do histórico sem verificar o disco de
novo. O 4-step **já é `LONGTAKE_LORA_FILENAME`, o padrão do caminho
`minimax-longtake`** (`minimax_h3_backend.py`, node `LoraLoaderModelOnly`,
`N_LT_LORA`), carregado por cima do `Minimax-h3_Singularity_ref2va_v1.3_
Pruned_w4a8.safetensors`. Isso significa que ele está **validado
indiretamente** por toda a bateria do long-take (§3.117-3.124, filme do
CERCO EM SEUL entregue) — não foi testado isolado (sem o resto do grafo
long-take), mas roda em produção real há dias. Há também um teste manual
avulso do usuário fora desta auditoria: `ComfyUI/output/turbo_test_4step_
00001_.mp4` (2026-09-24, fora do `run_decupagem`, não documentado até
agora).

## LongCat-Video-Avatar 1.5 — motor dos planos de fala (`--video-engine longcat`)

ComfyUI SEPARADO em `I:\LongCat-Video\ComfyUI` (venv próprio,
porta **8190**, WanVideoWrapper do Kijai). `longcat_video_backend.py` dirige por HTTP e
disputa a MESMA 3090 — desligue o 8188 antes (`gpu_watchdog.free_port(8188)`).

**Ligado na decupagem desde 2026-09-13** (`run_decupagem --video-engine longcat`, UI 7913):
planos sem fala vão para o LTX, planos de fala para o LongCat; o `render_shots` faz os de
ação primeiro e troca 8188 → 8190 sozinho uma vez. VALIDADO: wide no LTX 699 s + fala no
LongCat 772 s, SyncNet 9,43. ⚠️ O LongCat entrega o áudio **2 quadros (80 ms) fora**
(offset −2 constante no SyncNet). `LONGCAT_DIT_15` troca o checkpoint (GGUF Q8 em
`diffusion_models/Avatar/`). Ver `MEMORIAL.md` §3.85.

- **Padrão `--variant 1.5`**: `diffusion_models/Avatar/LongCat-Avatar-15_bf16` (31,7 GB,
  Kijai/WanVideo_comfy) + LoRA `LongCat-Avatar-15_dmd_distill_lora_rank128_bf16` (0,9, sem
  merge) + Whisper `HuMo\whisper_large_v3_encoder_fp16` + `LongCatAvatarWhisperEmbeds`;
  scheduler `longcat_distill_euler`, 12 passos, shift 12, CFG 1. `--variant 1.0` =
  wav2vec2, 20 passos, CFG 3 (94 min para 4 s).
- ⚠️ **36 blocos em swap, não 25**: com 25 a VRAM transborda para a memória compartilhada
  do WDDM e o 1º passo passa de 18 min. `LONGCAT_BLOCKS_TO_SWAP` sobrepõe.
- MEDIDO 2026-09-13 a 960x544: **57/101/125/177 quadros = 9/18/24/42 min** (~44/88/117/199
  s/passo). **Sync pelo SyncNet** (LSE-C; os números de correlação antigos desta seção eram do
  proxy, que errava): LongCat 8,86 / 7,89 / 9,43 / 7,3 contra LTX + LatentSync 7,04 / 8,54 /
  9,73 / 4,38. Empatam em fala comum; **na fala longa e hesitante o LongCat vence** (7,3 × 4,38),
  ao custo de ~3× o tempo.
- **GGUF Q8 não ganha do bf16** (2026-09-14, 177 quadros): 2533 s / conf 7,28 contra 2558 s /
  7,31 — com 36 blocos em swap o gargalo não é o tamanho do checkpoint. bf16 segue padrão.
- ⚠️ O fp8 em `weights/LongCat-Video-Avatar-1.5/base_model_fp8` é **optimum-quanto**: o
  ComfyUI não carrega, e no caminho Python oficial mede 4h23min por passo.
- Os "Windows fatal exception 0xc0000139" e `ConnectionResetError` no
  `comfyui_server.log` são import de flash_attn e cliente HTTP fechando — não são crash.

Ver `MEMORIAL.md` §3.83.

## Roteiro → filme (`script_pipeline/`)

Duas variantes do mesmo pipeline, escolhidas pelo comando, não por acaso. Elas
compartilham parse, cast, TTS, lip-sync, mistura e montagem; divergem só na
**unidade de renderização**.

| | UI screenplay (porta 7810) | decupagem (`start_decupagem.bat`) |
|---|---|---|
| unidade | uma FALA | um PLANO |
| still | um por CENA | um por PLANO, fixando o enquadramento |
| ações sem fala | não viram clipe | viram |
| estado | **validado** | **até o animatic**; modo espacial validado em 2026-09-20, vídeo final ainda não |

    [1] parse   [2] cast   [E] emoção   [4] TTS   [S] estrutura
      -> [L] lacunas do roteiro   [P] decupagem   [M] movimento + complexidade + previs 3D
      -> [5] render_scenes  OU  [5-D] render_shots_stage
      -> [6] lipsync  [7] mix  [8] assemble  [9] verify

### Roteiro completo, lacunas, complexidade e elenco (desde 2026-09-27)

Auditoria geral em `AUDITORIA_GERAL_2026-09-27.md`; o porquê em `MEMORIAL.md` §3.131-3.134.

- **O roteiro precisa dizer** agente + ação + alvo + resultado, posição, objetos, estado que
  persiste, e todos que aparecem, inclusive sem fala. Guia com modelo de cena:
  `script_pipeline/GUIA_ROTEIRO.md`.
- **[L] `screenplay_gaps.py`** roda antes da decupagem, nos dois caminhos, e grava
  `parse/lacunas.md`. Só relata; `--lacunas bloquear` para a corrida, `--lacunas-sem-llm`
  deixa só a parte determinística.
- **[M] `shot_complexity.py`** grava `shots/complexity_report.json`: simples (still basta),
  média (revisar `shots/blocking_preview.png`) ou complexa (previs 3D de baixa resolução, ou
  dividir o plano). As recomendações dependem do motor.
- **`[3D previs]` `previs_spec.py`** (desde 2026-09-27, MEMORIAL §3.135) gera o spec espacial
  SOZINHO a partir da decupagem + elenco + movimento + complexidade, e renderiza manequins/câmera
  no Blender (CPU, sem difusão) para os planos do nível pedido: folha `shots/previs_3d.png` e
  clipe `shots/previs_3d.mp4`. `--previs3d {complexas,medias,todos,off}` (padrão `complexas`),
  `--previs3d-stills` liga o quadro inicial como blocking do still (exige `--image-engine flux`;
  ignorado com `--spatial-spec` manual), `--previs3d-intergen` usa o InterGen no clipe de dupla em
  contato. Só relata/render diagnóstico; nunca bloqueia.
- **Movimento é aplicado SEMPRE** que a corrida passa por [M]. Antes, `--ate animatic`
  regerava o `shot_plan` sem ele, e o filme do CERCO v1 saiu sem movimento.
- **Figurantes** (quem age sem nome próprio): `cast.json` com `"extra": true` e `"aliases"`
  nos dois idiomas. O `cast_characters` os cria sozinho (LLM junta sinônimos da mesma pessoa).
- **Voz**: gênero pela foto > LLM > texto; idade pelo roteiro (a foto só se o texto cala);
  faixa etária, sem repetir voz. `cast_characters --revoice`, `"voice_locked": true`.
- **Emoção para vídeo** tem uma regra única, `shot_plan.emocao_para_video`: sem sorriso em
  cena de ação, e a versão de fala não ocupa a boca. Todos os caminhos usam essa função.
- **`verify_output`** avisa `GATES_NOT_APPROVED`: arquivo íntegro não significa filme aprovado.
- **MiniMax long-take**: no máximo 4 planos por take (`MINIMAX_LONGTAKE_MAX_PLANOS`). Um take
  que falha refaz os planos como clipes avulsos.

### Continuidade espacial 3D na decupagem

É um modo opt-in da `decupagem_ui.py`, no acordeão **Continuidade espacial 3D
(opcional)**. Ele não liga sozinho para projetos antigos. Requisitos:

- projeto mestre persistente vinculado à corrida;
- `Motor das imagens = flux`;
- spec JSON com exatamente um binding para cada ID único do
  `parse/shot_plan.json`;
- Blender acessível e dependências de `requirements-spatial.txt` instaladas.

Campos da WebUI: **Ativar estado espacial persistente**, caminho do spec,
**Força de transformação do blocking** (`0.65` padrão; `0.78` usado no teste
fotográfico de cabine) e **Animatic diagnóstico**. O diagnóstico encaminha
`--no-visual-audit`; serve para inspecionar um bloqueio e nunca deve ser lido
como aprovação para vídeo final.

CLI equivalente:

```powershell
.venv\Scripts\python.exe -m script_pipeline.run_decupagem ^
  --run-dir <RUN> --ate animatic --image-engine flux --video-engine ltx ^
  --spatial-spec <RUN>\world\spatial_spec.json --spatial-denoise 0.78
```

Para o Voo 702, gere o spec depois de o plano estar conformado e estável:

```powershell
.venv\Scripts\python.exe -m script_pipeline.voo702_spatial ^
  --run <RUN> --output <RUN>\world\voo702_spatial_spec.json
```

Teste validado em 2026-09-20: quatro planos contíguos da cabine, 384×256,
24 fps, 480 frames editoriais, 20,000 s e três falas. As fotos externas de
HA-EUN, MIN-JUN, JI-HO e SEO-YEON foram usadas; o gate aprovou 4/4 stills e
`shots/animatic_audit.json` ficou `ok` (3/3 falas). A corrida
`20260919_073649_VOO_702_-_CÉU_TURBULENTO` guarda o plano integral de 69 planos
em `parse/shot_plan.full.json`, o recorte anterior do cockpit em
`parse/shot_plan.cockpit20.json` e o recorte testado em `parse/shot_plan.json`.

`attach_to_run()` rejeita IDs ausentes ou duplicados. `conform_plan()` acrescenta
um contador determinístico quando mais de um plano cobre a mesma unidade-fonte.
Bindings com evento temporal devem usar
`script_pipeline.spatial_pipeline`, pois a decupagem comum aceita apenas estado
estático. Em closes, a referência espacial entra antes da identidade; em planos
abertos, o blocking continua como latente img2img. Detalhes e artefatos:
`script_pipeline/SPATIAL_PIPELINE.md` e `MEMORIAL.md` §3.97.

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

**Plano de fala abaixo de 704 px de altura sai só em close** (desde 2026-09-13,
`--dialogue-framing auto`). MEDIDO: a 960x544 o close dá rosto ~277 px e sync
mediana +0,25; o medium dá ~203 px e +0,05 — o LatentSync recorta 512x512 e rosto
pequeno chega sem sinal de boca. `extreme_close` nunca vai para fala (só olhos
no quadro). `--dialogue-framing close|livre` força ou solta. Ver `MEMORIAL.md` §3.80.

⚠️ **O NOME do enquadramento não basta para o FLUX.** "close-up, the face filling most of
the frame" com descritor longo + gesto de corpo ("crosses her arms") saiu plano médio, e
nenhum motor de lip-sync salvou (−0,19 a +0,10). O close agora leva limite físico ("top of
the head to the shoulders, mouth clearly visible, no hands or waist") e **nunca gesto de
corpo**: LatentSync foi a +0,17 / +0,38. Se um close reprovar no sync, olhe o STILL antes do
motor. Ver `MEMORIAL.md` §3.81.

**Plano de fala: emoção sem ocupar a boca** (desde 2026-09-13). `EMOCAO_VISIVEL_FALA` troca
"open smile"/"mouth wide" por olhos e sobrancelha, e o `video_prompt` de close perde gesto de
corpo e riso. Sem isso o LTX fez uma fala `alegre` virar gargalhada (SyncNet 8,5 → 9,7 com o
conserto). Ver `MEMORIAL.md` §3.85.

**O TTS agora roda sempre e refaz só a fala que mudou** (`dialogue/<id>.key`: texto, emoção,
tomada, motor). Antes pulava quando `lines.json` existia e a emoção dirigida nunca chegava à
voz. ⚠️ Tomada emotiva muda a DURAÇÃO: a "confusa" levou uma fala de 3,97 s para 7,09 s (51%
silêncio) — o plano cresce junto e o SyncNet cai por falta de trecho falado, não por sync ruim.

**Sincronia se mede com SyncNet, não com o proxy.** `script_pipeline/syncnet_audit.py` usa o
avaliador do próprio LatentSync (`LatentSync/.conda_env`, `syncnet_v2.model`); roda dentro do
`lipsync_scenes` e decide o resumo. LSE-C ≥ 3 = ok; offset em quadros a 25 fps. O
`lipsync_audit` (correlação boca × volume) reprovou 3 planos que o SyncNet aprovou com 4,4–9,7 —
não decida por ele. `--no-syncnet` desliga.

**Relatórios depois da montagem (só medem, nada é reescrito):** `8b identidade`
(`clip_identity_audit`, ArcFace nos clipes contra o still e o personagem, limiar 0,35) e
`8c doctor` (`video_doctor analyze` no `final/movie.mp4`, detecção de corte ligada). A/B de
LoRA reprodutível: `python -m script_pipeline.lora_ab --run-dir DIR --shots 3 --config
"bhm:better-human-motion:0.6"` (distilled por padrão; saída em `<run>/lora_ab/`).

Scripts antigos sem referência saíram da raiz para `_arquivo/` (ver `_arquivo/LEIAME.md`).

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
