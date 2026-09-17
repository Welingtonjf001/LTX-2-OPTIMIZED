# Relatório por script

[Relatório principal e critérios](RELATORIO.md). **348 fichas**, uma por arquivo do inventário. Triagem automatizada não substitui revisão funcional. Os sinais abaixo são oportunidades de investigação, não bugs confirmados.

## Hermes.bat

**Arquivo:** [Hermes.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/Hermes.bat>) · 99 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L6, L7, L8, L9, L10, L12 (+4); gestão de processos: L42, L43.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## HermesCleanup.ps1

**Arquivo:** [HermesCleanup.ps1](<E:/Users/home/Documents/LTX-2-OPTIMIZED/HermesCleanup.ps1>) · 44 linhas · **Cobertura:** Revisão dirigida + triagem.

**Verificação:** Parser PowerShell: OK; nenhuma execução.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L3, L25, L28; gestão de processos: L36, L37, L43.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## Qwen3CoderNextBrain.bat

**Arquivo:** [Qwen3CoderNextBrain.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/Qwen3CoderNextBrain.bat>) · 38 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L5, L35.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## SystemDoctor.bat

**Arquivo:** [SystemDoctor.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/SystemDoctor.bat>) · 33 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L5, L6, L7, L15, L22 (+2).

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## _arquivo/_bench_extract.py

**Arquivo:** [_arquivo/_bench_extract.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_bench_extract.py>) · 30 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Compara motores de LLM na tarefa de extracao prosa -> roteiro, com criterios objetivos (nao impressao): falas preservadas COMO DIALOGO PARSEADO, planos recuperados, e tempo.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L19.

**Melhoria sugerida:** Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.

## _arquivo/_gen_base640.py

**Arquivo:** [_arquivo/_gen_base640.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_gen_base640.py>) · 13 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.

## _arquivo/_gen_princesa.py

**Arquivo:** [_arquivo/_gen_princesa.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_gen_princesa.py>) · 38 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `vram`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L11; captura ampla de exceções: L14, L27; subprocesso bloqueante sem timeout explícito: L11.

**Melhoria sugerida:** Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.

## _arquivo/_gen_twostage.py

**Arquivo:** [_arquivo/_gen_twostage.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_gen_twostage.py>) · 16 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.

## _arquivo/_patch_screenplay.py

**Arquivo:** [_arquivo/_patch_screenplay.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_patch_screenplay.py>) · 22 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.

## _arquivo/_smoke_gguf.py

**Arquivo:** [_arquivo/_smoke_gguf.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_smoke_gguf.py>) · 7 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _arquivo/_smoke_v3.py

**Arquivo:** [_arquivo/_smoke_v3.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_smoke_v3.py>) · 7 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _arquivo/_test_camera.py

**Arquivo:** [_arquivo/_test_camera.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_test_camera.py>) · 28 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** O LTX obedece a vocabulario de camera? Mesma cena, mesmo seed, so a especificacao muda. Testa os dois eixos mais basicos da gramatica: ESCALA (wide<->close) e ALTURA (contra-plongee<->plongee).

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L26.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _arquivo/_test_storyplay25_disney.py

**Arquivo:** [_arquivo/_test_storyplay25_disney.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_test_storyplay25_disney.py>) · 52 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Ciclo completo do storyplay25, chamando as MESMAS funcoes que os botoes da UI chamam (do_parse -> do_storyboard -> do_video), com a cena da Lyra em 30s -- acima do limiar de legendas medido (MEMORIAL.md 3.11).

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _arquivo/_test_storyplay25_e2e.py

**Arquivo:** [_arquivo/_test_storyplay25_e2e.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/_test_storyplay25_e2e.py>) · 51 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Ciclo completo do storyplay25, chamando as MESMAS funcoes que os botoes da UI chamam (do_parse -> do_storyboard -> do_video), com a cena da Lyra em 30s -- acima do limiar de legendas medido (MEMORIAL.md 3.11).

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _arquivo/v2_output_folder_test.py

**Arquivo:** [_arquivo/v2_output_folder_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_arquivo/v2_output_folder_test.py>) · 20 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L5, L6.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _atribuicao_338.py

**Arquivo:** [_atribuicao_338.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_atribuicao_338.py>) · 130 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Experimento de atribuicao: QUAL fator fez o clipe T2V ganhar da decupagem? O clipe de referencia (prompt do usuario, T2V continuo, sem still, seed 77) foi julgado "o melhor de todos". Ele muda CINCO coisas ao mesmo tempo em relacao a cadeia

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L22, L28, L29; gestão de processos: L121; subprocesso bloqueante sem timeout explícito: L121.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## _make_25_uis.py

**Arquivo:** [_make_25_uis.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_make_25_uis.py>) · 167 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Generate the *_25 UI variants from their 2.3 originals. Each 2.5 UI is the 2.3 UI with a small, explicit set of substitutions: the generation subprocess is pointed at `ltx_pipelines_25` (the LTX-2.5 shim), the displayed model paths point at

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L153.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## _run_screenplay_batch.py

**Arquivo:** [_run_screenplay_batch.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_run_screenplay_batch.py>) · 123 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Queue the remaining 5 screenplay prompts through the validated LTX-2.5 single-stage T2V/I2V ComfyUI workflow, one at a time, waiting for each.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L13; captura ampla de exceções: L71.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## _screenplay_prompts.py

**Arquivo:** [_screenplay_prompts.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_screenplay_prompts.py>) · 39 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** The 6 screenplay prompts (LTX-2.3 prompt-guide style) provided by the user, plus shared negative prompt. #2 (western) already ran; this covers the rest.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## _test_longtake.py

**Arquivo:** [_test_longtake.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_test_longtake.py>) · 54 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Where does a single continuous LTX-2.5 take break? Longest we had actually generated was 433 frames (18s). The screenplay UI caps max_clip_seconds at 15 with the note "clipes longos travam o upsampler" -- a limit inherited from the 2.3 nati

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L53.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _test_lora_ab.py

**Arquivo:** [_test_lora_ab.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_test_lora_ab.py>) · 78 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Teste A/B do LoRA "Multi-Ref Character Storyboard V2": mesma imagem de referencia (rosto da espadachim, extraido do beat de close-up da cena wuxia), mesmo prompt (uma cena NOVA, nao a que gerou a referencia), mesma seed -- so a forca do LoR

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** remoção de arquivos: L70.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _test_minimax_duration_cap.py

**Arquivo:** [_test_minimax_duration_cap.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_test_minimax_duration_cap.py>) · 195 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Teste standalone (fora do pipeline de producao): plano longo vs plano dividido, no MiniMax H3 -- ver conversa 2026-09-03, pergunta "limitar ou deixar variavel". NAO integra em render_shots.py ainda; e um teste isolado pra decidir SE vale a 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _test_nag_subtitles.py

**Arquivo:** [_test_nag_subtitles.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_test_nag_subtitles.py>) · 175 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Controlled A/B test: does NAG actually remove the baked-in subtitles? Background (why the negative prompt alone never worked): the LTX-2.5 *distilled* workflow samples at **CFG = 1**. Classifier-free guidance at scale 1 reduces to `uncond +

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L28.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## _test_noquotes.py

**Arquivo:** [_test_noquotes.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/_test_noquotes.py>) · 39 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Decisive test for the burned-in subtitles: same scene, dialogue described INDIRECTLY (no quoted lines) instead of verbatim between quotes. Everything else identical (seed, size, duration, negative prompt, distilled variant). If the subtitle

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## analyze_music.py

**Arquivo:** [analyze_music.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/analyze_music.py>) · 55 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## animar_corpo.py

**Arquivo:** [animar_corpo.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/animar_corpo.py>) · 91 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Identidade da FOTO + movimento do VIDEO -> sequencia de malhas SMPL-X. conda run -n mhr python animar_corpo.py <smplx_params.npz> <hmr4d_results.pt> <saida> [--frames N] Por que isto funciona sem rigging nenhum: o SMPL-X e' um modelo PARAME

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L30.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## audio_fx.py

**Arquivo:** [audio_fx.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/audio_fx.py>) · 189 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Shared vocal spatialization (room/environment reverb) for the talking-head pipeline, used by music_maker_ui_v2.py / v3.py / GGUF / TensorRT variants. Design constraint that decides WHERE this must be called from: Wav2Lip drives mouth moveme

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L60; gestão de processos: L129, L160; captura ampla de exceções: L31, L108, L137; subprocesso bloqueante sem timeout explícito: L160, L129.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## character3d_webui.py

**Arquivo:** [character3d_webui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/character3d_webui.py>) · 189 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** MVP local: fotos -> referências FLUX -> pipeline 3D. The heavy model runners are intentionally optional. This UI can be used to prepare a reproducible job and will call ComfyUI when it is running.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** B008 L92. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L12; gestão de processos: L65, L68, L69, L111, L144; subprocesso bloqueante sem timeout explícito: L111.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## character_sheet_flux_webui.py

**Arquivo:** [character_sheet_flux_webui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/character_sheet_flux_webui.py>) · 176 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `start_backend`, `stop_backend`, `upload_image`, `add_node`, `build_prompt`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L15, L19, L23; gestão de processos: L44, L60; captura ampla de exceções: L32, L49.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## comfy_gguf_patch.py

**Arquivo:** [comfy_gguf_patch.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_gguf_patch.py>) · 172 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Patch the API-format LTX-2.3 'Full' workflow to run from a GGUF UNET using only the model files present in this installation. The upstream example workflow assumes a single ``ltx-2.3-22b-dev.safetensors`` checkpoint (model+clip+vae), a dist

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## comfy_ingredients_patch.py

**Arquivo:** [comfy_ingredients_patch.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_ingredients_patch.py>) · 166 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Patch the API-format LTX-2.3 IC-LoRA "Ingredients" workflow to run from a GGUF UNET using only the model files present in this installation. Reference-image-conditioned generation: given ONE reference image (a person, object, or scene compo

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## comfy_run.py

**Arquivo:** [comfy_run.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_run.py>) · 84 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Submit an API-format workflow to ComfyUI, wait for it, report outputs/errors. Usage: python comfy_run.py --workflow api.json [--server http://127.0.0.1:8188] [--timeout 1800]

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L3, L30; captura ampla de exceções: L46.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## comfy_union_depth_patch.py

**Arquivo:** [comfy_union_depth_patch.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_union_depth_patch.py>) · 146 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Union-Control com DOIS guias encadeados: pose + profundidade. Experimento. A IC-LoRA Union-Control foi treinada com Canny + Depth + Pose, mas o workflow tem UM no `LTXAddVideoICLoRAGuide` -- ou seja, ela aceita um tipo de controle por vez, 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L84; remoção de arquivos: L138; gestão de processos: L82; subprocesso bloqueante sem timeout explícito: L82.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## comfy_union_ingredients_patch.py

**Arquivo:** [comfy_union_ingredients_patch.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_union_ingredients_patch.py>) · 157 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Union-Control (pose) + Ingredients (personagem) no MESMO workflow. python comfy_union_ingredients_patch.py --pose bundle/pose.mp4 --reference personagem.png --prompt "..." --out _api_combo.json Por que um patch novo em vez de reusar um dos 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## comfy_union_patch.py

**Arquivo:** [comfy_union_patch.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_union_patch.py>) · 195 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Patch the API-format LTX-2.3 IC-LoRA *Union-Control* workflow to run from a ChoreoEngine pose bundle on the local GGUF stack. The official workflow (example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_ Distilled.json) derives the control vid

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L14.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## comfy_workflow_tool.py

**Arquivo:** [comfy_workflow_tool.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfy_workflow_tool.py>) · 295 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Convert a ComfyUI *UI-format* workflow into *API-format*, prune it to the subgraph feeding a chosen output node, and optionally swap the diffusion-model loader for the GGUF loader. The UI format (what the editor saves) stores nodes with pos

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L13, L269.

**Melhoria sugerida:** Dividir `convert` (L73, 162 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## comfyui_workflows/export_minimax_h3_corrected.py

**Arquivo:** [comfyui_workflows/export_minimax_h3_corrected.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/comfyui_workflows/export_minimax_h3_corrected.py>) · 41 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Export the supplied prompt into the correct API graph and official UI template.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## continuous_chain.py

**Arquivo:** [continuous_chain.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/continuous_chain.py>) · 517 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Camada 0 do plano de video continuo por minutos (MEMORIAL.md, secao "Video continuo por encadeamento" -- ver tambem CLAUDE.md, que so documenta 2.3/2.5 por clipe curto). Generaliza para fora do roteiro/screenplay o mecanismo de chain-contin

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L79, L177; gestão de processos: L178, L230, L231; subprocesso bloqueante sem timeout explícito: L178.

**Melhoria sugerida:** Dividir `main` (L333, 181 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## convert.py

**Arquivo:** [convert.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/convert.py>) · 169 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_layer_pattern`, `dtype_to_str`, `str_to_dtype`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L76, L96, L166.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## daz_integration/__init__.py

**Arquivo:** [daz_integration/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/__init__.py>) · 2 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Local LLM -> DAZ Studio character integration.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## daz_integration/app.py

**Arquivo:** [daz_integration/app.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/app.py>) · 112 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `build`, `preview_from_options`, `main`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L69, L71, L98; captura ampla de exceções: L40, L57.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## daz_integration/character_schema.py

**Arquivo:** [daz_integration/character_schema.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/character_schema.py>) · 39 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `to_json`, `from_dict`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## daz_integration/daz_controller.py

**Arquivo:** [daz_integration/daz_controller.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/daz_controller.py>) · 100 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `resolve_figure_preset`, `_daz_path`, `create_script`, `run_daz`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L8, L11; gestão de processos: L85, L97, L98; subprocesso bloqueante sem timeout explícito: L85.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## daz_integration/inspect/dump_exporters.dsa

**Arquivo:** [daz_integration/inspect/dump_exporters.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/inspect/dump_exporters.dsa>) · 14 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L1.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/inspect/dump_skeleton.dsa

**Arquivo:** [daz_integration/inspect/dump_skeleton.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/inspect/dump_skeleton.dsa>) · 49 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2, L3, L4.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/inspect/export_fbx.dsa

**Arquivo:** [daz_integration/inspect/export_fbx.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/inspect/export_fbx.dsa>) · 17 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/inspect/export_fbx_tentativas_1a5.dsa

**Arquivo:** [daz_integration/inspect/export_fbx_tentativas_1a5.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/inspect/export_fbx_tentativas_1a5.dsa>) · 51 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L1, L2, L40.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/local_llm.py

**Arquivo:** [daz_integration/local_llm.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/local_llm.py>) · 44 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_extract_json`, `interpret`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L21.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## daz_integration/mcp_server.py

**Arquivo:** [daz_integration/mcp_server.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/mcp_server.py>) · 19 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `generate_character`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## daz_integration/runs/daz_character_23ee943f.dsa

**Arquivo:** [daz_integration/runs/daz_character_23ee943f.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/daz_character_23ee943f.dsa>) · 18 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/daz_character_362bcab3.dsa

**Arquivo:** [daz_integration/runs/daz_character_362bcab3.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/daz_character_362bcab3.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/daz_character_e8056048.dsa

**Arquivo:** [daz_integration/runs/daz_character_e8056048.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/daz_character_e8056048.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/daz_character_ffdc1d8e.dsa

**Arquivo:** [daz_integration/runs/daz_character_ffdc1d8e.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/daz_character_ffdc1d8e.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/full_pipeline/daz_character_5e213044.dsa

**Arquivo:** [daz_integration/runs/full_pipeline/daz_character_5e213044.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/full_pipeline/daz_character_5e213044.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/load_check/daz_character_e050aa1c.dsa

**Arquivo:** [daz_integration/runs/load_check/daz_character_e050aa1c.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/load_check/daz_character_e050aa1c.dsa>) · 21 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/preview_test/daz_character_6f6be7fa.dsa

**Arquivo:** [daz_integration/runs/preview_test/daz_character_6f6be7fa.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/preview_test/daz_character_6f6be7fa.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## daz_integration/runs/test_real/daz_character_8e2231b1.dsa

**Arquivo:** [daz_integration/runs/test_real/daz_character_8e2231b1.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/test_real/daz_character_8e2231b1.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## daz_integration/runs/viewport_test/daz_character_ed04dcea.dsa

**Arquivo:** [daz_integration/runs/viewport_test/daz_character_ed04dcea.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/runs/viewport_test/daz_character_ed04dcea.dsa>) · 18 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## daz_integration/start_daz_character.bat

**Arquivo:** [daz_integration/start_daz_character.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/start_daz_character.bat>) · 4 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## daz_integration/test_run/daz_character_b6695293.dsa

**Arquivo:** [daz_integration/test_run/daz_character_b6695293.dsa](<E:/Users/home/Documents/LTX-2-OPTIMIZED/daz_integration/test_run/daz_character_b6695293.dsa>) · 12 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.

## decupagem_ui.py

**Arquivo:** [decupagem_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/decupagem_ui.py>) · 1756 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** UI da cadeia de DECUPAGEM (roteiro -> filme), porta 7913. POR QUE ESTA UI EXISTE O `start_decupagem.bat` roda a cadeia inteira e termina -- e por vinte minutos a unica coisa que a pessoa ve e console. Os artefatos intermediarios existem tod

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L126, L696, L700, L742, L746, L923 (+2); remoção de arquivos: L519, L879, L982, L986; gestão de processos: L65, L933, L1135, L1136, L1176; captura ampla de exceções: L129; subprocesso bloqueante sem timeout explícito: L933.

**Melhoria sugerida:** Dividir `build` (L1217, 536 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## film_maker_ui_v4.py

**Arquivo:** [film_maker_ui_v4.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/film_maker_ui_v4.py>) · 607 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `extract_frame`, `extract_first_frame`, `extract_last_frame`, `call_gemini`, `process_chain_generation`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** E722 L449. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L354, L355, L361, L446; captura ampla de exceções: L182, L385, L449; subprocesso bloqueante sem timeout explícito: L446, L361.

**Melhoria sugerida:** Dividir `process_chain_generation` (L186, 205 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## film_maker_ui_v4_25.py

**Arquivo:** [film_maker_ui_v4_25.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/film_maker_ui_v4_25.py>) · 623 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LTX-2.5 variant of film_maker_ui_v4.py -- GENERATED by _make_25_uis.py, do not hand-edit without also updating that script (or it will be overwritten on regeneration). Differences from the 2.3 original: - Generation goes through `ltx_pipeli

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** E722 L465. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L370, L371, L377, L462; captura ampla de exceções: L198, L401, L465; subprocesso bloqueante sem timeout explícito: L462, L377.

**Melhoria sugerida:** Dividir `process_chain_generation` (L202, 205 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## generate_sd35.py

**Arquivo:** [generate_sd35.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/generate_sd35.py>) · 187 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Generate an image with SD 3.5 via ComfyUI API.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L6, L8; captura ampla de exceções: L42.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## generate_upscale.ps1

**Arquivo:** [generate_upscale.ps1](<E:/Users/home/Documents/LTX-2-OPTIMIZED/generate_upscale.ps1>) · 42 linhas · **Cobertura:** Revisão dirigida + triagem.

**Verificação:** Parser PowerShell: OK; nenhuma execução.

- **P1 [A12](RELATORIO.md#a12): Scripts de upscale anunciam sucesso sem verificar executáveis. Falhas podem chegar até mensagens PRONTO/Instalação concluída; os wrappers de upscale ainda apagam intermediários após falha. Correção: Verificar LASTEXITCODE imediatamente e validar saída antes de continuar; preservar intermediários em falha. Aplicar o mesmo contrato ao download.
- **P2 [A13](RELATORIO.md#a13): Upcale compartilha temporários destrutivos entre execuções. Duas execuções simultâneas podem apagar quadros ou substituir vídeos uma da outra. Correção: Criar diretório único por execução; validar confinamento antes da limpeza e remover só os arquivos da própria execução.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L7, L11; remoção de arquivos: L41.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## gguf_backend.py

**Arquivo:** [gguf_backend.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/gguf_backend.py>) · 478 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** GGUF/ComfyUI generation backend for the music-video UIs. Keeps a single ComfyUI server process alive for the whole session (started on first use, reused across every scene) so the 22B transformer is loaded ONCE instead of once per scene lik

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L49; gestão de processos: L132, L149, L182; captura ampla de exceções: L105, L151, L384, L400.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## hunyuan_worker.py

**Arquivo:** [hunyuan_worker.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/hunyuan_worker.py>) · 37 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L12, L13.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## img2threejs_viewer_ui.py

**Arquivo:** [img2threejs_viewer_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/img2threejs_viewer_ui.py>) · 346 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** UI Gradio para o skill img2threejs: pipeline imagem->3D, visualizador e exportador de modelo. Por que existe: o img2threejs (clonado em `E:\Users\home\Documents\img2threejs`, junction em `~/.claude/skills/img2threejs`) roda como skill do Cl

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L57, L67, L98; gestão de processos: L92, L95, L117, L121; captura ampla de exceções: L97, L123, L211.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## install_character3d.ps1

**Arquivo:** [install_character3d.ps1](<E:/Users/home/Documents/LTX-2-OPTIMIZED/install_character3d.ps1>) · 11 linhas · **Cobertura:** Revisão dirigida + triagem.

**Verificação:** Parser PowerShell: OK; nenhuma execução.

- **P1 [A12](RELATORIO.md#a12): Scripts de upscale anunciam sucesso sem verificar executáveis. Falhas podem chegar até mensagens PRONTO/Instalação concluída; os wrappers de upscale ainda apagam intermediários após falha. Correção: Verificar LASTEXITCODE imediatamente e validar saída antes de continuar; preservar intermediários em falha. Aplicar o mesmo contrato ao download.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L1, L7.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## krea2_test_ui.py

**Arquivo:** [krea2_test_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/krea2_test_ui.py>) · 236 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `check_models`, `check_comfyui_connection`, `get_comfyui_workflow`, `queue_prompt`, `wait_for_image`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L13, L170.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## latentsync_syncnet_eval.py

**Arquivo:** [latentsync_syncnet_eval.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/latentsync_syncnet_eval.py>) · 56 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Evaluate one LatentSync result with the official SyncNet confidence metric.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## latentsync_v3_smoke_test.py

**Arquivo:** [latentsync_v3_smoke_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/latentsync_v3_smoke_test.py>) · 30 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Six-second end-to-end LatentSync smoke test for Music Video V3.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## longcat_video_backend.py

**Arquivo:** [longcat_video_backend.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/longcat_video_backend.py>) · 379 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LongCat-Video-Avatar via um ComfyUI SEPARADO do LTX/MiniMax H3. Mesmo padrao de `ltx25_backend.py`/`minimax_h3_backend.py` -- dirige por HTTP uma instalacao de ComfyUI independente (outro venv, outra porta: 8190), disputando a MESMA 3090 fi

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L44, L47; gestão de processos: L100, L121, L124; captura ampla de exceções: L95.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## lora_storyboard_backend.py

**Arquivo:** [lora_storyboard_backend.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/lora_storyboard_backend.py>) · 286 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** ComfyUI backend for testing a LoRA on the LTX 2.3 route -- built for the "Multi-Ref Character Storyboard V2" LoRA (models/loras/), but generic to any LoraLoaderModelOnly-compatible LoRA on the 2.3 checkpoint. Why a hand-built flat graph ins

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** remoção de arquivos: L284; gestão de processos: L99; captura ampla de exceções: L214, L219.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## lora_storyboard_encode.py

**Arquivo:** [lora_storyboard_encode.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/lora_storyboard_encode.py>) · 123 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Encode a prompt with the NATIVE LTX-2.3 Gemma text encoder (ltx_pipelines' ModelLedger -- the same code path proven to work, ~146s cold) and write the result in the safetensors format `ComfyUI-LTXVideo`'s LTXVSaveConditioning/ LTXVLoadCondi

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## lora_storyboard_test_ui.py

**Arquivo:** [lora_storyboard_test_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/lora_storyboard_test_ui.py>) · 164 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** WebUI de teste para o LoRA `ltx_2.3_Multi-Ref Character Storyboard_V2` (e qualquer outro LoRA em models/loras/) sobre o checkpoint LTX 2.3. Por que uma UI dedicada em vez de encaixar isso nas UIs de produção (music_maker_ui, screenplay_ui e

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L78.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## ltx25_backend.py

**Arquivo:** [ltx25_backend.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/ltx25_backend.py>) · 1370 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LTX-2.5 generation backend for the *_25 UIs. Why this exists (and why the 2.5 UIs can't just reuse the 2.3 code path): `ltx_pipelines` has no LTX-2.5 support -- no CLI for the split per-component checkpoints, no Gemma4 text encoder in `ltx_

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L48, L482, L1210; remoção de arquivos: L327, L338, L1194, L1304; gestão de processos: L386, L388, L418, L1215, L1265, L1299; captura ampla de exceções: L256, L291, L390; subprocesso bloqueante sem timeout explícito: L1215, L1265, L1299.

**Melhoria sugerida:** Dividir `build_workflow` (L783, 222 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## ltx_loras.py

**Arquivo:** [ltx_loras.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/ltx_loras.py>) · 499 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Catalogo dos LoRAs de VIDEO do LTX (2.3/2.5): o que existe, onde fica, como se usa. Por que um catalogo em codigo e nao so arquivos numa pasta: um LoRA de video nao e so um peso. Ele tem TIPO (LoRA comum, IC-LoRA guiado por imagem, por vide

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L278, L415, L416; remoção de arquivos: L431; captura ampla de exceções: L384, L418, L429.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## ltx_pipelines_25.py

**Arquivo:** [ltx_pipelines_25.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/ltx_pipelines_25.py>) · 215 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Drop-in CLI shim: same arguments as `ltx_pipelines.distilled` / `ltx_pipelines.music_to_video`, but generates with **LTX-2.5** through `ltx25_backend` (ComfyUI route). Why a shim instead of porting each UI's generation code: every UI in thi

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L206.

**Melhoria sugerida:** Dividir `main` (L52, 160 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## mhr_to_smplx.py

**Arquivo:** [mhr_to_smplx.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/mhr_to_smplx.py>) · 119 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Ponte: saida do SAM 3D Body (MHR) -> parametros SMPL-X. conda run -n mhr python mhr_to_smplx.py <entrada.npz / pasta> <saida> IMPORTANTE -- este script roda no ambiente conda `mhr`, NAO no Python global, e exige `PYTHONNOUSERSITE=1`. Os doi

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L30.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## minimax_h3_backend.py

**Arquivo:** [minimax_h3_backend.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/minimax_h3_backend.py>) · 668 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** MiniMax H3 (Reference-to-Video) via a ComfyUI SEPARADO do LTX 2.5. Mesmo padrao do `ltx25_backend.py` -- converter workflow oficial para API via `comfy_workflow_tool`, escrever nos nos conhecidos por ID, submeter e esperar -- mas para uma i

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L48, L51; remoção de arquivos: L213, L224, L629; gestão de processos: L235, L265, L340, L342; captura ampla de exceções: L168, L188, L344.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## minimax_h3_test_ui.py

**Arquivo:** [minimax_h3_test_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/minimax_h3_test_ui.py>) · 209 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Gradio test bench for the installed MiniMax H3 ComfyUI (port 8189).

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L17, L18; captura ampla de exceções: L107, L169.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## music_maker_ui.py

**Arquivo:** [music_maker_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui.py>) · 1037 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `load_audio_compatible`, `save_audio_compatible`, `beat_aligned_boundaries`, `load_lyrics_segments`, `lyrics_for_window`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** E722 L825. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L72, L699, L725; remoção de arquivos: L744; gestão de processos: L75, L194, L222, L240, L281, L645 (+5); captura ampla de exceções: L85, L92, L160, L195, L205, L225 (+8); subprocesso bloqueante sem timeout explícito: L281, L75, L194, L222, L240, L822.

**Melhoria sugerida:** Dividir `process_chain_generation` (L524, 239 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## music_maker_ui_gguf.py

**Arquivo:** [music_maker_ui_gguf.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_gguf.py>) · 1739 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `v2_status`, `run_subdir`, `vocal_cache_key`, `create_generation_folder`, `load_audio_compatible`.

**Verificação:** OK (Python 3.12).

- **P1 [A01](RELATORIO.md#a01): Fallback de separação vocal aborta por variável local não inicializada. Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original. Correção: Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.

**Alertas Ruff para triagem:** F823 L592, B023 L1054, B023 L1087, E722 L1359. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L164, L532, L1176, L1195, L1248, L1604 (+1); remoção de arquivos: L1268; gestão de processos: L167, L349, L377, L395, L435, L498 (+7); captura ampla de exceções: L177, L184, L313, L350, L360, L380 (+15); subprocesso bloqueante sem timeout explícito: L435, L540, L167, L349, L377, L395.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## music_maker_ui_v2.py

**Arquivo:** [music_maker_ui_v2.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v2.py>) · 1739 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `v2_status`, `run_subdir`, `vocal_cache_key`, `create_generation_folder`, `load_audio_compatible`.

**Verificação:** OK (Python 3.12).

- **P1 [A01](RELATORIO.md#a01): Fallback de separação vocal aborta por variável local não inicializada. Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original. Correção: Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.

**Alertas Ruff para triagem:** F823 L589, E722 L1384. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L161, L529, L1176, L1195, L1248, L1595 (+1); remoção de arquivos: L1268; gestão de processos: L164, L346, L374, L392, L432, L495 (+10); captura ampla de exceções: L174, L181, L310, L347, L357, L377 (+15); subprocesso bloqueante sem timeout explícito: L432, L537, L164, L346, L374, L392.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## music_maker_ui_v2_25.py

**Arquivo:** [music_maker_ui_v2_25.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v2_25.py>) · 1795 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** LTX-2.5 variant of music_maker_ui_v2.py -- GENERATED by _make_25_uis.py, do not hand-edit without also updating that script (or it will be overwritten on regeneration). Differences from the 2.3 original: - Generation goes through `ltx_pipel

**Verificação:** OK (Python 3.12).

- **P1 [A01](RELATORIO.md#a01): Fallback de separação vocal aborta por variável local não inicializada. Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original. Correção: Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.

**Alertas Ruff para triagem:** F823 L623, E722 L1433. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L195, L563, L1210, L1229, L1293, L1651 (+1); remoção de arquivos: L1314; gestão de processos: L198, L380, L408, L426, L466, L529 (+10); captura ampla de exceções: L208, L215, L344, L381, L391, L411 (+15); subprocesso bloqueante sem timeout explícito: L466, L571, L198, L380, L408, L426.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## music_maker_ui_v3.py

**Arquivo:** [music_maker_ui_v3.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v3.py>) · 1951 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `v2_status`, `run_subdir`, `vocal_cache_key`, `apply_low_resolution_test_profile`, `create_generation_folder`.

**Verificação:** OK (Python 3.12).

- **P1 [A01](RELATORIO.md#a01): Fallback de separação vocal aborta por variável local não inicializada. Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original. Correção: Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.

**Alertas Ruff para triagem:** F823 L766, E722 L1570. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L29, L169, L487, L706, L1362, L1381 (+3); remoção de arquivos: L1454; gestão de processos: L172, L352, L380, L398, L443, L492 (+20); captura ampla de exceções: L182, L189, L316, L353, L363, L383 (+18); subprocesso bloqueante sem timeout explícito: L492, L497, L639, L714, L172, L352.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## music_maker_ui_v3_25.py

**Arquivo:** [music_maker_ui_v3_25.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v3_25.py>) · 1967 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** LTX-2.5 variant of music_maker_ui_v3.py -- GENERATED by _make_25_uis.py, do not hand-edit without also updating that script (or it will be overwritten on regeneration). Differences from the 2.3 original: - Generation goes through `ltx_pipel

**Verificação:** OK (Python 3.12).

- **P1 [A01](RELATORIO.md#a01): Fallback de separação vocal aborta por variável local não inicializada. Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original. Correção: Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.

**Alertas Ruff para triagem:** F823 L782, E722 L1586. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L45, L185, L503, L722, L1378, L1397 (+3); remoção de arquivos: L1470; gestão de processos: L188, L368, L396, L414, L459, L508 (+20); captura ampla de exceções: L198, L205, L332, L369, L379, L399 (+18); subprocesso bloqueante sem timeout explícito: L508, L513, L655, L730, L188, L368.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## packages/ltx-core/src/ltx_core/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/__init__.py>) · 0 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/components/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/__init__.py>) · 10 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Diffusion pipeline components. Submodules: diffusion_steps - Diffusion stepping algorithms (EulerDiffusionStep) guiders - Guidance strategies (CFGGuider, STGGuider, APG variants) noisers - Noise samplers (GaussianNoiser) patchifiers - Laten

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/components/diffusion_steps.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/diffusion_steps.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/diffusion_steps.py>) · 95 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `step`, `get_sde_coeff`, `step`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/components/guiders.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/guiders.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/guiders.py>) · 364 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_params_for_sigma_from_sorted_dict`, `create_multimodal_guider_factory`, `projection_coef`, `delta`, `enabled`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/components/noisers.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/noisers.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/noisers.py>) · 35 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__call__`, `__init__`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/components/patchifiers.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/patchifiers.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/patchifiers.py>) · 348 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_pixel_coords`, `__init__`, `patch_size`, `get_token_count`, `patchify`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/components/protocols.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/protocols.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/protocols.py>) · 101 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `patchify`, `unpatchify`, `patch_size`, `get_patch_grid_bounds`, `execute`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/components/schedulers.py

**Arquivo:** [packages/ltx-core/src/ltx_core/components/schedulers.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/components/schedulers.py>) · 130 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_precalculate_model_sampling_sigmas`, `flux_time_shift`, `execute`, `execute`, `execute`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L94.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-core/src/ltx_core/conditioning/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/__init__.py>) · 19 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Conditioning utilities: latent state, tools, and conditioning types.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/conditioning/exceptions.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/exceptions.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/exceptions.py>) · 4 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/conditioning/item.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/item.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/item.py>) · 20 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `apply_to`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/conditioning/mask_utils.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/mask_utils.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/mask_utils.py>) · 210 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Utilities for building 2D self-attention masks for conditioning items.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/conditioning/types/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/types/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/types/__init__.py>) · 13 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Conditioning type implementations.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/conditioning/types/attention_strength_wrapper.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/types/attention_strength_wrapper.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/types/attention_strength_wrapper.py>) · 71 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Wrapper conditioning item that adds attention masking to any inner conditioning.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py>) · 70 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `apply_to`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py>) · 44 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `apply_to`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py

**Arquivo:** [packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py>) · 91 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Reference video conditioning for IC-LoRA inference.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/guidance/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/guidance/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/guidance/__init__.py>) · 15 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Guidance and perturbation utilities for attention manipulation.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/guidance/perturbations.py

**Arquivo:** [packages/ltx-core/src/ltx_core/guidance/perturbations.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/guidance/perturbations.py>) · 79 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `is_perturbed`, `is_perturbed`, `empty`, `mask`, `mask_like`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/loader/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/__init__.py>) · 48 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Loader utilities for model weights, LoRAs, and safetensor operations.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/loader/fuse_loras.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/fuse_loras.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/fuse_loras.py>) · 126 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `calculate_weight_float8_`, `_prepare_deltas`, `apply_loras`, `fused_add_round_launch`, `fused_add_round_launch`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L37.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-core/src/ltx_core/loader/kernels.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/kernels.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/kernels.py>) · 151 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `fused_add_round_kernel`, `fused_add_round_kernel`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L75.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-core/src/ltx_core/loader/module_ops.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/module_ops.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/module_ops.py>) · 14 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/loader/primitives.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/primitives.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/primitives.py>) · 109 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `footprint`, `metadata`, `load`, `meta_model`, `build`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/loader/registry.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/registry.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/registry.py>) · 84 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `add`, `pop`, `get`, `clear`, `add`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/loader/sd_ops.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/sd_ops.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/sd_ops.py>) · 127 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__call__`, `with_replacement`, `with_matching`, `with_kv_operation`, `apply_to_key`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/loader/sft_loader.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/sft_loader.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/sft_loader.py>) · 66 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `metadata`, `load`, `__init__`, `metadata`, `load`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py

**Arquivo:** [packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py>) · 158 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `lora`, `model_config`, `meta_model`, `load_sd`, `_return_model`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/__init__.py>) · 8 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Model definitions for LTX-2.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/model/audio_vae/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/__init__.py>) · 29 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Audio VAE model components.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/model/audio_vae/attention.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/attention.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/attention.py>) · 71 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `make_attn`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py>) · 508 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `build_mid_block`, `run_mid_block`, `encode_audio`, `decode_audio`, `__init__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py>) · 110 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `make_conv2d`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/causality_axis.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/causality_axis.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/causality_axis.py>) · 10 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/downsample.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/downsample.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/downsample.py>) · 110 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `build_downsampling_path`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/model_configurator.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/model_configurator.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/model_configurator.py>) · 200 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_vocoder_from_config`, `_strip_vocoder_prefix`, `from_config`, `from_config`, `from_config`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/ops.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/ops.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/ops.py>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `resample_audio`, `waveform_to_mel`, `__init__`, `un_normalize`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/resnet.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/resnet.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/resnet.py>) · 176 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`, `__init__`, `forward`, `__init__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/upsample.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/upsample.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/upsample.py>) · 106 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `build_upsampling_path`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py>) · 575 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_padding`, `_sinc`, `kaiser_sinc_filter1d`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/common/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/common/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/common/__init__.py>) · 9 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Common model utilities.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/model/common/normalization.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/common/normalization.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/common/normalization.py>) · 59 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `build_normalization_layer`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/model_protocol.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/model_protocol.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/model_protocol.py>) · 10 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `from_config`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/__init__.py>) · 18 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Transformer model components.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/model/transformer/adaln.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/adaln.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/adaln.py>) · 45 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `adaln_embedding_coefficient`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L22.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-core/src/ltx_core/model/transformer/attention.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/attention.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/attention.py>) · 260 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__call__`, `__call__`, `__call__`, `__call__`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py>) · 17 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py>) · 10 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/modality.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/modality.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/modality.py>) · 40 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/model.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/model.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/model.py>) · 508 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `is_video_enabled`, `is_audio_enabled`, `__init__`, `_adaln_embedding_coefficient`, `_init_video`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py>) · 158 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_build_caption_projections`, `from_config`, `from_config`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/rope.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/rope.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/rope.py>) · 235 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `apply_rotary_emb`, `apply_interleaved_rotary_emb`, `apply_split_rotary_emb`, `apply_split_rotary_emb_`, `generate_freq_grid_np`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/transformer/text_projection.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/text_projection.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/text_projection.py>) · 38 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `create_caption_projection`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L8.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py>) · 143 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_timestep_embedding`, `__init__`, `forward`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L122.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-core/src/ltx_core/model/transformer/transformer.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/transformer.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/transformer.py>) · 620 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `apply_cross_attention_adaln`, `__init__`, `get_ada_values`, `get_av_ca_ada_values`, `_apply_text_cross_attention`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Dividir `forward_` (L382, 217 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py>) · 301 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `_prepare_timestep`, `_prepare_context`, `_prepare_attention_mask`, `_prepare_self_attention_mask`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/upsampler/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/__init__.py>) · 10 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Latent upsampler model components.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/model/upsampler/blur_downsample.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/blur_downsample.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/blur_downsample.py>) · 53 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`, `_apply_2d`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/upsampler/model.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/model.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/model.py>) · 142 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `upsample_video`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/upsampler/model_configurator.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/model_configurator.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/model_configurator.py>) · 30 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `from_config`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/upsampler/pixel_shuffle.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/pixel_shuffle.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/pixel_shuffle.py>) · 54 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/upsampler/res_block.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/res_block.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/res_block.py>) · 37 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/upsampler/spatial_rational_resampler.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/upsampler/spatial_rational_resampler.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/upsampler/spatial_rational_resampler.py>) · 47 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_rational_for_scale`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/__init__.py>) · 24 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Video VAE package.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/model/video_vae/convolution.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/convolution.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/convolution.py>) · 317 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `make_conv_nd`, `make_linear_nd`, `__init__`, `reset_parameters`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/enums.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/enums.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/enums.py>) · 20 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/model_configurator.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/model_configurator.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/model_configurator.py>) · 79 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `from_config`, `from_config`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/normalization.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/normalization.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/normalization.py>) · 3 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/ops.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/ops.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/ops.py>) · 82 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `patchify`, `unpatchify`, `__init__`, `un_normalize`, `normalize`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/resnet.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/resnet.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/resnet.py>) · 277 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `_feed_spatial_noise`, `forward`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/sampling.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/sampling.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/sampling.py>) · 123 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/tiling.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/tiling.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/tiling.py>) · 291 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `compute_trapezoidal_mask_1d`, `compute_rectangular_mask_1d`, `default_split_operation`, `default_mapping_operation`, `create_tiles_from_intervals_and_mappers`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py

**Arquivo:** [packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py>) · 1221 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_make_encoder_block`, `prepare_tiles_for_encoding`, `_make_decoder_block`, `decode_video`, `get_video_chunks_number`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/quantization/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/quantization/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/quantization/__init__.py>) · 16 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/quantization/fp8_cast.py

**Arquivo:** [packages/ltx-core/src/ltx_core/quantization/fp8_cast.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/quantization/fp8_cast.py>) · 172 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `calculate_weight_float8`, `_fused_add_round_launch`, `_naive_weight_or_bias_downcast`, `_upcast_and_round`, `_replace_fwd_with_upcast`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/quantization/fp8_scaled_mm.py

**Arquivo:** [packages/ltx-core/src/ltx_core/quantization/fp8_scaled_mm.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/quantization/fp8_scaled_mm.py>) · 207 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `quantize_weight_to_fp8_per_tensor`, `_should_skip_layer`, `_linear_to_fp8linear`, `_apply_fp8_prepare_to_model`, `_create_transpose_kv_operation`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/quantization/policy.py

**Arquivo:** [packages/ltx-core/src/ltx_core/quantization/policy.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/quantization/policy.py>) · 39 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `fp8_cast`, `fp8_scaled_mm`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/__init__.py>) · 1 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** CLIP/text encoder model components.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/__init__.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/__init__.py>) · 33 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Gemma text encoder components.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/config.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/config.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/config.py>) · 75 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `to_dict`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py>) · 261 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `forward`, `__init__`, `_replace_padded_with_learnable_registers`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_processor.py>) · 89 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `convert_to_additive_mask`, `_to_binary_mask`, `__init__`, `create_embeddings`, `process_hidden_states`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py>) · 202 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_load_system_prompt`, `_cat_with_padding`, `_pad_inputs_for_attention_alignment`, `module_ops_from_gemma_root`, `__init__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py>) · 181 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_create_feature_extractor`, `create_and_populate`, `from_config`, `from_config`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py>) · 141 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_norm_and_concat_padded_batch`, `norm_and_concat_per_token_rms`, `_rescale_norm`, `__init__`, `forward`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py

**Arquivo:** [packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/text_encoders/gemma/tokenizer.py>) · 64 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `tokenize_with_weights`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/tools.py

**Arquivo:** [packages/ltx-core/src/ltx_core/tools.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/tools.py>) · 190 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `create_initial_state`, `patchify`, `unpatchify`, `clear_conditioning`, `create_initial_state`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/types.py

**Arquivo:** [packages/ltx-core/src/ltx_core/types.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/types.py>) · 209 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `default`, `to_torch_shape`, `from_torch_shape`, `token_count`, `mask_shape`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-core/src/ltx_core/utils.py

**Arquivo:** [packages/ltx-core/src/ltx_core/utils.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-core/src/ltx_core/utils.py>) · 62 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `rms_norm`, `check_config_value`, `to_velocity`, `to_denoised`, `find_matching_file`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/__init__.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/__init__.py>) · 26 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LTX-2 Pipelines: High-level video generation pipelines and utilities. This package provides ready-to-use pipelines for video generation: - TI2VidOneStagePipeline: Text/image-to-video in a single stage - TI2VidTwoStagesPipeline: Two-stage ge

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-pipelines/src/ltx_pipelines/distilled.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/distilled.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/distilled.py>) · 440 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`, `__init__`, `get_interpolated_sigmas`, `get_interpolated_sigmas2`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L168, L200.

**Melhoria sugerida:** Dividir `__call__` (L109, 278 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py>) · 476 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`, `_load_mask_video`, `_read_lora_reference_downscale_factor`, `__init__`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L470.

**Melhoria sugerida:** Dividir `__call__` (L111, 178 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py>) · 302 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`, `__init__`, `__call__`, `first_stage_denoising_loop`, `second_stage_denoising_loop`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Dividir `__call__` (L77, 169 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/music_to_video.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/music_to_video.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/music_to_video.py>) · 569 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `load_audio_input`, `main`, `__init__`, `encode_audio_latents`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** F821 L265. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L61; gestão de processos: L62; captura ampla de exceções: L201, L236; subprocesso bloqueante sem timeout explícito: L62.

**Melhoria sugerida:** Dividir `__call__` (L144, 370 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/music_to_video_v2.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/music_to_video_v2.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/music_to_video_v2.py>) · 530 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `load_audio_input`, `main`, `__init__`, `encode_audio_latents`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** F821 L318, F821 L431. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L214.

**Melhoria sugerida:** Dividir `__call__` (L146, 321 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py>) · 220 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`, `__init__`, `__call__`, `first_stage_denoising_loop`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py>) · 363 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`, `__init__`, `__call__`, `first_stage_denoising_loop`, `second_stage_denoising_loop`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L142.

**Melhoria sugerida:** Dividir `__call__` (L83, 224 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/utils/__init__.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/__init__.py>) · 35 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-pipelines/src/ltx_pipelines/utils/args.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/args.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/args.py>) · 593 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `resolve_path`, `detect_checkpoint_path`, `basic_arg_parser`, `default_1_stage_arg_parser`, `default_2_stage_arg_parser`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py>) · 143 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `detect_params`, `stage_2_height`, `stage_2_width`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L116.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py>) · 697 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_device`, `cleanup_memory`, `encode_prompts`, `combined_image_conditionings`, `video_head_conditionings`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py>) · 391 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `resize_aspect_ratio_preserving`, `resize_and_center_crop`, `normalize_latent`, `load_image_conditioning`, `load_video_conditioning`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py>) · 343 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `__init__`, `build_model_builders`, `_target_device`, `with_additional_loras`, `with_loras`.

**Verificação:** OK (Python 3.12).

- **P3 [A16](RELATORIO.md#a16): Método audio_encoder duplicado na mesma classe. Sem diferença funcional atual, mas uma correção aplicada só na primeira cópia não terá efeito. Correção: Manter uma implementação única; adicionar F811 ao lint de regressão.

**Alertas Ruff para triagem:** F811 L318. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## packages/ltx-pipelines/src/ltx_pipelines/utils/prompt_cache.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/prompt_cache.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/prompt_cache.py>) · 24 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `prompt_cache_path`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py>) · 62 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `phi`, `get_res2s_coefficients`, `get_phi`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py>) · 365 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `euler_denoising_loop`, `gradient_estimating_euler_denoising_loop`, `_channelwise_normalize`, `_get_new_noise`, `_inject_sde_noise`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L88.

**Melhoria sugerida:** Dividir `res2s_audio_video_denoising_loop` (L173, 193 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-pipelines/src/ltx_pipelines/utils/types.py

**Arquivo:** [packages/ltx-pipelines/src/ltx_pipelines/utils/types.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/types.py>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `__call__`, `__call__`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/scripts/caption_videos.py

**Arquivo:** [packages/ltx-trainer/scripts/caption_videos.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/caption_videos.py>) · 486 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Auto-caption videos with audio using multimodal models. This script provides a command-line interface for generating captions for videos (including audio) using multimodal models. It supports: - Qwen2.5-Omni: Local model for audio-visual ca

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L158, L329.

**Melhoria sugerida:** Dividir `main` (L335, 148 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-trainer/scripts/compute_reference.py

**Arquivo:** [packages/ltx-trainer/scripts/compute_reference.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/compute_reference.py>) · 288 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Compute reference videos for IC-LoRA training. This script provides a command-line interface for generating reference videos to be used for IC-LoRA training. Note that it reads and writes to the same file (the output of caption_videos.py), 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L99, L212.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-trainer/scripts/decode_latents.py

**Arquivo:** [packages/ltx-trainer/scripts/decode_latents.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/decode_latents.py>) · 341 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Decode precomputed video latents back into videos using the VAE. This script loads latent files saved during preprocessing and decodes them back into video clips using the same VAE model. Basic usage: python scripts/decode_latents.py /path/

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L120, L224.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-trainer/scripts/inference.py

**Arquivo:** [packages/ltx-trainer/scripts/inference.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/inference.py>) · 443 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** CLI script for running LTX video/audio generation inference. Usage: # Text-to-Video + Audio (default behavior) python scripts/inference.py --checkpoint path/to/model.safetensors --text-encoder-path path/to/gemma --prompt "A cat playing with

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Dividir `main` (L129, 311 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-trainer/scripts/process_captions.py

**Arquivo:** [packages/ltx-trainer/scripts/process_captions.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/process_captions.py>) · 415 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Compute text embeddings for video generation training. This module provides functionality for processing text captions, including: - Loading captions from various file formats (CSV, JSON, JSONL) - Cleaning and preprocessing text (removing L

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/scripts/process_dataset.py

**Arquivo:** [packages/ltx-trainer/scripts/process_dataset.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/process_dataset.py>) · 276 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Preprocess a video dataset by computing video clips latents and text captions embeddings. This script provides a command-line interface for preprocessing video datasets by computing latent representations of video clips and text embeddings 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/scripts/process_videos.py

**Arquivo:** [packages/ltx-trainer/scripts/process_videos.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/process_videos.py>) · 828 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Compute latent representations for video generation training. This module provides functionality for processing video and image files, including: - Loading videos/images from various file formats (CSV, JSON, JSONL) - Resizing, cropping, and

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L202, L297.

**Melhoria sugerida:** Dividir `compute_latents` (L430, 178 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-trainer/scripts/split_scenes.py

**Arquivo:** [packages/ltx-trainer/scripts/split_scenes.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/split_scenes.py>) · 417 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Split video into scenes using PySceneDetect. This script provides a command-line interface for splitting videos into scenes using various detection algorithms. It supports multiple detection methods, preview image generation, and customizab

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L283.

**Melhoria sugerida:** Dividir `detect_and_split_scenes` (L153, 154 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-trainer/scripts/train.py

**Arquivo:** [packages/ltx-trainer/scripts/train.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/scripts/train.py>) · 64 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Train LTXV models using configuration from YAML files. This script provides a command-line interface for training LTXV models using either LoRA fine-tuning or full model fine-tuning. It loads configuration from a YAML file and passes it to 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L54.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-trainer/src/ltx_trainer/__init__.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/__init__.py>) · 44 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-trainer/src/ltx_trainer/captioning.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/captioning.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/captioning.py>) · 401 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Audio-visual media captioning using multimodal models. This module provides captioning capabilities for videos with audio using: - Qwen2.5-Omni: Local model supporting text, audio, image, and video inputs (default) - Gemini Flash: Cloud-bas

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L115.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-trainer/src/ltx_trainer/config.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/config.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/config.py>) · 472 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `_get_strategy_discriminator`, `validate_model_path`, `validate_video_dims`, `validate_images`, `validate_reference_videos`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L44.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-trainer/src/ltx_trainer/config_display.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/config_display.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/config_display.py>) · 155 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Display utilities for training configuration. This module provides formatted console output for LtxTrainerConfig.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Dividir `print_config` (L12, 144 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-trainer/src/ltx_trainer/datasets.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/datasets.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/datasets.py>) · 270 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `__len__`, `__getitem__`, `__init__`, `_setup_data_root`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L235.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## packages/ltx-trainer/src/ltx_trainer/hf_hub_utils.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/hf_hub_utils.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/hf_hub_utils.py>) · 208 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `push_to_hub`, `convert_video_to_gif`, `_create_model_card`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L131, L141; captura ampla de exceções: L37, L76, L105, L169.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## packages/ltx-trainer/src/ltx_trainer/model_loader.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/model_loader.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/model_loader.py>) · 336 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Model loader for LTX-2 trainer using the new ltx-core package. This module provides a unified interface for loading LTX-2 model components for training, using SingleGPUModelBuilder from ltx-core. Example usage: # Load individual components 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/progress.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/progress.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/progress.py>) · 236 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Progress tracking for LTX training. This module provides a unified progress display for training and validation sampling, encapsulating all Rich progress bar logic in one place.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/quantization.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/quantization.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/quantization.py>) · 90 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `quantize_model`, `_quanto_type_map`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py>) · 128 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `example`, `sample`, `sample_for`, `__init__`, `sample`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/trainer.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/trainer.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/trainer.py>) · 955 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `__init__`, `train`, `_training_step`, `_load_text_encoder_and_cache_embeddings`, `_load_models`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L845; remoção de arquivos: L908.

**Melhoria sugerida:** Dividir `train` (L93, 213 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## packages/ltx-trainer/src/ltx_trainer/training_strategies/__init__.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/training_strategies/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/training_strategies/__init__.py>) · 58 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Training strategies for different conditioning modes. This package implements the Strategy Pattern to handle different training modes: - Text-to-video training (standard generation, optionally with audio) - Video-to-video training (IC-LoRA 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/training_strategies/base_strategy.py>) · 253 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Base class for training strategies. This module defines the abstract base class that all training strategies must implement, along with the base configuration class.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/training_strategies/text_to_video.py>) · 289 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Text-to-video training strategy. This strategy implements standard text-to-video generation training where: - Only target latents are used (no reference videos) - Standard noise application and loss computation - Supports first frame condit

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py>) · 223 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Video-to-video training strategy for IC-LoRA. This strategy implements training with reference video conditioning where: - Reference latents (clean) are concatenated with target latents (noised) - Video coordinates handle both reference and

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/utils.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/utils.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/utils.py>) · 118 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_gpu_memory_gb`, `open_image_as_srgb`, `save_image`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L23, L34; subprocesso bloqueante sem timeout explícito: L23.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## packages/ltx-trainer/src/ltx_trainer/validation_sampler.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/validation_sampler.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/validation_sampler.py>) · 817 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Validation sampling for LTX-2 training using ltx-core components. This module provides a simplified validation pipeline for generating samples during training, using the new ltx-core components (VideoLatentTools, AudioLatentTools, LatentSta

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## packages/ltx-trainer/src/ltx_trainer/video_utils.py

**Arquivo:** [packages/ltx-trainer/src/ltx_trainer/video_utils.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-trainer/src/ltx_trainer/video_utils.py>) · 159 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Video I/O utilities using PyAV. This module provides functions for reading and writing video files using PyAV, with optional audio support.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## repair_comfyui.bat

**Arquivo:** [repair_comfyui.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/repair_comfyui.bat>) · 15 linhas · **Cobertura:** Revisão dirigida + triagem.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

- **P1 [A14](RELATORIO.md#a14): Reparo encerra todos os processos Python e altera ambiente sem controle. Pode interromper outras gerações/servidores e substituir a combinação local de bibliotecas documentada pelo projeto; a mensagem final não garante reparo. Correção: Restringir encerramento a PID/linha de comando do ComfyUI; registrar versões, testar reparo em ambiente separado e interromper após qualquer falha.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L6, L11; gestão de processos: L3.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## roop_backend.py

**Arquivo:** [roop_backend.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/roop_backend.py>) · 113 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** roop-unleashed face-swap backend: an optional post-processing step run AFTER a scene's video is generated (by either the fp8 or GGUF backend). roop-unleashed lives in its own fully isolated environment (E:\Users\home\Documents\roop\roop-unl

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L5, L30; gestão de processos: L91, L93, L102.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## run_krea2_integrated.bat

**Arquivo:** [run_krea2_integrated.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/run_krea2_integrated.bat>) · 35 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L13; gestão de processos: L32.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## run_krea2_ui.bat

**Arquivo:** [run_krea2_ui.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/run_krea2_ui.bat>) · 13 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## sam_body_worker.py

**Arquivo:** [sam_body_worker.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/sam_body_worker.py>) · 163 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Headless SAM 3D Body inference for the unified local WebUI. Grava DOIS artefatos por imagem: - `sam_body_NN.obj` — malha estatica, para o visualizador da UI e para levar ao Blender/Hunyuan como geometria. - `sam_body_NN.npz` — o RIG: 127 ju

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L39; captura ampla de exceções: L50.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## screenplay_to_video.py

**Arquivo:** [screenplay_to_video.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/screenplay_to_video.py>) · 316 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Screenplay -> video pipeline (v1) -- top-level CLI orchestrator. Turns a plain-text screenplay into a movie by running nine stages in order, each as its own subprocess (matching this project's established convention of isolating heavy-model

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L153.

**Melhoria sugerida:** Dividir `main` (L162, 151 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## screenplay_ui.py

**Arquivo:** [screenplay_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/screenplay_ui.py>) · 1016 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Gradio UI for the screenplay-to-video pipeline (script_pipeline/), built on the same structural patterns already established in music_maker_ui_v2.py / webui_v2.py / webui_v3.py: - Global mutable state (CURRENT_RUN_DIR, IS_PROCESSING, CURREN

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L509, L513; gestão de processos: L81, L160, L161, L208, L585; captura ampla de exceções: L209; subprocesso bloqueante sem timeout explícito: L208.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## script_pipeline/__init__.py

**Arquivo:** [script_pipeline/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/__init__.py>) · 15 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Screenplay -> video pipeline (v1). Turns a plain-text screenplay into a movie: parses scenes/characters/dialogue, generates a storyboard image per scene, synthesizes character voices, renders each scene in LTX conditioned on its storyboard 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.

## script_pipeline/assemble_final.py

**Arquivo:** [script_pipeline/assemble_final.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/assemble_final.py>) · 273 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Stage [8]: concatenate every clip, in script order, into the final movie. Same ffmpeg concat-demuxer pattern already established in music_maker_ui_v2/v3.py: try a fast stream-copy concat first, fall back to a full re-encode if the clips' co

**Verificação:** OK (Python 3.12).

- **P1 [A09](RELATORIO.md#a09): Lista de concatenação não suporta caminhos Unicode nem apóstrofos. Pastas/arquivos com ação, João etc. interrompem a montagem; apóstrofos quebram a sintaxe da lista ffconcat. Correção: Gravar UTF-8 sem BOM e escapar nomes conforme ffconcat, ou criar nomes intermediários controlados e únicos.
- **P2 [A20](RELATORIO.md#a20): Normalização de concat sobrescreve entradas com mesmo basename. Ao concatenar a/clip.mp4 e b/clip.mp4, a segunda normalização substitui a primeira; a lista final aponta duas vezes para o mesmo arquivo. Correção: Usar índice estável ou hash do caminho completo/conteúdo no nome intermediário.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L120, L195; gestão de processos: L33, L44, L112, L137, L144, L203; subprocesso bloqueante sem timeout explícito: L33, L44, L112, L137, L203, L144.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/cast_characters.py

**Arquivo:** [script_pipeline/cast_characters.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/cast_characters.py>) · 650 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Character registry: one entry per character, with a visual descriptor (for storyboard/scene prompts) and a default voice assignment (for TTS) -- written as an editable ``cast.json`` the user can hand-tune before rendering. Deterministic by 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L27, L472; captura ampla de exceções: L484.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/character_sheet.py

**Arquivo:** [script_pipeline/character_sheet.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/character_sheet.py>) · 185 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Fase A da etapa de consistência de personagem (2026-09-08, pedido do usuário depois de assistir ao filme e apontar drift de rosto/acessório). PROBLEMA: `render_shots.py` já documenta que "o primeiro still de cada personagem vira reference_i

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L73.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/clip_identity_audit.py

**Arquivo:** [script_pipeline/clip_identity_audit.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/clip_identity_audit.py>) · 123 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Identidade do personagem ENTRE CLIPES, medida por embedding facial. Item 10 do MEMORIAL §7 ("identidade entre cenas, sem ferramenta de medida") -- a ferramenta ja existia para STILLS (`consistency_audit`, insightface/ArcFace, validado em §3

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/consistency_audit.py

**Arquivo:** [script_pipeline/consistency_audit.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/consistency_audit.py>) · 168 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Auditoria automatica de consistencia visual entre stills, via embedding facial (insightface/buffalo_l, ArcFace). Nasceu de um pedido do usuario depois de comparar 4 motores de storyboard a olho (2026-09-03) -- ate aqui a unica forma de sabe

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/dialogue_tts.py

**Arquivo:** [script_pipeline/dialogue_tts.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/dialogue_tts.py>) · 301 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Unified dialogue-TTS interface: engine="auto"/"xtts"/"qwen"/"fish", mirroring the engine-selection + fallback pattern already proven in tensorxx_ge/lipsync.py (auto/latentsync/wav2lip). xtts/qwen run in their OWN isolated venv via subproces

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L34, L36, L38, L40, L50, L52; gestão de processos: L145, L146; captura ampla de exceções: L216.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/emotion_director.py

**Arquivo:** [script_pipeline/emotion_director.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/emotion_director.py>) · 303 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Direção de voz: escolhe a EMOÇÃO de cada fala e a VOZ de cada personagem. POR QUE ISTO EXISTE A biblioteca de vozes já traz 12 vozes emotivas com 17 takes cada -- raiva, surpresa, angústia, alegria, medo, desdém, grito -- e `synthesize_dial

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L52, L144, L151, L199, L204; captura ampla de exceções: L104.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/generate_storyboards.py

**Arquivo:** [script_pipeline/generate_storyboards.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/generate_storyboards.py>) · 963 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Storyboard image generation: one txt2img render per scene, driven headlessly through the project's already-existing ComfyUI instance (see comfy_run.py -- this module inlines the same submit/poll pattern rather than shelling out to it, so it

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L25, L836; remoção de arquivos: L796; gestão de processos: L361, L365, L472, L490, L492, L522; captura ampla de exceções: L335.

**Melhoria sugerida:** Dividir `main` (L800, 160 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## script_pipeline/gpu_watchdog.py

**Arquivo:** [script_pipeline/gpu_watchdog.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/gpu_watchdog.py>) · 281 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Observador compartilhado para os servidores ComfyUI (LTX 2.3/2.5, MiniMax H3) e para qualquer launcher (.bat/UI) que precise liberar a porta e a VRAM antes de subir de novo. POR QUE ISTO EXISTE Toda trava real medida nesta maquina ate agora

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** remoção de arquivos: L149, L162; gestão de processos: L25, L50, L57, L73, L85, L180; captura ampla de exceções: L58, L91, L220, L250, L267; subprocesso bloqueante sem timeout explícito: L73, L57.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## script_pipeline/guide_leak_audit.py

**Arquivo:** [script_pipeline/guide_leak_audit.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/guide_leak_audit.py>) · 78 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Auditoria de VAZAMENTO DE GUIA: corte seco dentro de UM clipe. VISTO 2026-09-13 (corrida teste_webui_loras, w4a8 + IC-LoRA MSR): nos planos de 73 quadros com dois sujeitos, o video copiava a sequencia de referencia do MSR quadro a quadro --

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/ic_references.py

**Arquivo:** [script_pipeline/ic_references.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/ic_references.py>) · 273 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Referencias visuais para IC-LoRA na passada de VIDEO: folha de ingredientes e pseudo-video do MSR (Multiple Subject Reference). Por que isso existe: o still do FLUX fixa o PRIMEIRO quadro e o personagem deriva depois (MEMORIAL 3.38: a ancor

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/import_reference.py

**Arquivo:** [script_pipeline/import_reference.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/import_reference.py>) · 158 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Importa foto de referência EXTERNA (gerada por outra ferramenta, ex. ChatGPT/ DALL-E) para um personagem do cast, no formato que `character_sheet.py` produz nativamente: rosto detectável, enquadramento cintura/peito para cima, sem overlay. 

**Verificação:** OK (Python 3.12).

- **P2 [A11](RELATORIO.md#a11): Importação sem rosto falha em pasta nova. Na primeira importação de uma foto sem rosto detectável, o comportamento prometido de salvar com aviso termina em FileNotFoundError. Correção: Criar a pasta antes da bifurcação; testar referência com e sem rosto.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/import_voices.py

**Arquivo:** [script_pipeline/import_voices.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/import_voices.py>) · 242 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Importa vozes novas para a biblioteca do XTTS, na convenção que já existe. A CONVENÇÃO (não é escolha minha -- é o que `voice_library` já lê) speakers/01_vozes_emotivas/<ID_DA_VOZ>/<NN>_<emocao>.wav - `ID_DA_VOZ` começa com F ou M: `voice_l

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L72.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/lipsync_audit.py

**Arquivo:** [script_pipeline/lipsync_audit.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/lipsync_audit.py>) · 141 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Auditoria de qualidade do lip-sync aplicado -- pedido do usuario 2026-09-09 depois de assistir ao filme "Palacio de Esmeralda" e perguntar se a sincronizacao era auditada. Ate aqui, `lipsync_scenes.py` so verificava se o processo TECNICO ro

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L109, L117.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## script_pipeline/lipsync_scenes.py

**Arquivo:** [script_pipeline/lipsync_scenes.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/lipsync_scenes.py>) · 233 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Stage [6]: lip-sync each rendered dialogue clip to its own dry TTS line, via the project's already-existing tensorxx_ge/lipsync.py (LatentSync 1.6 first, Wav2Lip fallback -- unmodified, no fork needed: it's already generic to "one face, one

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L65, L161, L179.

**Melhoria sugerida:** Dividir `main` (L70, 160 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## script_pipeline/llm_workers/gemma4_worker.py

**Arquivo:** [script_pipeline/llm_workers/gemma4_worker.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/llm_workers/gemma4_worker.py>) · 97 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Gemma 4 E2B batch worker. Runs INSIDE gemma4_env (isolated venv with transformers 5.14.1 -- Gemma3's Gemma3ForConditionalGeneration import needs transformers 4.x, and Gemma 4's config.json requires transformers_version>=5.5.0.dev0; the two 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L88.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## script_pipeline/lora_ab.py

**Arquivo:** [script_pipeline/lora_ab.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/lora_ab.py>) · 156 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** A/B reprodutivel de LoRA de video: mesmo plano, mesma seed, com e sem. Pedido da auditoria de 2026-09-13. Os testes de LoRA ate aqui foram scripts avulsos no scratchpad e nunca compararam com e sem na mesma seed -- por isso "o better-human-

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** remoção de arquivos: L146; gestão de processos: L109; subprocesso bloqueante sem timeout explícito: L109.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## script_pipeline/mix_audio.py

**Arquivo:** [script_pipeline/mix_audio.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/mix_audio.py>) · 271 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Stage [7]: ambient bed + reverb/spatialization, always AFTER lip-sync -- same ordering discipline audio_fx.py already documents (reverb would smear the onsets lip-sync keyed off of if applied first). Unlike the music-video pipeline, each cl

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L37; gestão de processos: L42, L53, L108, L141, L176, L187; subprocesso bloqueante sem timeout explícito: L42, L53, L108, L141, L187, L176.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## script_pipeline/nvidia_llm.py

**Arquivo:** [script_pipeline/nvidia_llm.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/nvidia_llm.py>) · 153 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Motor de LLM opcional via API da NVIDIA (build.nvidia.com / integrate.api.nvidia.com), mesmo contrato JSON-in/JSON-out de `story_structure._call_ollama` -- pedido do usuario 2026-09-12 depois de `cast_characters.py` ter alucinado a aparenci

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L37; captura ampla de exceções: L69, L134.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/parse_screenplay.py

**Arquivo:** [script_pipeline/parse_screenplay.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/parse_screenplay.py>) · 1302 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Screenplay parsing: deterministic structure first, LLM enrichment second. Structure (scene headings, character cues, dialogue, parentheticals) is extracted by regex/heuristics -- screenplay formatting is regular enough that this is reliable

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L1069; gestão de processos: L1006, L1007; captura ampla de exceções: L948, L959, L1128.

**Melhoria sugerida:** Dividir `parse_structure` (L302, 144 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## script_pipeline/postprod_v2v.py

**Arquivo:** [script_pipeline/postprod_v2v.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/postprod_v2v.py>) · 129 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Estagio [7b], opcional: pos-producao por IC-LoRA V2V sobre os clipes ja mixados. Roda DEPOIS do mix e ANTES da montagem: o audio de cada clipe ja esta pronto e e so recolocado sobre a imagem refeita (ltx25_backend.generate_v2v, audio="remux

**Verificação:** OK (Python 3.12).

- **P1 [A10](RELATORIO.md#a10): Pós-produção pode ressuscitar manifesto e vídeos antigos. Após remontar a mixagem ou alterar os clipes da corrida, a pós-produção pode recuperar a seleção anterior e reaproveitar efeitos obsoletos. Correção: Guardar assinatura do manifesto de entrada e conteúdo; atualizar snapshot quando a origem muda, mantendo explicitamente separado o manifesto pré e pós.

**Alertas Ruff para triagem:** B023 L109. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L45; gestão de processos: L46; captura ampla de exceções: L113; subprocesso bloqueante sem timeout explícito: L46.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/prompt_polish.py

**Arquivo:** [script_pipeline/prompt_polish.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/prompt_polish.py>) · 339 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Estagio opcional [P2]: mede e reescreve os prompts de video de um shot_plan. O `shot_plan` produz prompts CORRETOS mas em lista de fragmentos: acao, emocao, descritor, movimento e look colados por pontos. O prompt bom para o LTX -- ver MEMO

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/prose_to_screenplay.py

**Arquivo:** [script_pipeline/prose_to_screenplay.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/prose_to_screenplay.py>) · 491 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Turn free-form prose (a treatment, a synopsis, or an LTX audiovisual prompt) into screenplay formatting that parse_screenplay's deterministic pass can read. WHY THIS IS A SEPARATE, EXPLICIT STEP parse_screenplay's core promise is that struc

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L346, L358; gestão de processos: L330; captura ampla de exceções: L242, L406, L409, L460.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/render_scenes.py

**Arquivo:** [script_pipeline/render_scenes.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_scenes.py>) · 809 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Stage [5]: render one LTX video clip per dialogue line (shot-per-speaker; v1 does not attempt multi-character-speaking-at-once shots -- see plan), or one clip for the whole scene when it has no dialogue (establishing/action-only beats). Use

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L243, L278, L355, L409, L684; gestão de processos: L244, L279, L374, L570, L571; captura ampla de exceções: L364, L417; subprocesso bloqueante sem timeout explícito: L279, L244, L374.

**Melhoria sugerida:** Dividir `main` (L652, 154 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## script_pipeline/render_shots.py

**Arquivo:** [script_pipeline/render_shots.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py>) · 1197 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Executa um `shot_plan`: para cada plano, gera o STILL e depois o VÍDEO condicionado por ele. É a ponte que faltava entre decupagem e imagem. POR QUE DOIS ESTÁGIOS POR PLANO, E NÃO UM PROMPT SÓ MEDIDO 2026-08-26 (MEMORIAL.md §3.21): o LTX tr

**Verificação:** OK (Python 3.12).

- **P1 [A02](RELATORIO.md#a02): Cache de áudio e imagem identifica conteúdo apenas pelo tamanho. Vídeo antigo pode ser reutilizado após mudar voz, fala ou imagem, causando identidade ou sincronia incorretas. Correção: Usar hash de conteúdo, com memoização por metadados se necessário. Não tratar ausência e erro de leitura como a mesma identidade.
- **P1 [A03](RELATORIO.md#a03): Cache de still não invalida referências e parâmetros de amostragem. Trocar a sheet, importar outra foto ou ajustar qualidade pode devolver exatamente o still anterior. A justificativa no docstring não propaga a chave da referência ao dependente. Correção: Serializar a configuração efetiva e os hashes de ambas as referências; separar geração de reavaliação de consistência.
- **P1 [A04](RELATORIO.md#a04): Cache de vídeo omite configuração e usa grade LTX para todos os motores. Certas mudanças são ignoradas; em outros casos clipes válidos são regenerados indefinidamente por divergência de grade ou pós-efeito. Correção: Chave versionada com configuração resolvida; registrar frames/fps realmente produzidos e validar por contrato específico do motor, incluindo pós-processamento.
- **P1 [A05](RELATORIO.md#a05): Encadeamento MiniMax descarta o último frame quando já há duas referências. O modo comum com still + sheet deixa de encadear movimento e cenário, embora o log anuncie encadeamento. Correção: Escolher explicitamente sheet + último frame nos segmentos posteriores; definir a prioridade dos dois slots.
- **P1 [A06](RELATORIO.md#a06): Encadeamento LTX reduz duração e condiciona áudio apenas no primeiro trecho. Um plano de 401 frames vira 387 frames (14 a menos, ~0,58s a 24fps). A continuação da fala deixa de condicionar os segmentos seguintes. Correção: Planejar os segmentos em tempo, cortar o WAV por segmento, distribuir o restante e aparar/padronizar o resultado para a duração contratada. Adequar também a guia IC por segmento.
- **P1 [A07](RELATORIO.md#a07): Falha da concatenação pode ser registrada como clipe bem-sucedido. Manifesto pode marcar ok=true para vídeo inexistente ou para uma saída antiga após concat falhar. Correção: Levantar erro quando concat retornar False; gerar em temporário e promover somente após validar arquivo, streams e duração; validar o arquivo ao montar manifesto.
- **P1 [A08](RELATORIO.md#a08): Plano sem still desaparece da contagem de falhas. Um conjunto parcial ou vazio pode ser marcado como render concluído, permitindo montagem incompleta. Correção: Registrar uma entrada de falha por plano solicitado e comparar com o conjunto esperado. No modo stills-only também retornar falha quando faltar saída solicitada.
- **P1 [A17](RELATORIO.md#a17): Modo combinado MiniMax desliga o servidor necessário aos stills. A invocação combinada sem stills em cache tenta gerar imagens num servidor desligado. O orquestrador em duas passadas evita essa rota, mas a função/CLI avulsa a expõe. Correção: Gerar todos os stills antes da troca de servidor, ou exigir explicitamente duas fases para MiniMax.

**Alertas Ruff para triagem:** B023 L630, B023 L636, B023 L636, B023 L645, B023 L649, B023 L671, B023 L675, B023 L679, B023 L679, B023 L685, B023 L687, B023 L689, B023 L690. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L48, L49, L246, L376, L377, L387 (+5); remoção de arquivos: L188, L286, L1078; gestão de processos: L161, L185, L906, L971, L979, L1074; captura ampla de exceções: L206, L813; subprocesso bloqueante sem timeout explícito: L161, L185, L906, L971, L979, L1074.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/render_shots_stage.py

**Arquivo:** [script_pipeline/render_shots_stage.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots_stage.py>) · 323 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Estágio [5-D]: variante DECUPADA do estágio 5, plugável no pipeline validado. O QUE ISTO É, E O QUE NÃO É `render_scenes.py` (estágio 5) já renderiza um clipe por fala, com duração vinda do TTS e o storyboard da cena como condicionamento. E

**Verificação:** OK (Python 3.12).

- **P1 [A07](RELATORIO.md#a07): Falha da concatenação pode ser registrada como clipe bem-sucedido. Manifesto pode marcar ok=true para vídeo inexistente ou para uma saída antiga após concat falhar. Correção: Levantar erro quando concat retornar False; gerar em temporário e promover somente após validar arquivo, streams e duração; validar o arquivo ao montar manifesto.
- **P1 [A08](RELATORIO.md#a08): Plano sem still desaparece da contagem de falhas. Um conjunto parcial ou vazio pode ser marcado como render concluído, permitindo montagem incompleta. Correção: Registrar uma entrada de falha por plano solicitado e comparar com o conjunto esperado. No modo stills-only também retornar falha quando faltar saída solicitada.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/run_decupagem.py

**Arquivo:** [script_pipeline/run_decupagem.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/run_decupagem.py>) · 502 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Orquestrador da variante DECUPADA. Não substitui nada; declara um caminho. POR QUE UM ORQUESTRADOR SEPARADO Havia dois caminhos fazendo coisas sobrepostas sem que nenhum documento dissesse isso -- que é a pior das configurações possíveis, p

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L73; subprocesso bloqueante sem timeout explícito: L73.

**Melhoria sugerida:** Dividir `main` (L84, 415 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## script_pipeline/run_folder.py

**Arquivo:** [script_pipeline/run_folder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/run_folder.py>) · 125 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Run-folder convention for the screenplay pipeline. Mirrors ``create_generation_folder``/``run_subdir`` in ``music_maker_ui_v2.py``: one timestamped folder per run under ``outputs/screenplay/``, with a fixed set of stage subfolders and a JSO

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/shot_plan.py

**Arquivo:** [script_pipeline/shot_plan.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/shot_plan.py>) · 1336 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Decupagem: transforma cenas + estrutura dramática em uma lista de PLANOS, cada um com câmera concreta. A camada que faltava. `parse_screenplay` produz `shot_list`, mas ele só ORDENA conteúdo -- `{"type":"dialogue","line_index":3}` -- sem di

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L1032; captura ampla de exceções: L1075.

**Melhoria sugerida:** Dividir `plan_scene` (L696, 191 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## script_pipeline/story_structure.py

**Arquivo:** [script_pipeline/story_structure.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/story_structure.py>) · 400 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Leitura do roteiro INTEIRO, de uma vez, para produzir a camada que falta: estrutura dramática e continuidade entre cenas. POR QUE ISTO É UMA ETAPA SEPARADA, E NÃO UM PARÂMETRO DO PARSE O enriquecimento de `parse_screenplay` roda **uma chama

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L70, L162, L187; captura ampla de exceções: L237, L241.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/storyboard_audit.py

**Arquivo:** [script_pipeline/storyboard_audit.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/storyboard_audit.py>) · 105 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Relatorio agregado dos stills de uma corrida -- ponto de revisao ANTES do estagio de video, pedido do usuario 2026-09-10 depois de dois defeitos reais passarem direto pro vídeo sem aviso nenhum: um still com dois personagens onde o segundo 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## script_pipeline/strip_subtitles.py

**Arquivo:** [script_pipeline/strip_subtitles.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/strip_subtitles.py>) · 142 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Remove LTX-2.3's burned-in subtitle band by cropping it off. WHY THIS EXISTS -- and why it is a crop rather than a fix at the source. LTX-2.3 paints garbled subtitles over any clip containing speech. The text is never readable ("Yau oughe w

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** gestão de processos: L47, L68, L90; subprocesso bloqueante sem timeout explícito: L47, L90, L68.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## script_pipeline/syncnet_audit.py

**Arquivo:** [script_pipeline/syncnet_audit.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/syncnet_audit.py>) · 112 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Sincronia labial medida pelo SyncNet (LSE-C / LSE-D), o avaliador padrao da area. Pedido da auditoria de 2026-09-13. O `lipsync_audit` correlaciona movimento do terco inferior do rosto com o volume do audio: e um proxy, e mediu mal rosto a 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L34; remoção de arquivos: L97; gestão de processos: L75, L82, L94; subprocesso bloqueante sem timeout explícito: L75.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## script_pipeline/synthesize_dialogue.py

**Arquivo:** [script_pipeline/synthesize_dialogue.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/synthesize_dialogue.py>) · 208 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Stage [4]: synthesize one dry (unprocessed) WAV per dialogue line via dialogue_tts.py. Builds one TTS job per (scene, line), carrying the character's cast.json voice assignment and a delivery hint (the line's parenthetical if present, else 

**Verificação:** OK (Python 3.12).

- **P2 [A19](RELATORIO.md#a19): Cache TTS não acompanha alteração da referência no mesmo caminho. Substituir a amostra de voz mantendo o nome pode reutilizar a voz antiga. Mudanças no motor resolvido exigem política explícita de invalidação. Correção: Incluir hash das referências e versão/configuração do motor; persistir a resolução de auto para diagnóstico.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/tts_workers/fish_worker.py

**Arquivo:** [script_pipeline/tts_workers/fish_worker.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/tts_workers/fish_worker.py>) · 121 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Fish Speech batch worker. Roda DENTRO do venv proprio do fish-speech (nao o principal do LTX) -- mesmo motivo do xtts_worker.py: ormsgpack/requests/ fish_speech.utils.schema so existem la, e chamar via subprocesso evita ter que casar as dep

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L37; captura ampla de exceções: L91, L111.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/tts_workers/qwen_worker.py

**Arquivo:** [script_pipeline/tts_workers/qwen_worker.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/tts_workers/qwen_worker.py>) · 108 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Qwen3-TTS batch worker. Runs INSIDE Qwen3-TTS's own isolated venv (.venv), not the main project venv -- invoked via subprocess by dialogue_tts.py. v1 scope: only the CustomVoice checkpoint (preset timbres + free-text emotion/delivery "instr

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L28; captura ampla de exceções: L77, L96.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/tts_workers/xtts_worker.py

**Arquivo:** [script_pipeline/tts_workers/xtts_worker.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/tts_workers/xtts_worker.py>) · 268 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** XTTS-v2 batch worker. Runs INSIDE xtts's own isolated venv (webui/venv), not the main project venv -- invoked via subprocess by dialogue_tts.py, mirroring how tensorxx_ge/lipsync.py shells out to LatentSync's own conda env. Loads the model 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L26; remoção de arquivos: L235; captura ampla de exceções: L256.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## script_pipeline/verify_output.py

**Arquivo:** [script_pipeline/verify_output.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/verify_output.py>) · 328 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Stage [9]: post-render self-review -- check the finished film against the defects this pipeline has actually shipped, and fail loudly instead of quietly. Every check here exists because the corresponding bug reached the user in a real run: 

**Verificação:** OK (Python 3.12).

- **P1 [A15](RELATORIO.md#a15): Verificador não detecta clipes ausentes do filme planejado. Um filme truncado por planos que já faltavam no manifesto/arquivo pode passar na checagem de duração. Correção: Emitir MISSING_CLIP e reconciliar IDs do shot_plan, manifesto e arquivos, respeitando seleção parcial declarada. Usar duração planejada como controle independente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L39, L80, L81; gestão de processos: L47, L59, L75, L96; subprocesso bloqueante sem timeout explícito: L47, L59, L75, L96.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## script_pipeline/voice_library.py

**Arquivo:** [script_pipeline/voice_library.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/voice_library.py>) · 220 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Discovery and emotion-matching over the xtts speaker library. The library grew three tiers, and only the first was ever visible to the pipeline: 1. FLAT legacy clips at the root: male.wav, female.wav, calm_female.wav, sandra.mp3 plus 24 per

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L32.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## scripts_2.5/dit_bridge.py

**Arquivo:** [scripts_2.5/dit_bridge.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/dit_bridge.py>) · 137 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LTX-2.5 DiT (transformer) + embeddings-processor bridge. Reuses existing ltx_core building blocks UNMODIFIED (LTXModelConfigurator, Embeddings1DConnector[Configurator], FeatureExtractorV2, EmbeddingsProcessor) -- these already implement the

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L24, L41, L44.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## scripts_2.5/dit_forward_test.py

**Arquivo:** [scripts_2.5/dit_forward_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/dit_forward_test.py>) · 124 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** End-to-end smoke test: encode a prompt with Gemma25TextEncoder, run it through the (Gemma4-sized) EmbeddingsProcessor, and do ONE forward pass through the real LTX-2.5 transformer with a synthetic random video latent. This validates that th

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## scripts_2.5/gemma4_text_encoder.py

**Arquivo:** [scripts_2.5/gemma4_text_encoder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/gemma4_text_encoder.py>) · 87 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Standalone LTX-2.5 text encoder (Gemma 4 12B, "gemma4_unified"). LTX-2.5's transformer (DiT) is not wired into ltx_core yet -- see CLAUDE.md, section "LTX-2.5". This module only covers the text encoder half, which is validated and works sta

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L21, L23.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## scripts_2.5/generate_latent_test.py

**Arquivo:** [scripts_2.5/generate_latent_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/generate_latent_test.py>) · 132 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Multi-step LTX-2.5 T2V latent generation (no VAE decode yet -- see CLAUDE.md / MEMORIAL.md, section "LTX-2.5": the 2.5 video VAE decoder is a new, undocumented architecture (NADiffusionDecoder) and hasn't been implemented). This runs the RE

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## scripts_2.5/na_diffusion_decoder.py

**Arquivo:** [scripts_2.5/na_diffusion_decoder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/na_diffusion_decoder.py>) · 278 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Best-effort reimplementation of LTX-2.5's video VAE decoder ("NADiffusionDecoder"). WARNING -- confidence level is much lower than every other module in scripts_2.5/. Unlike the Gemma4 encoder (found ready-made in ComfyUI) and the DiT embed

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.

## scripts_2.5/vae_decode_official.py

**Arquivo:** [scripts_2.5/vae_decode_official.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/vae_decode_official.py>) · 85 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Decode the saved LTX-2.5 latent using the REAL NADiffusionDecoder from ComfyUI core (comfy/ldm/lightricks/vae/na_diffusion_decoder.py), pulled in via updating the vendored ComfyUI to upstream master on 2026-08-22. Supersedes the hand-revers

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L16, L22.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## scripts_2.5/vae_decode_test.py

**Arquivo:** [scripts_2.5/vae_decode_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/scripts_2.5/vae_decode_test.py>) · 106 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Load na_diffusion_decoder.NADiffusionDecoder from the real 2.5 checkpoint and run a forward pass against the latent produced by generate_latent_test.py. See na_diffusion_decoder.py's module docstring for the confidence caveat: this architec

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L20.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## start_character3d.bat

**Arquivo:** [start_character3d.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_character3d.bat>) · 9 linhas · **Cobertura:** Revisão dirigida + triagem.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

- **P2 [A18](RELATORIO.md#a18): Launcher Character3D depende do diretório de quem chama. Quando chamado por terminal/atalho com outro diretório de trabalho, Python não encontra o arquivo, embora o launcher ainda aguarde a porta. Correção: Resolver o diretório pelo próprio .bat e explicitar o interpretador apropriado para a integração.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2, L4, L9.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## start_character_sheet_flux.bat

**Arquivo:** [start_character_sheet_flux.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_character_sheet_flux.bat>) · 11 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4; gestão de processos: L6, L8.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_cinema.bat

**Arquivo:** [start_cinema.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_cinema.bat>) · 36 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L3, L27; gestão de processos: L20, L33.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_cinema_25.bat

**Arquivo:** [start_cinema_25.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_cinema_25.bat>) · 67 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L15, L56, L58; gestão de processos: L52, L65.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_comfyui_ltx.bat

**Arquivo:** [start_comfyui_ltx.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_comfyui_ltx.bat>) · 33 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L3; gestão de processos: L15, L31.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_decupagem.bat

**Arquivo:** [start_decupagem.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_decupagem.bat>) · 142 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L17.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_decupagem_ui.bat

**Arquivo:** [start_decupagem_ui.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_decupagem_ui.bat>) · 26 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L19.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_hunyuan3d_download.bat

**Arquivo:** [start_hunyuan3d_download.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_hunyuan3d_download.bat>) · 5 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_img2threejs_viewer.bat

**Arquivo:** [start_img2threejs_viewer.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_img2threejs_viewer.bat>) · 29 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L21, L22; gestão de processos: L17, L27.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_lora_storyboard_test.bat

**Arquivo:** [start_lora_storyboard_test.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_lora_storyboard_test.bat>) · 30 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L24; gestão de processos: L20, L28.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## start_minimax_h3_test.bat

**Arquivo:** [start_minimax_h3_test.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_minimax_h3_test.bat>) · 30 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L27, L28.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## start_music_video.bat

**Arquivo:** [start_music_video.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video.bat>) · 54 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L5, L14, L40, L42; gestão de processos: L48.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_music_video_gguf.bat

**Arquivo:** [start_music_video_gguf.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video_gguf.bat>) · 43 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L32, L34, L35; gestão de processos: L28, L41.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_music_video_v2.bat

**Arquivo:** [start_music_video_v2.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video_v2.bat>) · 57 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L40, L48, L49, L50; gestão de processos: L45, L55.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_music_video_v2.prev.bat

**Arquivo:** [start_music_video_v2.prev.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video_v2.prev.bat>) · 18 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L14, L15, L16, L17.

**Melhoria sugerida:** Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.

## start_music_video_v2_25.bat

**Arquivo:** [start_music_video_v2_25.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video_v2_25.bat>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L15, L62, L64; gestão de processos: L58, L71.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_music_video_v3.bat

**Arquivo:** [start_music_video_v3.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video_v3.bat>) · 60 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L39, L40, L42, L51, L52 (+1); gestão de processos: L47, L58.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_music_video_v3_25.bat

**Arquivo:** [start_music_video_v3_25.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_music_video_v3_25.bat>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L15, L62, L64; gestão de processos: L58, L71.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_qwen_claude_web.bat

**Arquivo:** [start_qwen_claude_web.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_qwen_claude_web.bat>) · 9 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_screenplay_25.bat

**Arquivo:** [start_screenplay_25.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_screenplay_25.bat>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L13, L66, L67, L68; gestão de processos: L62, L71.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_screenplay_ui.bat

**Arquivo:** [start_screenplay_ui.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_screenplay_ui.bat>) · 34 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L12, L25, L26, L27; gestão de processos: L22, L32.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_storyplay25.bat

**Arquivo:** [start_storyplay25.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_storyplay25.bat>) · 78 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L22, L67, L69; gestão de processos: L63, L76.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_tensorRT_studio.bat

**Arquivo:** [start_tensorRT_studio.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_tensorRT_studio.bat>) · 25 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L23.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_tensorRT_v2.bat

**Arquivo:** [start_tensorRT_v2.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_tensorRT_v2.bat>) · 50 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L35, L48; gestão de processos: L45.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_tensorRT_v3.bat

**Arquivo:** [start_tensorRT_v3.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_tensorRT_v3.bat>) · 54 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L37, L38, L39, L52; gestão de processos: L49.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_tensorRT_webui.bat

**Arquivo:** [start_tensorRT_webui.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_tensorRT_webui.bat>) · 25 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L23.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_tensorxx_ge.bat

**Arquivo:** [start_tensorxx_ge.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_tensorxx_ge.bat>) · 22 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_video_doctor.bat

**Arquivo:** [start_video_doctor.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_video_doctor.bat>) · 42 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L16, L33, L35; gestão de processos: L29, L40.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_webui.bat

**Arquivo:** [start_webui.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_webui.bat>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L5, L26, L41, L42, L64, L65; gestão de processos: L50, L71.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## start_webui_25.bat

**Arquivo:** [start_webui_25.bat](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_webui_25.bat>) · 67 linhas · **Cobertura:** Triagem estática/textual.

**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.

**Verificação:** Inspeção textual; não executado.

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L4, L15, L56, L58; gestão de processos: L52, L65.

**Melhoria sugerida:** Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.

## storyplay25.py

**Arquivo:** [storyplay25.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/storyplay25.py>) · 541 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** StoryPlay 2.5 -- storyboard com QUADROS INTERMEDIARIOS visiveis, por cena. Diferenca em relacao ao pipeline de screenplay existente: la, o estagio `storyboard` gera UMA imagem por cena, usada apenas como primeiro frame do clipe. Aqui, cada 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L65; gestão de processos: L111, L137; captura ampla de exceções: L157, L441; subprocesso bloqueante sem timeout explícito: L111, L137.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## symplectic_experiment/agregar_cv.py

**Arquivo:** [symplectic_experiment/agregar_cv.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/agregar_cv.py>) · 96 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Agrega a validação cruzada por famílias de `train_latent_dynamics.py`. Lê <base>/cv_f*/resultados.json e compara dois modelos pareando por (dobra, semente). A diferença é MSE(B) - MSE(A), então positivo quer dizer que A foi melhor. Dois nív

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/baseline_predictor.py

**Arquivo:** [symplectic_experiment/baseline_predictor.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/baseline_predictor.py>) · 39 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Baseline SEM estrutura física — a linha decisiva do experimento. Um preditor temporal compacto que mapeia (q, p, ctx) -> (q', p') diretamente, sem Hamiltoniano, sem integrador simplético, sem conservação. Mesma dimensão de estado, mesmo dec

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/decode_predictions.py

**Arquivo:** [symplectic_experiment/decode_predictions.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/decode_predictions.py>) · 163 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Passo 3 do teste em vídeo real: decodificar com o decoder CONGELADO os latentes previstos por `train_latent_dynamics.py`, e medir no espaço de imagem. Duas comparações, para separar as fontes de erro: previsto decodificado vs latente REAL d

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L60.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/extract_latents.py

**Arquivo:** [symplectic_experiment/extract_latents.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/extract_latents.py>) · 118 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Passo 1 do teste em vídeo real: latentes do VAE do LTX 2.5 para os clipes do professor (vídeos já gerados pelo LTX 2.5 neste repositório). Formato MEDIDO no ida e volta (vae_roundtrip.py, 2026-09-13): um clipe de 121 quadros 960x544 vira la

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L103.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/hamiltonian.py

**Arquivo:** [symplectic_experiment/hamiltonian.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/hamiltonian.py>) · 116 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Blocos hamiltonianos latentes: energia cinética/potencial e integrador simplético. Ver README.md deste diretório para o que isto valida e o que NÃO valida.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/inr_decoder.py

**Arquivo:** [symplectic_experiment/inr_decoder.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/inr_decoder.py>) · 73 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Decoder INR modulado (Fourier features + FiLM/SIREN) sobre a coordenada latente q(t). Resolução de saída é definida pela malha de coordenadas passada em forward(), não pela arquitetura — permite renderizar em resoluções diferentes das usada

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/pf_ode.py

**Arquivo:** [symplectic_experiment/pf_ode.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/pf_ode.py>) · 74 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Probability Flow ODE com precondicionamento EDM (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models"). Isto é a formulação determinística REAL e correta de "vídeo como trajetória contínua" — ao contrário 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/rectified_flow.py

**Arquivo:** [symplectic_experiment/rectified_flow.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/rectified_flow.py>) · 88 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Conditional Flow Matching / Rectified Flow (Lipman et al. 2022; Liu et al. 2022 "Flow Straight and Fast") COM precondicionamento — análogo ao EDM (Karras et al. 2022), adaptado ao caminho reto do flow matching. Por que o precondicionamento 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/score_model.py

**Arquivo:** [symplectic_experiment/score_model.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/score_model.py>) · 43 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Rede de score toy: F_theta(x_in, c_noise) -> predição bruta, mesma forma de x. A rede NUNCA vê o eixo de tempo físico do vídeo diretamente — só o nível de ruído sigma (c_noise). O vídeo inteiro (todos os frames concatenados) é UM ponto x no

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/toy_dataset.py

**Arquivo:** [symplectic_experiment/toy_dataset.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/toy_dataset.py>) · 47 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Vídeo sintético: um disco colorido quicando elasticamente numa caixa 2D. Escolhido de propósito por ser um sistema aproximadamente conservativo (fora dos instantes de colisão) — é o caso mais favorável possível para um integrador hamiltonia

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/toy_video_distribution.py

**Arquivo:** [symplectic_experiment/toy_video_distribution.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/toy_video_distribution.py>) · 49 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Distribuição de vídeos sintéticos (disco quicando), para treinar um score model de verdade — não um único vídeo fixo. Posição inicial, velocidade e cor são amostradas aleatoriamente por vídeo: p(x) é a distribuição sobre TODAS as trajetória

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_compare.py

**Arquivo:** [symplectic_experiment/train_compare.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_compare.py>) · 189 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Compara núcleo hamiltoniano vs. preditor compacto SEM estrutura física, sob condições idênticas (mesmo dado, mesma semente, mesmo decoder, mesmo orçamento de treino), medindo duas coisas: 1. reconstrução DENTRO do horizonte de treino 2. rec

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_dissipative.py

**Arquivo:** [symplectic_experiment/train_dissipative.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_dissipative.py>) · 233 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Linha 4 da reanálise: o ganho da estrutura sobrevive quando o sistema DISSIPA? `train_hybrid.py` confirmou (20 sementes, p = 0,005) que o hamiltoniano puro acompanha a trajetória +29% mais tempo que um aluno do mesmo tamanho sem estrutura. 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_dynamics_only.py

**Arquivo:** [symplectic_experiment/train_dynamics_only.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_dynamics_only.py>) · 191 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Comparação LIMPA: núcleo hamiltoniano vs. preditor compacto sem estrutura, em espaço de estado puro — SEM decoder, SEM aprender o espaço latente. Por que esta versão existe: as duas tentativas anteriores (train_toy.py, train_compare.py) apr

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_hybrid.py

**Arquivo:** [symplectic_experiment/train_hybrid.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_hybrid.py>) · 373 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Linha 1 da reanálise: o núcleo hamiltoniano passa a valer quando os EVENTOS DESCONTÍNUOS são tratados à parte? `train_dynamics_only.py` mediu que o hamiltoniano puro perde por ~3 ordens de grandeza para uma rede sem estrutura, e a causa é a

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_latent_dynamics.py

**Arquivo:** [symplectic_experiment/train_latent_dynamics.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_latent_dynamics.py>) · 474 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Primeiro teste da tese em VÍDEO REAL: um propagador compacto sobre latentes do VAE do LTX 2.5 (decoder congelado) prevê a evolução da cena melhor com estrutura hamiltoniana do que sem? Dados: latentes de `extract_latents.py`, clipes do LTX 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** B008 L267, B008 L357, B008 L358, B023 L400. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_pf_ode.py

**Arquivo:** [symplectic_experiment/train_pf_ode.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_pf_ode.py>) · 111 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Treina um score model real (EDM denoising score matching) sobre a distribuição de vídeos sintéticos, depois demonstra a propriedade que a tentativa hamiltoniana prometia mas não conseguia sustentar de forma válida: uma trajetória determinís

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_rectified_flow.py

**Arquivo:** [symplectic_experiment/train_rectified_flow.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_rectified_flow.py>) · 97 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Treina o campo de velocidade (agora com precondicionamento — ver rectified_flow.py) via Conditional Flow Matching sobre a mesma distribuição de vídeos sintéticos usada em train_pf_ode.py, e roda o MESMO teste de round-trip + diagnóstico ant

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/train_toy.py

**Arquivo:** [symplectic_experiment/train_toy.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/train_toy.py>) · 108 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Sanity check: o integrador simplético + decoder INR conseguem aprender a reconstruir um vídeo sintético de brinquedo (disco quicando)? Isto NÃO treina sobre pesos do LTX nem usa texto/áudio condicionante — ver README.md deste diretório para

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/vae_roundtrip.py

**Arquivo:** [symplectic_experiment/vae_roundtrip.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/vae_roundtrip.py>) · 147 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Passo 0 do teste em vídeo real: o VAE do LTX 2.5 faz ida e volta sem estragar o vídeo, rodando FORA do Windows (Linux, Python 3.13, torch 2.11, RTX 4070)? Sem isso, nenhum preditor treinado sobre latentes pode ser avaliado: o decoder congel

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L130.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## symplectic_experiment/velocity_model.py

**Arquivo:** [symplectic_experiment/velocity_model.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/symplectic_experiment/velocity_model.py>) · 42 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Rede de velocidade toy para Rectified Flow / Flow Matching. Ao contrário do score model (que recebe sigma, nível de ruído), esta rede recebe t em [0,1], a posição ao longo do caminho reto entre ruído (t=0) e dado (t=1). Mesma observação do 

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.

## test_generation.py

**Arquivo:** [test_generation.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/test_generation.py>) · 100 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `test_gen`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L5; captura ampla de exceções: L95.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## test_quem_slice_remote.py

**Arquivo:** [test_quem_slice_remote.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/test_quem_slice_remote.py>) · 11 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## tests/__init__.py

**Arquivo:** [tests/__init__.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/__init__.py>) · 0 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_auditoria_20260913.py

**Arquivo:** [tests/test_auditoria_20260913.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_auditoria_20260913.py>) · 81 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Itens da auditoria de scripts de 2026-09-13 que dao para testar sem GPU.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_limpar_cache_stills.py

**Arquivo:** [tests/test_limpar_cache_stills.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_limpar_cache_stills.py>) · 54 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Limpador de cache de stills da decupagem_ui (2026-09-13). Sem GPU, sem rede: monta uma corrida falsa em tmp e confere o que some e o que fica.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_ltx_loras.py

**Arquivo:** [tests/test_ltx_loras.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_ltx_loras.py>) · 233 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LoRAs de video: catalogo, referencias IC e enxertos no grafo 2.5 -- sem GPU e sem ComfyUI no ar. O grafo base (fixture) e o que comfy_workflow_tool.convert() produz do workflow oficial LTX-2.5_T2V_I2V_Single_Stage_Distilled contra o /object

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L53, L54, L195.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_minimax_h3_test_ui.py

**Arquivo:** [tests/test_minimax_h3_test_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_minimax_h3_test_ui.py>) · 43 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `test_av_contract_and_separate_decoders`, `test_turbo_and_references_are_connected`, `test_frame_grid_and_invalid_parameters`, `test_missing_node_is_reported_before_submission`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_optimization_paths.py

**Arquivo:** [tests/test_optimization_paths.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_optimization_paths.py>) · 76 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `test_ingredients_resize_uses_requested_dimensions`, `test_ingredients_resize_preserves_legacy_fallback`, `test_prompt_cache_path_matches_native_cache_contract`, `test_accelerate_keeps_ltx_blocks_indivisible`, `test_attention_norms_materialize_from_accelerate_offload`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_pipeline_smoke.py

**Arquivo:** [tests/test_pipeline_smoke.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_pipeline_smoke.py>) · 233 linhas · **Cobertura:** Revisão dirigida + triagem.

**Finalidade declarada no código:** Suíte rápida, sem GPU e sem rede -- roda em segundos. Cobre as funções PURAS dos pontos que mais quebraram em silêncio esta sessão: seleção de saída do ComfyUI, deteccão de corte do doctor, score de prompt, seleção de índices e resolução de

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## tests/test_tensorxx_ge.py

**Arquivo:** [tests/test_tensorxx_ge.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/tests/test_tensorxx_ge.py>) · 85 linhas · **Cobertura:** Revisão dirigida + triagem.

**Pontos de entrada/rotinas identificados:** `setUp`, `test_capture_round_trip_keeps_ltx_contract`, `test_cache_key_is_order_independent`, `test_projected_cache_excludes_raw_hidden_states`, `test_tensor_comparison_reports_exact_match`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.

## transcribe_vocals.py

**Arquivo:** [transcribe_vocals.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/transcribe_vocals.py>) · 46 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `main`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** captura ampla de exceções: L32.

**Melhoria sugerida:** Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.

## upscale_film.ps1

**Arquivo:** [upscale_film.ps1](<E:/Users/home/Documents/LTX-2-OPTIMIZED/upscale_film.ps1>) · 40 linhas · **Cobertura:** Revisão dirigida + triagem.

**Verificação:** Parser PowerShell: OK; nenhuma execução.

- **P1 [A12](RELATORIO.md#a12): Scripts de upscale anunciam sucesso sem verificar executáveis. Falhas podem chegar até mensagens PRONTO/Instalação concluída; os wrappers de upscale ainda apagam intermediários após falha. Correção: Verificar LASTEXITCODE imediatamente e validar saída antes de continuar; preservar intermediários em falha. Aplicar o mesmo contrato ao download.
- **P2 [A13](RELATORIO.md#a13): Upcale compartilha temporários destrutivos entre execuções. Duas execuções simultâneas podem apagar quadros ou substituir vídeos uma da outra. Correção: Criar diretório único por execução; validar confinamento antes da limpeza e remover só os arquivos da própria execução.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L2, L5, L9; remoção de arquivos: L17, L39.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## upscale_video.ps1

**Arquivo:** [upscale_video.ps1](<E:/Users/home/Documents/LTX-2-OPTIMIZED/upscale_video.ps1>) · 35 linhas · **Cobertura:** Revisão dirigida + triagem.

**Verificação:** Parser PowerShell: OK; nenhuma execução.

- **P1 [A12](RELATORIO.md#a12): Scripts de upscale anunciam sucesso sem verificar executáveis. Falhas podem chegar até mensagens PRONTO/Instalação concluída; os wrappers de upscale ainda apagam intermediários após falha. Correção: Verificar LASTEXITCODE imediatamente e validar saída antes de continuar; preservar intermediários em falha. Aplicar o mesmo contrato ao download.
- **P2 [A13](RELATORIO.md#a13): Upcale compartilha temporários destrutivos entre execuções. Duas execuções simultâneas podem apagar quadros ou substituir vídeos uma da outra. Correção: Criar diretório único por execução; validar confinamento antes da limpeza e remover só os arquivos da própria execução.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L10; remoção de arquivos: L20, L34.

**Melhoria sugerida:** Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.

## v2_generation_smoke_test.py

**Arquivo:** [v2_generation_smoke_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/v2_generation_smoke_test.py>) · 41 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L5, L9, L10.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## video_doctor.py

**Arquivo:** [video_doctor.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/video_doctor.py>) · 916 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Diagnóstico e reparo temporal pós-geração para clipes do LTX. O problema: o LTX produz artefatos que duram poucos frames -- rosto ou mão "derretendo", textura pulsando, um frame isolado destoando dos vizinhos. Regerar a cena inteira por cau

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L83; remoção de arquivos: L156, L591, L697; gestão de processos: L115, L153, L575; captura ampla de exceções: L585, L691; subprocesso bloqueante sem timeout explícito: L115, L153, L575.

**Melhoria sugerida:** Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.

## video_doctor_ui.py

**Arquivo:** [video_doctor_ui.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/video_doctor_ui.py>) · 181 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** Aba Gradio do video_doctor, para embutir nas UIs de geração. Por que existe como módulo separado: as UIs de 2.3 e 2.5 são arquivos grandes e independentes (music_maker v2/v3, web_ui, film_maker, storyplay). Duplicar a interface de diagnósti

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L75, L101; captura ampla de exceções: L74, L100.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## wav2lip_scene_smoke_test.py

**Arquivo:** [wav2lip_scene_smoke_test.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/wav2lip_scene_smoke_test.py>) · 10 linhas · **Cobertura:** Triagem estática/textual.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Melhoria sugerida:** Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.

## web_ui_v2.py

**Arquivo:** [web_ui_v2.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/web_ui_v2.py>) · 321 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `get_preset_frames`, `run_generation`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L137; gestão de processos: L142, L144, L145; captura ampla de exceções: L164.

**Melhoria sugerida:** Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.

## web_ui_v4.py

**Arquivo:** [web_ui_v4.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/web_ui_v4.py>) · 708 linhas · **Cobertura:** Triagem estática/textual.

**Pontos de entrada/rotinas identificados:** `build_ltx_prompt_text`, `get_preset_frames`, `estimate_vram_text`, `normalize_ltx_frames`, `extract_last_frame`.

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** E722 L403. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L287, L347, L363; gestão de processos: L185, L193, L204, L288, L289, L396 (+1); captura ampla de exceções: L386, L399, L403; subprocesso bloqueante sem timeout explícito: L185, L193, L204, L396.

**Melhoria sugerida:** Dividir `process_job_logic` (L210, 180 linhas) por responsabilidades e testar seus contratos de entrada/saída.

## web_ui_v4_25.py

**Arquivo:** [web_ui_v4_25.py](<E:/Users/home/Documents/LTX-2-OPTIMIZED/web_ui_v4_25.py>) · 738 linhas · **Cobertura:** Triagem estática/textual.

**Finalidade declarada no código:** LTX-2.5 variant of web_ui_v4.py -- GENERATED by _make_25_uis.py, do not hand-edit without also updating that script (or it will be overwritten on regeneration). Differences from the 2.3 original: - Generation goes through `ltx_pipelines_25`

**Verificação:** OK (Python 3.12).

**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.

**Alertas Ruff para triagem:** E722 L421. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.

**Sinais estáticos (não confirmam defeito):** caminhos absolutos/configuração local: L305, L365, L381; gestão de processos: L202, L210, L221, L306, L307, L414 (+1); captura ampla de exceções: L404, L417, L421; subprocesso bloqueante sem timeout explícito: L202, L210, L221, L414.

**Melhoria sugerida:** Dividir `process_job_logic` (L227, 181 linhas) por responsabilidades e testar seus contratos de entrada/saída.
