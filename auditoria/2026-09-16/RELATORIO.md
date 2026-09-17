# Auditoria de scripts — 16/09/2026

Foram catalogados **348 scripts (74.585 linhas)**: 292 Python, 37 BAT, 5 PowerShell e 14 DAZ Script. A revisão encontrou **20 achados priorizados**, agrupando ocorrências da mesma causa. Nenhum foi classificado como P0. O risco predominante é produzir/reutilizar artefatos incorretos ou declarar sucesso com saídas incompletas.

## Escopo e limites

Todos os 348 arquivos receberam triagem estática/textual e uma ficha individual. **36 scripts** receberam também revisão dirigida de trechos e contratos; isso não significa leitura manual integral dos 348 scripts. As fichas distinguem os níveis de cobertura. “Sem defeito confirmado” não é certificação funcional.

Inventário: união dos arquivos versionados (`git ls-files`) com os visíveis a `rg --files`. Inclui o arquivo novo não versionado `script_pipeline/import_reference.py` e scripts históricos versionados. Exclui código ignorado de terceiros, ambientes, modelos e backups não versionados; por exemplo ComfyUI, tensorxx_ge e tools. Imports desses componentes exercitados pelos testes não equivalem a auditoria deles. Os scripts desta própria auditoria estão fora da contagem.

O estado analisado é o da árvore de trabalho, incluindo nove arquivos já modificados e um novo arquivo. Não foram alterados scripts operacionais. Não houve inferência, treino, downloads, reparos ou encerramento de servidores. Desempenho GPU, qualidade audiovisual, APIs externas e integrações DAZ não foram validados de ponta a ponta.

## Evidências executadas

- 292 arquivos Python: parsing AST e compilação em memória com Python 3.12, sem importá-los; nenhum erro de sintaxe. A compatibilidade específica com Python 3.11 não foi executada.
- 5 arquivos PowerShell: parser nativo, sem executar; nenhum erro de sintaxe. BAT e DSA: inspeção textual, sem parser/runtime dedicado.
- Suíte existente: **60 passed, 1 deselected, 4 warnings**, 31,82s. O teste de offload CUDA foi excluído para não ocupar GPU. Não foi executado pytest na raiz porque há scripts de geração pesada chamados test/smoke fora de tests/.
- Reproduções isoladas: **9 casos confirmados**, incluindo o mesmo defeito em cinco interfaces. Nesses testes, PASS significa que o defeito atual foi reproduzido, não que o produto está correto. Dependências GPU/rede são simuladas.
- Ruff 0.3.4 com configuração isolada: 40 alertas na seleção principal; segunda seleção para ampliar Pyflakes/Bugbear. A configuração do projeto solicita Ruff >=0.14.3; este resultado não representa lint completo com a versão esperada.

Comandos e resultados: [pytest.txt](pytest.txt), [reproduzir.py](reproduzir.py), [reproducoes.txt](reproducoes.txt), [coletar.py](coletar.py), [inventario.json](inventario.json), [powershell.json](powershell.json), [ruff.json](ruff.json), [ruff_adicional.json](ruff_adicional.json). Os JSONs são evidências locais e podem ser ignorados pela regra global `*.json` do Git.

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q --disable-warnings -k "not attention_norms_materialize_from_accelerate_offload" --tb=short
.venv/Scripts/python.exe auditoria/2026-09-16/reproduzir.py
python auditoria/2026-09-16/coletar.py
```

## Índice dos achados

P1 = corrigir antes de confiar em produção/reuso; P2 = falha condicionada ou robustez; P3 = manutenção.

| ID | Prioridade | Achado |
|---|---|---|
| [A01](#a01) | P1 | Fallback de separação vocal aborta por variável local não inicializada |
| [A02](#a02) | P1 | Cache de áudio e imagem identifica conteúdo apenas pelo tamanho |
| [A03](#a03) | P1 | Cache de still não invalida referências e parâmetros de amostragem |
| [A04](#a04) | P1 | Cache de vídeo omite configuração e usa grade LTX para todos os motores |
| [A05](#a05) | P1 | Encadeamento MiniMax descarta o último frame quando já há duas referências |
| [A06](#a06) | P1 | Encadeamento LTX reduz duração e condiciona áudio apenas no primeiro trecho |
| [A07](#a07) | P1 | Falha da concatenação pode ser registrada como clipe bem-sucedido |
| [A08](#a08) | P1 | Plano sem still desaparece da contagem de falhas |
| [A09](#a09) | P1 | Lista de concatenação não suporta caminhos Unicode nem apóstrofos |
| [A10](#a10) | P1 | Pós-produção pode ressuscitar manifesto e vídeos antigos |
| [A11](#a11) | P2 | Importação sem rosto falha em pasta nova |
| [A12](#a12) | P1 | Scripts de upscale anunciam sucesso sem verificar executáveis |
| [A13](#a13) | P2 | Upcale compartilha temporários destrutivos entre execuções |
| [A14](#a14) | P1 | Reparo encerra todos os processos Python e altera ambiente sem controle |
| [A15](#a15) | P1 | Verificador não detecta clipes ausentes do filme planejado |
| [A16](#a16) | P3 | Método audio_encoder duplicado na mesma classe |
| [A17](#a17) | P1 | Modo combinado MiniMax desliga o servidor necessário aos stills |
| [A18](#a18) | P2 | Launcher Character3D depende do diretório de quem chama |
| [A19](#a19) | P2 | Cache TTS não acompanha alteração da referência no mesmo caminho |
| [A20](#a20) | P2 | Normalização de concat sobrescreve entradas com mesmo basename |

## Achados detalhados

<a id="a01"></a>
### A01 · P1 · Fallback de separação vocal aborta por variável local não inicializada

**Arquivos:** [music_maker_ui_v2.py:589](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v2.py:589>), [music_maker_ui_v2_25.py:623](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v2_25.py:623>), [music_maker_ui_v3.py:766](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v3.py:766>), [music_maker_ui_v3_25.py:782](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_v3_25.py:782>), [music_maker_ui_gguf.py:592](<E:/Users/home/Documents/LTX-2-OPTIMIZED/music_maker_ui_gguf.py:592>).

**Evidência:** `slice_audio()` usa `CURRENT_LOG +=` no except, mas não declara CURRENT_LOG como global. Se Demucs falha, o handler lança UnboundLocalError e o ASR alternativo não é chamado.

**Impacto:** Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original.

**Correção sugerida:** Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.

**Validação:** Reproduzido nas cinco interfaces: test_01; o ASR simulado recebeu zero chamadas.

<a id="a02"></a>
### A02 · P1 · Cache de áudio e imagem identifica conteúdo apenas pelo tamanho

**Arquivos:** [script_pipeline/render_shots.py:108](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:108>).

**Evidência:** `_audio_key()` devolve somente st_size e também é usado para o still. Arquivos diferentes com o mesmo tamanho geram a mesma chave; WAVs PCM de igual duração frequentemente têm tamanhos iguais.

**Impacto:** Vídeo antigo pode ser reutilizado após mudar voz, fala ou imagem, causando identidade ou sincronia incorretas.

**Correção sugerida:** Usar hash de conteúdo, com memoização por metadados se necessário. Não tratar ausência e erro de leitura como a mesma identidade.

**Validação:** test_08 substitui AAAA por BBBB no mesmo caminho e comprova chave idêntica.

<a id="a03"></a>
### A03 · P1 · Cache de still não invalida referências e parâmetros de amostragem

**Arquivos:** [script_pipeline/render_shots.py:76](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:76>).

**Evidência:** `_still_key()` usa só o basename da referência; a segunda referência também usa apenas nome (linha 233). Seed, steps, CFG, guidance, CLIP, VAE, weight_dtype e limiar de consistência não entram na chave.

**Impacto:** Trocar a sheet, importar outra foto ou ajustar qualidade pode devolver exatamente o still anterior. A justificativa no docstring não propaga a chave da referência ao dependente.

**Correção sugerida:** Serializar a configuração efetiva e os hashes de ambas as referências; separar geração de reavaliação de consistência.

**Validação:** test_09 comprova colisão entre cast_a/ref.png e cast_b/ref.png; omissões conferidas no código.

<a id="a04"></a>
### A04 · P1 · Cache de vídeo omite configuração e usa grade LTX para todos os motores

**Arquivos:** [script_pipeline/render_shots.py:532](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:532>).

**Evidência:** A chave não inclui seed, fps, resolução de vídeo, variante/checkpoint efetivo, turbo, megapixels, aspect ratio, minimax_ref_audio ou configuração Realism/SageAttention. A reutilização exige `_clip_frames(...) == shot["frames"]`, mesmo para MiniMax (5+17k), LongCat (4k+1 a 25fps), chain e freeze.

**Impacto:** Certas mudanças são ignoradas; em outros casos clipes válidos são regenerados indefinidamente por divergência de grade ou pós-efeito.

**Correção sugerida:** Chave versionada com configuração resolvida; registrar frames/fps realmente produzidos e validar por contrato específico do motor, incluindo pós-processamento.

**Validação:** Revisão do produtor da chave, condição de cache e chamadas dos três backends; não medido com modelos.

<a id="a05"></a>
### A05 · P1 · Encadeamento MiniMax descarta o último frame quando já há duas referências

**Arquivos:** [script_pipeline/render_shots.py:642](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:642>).

**Evidência:** `(refs_j + [frame_anterior])[:2]` mantém still e sheet e descarta o terceiro elemento, que é justamente o frame de continuidade.

**Impacto:** O modo comum com still + sheet deixa de encadear movimento e cenário, embora o log anuncie encadeamento.

**Correção sugerida:** Escolher explicitamente sheet + último frame nos segmentos posteriores; definir a prioridade dos dois slots.

**Validação:** test_04: três chamadas; a segunda mantém [still.png, sheet.png] sem o frame anterior.

<a id="a06"></a>
### A06 · P1 · Encadeamento LTX reduz duração e condiciona áudio apenas no primeiro trecho

**Arquivos:** [script_pipeline/render_shots.py:671](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:671>).

**Evidência:** Todos os segmentos usam o mesmo número de frames arredondado para baixo; o resto é descartado. `audio_conditioning=wav_cond if j == 0 else None` envia o WAV inteiro só ao primeiro segmento.

**Impacto:** Um plano de 401 frames vira 387 frames (14 a menos, ~0,58s a 24fps). A continuação da fala deixa de condicionar os segmentos seguintes.

**Correção sugerida:** Planejar os segmentos em tempo, cortar o WAV por segmento, distribuir o restante e aparar/padronizar o resultado para a duração contratada. Adequar também a guia IC por segmento.

**Validação:** test_05 confirma 387 frames e condicionamento [voice.wav, None, None].

<a id="a07"></a>
### A07 · P1 · Falha da concatenação pode ser registrada como clipe bem-sucedido

**Arquivos:** [script_pipeline/render_shots.py:659](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:659>), [script_pipeline/render_shots_stage.py:77](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots_stage.py:77>).

**Evidência:** Os dois helpers chain ignoram o booleano de concat_videos (linhas 659 e 699). O chamador grava a chave e inclui o caminho como sucesso. build_clips_manifest usa bool(caminho), sem verificar o arquivo.

**Impacto:** Manifesto pode marcar ok=true para vídeo inexistente ou para uma saída antiga após concat falhar.

**Correção sugerida:** Levantar erro quando concat retornar False; gerar em temporário e promover somente após validar arquivo, streams e duração; validar o arquivo ao montar manifesto.

**Validação:** test_06 força concat=False sem exceção; test_07 comprova ok=true com nonexistent.mp4.

<a id="a08"></a>
### A08 · P1 · Plano sem still desaparece da contagem de falhas

**Arquivos:** [script_pipeline/render_shots.py:491](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:491>), [script_pipeline/render_shots_stage.py:316](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots_stage.py:316>).

**Evidência:** Still ausente/falhado usa continue antes de inserir uma entrada em feitos. O estágio considera sucesso quando ok == len(manifesto), inclusive 0 == 0.

**Impacto:** Um conjunto parcial ou vazio pode ser marcado como render concluído, permitindo montagem incompleta.

**Correção sugerida:** Registrar uma entrada de falha por plano solicitado e comparar com o conjunto esperado. No modo stills-only também retornar falha quando faltar saída solicitada.

**Validação:** Inspeção do produtor de feitos e da condição final; sucesso vazio demonstrável diretamente pela condição.

<a id="a09"></a>
### A09 · P1 · Lista de concatenação não suporta caminhos Unicode nem apóstrofos

**Arquivos:** [script_pipeline/assemble_final.py:131](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/assemble_final.py:131>).

**Evidência:** concat_list.txt é gravado em ASCII; cada caminho é envolvido por apóstrofos sem escape.

**Impacto:** Pastas/arquivos com ação, João etc. interrompem a montagem; apóstrofos quebram a sintaxe da lista ffconcat.

**Correção sugerida:** Gravar UTF-8 sem BOM e escapar nomes conforme ffconcat, ou criar nomes intermediários controlados e únicos.

**Validação:** test_02 reproduz UnicodeEncodeError antes de executar ffmpeg; ramo de apóstrofo revisado estaticamente.

<a id="a10"></a>
### A10 · P1 · Pós-produção pode ressuscitar manifesto e vídeos antigos

**Arquivos:** [script_pipeline/postprod_v2v.py:83](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/postprod_v2v.py:83>).

**Evidência:** mixed_clips_pre_post.json é criado só uma vez e sempre usado como origem. A chave do efeito contém apenas basename do vídeo, força e guia, sem conteúdo, prompt ou modelo.

**Impacto:** Após remontar a mixagem ou alterar os clipes da corrida, a pós-produção pode recuperar a seleção anterior e reaproveitar efeitos obsoletos.

**Correção sugerida:** Guardar assinatura do manifesto de entrada e conteúdo; atualizar snapshot quando a origem muda, mantendo explicitamente separado o manifesto pré e pós.

**Validação:** Revisão das linhas 83–110; falta teste de duas execuções após mudar a mixagem.

<a id="a11"></a>
### A11 · P2 · Importação sem rosto falha em pasta nova

**Arquivos:** [script_pipeline/import_reference.py:80](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/import_reference.py:80>).

**Evidência:** O ramo bbox is None salva antes de criar out_path.parent; mkdir só existe no ramo com rosto. O CLI também não cria refs_dir antes da chamada.

**Impacto:** Na primeira importação de uma foto sem rosto detectável, o comportamento prometido de salvar com aviso termina em FileNotFoundError.

**Correção sugerida:** Criar a pasta antes da bifurcação; testar referência com e sem rosto.

**Validação:** test_03 reproduz com PNG temporário e detector simulado sem rosto.

<a id="a12"></a>
### A12 · P1 · Scripts de upscale anunciam sucesso sem verificar executáveis

**Arquivos:** [upscale_video.ps1:24](<E:/Users/home/Documents/LTX-2-OPTIMIZED/upscale_video.ps1:24>), [generate_upscale.ps1:39](<E:/Users/home/Documents/LTX-2-OPTIMIZED/generate_upscale.ps1:39>), [upscale_film.ps1:25](<E:/Users/home/Documents/LTX-2-OPTIMIZED/upscale_film.ps1:25>), [install_character3d.ps1:7](<E:/Users/home/Documents/LTX-2-OPTIMIZED/install_character3d.ps1:7>).

**Evidência:** As chamadas nativas de ffmpeg/Real-ESRGAN, o PowerShell filho e downloads não têm verificação consistente de LASTEXITCODE. ErrorActionPreference=Stop não garante tratamento de exit code nativo no Windows PowerShell invocado.

**Impacto:** Falhas podem chegar até mensagens PRONTO/Instalação concluída; os wrappers de upscale ainda apagam intermediários após falha.

**Correção sugerida:** Verificar LASTEXITCODE imediatamente e validar saída antes de continuar; preservar intermediários em falha. Aplicar o mesmo contrato ao download.

**Validação:** Inspeção PowerShell; parser passou, executáveis e downloads não foram disparados.

<a id="a13"></a>
### A13 · P2 · Upcale compartilha temporários destrutivos entre execuções

**Arquivos:** [upscale_video.ps1:18](<E:/Users/home/Documents/LTX-2-OPTIMIZED/upscale_video.ps1:18>), [generate_upscale.ps1:15](<E:/Users/home/Documents/LTX-2-OPTIMIZED/generate_upscale.ps1:15>), [upscale_film.ps1:16](<E:/Users/home/Documents/LTX-2-OPTIMIZED/upscale_film.ps1:16>).

**Evidência:** tools/_upscale_tmp, _gen_lowres.mp4 e tools/_film_up são fixos. A pasta de trabalho é removida recursivamente no início/fim.

**Impacto:** Duas execuções simultâneas podem apagar quadros ou substituir vídeos uma da outra.

**Correção sugerida:** Criar diretório único por execução; validar confinamento antes da limpeza e remover só os arquivos da própria execução.

**Validação:** Risco de concorrência diretamente identificado; não provocado para preservar trabalhos existentes.

<a id="a14"></a>
### A14 · P1 · Reparo encerra todos os processos Python e altera ambiente sem controle

**Arquivos:** [repair_comfyui.bat:3](<E:/Users/home/Documents/LTX-2-OPTIMIZED/repair_comfyui.bat:3>).

**Evidência:** `taskkill /F /IM python.exe` encerra todos os Python do usuário. Depois o script atualiza dependências e trio torch pelo índice cu124, sem pins nem verificação de errorlevel.

**Impacto:** Pode interromper outras gerações/servidores e substituir a combinação local de bibliotecas documentada pelo projeto; a mensagem final não garante reparo.

**Correção sugerida:** Restringir encerramento a PID/linha de comando do ComfyUI; registrar versões, testar reparo em ambiente separado e interromper após qualquer falha.

**Validação:** Inspeção textual apenas; o reparo não foi executado.

<a id="a15"></a>
### A15 · P1 · Verificador não detecta clipes ausentes do filme planejado

**Arquivos:** [script_pipeline/verify_output.py:173](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/verify_output.py:173>).

**Evidência:** Clipes sem caminho ou arquivo são pulados. total_clip_seconds soma apenas os existentes; SHORT_FILM compara o filme com essa soma reduzida, sem reconciliar o plano solicitado.

**Impacto:** Um filme truncado por planos que já faltavam no manifesto/arquivo pode passar na checagem de duração.

**Correção sugerida:** Emitir MISSING_CLIP e reconciliar IDs do shot_plan, manifesto e arquivos, respeitando seleção parcial declarada. Usar duração planejada como controle independente.

**Validação:** Revisão do loop de clipes e teste SHORT_FILM; não renderizado filme completo.

<a id="a16"></a>
### A16 · P3 · Método audio_encoder duplicado na mesma classe

**Arquivos:** [packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py:318](<E:/Users/home/Documents/LTX-2-OPTIMIZED/packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py:318>).

**Evidência:** audio_encoder é definido nas linhas 298 e 318 com corpo idêntico; a segunda definição sobrescreve a primeira.

**Impacto:** Sem diferença funcional atual, mas uma correção aplicada só na primeira cópia não terá efeito.

**Correção sugerida:** Manter uma implementação única; adicionar F811 ao lint de regressão.

**Validação:** Confirmado por AST/Ruff e leitura das duas definições.

<a id="a17"></a>
### A17 · P1 · Modo combinado MiniMax desliga o servidor necessário aos stills

**Arquivos:** [script_pipeline/render_shots.py:389](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/render_shots.py:389>).

**Evidência:** Sem stills_only/videos_only, render garante o ComfyUI 8188 e logo o encerra para MiniMax, antes do loop que chama _still_for_shot. generate_scene_storyboard submete ao 8188 sem reiniciá-lo.

**Impacto:** A invocação combinada sem stills em cache tenta gerar imagens num servidor desligado. O orquestrador em duas passadas evita essa rota, mas a função/CLI avulsa a expõe.

**Correção sugerida:** Gerar todos os stills antes da troca de servidor, ou exigir explicitamente duas fases para MiniMax.

**Validação:** Revisão cruzada de render_shots e generate_storyboards; sem parada real de serviços.

<a id="a18"></a>
### A18 · P2 · Launcher Character3D depende do diretório de quem chama

**Arquivos:** [start_character3d.bat:8](<E:/Users/home/Documents/LTX-2-OPTIMIZED/start_character3d.bat:8>).

**Evidência:** Executa python character3d_webui.py sem cd /d "%~dp0" e sem caminho absoluto do script.

**Impacto:** Quando chamado por terminal/atalho com outro diretório de trabalho, Python não encontra o arquivo, embora o launcher ainda aguarde a porta.

**Correção sugerida:** Resolver o diretório pelo próprio .bat e explicitar o interpretador apropriado para a integração.

**Validação:** Inspeção textual; a interface não foi iniciada.

<a id="a19"></a>
### A19 · P2 · Cache TTS não acompanha alteração da referência no mesmo caminho

**Arquivos:** [script_pipeline/synthesize_dialogue.py:154](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/synthesize_dialogue.py:154>).

**Evidência:** _chave inclui xtts_speaker_wav como string, sem assinatura do arquivo; engine=auto também representa o pedido, não a resolução do motor.

**Impacto:** Substituir a amostra de voz mantendo o nome pode reutilizar a voz antiga. Mudanças no motor resolvido exigem política explícita de invalidação.

**Correção sugerida:** Incluir hash das referências e versão/configuração do motor; persistir a resolução de auto para diagnóstico.

**Validação:** Revisão da chave e dos testes existentes, que cobrem emoção mas não sobrescrita do WAV de referência.

<a id="a20"></a>
### A20 · P2 · Normalização de concat sobrescreve entradas com mesmo basename

**Arquivos:** [script_pipeline/assemble_final.py:79](<E:/Users/home/Documents/LTX-2-OPTIMIZED/script_pipeline/assemble_final.py:79>).

**Evidência:** O destino da normalização usa apenas Path(video_path).stem + _norm.mp4 dentro de uma pasta comum.

**Impacto:** Ao concatenar a/clip.mp4 e b/clip.mp4, a segunda normalização substitui a primeira; a lista final aponta duas vezes para o mesmo arquivo.

**Correção sugerida:** Usar índice estável ou hash do caminho completo/conteúdo no nome intermediário.

**Validação:** Colisão determinística na construção do caminho; rota principal costuma usar IDs únicos, reduzindo a incidência.

## Alertas que não foram promovidos a bugs

- F821 em music_to_video.py/music_to_video_v2.py: as closures usam transformer enquanto o objeto ainda está atribuído; del ocorre após o denoising síncrono. Alerta da versão antiga do analisador, não um NameError confirmado.
- B023 nas closures de render_shots/postprod_v2v e callbacks GGUF: uso no ciclo corrente, sem demonstração de execução tardia. Os bugs de chain descritos acima são diferentes desse alerta.
- B008 em gr.Progress: padrão de injeção do Gradio; não classificado automaticamente como defeito. range e funções auxiliares de experimento também exigem contexto.
- E722/exceções amplas, caminhos fixos e subprocessos sem timeout são sinais para robustez, não prova isolada de falha. As fichas trazem as linhas para inspeção.

## Melhorias de arquitetura e consistência

1. Centralizar assinaturas de cache com versão, conteúdo das entradas e configuração efetiva. Os achados A02/A03/A04/A10/A19 são variações do mesmo problema.
2. Definir um contrato de saída por plano: IDs esperados, caminho validado, hash, motor/variante, frames, fps, áudio e status explícito. Reconciliar esse contrato até verify_output.
3. Extrair o núcleo comum das cinco UIs music_maker. O mesmo erro replicado e funções muito extensas tornam as correções divergentes.
4. Centralizar configuração local (raízes, FFmpeg, Python, portas, GPUs) sem presumir que o índice CUDA coincide com nvidia-smi. Manter overrides explícitos por integração.
5. Padronizar escrita atômica de manifestos e travas por corrida; run_folder grava JSON diretamente e as UIs usam estado global. A concorrência não foi exercitada nesta auditoria.
6. Em run_decupagem, a verificação final é informativa (`obrigatorio=False`, sem `--strict`). Oferecer modo de validação obrigatória que propague falhas de qualidade, mantendo rascunhos permissivos quando solicitado.
7. Alinhar opções entre backend e CLI: minimax_h3_backend contém int8convrot, mas run_decupagem limita variantes a fp8int8/w4a8/gguf-q4km. Explicitar se a exclusão experimental é deliberada.
8. Acrescentar regressões dos casos reproduzidos à suíte normal após as correções, invertendo as expectativas para o comportamento correto; acrescentar cenários de cache, seleção parcial, falha de ffmpeg e retomada.

## Sequência sugerida de correção

Primeiro: A07/A08/A15 (não declarar sucesso incompleto) e A14 (reparo destrutivo). Depois: A02/A03/A04/A10/A19 (cache), A05/A06/A17 (encadeamento e ciclo de servidores), A01/A09/A11/A12 (falhas reproduzíveis de entrada/saída). Por fim, isolamento de temporários, launchers e manutenção.

## Relatório individual

[Abrir as 348 fichas por script](POR_SCRIPT.md). Cada ficha informa cobertura, sintaxe, achados associados, sinais com linhas e melhoria sugerida. [Inventário legível](INVENTARIO.tsv) inclui SHA-256 para fixar o estado auditado.
