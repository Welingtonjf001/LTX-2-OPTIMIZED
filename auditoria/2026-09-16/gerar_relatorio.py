"""Gera a entrega a partir do inventário e da triagem humana desta auditoria."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
inventory = json.loads((OUT/'inventario.json').read_text(encoding='utf-8'))
findings = []


def add(id, priority, title, refs, evidence, impact, fix, validation):
    findings.append(dict(id=id, priority=priority, title=title, refs=refs, evidence=evidence, impact=impact, fix=fix, validation=validation))


add('A01', 'P1', 'Fallback de separação vocal aborta por variável local não inicializada',
    [('music_maker_ui_v2.py',589),('music_maker_ui_v2_25.py',623),('music_maker_ui_v3.py',766),('music_maker_ui_v3_25.py',782),('music_maker_ui_gguf.py',592)],
    '`slice_audio()` usa `CURRENT_LOG +=` no except, mas não declara CURRENT_LOG como global. Se Demucs falha, o handler lança UnboundLocalError e o ASR alternativo não é chamado.',
    'Uma falha recuperável na separação vocal interrompe a segmentação inteira e esconde a causa original.',
    'Adicionar a declaração global ou usar uma função de logging comum; extrair a lógica compartilhada das cinco cópias.',
    'Reproduzido nas cinco interfaces: test_01; o ASR simulado recebeu zero chamadas.')
add('A02','P1','Cache de áudio e imagem identifica conteúdo apenas pelo tamanho', [('script_pipeline/render_shots.py',108)],
    '`_audio_key()` devolve somente st_size e também é usado para o still. Arquivos diferentes com o mesmo tamanho geram a mesma chave; WAVs PCM de igual duração frequentemente têm tamanhos iguais.',
    'Vídeo antigo pode ser reutilizado após mudar voz, fala ou imagem, causando identidade ou sincronia incorretas.',
    'Usar hash de conteúdo, com memoização por metadados se necessário. Não tratar ausência e erro de leitura como a mesma identidade.',
    'test_08 substitui AAAA por BBBB no mesmo caminho e comprova chave idêntica.')
add('A03','P1','Cache de still não invalida referências e parâmetros de amostragem', [('script_pipeline/render_shots.py',76)],
    '`_still_key()` usa só o basename da referência; a segunda referência também usa apenas nome (linha 233). Seed, steps, CFG, guidance, CLIP, VAE, weight_dtype e limiar de consistência não entram na chave.',
    'Trocar a sheet, importar outra foto ou ajustar qualidade pode devolver exatamente o still anterior. A justificativa no docstring não propaga a chave da referência ao dependente.',
    'Serializar a configuração efetiva e os hashes de ambas as referências; separar geração de reavaliação de consistência.',
    'test_09 comprova colisão entre cast_a/ref.png e cast_b/ref.png; omissões conferidas no código.')
add('A04','P1','Cache de vídeo omite configuração e usa grade LTX para todos os motores', [('script_pipeline/render_shots.py',532)],
    'A chave não inclui seed, fps, resolução de vídeo, variante/checkpoint efetivo, turbo, megapixels, aspect ratio, minimax_ref_audio ou configuração Realism/SageAttention. A reutilização exige `_clip_frames(...) == shot["frames"]`, mesmo para MiniMax (5+17k), LongCat (4k+1 a 25fps), chain e freeze.',
    'Certas mudanças são ignoradas; em outros casos clipes válidos são regenerados indefinidamente por divergência de grade ou pós-efeito.',
    'Chave versionada com configuração resolvida; registrar frames/fps realmente produzidos e validar por contrato específico do motor, incluindo pós-processamento.',
    'Revisão do produtor da chave, condição de cache e chamadas dos três backends; não medido com modelos.')
add('A05','P1','Encadeamento MiniMax descarta o último frame quando já há duas referências', [('script_pipeline/render_shots.py',642)],
    '`(refs_j + [frame_anterior])[:2]` mantém still e sheet e descarta o terceiro elemento, que é justamente o frame de continuidade.',
    'O modo comum com still + sheet deixa de encadear movimento e cenário, embora o log anuncie encadeamento.',
    'Escolher explicitamente sheet + último frame nos segmentos posteriores; definir a prioridade dos dois slots.',
    'test_04: três chamadas; a segunda mantém [still.png, sheet.png] sem o frame anterior.')
add('A06','P1','Encadeamento LTX reduz duração e condiciona áudio apenas no primeiro trecho', [('script_pipeline/render_shots.py',671)],
    'Todos os segmentos usam o mesmo número de frames arredondado para baixo; o resto é descartado. `audio_conditioning=wav_cond if j == 0 else None` envia o WAV inteiro só ao primeiro segmento.',
    'Um plano de 401 frames vira 387 frames (14 a menos, ~0,58s a 24fps). A continuação da fala deixa de condicionar os segmentos seguintes.',
    'Planejar os segmentos em tempo, cortar o WAV por segmento, distribuir o restante e aparar/padronizar o resultado para a duração contratada. Adequar também a guia IC por segmento.',
    'test_05 confirma 387 frames e condicionamento [voice.wav, None, None].')
add('A07','P1','Falha da concatenação pode ser registrada como clipe bem-sucedido', [('script_pipeline/render_shots.py',660),('script_pipeline/render_shots_stage.py',83)],
    'Os dois helpers chain ignoram o booleano de concat_videos (linhas 660 e 701). O chamador grava a chave e inclui o caminho como sucesso. build_clips_manifest usa bool(caminho), sem verificar o arquivo.',
    'Manifesto pode marcar ok=true para vídeo inexistente ou para uma saída antiga após concat falhar.',
    'Levantar erro quando concat retornar False; gerar em temporário e promover somente após validar arquivo, streams e duração; validar o arquivo ao montar manifesto.',
    'test_06 força concat=False sem exceção; test_07 comprova ok=true com nonexistent.mp4.')
add('A08','P1','Plano sem still desaparece da contagem de falhas', [('script_pipeline/render_shots.py',490),('script_pipeline/render_shots_stage.py',321)],
    'Still ausente/falhado usa continue antes de inserir uma entrada em feitos. O estágio considera sucesso quando ok == len(manifesto), inclusive 0 == 0.',
    'Um conjunto parcial ou vazio pode ser marcado como render concluído, permitindo montagem incompleta.',
    'Registrar uma entrada de falha por plano solicitado e comparar com o conjunto esperado. No modo stills-only também retornar falha quando faltar saída solicitada.',
    'Inspeção do produtor de feitos e da condição final; sucesso vazio demonstrável diretamente pela condição.')
add('A09','P1','Lista de concatenação não suporta caminhos Unicode nem apóstrofos', [('script_pipeline/assemble_final.py',131)],
    'concat_list.txt é gravado em ASCII; cada caminho é envolvido por apóstrofos sem escape.',
    'Pastas/arquivos com ação, João etc. interrompem a montagem; apóstrofos quebram a sintaxe da lista ffconcat.',
    'Gravar UTF-8 sem BOM e escapar nomes conforme ffconcat, ou criar nomes intermediários controlados e únicos.',
    'test_02 reproduz UnicodeEncodeError antes de executar ffmpeg; ramo de apóstrofo revisado estaticamente.')
add('A10','P1','Pós-produção pode ressuscitar manifesto e vídeos antigos', [('script_pipeline/postprod_v2v.py',83)],
    'mixed_clips_pre_post.json é criado só uma vez e sempre usado como origem. A chave do efeito contém apenas basename do vídeo, força e guia, sem conteúdo, prompt ou modelo.',
    'Após remontar a mixagem ou alterar os clipes da corrida, a pós-produção pode recuperar a seleção anterior e reaproveitar efeitos obsoletos.',
    'Guardar assinatura do manifesto de entrada e conteúdo; atualizar snapshot quando a origem muda, mantendo explicitamente separado o manifesto pré e pós.',
    'Revisão das linhas 83–110; falta teste de duas execuções após mudar a mixagem.')
add('A11','P2','Importação sem rosto falha em pasta nova', [('script_pipeline/import_reference.py',88)],
    'O ramo bbox is None salva antes de criar out_path.parent; mkdir só existe no ramo com rosto. O CLI também não cria refs_dir antes da chamada.',
    'Na primeira importação de uma foto sem rosto detectável, o comportamento prometido de salvar com aviso termina em FileNotFoundError.',
    'Criar a pasta antes da bifurcação; testar referência com e sem rosto.',
    'test_03 reproduz com PNG temporário e detector simulado sem rosto.')
add('A12','P1','Scripts de upscale anunciam sucesso sem verificar executáveis', [('upscale_video.ps1',27),('generate_upscale.ps1',41),('upscale_film.ps1',31),('install_character3d.ps1',7)],
    'As chamadas nativas de ffmpeg/Real-ESRGAN, o PowerShell filho e downloads não têm verificação consistente de LASTEXITCODE. ErrorActionPreference=Stop não garante tratamento de exit code nativo no Windows PowerShell invocado.',
    'Falhas podem chegar até mensagens PRONTO/Instalação concluída; os wrappers de upscale ainda apagam intermediários após falha.',
    'Verificar LASTEXITCODE imediatamente e validar saída antes de continuar; preservar intermediários em falha. Aplicar o mesmo contrato ao download.',
    'Inspeção PowerShell; parser passou, executáveis e downloads não foram disparados.')
add('A13','P2','Upcale compartilha temporários destrutivos entre execuções', [('upscale_video.ps1',20),('generate_upscale.ps1',16),('upscale_film.ps1',21)],
    'tools/_upscale_tmp, _gen_lowres.mp4 e tools/_film_up são fixos. A pasta de trabalho é removida recursivamente no início/fim.',
    'Duas execuções simultâneas podem apagar quadros ou substituir vídeos uma da outra.',
    'Criar diretório único por execução; validar confinamento antes da limpeza e remover só os arquivos da própria execução.',
    'Risco de concorrência diretamente identificado; não provocado para preservar trabalhos existentes.')
add('A14','P1','Reparo encerra todos os processos Python e altera ambiente sem controle', [('repair_comfyui.bat',3)],
    '`taskkill /F /IM python.exe` encerra todos os Python do usuário. Depois o script atualiza dependências e trio torch pelo índice cu124, sem pins nem verificação de errorlevel.',
    'Pode interromper outras gerações/servidores e substituir a combinação local de bibliotecas documentada pelo projeto; a mensagem final não garante reparo.',
    'Restringir encerramento a PID/linha de comando do ComfyUI; registrar versões, testar reparo em ambiente separado e interromper após qualquer falha.',
    'Inspeção textual apenas; o reparo não foi executado.')
add('A15','P1','Verificador não detecta clipes ausentes do filme planejado', [('script_pipeline/verify_output.py',173)],
    'Clipes sem caminho ou arquivo são pulados. total_clip_seconds soma apenas os existentes; SHORT_FILM compara o filme com essa soma reduzida, sem reconciliar o plano solicitado.',
    'Um filme truncado por planos que já faltavam no manifesto/arquivo pode passar na checagem de duração.',
    'Emitir MISSING_CLIP e reconciliar IDs do shot_plan, manifesto e arquivos, respeitando seleção parcial declarada. Usar duração planejada como controle independente.',
    'Revisão do loop de clipes e teste SHORT_FILM; não renderizado filme completo.')
add('A16','P3','Método audio_encoder duplicado na mesma classe', [('packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py',318)],
    'audio_encoder é definido nas linhas 298 e 318 com corpo idêntico; a segunda definição sobrescreve a primeira.',
    'Sem diferença funcional atual, mas uma correção aplicada só na primeira cópia não terá efeito.',
    'Manter uma implementação única; adicionar F811 ao lint de regressão.',
    'Confirmado por AST/Ruff e leitura das duas definições.')
add('A17','P1','Modo combinado MiniMax desliga o servidor necessário aos stills', [('script_pipeline/render_shots.py',372)],
    'Sem stills_only/videos_only, render garante o ComfyUI 8188 e logo o encerra para MiniMax, antes do loop que chama _still_for_shot. generate_scene_storyboard submete ao 8188 sem reiniciá-lo.',
    'A invocação combinada sem stills em cache tenta gerar imagens num servidor desligado. O orquestrador em duas passadas evita essa rota, mas a função/CLI avulsa a expõe.',
    'Gerar todos os stills antes da troca de servidor, ou exigir explicitamente duas fases para MiniMax.',
    'Revisão cruzada de render_shots e generate_storyboards; sem parada real de serviços.')
add('A18','P2','Launcher Character3D depende do diretório de quem chama', [('start_character3d.bat',8)],
    'Executa python character3d_webui.py sem cd /d "%~dp0" e sem caminho absoluto do script.',
    'Quando chamado por terminal/atalho com outro diretório de trabalho, Python não encontra o arquivo, embora o launcher ainda aguarde a porta.',
    'Resolver o diretório pelo próprio .bat e explicitar o interpretador apropriado para a integração.',
    'Inspeção textual; a interface não foi iniciada.')
add('A19','P2','Cache TTS não acompanha alteração da referência no mesmo caminho', [('script_pipeline/synthesize_dialogue.py',158)],
    '_chave inclui xtts_speaker_wav como string, sem assinatura do arquivo; engine=auto também representa o pedido, não a resolução do motor.',
    'Substituir a amostra de voz mantendo o nome pode reutilizar a voz antiga. Mudanças no motor resolvido exigem política explícita de invalidação.',
    'Incluir hash das referências e versão/configuração do motor; persistir a resolução de auto para diagnóstico.',
    'Revisão da chave e dos testes existentes, que cobrem emoção mas não sobrescrita do WAV de referência.')
add('A20','P2','Normalização de concat sobrescreve entradas com mesmo basename', [('script_pipeline/assemble_final.py',79)],
    'O destino da normalização usa apenas Path(video_path).stem + _norm.mp4 dentro de uma pasta comum.',
    'Ao concatenar a/clip.mp4 e b/clip.mp4, a segunda normalização substitui a primeira; a lista final aponta duas vezes para o mesmo arquivo.',
    'Usar índice estável ou hash do caminho completo/conteúdo no nome intermediário.',
    'Colisão determinística na construção do caminho; rota principal costuma usar IDs únicos, reduzindo a incidência.')

# Linhas conferidas na revisão final, apontando a expressão relevante.
line_corrections = {
    'A07': [659, 77], 'A08': [491, 316], 'A11': [80],
    'A12': [24, 39, 25, 7], 'A13': [18, 15, 16],
    'A17': [389], 'A19': [154],
}
for finding in findings:
    if finding['id'] in line_corrections:
        finding['refs'] = [(p, line) for (p, _), line in zip(finding['refs'], line_corrections[finding['id']])]
    if finding['id'] == 'A07':
        finding['evidence'] = finding['evidence'].replace('linhas 660 e 701', 'linhas 659 e 699')

reviewed = {p for f in findings for p,_ in f['refs']} | {
    'minimax_h3_backend.py','script_pipeline/generate_storyboards.py','script_pipeline/run_decupagem.py',
    'script_pipeline/run_folder.py','script_pipeline/lipsync_scenes.py','script_pipeline/mix_audio.py',
    'packages/ltx-pipelines/src/ltx_pipelines/music_to_video.py',
    'packages/ltx-pipelines/src/ltx_pipelines/music_to_video_v2.py',
    'packages/ltx-pipelines/src/ltx_pipelines/utils/prompt_cache.py','daz_integration/daz_controller.py',
    'HermesCleanup.ps1','tests/test_auditoria_20260913.py','tests/test_pipeline_smoke.py',
    'tests/test_minimax_h3_test_ui.py','tests/test_optimization_paths.py','tests/test_tensorxx_ge.py',
    'tests/test_limpar_cache_stills.py'}


def link(path,line=None):
    suffix = f':{line}' if line else ''
    return f'[{path}{suffix}](<{ROOT.as_posix()}/{path}{suffix}>)'


main = ['# Auditoria de scripts — 16/09/2026', '',
    'Foram catalogados **348 scripts (74.585 linhas)**: 292 Python, 37 BAT, 5 PowerShell e 14 DAZ Script. '
    'A revisão encontrou **20 achados priorizados**, agrupando ocorrências da mesma causa. Nenhum foi classificado como P0. '
    'O risco predominante é produzir/reutilizar artefatos incorretos ou declarar sucesso com saídas incompletas.', '',
    '## Escopo e limites', '',
    f'Todos os 348 arquivos receberam triagem estática/textual e uma ficha individual. **{len(reviewed)} scripts** receberam também revisão dirigida de trechos e contratos; '
    'isso não significa leitura manual integral dos 348 scripts. As fichas distinguem os níveis de cobertura. '
    '“Sem defeito confirmado” não é certificação funcional.', '',
    'Inventário: união dos arquivos versionados (`git ls-files`) com os visíveis a `rg --files`. '
    'Inclui o arquivo novo não versionado `script_pipeline/import_reference.py` e scripts históricos versionados. '
    'Exclui código ignorado de terceiros, ambientes, modelos e backups não versionados; por exemplo ComfyUI, tensorxx_ge e tools. '
    'Imports desses componentes exercitados pelos testes não equivalem a auditoria deles. Os scripts desta própria auditoria estão fora da contagem.', '',
    'O estado analisado é o da árvore de trabalho, incluindo nove arquivos já modificados e um novo arquivo. '
    'Não foram alterados scripts operacionais. Não houve inferência, treino, downloads, reparos ou encerramento de servidores. '
    'Desempenho GPU, qualidade audiovisual, APIs externas e integrações DAZ não foram validados de ponta a ponta.', '',
    '## Evidências executadas', '',
    '- 292 arquivos Python: parsing AST e compilação em memória com Python 3.12, sem importá-los; nenhum erro de sintaxe. A compatibilidade específica com Python 3.11 não foi executada.',
    '- 5 arquivos PowerShell: parser nativo, sem executar; nenhum erro de sintaxe. BAT e DSA: inspeção textual, sem parser/runtime dedicado.',
    '- Suíte existente: **60 passed, 1 deselected, 4 warnings**, 31,82s. O teste de offload CUDA foi excluído para não ocupar GPU. Não foi executado pytest na raiz porque há scripts de geração pesada chamados test/smoke fora de tests/.',
    '- Reproduções isoladas: **9 casos confirmados**, incluindo o mesmo defeito em cinco interfaces. Nesses testes, PASS significa que o defeito atual foi reproduzido, não que o produto está correto. Dependências GPU/rede são simuladas.',
    '- Ruff 0.3.4 com configuração isolada: 40 alertas na seleção principal; segunda seleção para ampliar Pyflakes/Bugbear. A configuração do projeto solicita Ruff >=0.14.3; este resultado não representa lint completo com a versão esperada.', '',
    'Comandos e resultados: [pytest.txt](pytest.txt), [reproduzir.py](reproduzir.py), [reproducoes.txt](reproducoes.txt), '
    '[coletar.py](coletar.py), [inventario.json](inventario.json), [powershell.json](powershell.json), '
    '[ruff.json](ruff.json), [ruff_adicional.json](ruff_adicional.json). Os JSONs são evidências locais e podem ser ignorados pela regra global `*.json` do Git.', '',
    '```powershell',
    '.venv/Scripts/python.exe -m pytest tests/ -q --disable-warnings -k "not attention_norms_materialize_from_accelerate_offload" --tb=short',
    '.venv/Scripts/python.exe auditoria/2026-09-16/reproduzir.py',
    'python auditoria/2026-09-16/coletar.py',
    '```', '',
    '## Índice dos achados', '',
    'P1 = corrigir antes de confiar em produção/reuso; P2 = falha condicionada ou robustez; P3 = manutenção.', '',
    '| ID | Prioridade | Achado |', '|---|---|---|']
for f in findings:
    main.append(f"| [{f['id']}](#{f['id'].lower()}) | {f['priority']} | {f['title']} |")
main += ['', '## Achados detalhados', '']
for f in findings:
    main += [f"<a id=\"{f['id'].lower()}\"></a>", f"### {f['id']} · {f['priority']} · {f['title']}", '',
             '**Arquivos:** '+', '.join(link(p,l) for p,l in f['refs'])+'.', '',
             '**Evidência:** '+f['evidence'], '', '**Impacto:** '+f['impact'], '',
             '**Correção sugerida:** '+f['fix'], '', '**Validação:** '+f['validation'], '']
main += ['## Alertas que não foram promovidos a bugs', '',
    '- F821 em music_to_video.py/music_to_video_v2.py: as closures usam transformer enquanto o objeto ainda está atribuído; del ocorre após o denoising síncrono. Alerta da versão antiga do analisador, não um NameError confirmado.',
    '- B023 nas closures de render_shots/postprod_v2v e callbacks GGUF: uso no ciclo corrente, sem demonstração de execução tardia. Os bugs de chain descritos acima são diferentes desse alerta.',
    '- B008 em gr.Progress: padrão de injeção do Gradio; não classificado automaticamente como defeito. range e funções auxiliares de experimento também exigem contexto.',
    '- E722/exceções amplas, caminhos fixos e subprocessos sem timeout são sinais para robustez, não prova isolada de falha. As fichas trazem as linhas para inspeção.', '',
    '## Melhorias de arquitetura e consistência', '',
    '1. Centralizar assinaturas de cache com versão, conteúdo das entradas e configuração efetiva. Os achados A02/A03/A04/A10/A19 são variações do mesmo problema.',
    '2. Definir um contrato de saída por plano: IDs esperados, caminho validado, hash, motor/variante, frames, fps, áudio e status explícito. Reconciliar esse contrato até verify_output.',
    '3. Extrair o núcleo comum das cinco UIs music_maker. O mesmo erro replicado e funções muito extensas tornam as correções divergentes.',
    '4. Centralizar configuração local (raízes, FFmpeg, Python, portas, GPUs) sem presumir que o índice CUDA coincide com nvidia-smi. Manter overrides explícitos por integração.',
    '5. Padronizar escrita atômica de manifestos e travas por corrida; run_folder grava JSON diretamente e as UIs usam estado global. A concorrência não foi exercitada nesta auditoria.',
    '6. Em run_decupagem, a verificação final é informativa (`obrigatorio=False`, sem `--strict`). Oferecer modo de validação obrigatória que propague falhas de qualidade, mantendo rascunhos permissivos quando solicitado.',
    '7. Alinhar opções entre backend e CLI: minimax_h3_backend contém int8convrot, mas run_decupagem limita variantes a fp8int8/w4a8/gguf-q4km. Explicitar se a exclusão experimental é deliberada.',
    '8. Acrescentar regressões dos casos reproduzidos à suíte normal após as correções, invertendo as expectativas para o comportamento correto; acrescentar cenários de cache, seleção parcial, falha de ffmpeg e retomada.', '',
    '## Sequência sugerida de correção', '',
    'Primeiro: A07/A08/A15 (não declarar sucesso incompleto) e A14 (reparo destrutivo). '
    'Depois: A02/A03/A04/A10/A19 (cache), A05/A06/A17 (encadeamento e ciclo de servidores), '
    'A01/A09/A11/A12 (falhas reproduzíveis de entrada/saída). Por fim, isolamento de temporários, launchers e manutenção.', '',
    '## Relatório individual', '',
    '[Abrir as 348 fichas por script](POR_SCRIPT.md). Cada ficha informa cobertura, sintaxe, '
    'achados associados, sinais com linhas e melhoria sugerida. [Inventário legível](INVENTARIO.tsv) inclui SHA-256 para fixar o estado auditado.', '']
(OUT/'RELATORIO.md').write_text('\n'.join(main), encoding='utf-8')

lint = json.loads((OUT/'ruff.json').read_text(encoding='utf-8'))
ps = json.loads((OUT/'powershell.json').read_text(encoding='utf-8-sig'))
ps_by_name = {r['file']:r['errors'] for r in ps}
per = ['# Relatório por script', '',
       '[Relatório principal e critérios](RELATORIO.md). **348 fichas**, uma por arquivo do inventário. '
       'Triagem automatizada não substitui revisão funcional. Os sinais abaixo são oportunidades de investigação, não bugs confirmados.', '']
tsv = ['script\tlinhas\tcobertura\tachados\tsha256']
changed=[]
for row in inventory:
    p=row['path']
    related=[f for f in findings if any(ref==p for ref,_ in f['refs'])]
    coverage='Revisão dirigida + triagem' if p in reviewed else 'Triagem estática/textual'
    if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=row['sha256']:
        changed.append(p)
    tsv.append('\t'.join([p,str(row['lines']),coverage,','.join(f['id'] for f in related),row['sha256']]))
    per += [f'## {p}', '', f'**Arquivo:** {link(p)} · {row["lines"]} linhas · **Cobertura:** {coverage}.', '']
    if row['purpose']:
        purpose=row['purpose'].replace('\n',' ').replace('|','/')
        per += ['**Finalidade declarada no código:** '+purpose, '']
    elif row['functions']:
        per += ['**Pontos de entrada/rotinas identificados:** '+', '.join('`'+v['name']+'`' for v in row['functions'][:5])+'.', '']
    elif p.endswith('.bat'):
        per += ['**Papel:** launcher/automação de ambiente; interpretador, cwd e propagação de exit code são os principais contratos a validar.', '']
    elif p.endswith('.dsa'):
        per += ['**Papel:** script DAZ; verificar seleção de figura, presets e retorno da exportação no DAZ. Arquivos sob runs/test_run são artefatos de execuções, não a implementação geradora.', '']
    syntax=row['syntax']
    if p.endswith('.ps1'):
        syntax='Parser PowerShell: OK; nenhuma execução.' if not ps_by_name.get(Path(p).name) else str(ps_by_name[Path(p).name])
    per += ['**Verificação:** '+syntax.rstrip('.')+'.', '']
    if related:
        for f in related:
            per += [f"- **{f['priority']} [{f['id']}](RELATORIO.md#{f['id'].lower()}): {f['title']}. {f['impact']} Correção: {f['fix']}"]
        per += ['']
    else:
        per += ['**Resultado:** nenhum defeito confirmado nesta cobertura; funcionamento integrado não demonstrado para este script individualmente.', '']
    entries=[e for e in lint if Path(e['filename']).resolve()==(ROOT/p).resolve()]
    if entries:
        per += ['**Alertas Ruff para triagem:** '+', '.join(f"{e['code']} L{e['location']['row']}" for e in entries)+'. Ver ressalvas do relatório principal; não somar estes alertas aos achados confirmados.', '']
    labels={'caminhos_fixos':'caminhos absolutos/configuração local','shell_true':'execução via shell', 'rede_ampla':'bind/compartilhamento de rede', 'remocao':'remoção de arquivos','processos':'gestão de processos','excecao_ampla':'captura ampla de exceções'}
    signals=[]
    for name,lines in row['signals'].items():
        signals.append(labels[name]+': '+', '.join('L'+str(n) for n in lines[:6])+(f' (+{len(lines)-6})' if len(lines)>6 else ''))
    if row.get('subprocess_without_timeout'):
        signals.append('subprocesso bloqueante sem timeout explícito: '+', '.join('L'+str(n) for n in row['subprocess_without_timeout'][:6]))
    if signals:
        per += ['**Sinais estáticos (não confirmam defeito):** '+'; '.join(signals)+'.', '']
    largest=max(row['functions'], key=lambda f:f['size'], default=None)
    if related:
        suggestion='Priorizar as correções vinculadas acima e cobrir o gatilho com teste de regressão.'
    elif p.startswith('tests/'):
        suggestion='Manter testes determinísticos e estender contratos de falha/cache. A aprovação da suíte não cobre inferência real.'
    elif '_test' in p or Path(p).name.startswith('test_') or 'smoke' in p:
        suggestion='Separar testes que carregam modelos dos testes rápidos; explicitar recursos, saída e critério de sucesso antes da execução.'
    elif p.startswith('_arquivo/') or p.endswith('.prev.bat'):
        suggestion='Identificar como histórico e documentar o substituto ativo para evitar execução acidental de versão obsoleta.'
    elif p.startswith('symplectic_experiment/'):
        suggestion='Registrar seed, configuração e versão dos dados junto às métricas; comparar em conjunto de validação independente.'
    elif p.endswith('.bat') or p.endswith('.ps1'):
        suggestion='Resolver cwd e interpretador explicitamente, conferir dependências e propagar exit code de cada etapa.'
    elif p.endswith('.dsa'):
        suggestion='Validar no runtime DAZ com cena de teste e retornar status verificável; manter alterações na fonte geradora quando aplicável.'
    elif largest and largest['size']>140:
        suggestion=f"Dividir `{largest['name']}` (L{largest['line']}, {largest['size']} linhas) por responsabilidades e testar seus contratos de entrada/saída."
    elif row.get('subprocess_without_timeout'):
        suggestion='Definir timeout/cancelamento compatível com a operação e preservar stderr/exit code; checar artefato antes de anunciar sucesso.'
    elif row['signals'].get('caminhos_fixos'):
        suggestion='Tornar caminhos locais configuráveis e falhar cedo com diagnóstico quando recurso não existir.'
    elif row['signals'].get('excecao_ampla'):
        suggestion='Distinguir falha recuperável de erro de programação e registrar contexto suficiente antes do fallback.'
    elif Path(p).name=='__init__.py':
        suggestion='Manter inicialização sem efeitos colaterais e testar somente contratos públicos exportados quando houver lógica.'
    else:
        suggestion='Cobrir invariantes de entrada/saída e casos-limite apropriados ao módulo; ausência de alertas estáticos não verifica comportamento numérico ou integração.'
    per += ['**Melhoria sugerida:** '+suggestion, '']
(OUT/'POR_SCRIPT.md').write_text('\n'.join(per),encoding='utf-8')
(OUT/'INVENTARIO.tsv').write_text('\n'.join(tsv)+'\n',encoding='utf-8')
print(json.dumps({'fichas':len(inventory),'revisao_dirigida':len(reviewed),'achados':len(findings),'prioridades':{p:sum(f['priority']==p for f in findings) for p in ('P1','P2','P3')},'alterados_desde_coleta':changed},ensure_ascii=False))
