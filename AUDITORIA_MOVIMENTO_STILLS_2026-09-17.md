# Auditoria de movimento, pose e stills

Auditoria do conteúdo local em 17/09/2026. Nenhum código de produção foi alterado e nenhum commit foi criado. P1 = corrigir antes de integrar; P2 = corrigir antes de depender do caso afetado.

## Achados confirmados

### 1. P1 — Cenas diferentes se sobrepõem no relógio do controlador

Local: `script_pipeline/motion_command_geometry.py:36`, em conjunto com `motion_conditioner.build_motion_score` e os dois runners.

O produtor reinicia `cursor = 0` por cena. O consumidor filtra apenas por ator e ordena todos os comandos por `start_s`, sem filtrar a cena nem aplicar deslocamentos temporais. Os runners calculam a duração pelo maior término, não pela sequência de cenas. Reproduzido com duas cenas de 2 s da ANA: ambas começam em 0; a duração calculada é 2 s; em t=1 apenas a primeira cena é selecionada. A segunda cena pode nunca executar. Trocas de formação também não têm um reset coordenado do agente.

Correção: executar uma cena explicitamente selecionada por vez, ou compilar uma timeline global com transições e transformação de coordenadas definidas. Testar duas cenas com o mesmo ator e durações diferentes.

### 2. P1 — Lacunas e entradas tardias executam comandos fora do horário

Local: `script_pipeline/motion_command_geometry.py:43-50`.

Quando não há comando ativo, `current_command` retorna o último comando da lista se o primeiro já começou. Reproduzido com comandos nos intervalos [0,1) e [5,6): em t=2 retorna o comando de t=5. Antes do primeiro comando, retorna esse primeiro comando antecipadamente. Lacunas são naturais porque o produtor emite comandos apenas para o sujeito de cada tomada. Um ator pode caminhar ou entrar em contato antes da hora.

Correção: representar explicitamente ausência de comando e produzir idle nas lacunas/antes da entrada. Definir também a política após o último comando, para não prolongar locomação indefinidamente enquanto outro ator continua.

### 3. P1 — As cores do vídeo não correspondem à convenção OpenPose declarada

Local: `script_pipeline/pose_video.py:128-135,169`.

A tabela é RGB, mas os bytes são enviados ao ffmpeg como `bgr24`. Um segmento pescoço–ombro direito produz [255,0,0] no buffer e chega ao vídeo como azul, em vez de vermelho. Além disso, todas as juntas são brancas, enquanto a referência usa uma cor por junta; os membros também diferem em forma/intensidade.

Correção: manter RGB e declarar `rgb24`, ou converter explicitamente para BGR; usar o rasterizador de referência ou validar equivalência visual com uma pose fixa. A divergência dos pixels é confirmada; seu impacto quantitativo na geração LTX/H3 ainda exige teste real.

Fontes primárias: [desenho do controlnet_aux](https://raw.githubusercontent.com/huggingface/controlnet_aux/master/src/controlnet_aux/open_pose/util.py) e [conversão do resultado em imagem RGB](https://raw.githubusercontent.com/huggingface/controlnet_aux/master/src/controlnet_aux/open_pose/__init__.py).

### 4. P1 — Adaptador e workflow H3 discordam sobre a duração

Local: `script_pipeline/motion_to_h3_controlnet.py:25-31`.

O adaptador escolhe o valor 17n+5 mais próximo; o node 131 do workflow efetivamente usado pelo backend arredonda para cima. Para 4,5 s, o adaptador gera 107 frames e o workflow pede 124. O `_fit_frames` do ControlNet instalado completa a diferença repetindo o último frame: 17 frames, aproximadamente 0,71 s de pose parada. O teste atual exige 108→107 e, portanto, consolida a divergência. O node aceita ainda 5 frames, enquanto o wrapper impõe mínimo de 22.

Correção: compartilhar o contrato de contagem entre adaptador e backend e testar diversas durações contra a expressão do workflow, não apenas 5 s, caso em que ambos produzem 124.

### 5. P2 — Escala é aplicada duas vezes às posições ao vivo

Local: `script_pipeline/motion_command_geometry.py:105,172`.

O controlador publica `live_positions` em coordenadas de mundo, já com a âncora escalada. `facing_for` e `walk_target` multiplicam novamente o destino retornado por `partner_anchor`, inclusive quando veio de `live_positions`. Reproduzido: ator em x=3, parceiro em x=2 e escala 2 produzem direção +X, embora o parceiro esteja à esquerda. O cálculo do freio usa as posições sem essa segunda escala, gerando ainda uma inconsistência entre direção e distância.

Correção: converter somente âncoras/destinos estáticos; posições ao vivo devem permanecer em metros de mundo. Testar escalas 0,5 e 2.

### 6. P2 — O freio ignora terceiros e locomação sem parceiro

Local: `script_pipeline/motion_command_geometry.py:162-168`.

A consulta de proximidade verifica apenas `command.partner`. Com BIA em (5,0) e CARLOS a 0,1 m do ator, o controlador continua andando em direção à BIA. Sem parceiro, nenhum vizinho é consultado. Logo, o diretor roda 2+ agentes, mas seu freio não oferece prevenção geral de sobreposição para 3+ agentes, nem para duas caminhadas sem parceiros definidos.

Correção: separar alvo de interação e consulta de proximidade a todos os outros atores ativos. Não confundir esse freio geométrico com a limitação já documentada de ausência de contato físico MuJoCo compartilhado.

### 7. P2 — A execução direta da CLI do adaptador H3 falha

Local: `script_pipeline/motion_to_h3_controlnet.py:20`.

Reproduzido na raiz do repo: `.venv/Scripts/python.exe script_pipeline/motion_to_h3_controlnet.py --help` falha com `ModuleNotFoundError: No module named 'script_pipeline'`. Os testes importam como pacote e não detectam isso.

Correção: padronizar e documentar `python -m script_pipeline.motion_to_h3_controlnet`, ou suportar a execução direta como os demais runners documentados. Adicionar smoke da CLI escolhida.

### 8. P2 — A CLI de storyboards ainda depende de ComfyUI com Z-Image

Local: `script_pipeline/generate_storyboards.py:916-923`.

O desvio para Z-Image existe em `generate_scene_storyboard`, mas `main` verifica/inicia ComfyUI antes de chegar a ele, independentemente do motor. Reproduzido com `--image-engine zimage` e ComfyUI indisponível, usando mock: retorno 1 e uma tentativa de iniciar ComfyUI. `--no-auto-start` também verifica a disponibilidade. Isso impede o motor independente de funcionar quando ComfyUI não está disponível.

Correção: aplicar à CLI a mesma guarda de arquitetura já presente em `render_shots`. Testar Z-Image com ComfyUI desligado sem executar geração pesada.

## Melhorias e riscos adicionais

- **Z-Image concorrente:** `../Z-Image/server.py` mantém pipelines globais e expõe handler síncrono sem lock. Requisições simultâneas podem disputar inicialização, scheduler e offload. Serializar carga e inferência; testar duas requisições concorrentes com pipeline simulado antes de testar GPU. Não foi reproduzida falha de inferência concorrente nesta auditoria.
- **Troca still→vídeo:** `render_shots` encerra instâncias ComfyUI em alguns caminhos, mas não encerra/libera Z-Image. O servidor também instancia separadamente txt2img e img2img. Medir RAM/VRAM após cada modalidade e definir liberação no ciclo de vida da pipeline. Não atribuir os 500 s/passo a isso sem medição.
- **Exportação MotionBricks:** os NPZ guardam apenas `qpos` local e `dt`; as âncoras de mundo não estão no arquivo. Guardar metadados de ator/cena, transformações e/ou `qpos_world` torna o resultado reutilizável sem reconstruir o contexto externo. A estatística final do diretor usa a primeira formação encontrada, incompatível com mudanças de cena.
- **Validação/diagnóstico:** validar fps positivos/finitos, T>0, última dimensão 3, dimensões de vídeo e intervalos do ControlNet. O ffmpeg descarta stderr; preservar esse diagnóstico facilita corrigir dimensões inválidas e falhas de encoder.
- **Guarda de referências:** a preservação em `character_sheet` está correta para o pedido. Como melhoria, filtrar atores antes de iniciar ComfyUI e alertar para referência inexistente, preservando a decisão de não sobrescrever sem `force`.
- **Documentação:** o controlador diz que `contact` vira idle, mas o mapa efetivo o converte em walk. Explicitar que é aproximação por caminhada, sem gesto de contato.

## Pendências esclarecidas e limites

- **InterGen:** `utils/preprocess.py:4` define `FPS = 30`; `tools/infer.py:55` renderiza a 30 fps. O checkout local sustenta 30 fps, não a suposição de 20. Sem contagem forçada, 210 frames a 20 fps viram 10,5 s, contra 7 s a 30 fps. Com contagem explícita, o resampler atual ignora fps e estica o movimento inteiro; portanto, essa correção isolada não muda o wrapper H3.
- **Inter-X:** revisão dos diffs de inferência, quaternion e visualização não encontrou outro bug confirmado. O rig padrão continua sendo esquemático; os testes aprovados não validam proporções reais de SMPL-X nem a licença do checkpoint.
- **ComfyUI H3:** HEAD confirmado em `cf5cc2b`, sem modificações locais no status consultado. Conferidos os contratos locais de grade temporal e ajuste dos frames. Não foi auditada integralmente a atualização upstream nem executado benchmark de `int8convrot`.
- **Geração final:** não foram executadas novas gerações pesadas LTX/H3. Carregar/aplicar o node sem erro continua insuficiente para validar aderência do vídeo à pose.
- **Estado Git:** Z-Image/character_sheet/storyboards/render_shots já estão no HEAD; último commit que os abrange é `ac887a8` (`Decupagem: two audit rounds, Z-Image still engine, MotionBricks bridge`). A afirmação de que tudo está não commitado não corresponde ao checkout encontrado. Esta auditoria não criou commits.

## Verificação executada

- 54 testes aprovados: geometry, pose_video, motion_to_h3_controlnet, motion_conditioner e minimax_h3_test_ui.
- 7 testes aprovados de `Inter-X/tests/test_hhi_visualization.py`, incluindo comparação com encoder/FK de referência e GIF.
- Reproduções pequenas: cenas sobrepostas, lacuna selecionando futuro, inversão de direção com escala, terceiro ator ignorado, pixel RGB interpretado como BGR, divergência 107/124 frames, CLI direta quebrada e dependência indevida de ComfyUI com Z-Image.
- Nenhum código de produção alterado. Este relatório é a única entrega adicionada ao repo.
