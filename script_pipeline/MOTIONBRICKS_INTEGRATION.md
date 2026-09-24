# MotionBricks + decupagem + LTX/MiniMax

`motion_conditioner.py` fica entre `shot_plan` e `render_shots`. Ele usa os
campos já determinados pela decupagem (`subject`, `co_subject`, `beat`,
`screen_side`, `framing`, `line_index`, duração e estilo) e gera dois artefatos.

1. `parse/motion_plan.json`: score por ator, com âncoras de blocking, duração,
   primitiva, alvo e distância mínima. É a entrada de um adaptador 3D.
2. `parse/shot_plan.json` (somente com `--apply`): preserva `video_prompt` e
   adiciona `motion_prompt`. LTX 2.5 e MiniMax H3 já consomem esse prompt pelo
   renderizador existente; nenhum endpoint novo é necessário.

## Uso

```powershell
python -m script_pipeline.motion_conditioner --plan RUN\parse\shot_plan.json --apply
python -m script_pipeline.run_decupagem --run-dir RUN --motion-conditioning --ate animatic
```

A etapa `[M movimento]` é opcional (`obrigatorio=False`, como `--camera-llm` e a
character-sheet): se falhar, a corrida segue com o `video_prompt` de sempre.
`--apply` é idempotente — rodar duas vezes não duplica a cláusula.

## Primitivas

`environment` (tomada sem sujeito), `speak`, `idle`, `gesture`, `turn`,
`locomote`, `pursue`, `reach`, `contact`, `takedown`, `protect`, `fire`.

Os verbos de ação têm contratos próprios: perseguição (`pursue`) mantém o eixo e
continua na mesma direção — não faz os dois agentes correrem um contra o outro;
queda controlada (`takedown`), proteção, puxar um parceiro e disparo não podem
cair silenciosamente em `idle`. O MotionBricks público só fornece clipes de
locomoção: perseguição/aproximação usam `walk`; disparo continua sem animação
esquelética de arma e depende da geração de vídeo e do gate visual. A auditoria
deve bloquear clipes em que a ação requerida não aparece.

Ao gerar vídeo (`--ate render` ou `--ate final`), o compilador roda
automaticamente e de forma obrigatória, mesmo sem `--motion-conditioning`. O
flag permanece para gerar o score ao parar antes do vídeo.

Regras que não são gosto:

- **Sem sujeito é sempre `environment`**, antes de olhar qualquer verbo. O
  texto de fallback de um plano de estabelecimento é a cena inteira e pode
  descrever outra pessoa andando — a v1 produzia "The subject must walk" num
  quadro vazio.
- **Close de fala é `speak`**, nunca gesto de corpo nem deslocamento: gesto de
  corpo num close tira a boca do quadro e derruba o lip-sync (MEMORIAL
  3.81/3.85). Em plano médio a mesma fala pode gesticular.
- Classificação por **regex com fronteira de palavra** e lemas cobertos. A v1
  usava substring solta e, MEDIDO no plano real dos piratas (13 tomadas),
  errou 4 das 9 com personagem: "gestures towards" → idle, "holds a spyglass"
  → idle, "looking out" → idle, e "hulls almost *touching*" → reach (o verbo
  era dos navios). A v2 acerta as quatro; `tests/test_motion_conditioner.py`
  fixa esses casos.
- `locomote` sem parceiro não tem destino (`short_path`, `destination: null`).
  A v1 mandava o ator "andar" até a própria âncora, que é ficar parado.

`story_editor.py` (Fix #6) reconstrói o `video_prompt` ao reescrever um beat
duplicado e recoloca o `motion_prompt` no fim — sem isso a cláusula sumia.

## Runtime MotionBricks nesta máquina (medido 2026-09-17)

Instalação em `E:\Users\home\Documents\GR00T-WholeBodyControl\motionbricks`,
venv próprio (torch 2.4+cu124, mujoco 3.13). Os quatro checkpoints estão
baixados em `out/` (G1-clip 7 MB, vqvae 286 MB, root 410 MB, pose 1,6 GB).
`datasets/` não existe — com `G1-clip.ckpt` presente não é necessário
(`navigation_demo` pula o dataloader).

O venv estava incompleto: faltavam `scipy`, `einops`, `hydra-core`,
`pytorch-lightning`, `matplotlib`, `colorlog`, `absl-py`, `keyboard`, `glfw`,
`pyopengl`, `etils[epath]`. Foram instalados em 2026-09-17 com **`torch==2.4.0`
e `numpy==1.26.4` fixados na mesma linha** — sem o pin, `scipy` sobe o numpy
para 2.x (torch 2.4 é compilado contra 1.x) e `pytorch-lightning`/`transformers`
tentam subir o torch. Duas restrições de versão que não são gosto:

- `vector-quantize-pytorch<1.20` (instalado 1.19.5): as versões novas dependem
  de `torch-einops-utils`, que exige torch ≥ 2.5 em todas as suas versões. O
  MotionBricks só usa `VectorQuantize`, estável desde muito antes.
- `scipy>=1.14,<1.15` (1.14.1): `mujoco_helper.py` chama
  `Rotation.from_quat(..., scalar_first=...)`, argumento que entrou no 1.14;
  o 1.14 ainda aceita numpy 1.26.

`transformers` e `zimage-native` nesse venv pedem torch ≥ 2.5 e só emitem
aviso — o MotionBricks não usa modelos do transformers. Nunca rode `pip` ali
com um processo do mesmo venv com torch importado: foi assim que o torch ficou
meio-desinstalado e precisou de reinstalação forçada (2,5 GB).

`script_pipeline/motionbricks_headless.py` é o runner sem viewer: instancia
`navigation_demo`, roda o laço `get_next_frame`/`generate_new_frames` com o
controlador `random` do repo e grava `qpos[T, 36]` do G1 em `.npz`. Roda com o
python do venv do MotionBricks, não com o `.venv` do LTX.

**MEDIDO 2026-09-17 (3090):** checkpoints carregados em 2,6 s; 120 quadros a
30 fps de representação gerados em 0,72 s (~167 fps no laço, incluindo
`mj_forward`); `qpos (120, 36)`; raiz deslocou 0 → 0,33 m sob o controlador
aleatório. O aviso "you are advised to provide all first 4 frames" é do
contexto inicial vazio e não afeta o resultado.

## Controlador por comandos (implementado 2026-09-17)

`motion_command_geometry.py` (lógica pura, numpy, testada no `.venv` do LTX —
`tests/test_motion_command_geometry.py`, 9 testes) + `motionbricks_command_
controller.py` (`MotionPlanController`, um wrapper fino que encaixa aquela
lógica na interface `generate_control_signals(viewer, mj_model, mj_data,
visualize, control_info)` que `navigation_demo` espera — roda no venv do
MotionBricks). Um agente por ATOR: filtra `commands[]` pelo campo `actor` e
avança pelo `start_s`/`duration_s` usando o tempo de simulação real
(`mj_model.opt.timestep` por passo).

```powershell
...\motionbricks\.venv\Scripts\python.exe script_pipeline/motionbricks_headless.py ^
    --motion-plan RUN\parse\motion_plan.json --actor "NOME" --out ator.npz
```

O numero de passos e derivado da duracao total dos comandos do ator (nao de
`--steps`, que so vale pro smoke aleatorio).

**MEDIDO 2026-09-17 (3090), plano sintetico idle(0-1s) → locomote ate um
parceiro a 2,77 m (1-5s) → speak(5-6s):** a raiz ficou parada nos primeiros
~1s, andou em linha reta e MONOTONICAMENTE reduziu a distancia ate o parceiro
(2,77 m → 0,10 m aos 4s, exatamente quando o comando `locomote` termina) e
assentou a ~0,03 m do alvo depois que o comando seguinte a deixou parada de
novo — sem overshoot alem do natural de uma caminhada freando. No plano real
dos piratas (SEO-YEON: so `speak`/`turn`, nunca `locomote`) a raiz ficou
corretamente parada o tempo todo (o clipe `idle` do checkpoint tem
`avg_root_vel: 0.0` por definicao) -- bom lembrete de que a maioria das cenas
de dialogo puro nunca exercita o caminho de caminhada.

**O limite do checkpoint publico continua o mesmo**: `speak`, `gesture`,
`reach` e `turn` viram o clipe `idle` (o agente para e so gira pra encarar o
alvo) porque `out/G1-clip.ckpt` so tem clipes de locomoção -- ver
`PRIMITIVE_TO_CLIP` em `motion_command_geometry.py`. Só `locomote` e
`contact` (que sempre tem parceiro) usam o clipe `walk` de verdade.

### Duas ou mais pessoas na mesma cena (verificado 2026-09-17)

Um agente por ator (`adapter_contract.multi_actor` ja avisava disso) -- cada
`MotionPlanController`/`navigation_demo` roda **num processo/simulacao
independente**. Isso funciona bem quando so UM personagem anda por vez
(verificado antes: caminha ate um parceiro parado e assenta perto dele). Com
**dois personagens andando um em direcao ao outro ao mesmo tempo**, apareceu
um bug real, so visivel rodando os dois de verdade:

- **1a versao**: `target["destination"]` de `approach_partner` era a ancora
  ORIGINAL do parceiro. Com os DOIS lados se aproximando, cada um andava ate
  a marca de ONDE O OUTRO COMECOU -- MEDIDO: ANA e BIA se cruzaram e
  TROCARAM de lado de tela inteiro, terminando **mais longe** (2,75 m) do
  que a distancia inicial (2,4 m). Corrigido: `_meeting_point()` em
  `motion_conditioner.py` poe o destino de cada um a `min_separation_m/2` do
  PONTO MEDIO entre as duas ancoras, do proprio lado -- nenhum dos dois cruza
  o centro por construcao (resolvido no JSON, sem precisar dos dois agentes
  se coordenarem em tempo real).
- **2a versao** (ainda no mesmo dia): a rede de seguranca que eu tinha posto
  em `walk_target()` (nunca chegar mais perto que `min_separation_m`) SOMAVA
  com o `_meeting_point` que ja tinha embutido metade dela -- os dois agentes
  liam "faltam 0,5 m, mas so paro a 1,0 m" e ficavam parados no lugar sem
  andar nada. Removida: a separacao segura e responsabilidade so do
  `_meeting_point` (na origem); `walk_target` so cuida da tolerancia de
  chegada (`ARRIVAL_EPS_M`).
- **Verificado depois do fix**: ANA (ancora -1,2) e BIA (ancora +1,2) andando
  uma em direcao a outra -- ANA ficou sempre com x negativo, BIA sempre
  positivo (screen_side nunca invertido), distancia entre elas oscilando
  1,3±0,6 m (a variacao normal do blend de locomocao deste checkpoint, ja
  vista no teste de 1 ator) em vez de convergir pra um numero fixo.

O que **continua sem verificar**: contato fisico de verdade (aperto de mao,
abraco) -- os dois agentes convergem pra perto um do outro, mas nenhum
detecta colisao real nem sincroniza o gesto; e o proprio `adapter_contract`
ja avisa que isso pede um "diretor de cena" resolvendo a trajetoria conjunta,
nao dois agentes cegos um pro outro.

### Diretor de cena: coordenacao em tempo real (implementado 2026-09-17)

`motionbricks_scene_director.py` roda N agentes (um `navigation_demo` por
ator, sem mundo MuJoCo compartilhado) no MESMO relogio de simulacao,
realimentando a posicao ATUAL de cada um nos outros a cada passo via
`live_positions` (dicionario compartilhado). Muda tres coisas em relacao ao
modo de 1 agente:

1. **Alvo passa a ser a posicao AO VIVO do parceiro**, nao o ponto
   pre-calculado da posicao inicial (`partner_anchor` em
   `motion_command_geometry.py` prefere `live_positions[partner]` quando
   disponivel).
2. **Freio de colisao ao vivo**: para assim que a distancia REAL cai abaixo
   de `min_separation_m + brake_margin_m` -- nao so quando chega no alvo
   pre-calculado.
3. **Correcao de referencial**: a simulacao do MotionBricks sempre NASCE
   cada agente em (0,0) local, nunca na ancora que `formation` atribuiu --
   com 1 agente isso passa despercebido; com 2+, eles nasciam sobrepostos.
   `MotionPlanController` agora desloca a posicao local pela propria ancora
   antes de qualquer calculo de distancia/destino/colisao (so translacao --
   nao muda direcao nenhuma, so onde "zero" fica).

```powershell
...\motionbricks\.venv\Scripts\python.exe script_pipeline/motionbricks_scene_director.py ^
    --motion-plan RUN\parse\motion_plan.json --actors "ANA,BIA" --out-dir saida\
```

**MEDIDO 2026-09-17 (3090), ANA e BIA se aproximando uma da outra:**

- **Sem o diretor** (cada uma so mirando o ponto pre-calculado da posicao
  ORIGINAL da outra): elas se CRUZAVAM e trocavam de lado de tela inteiro
  (ver secao anterior) -- corrigido na origem com `_meeting_point`.
- **Com o diretor**, calibracao do `--brake-margin` (quanto antes de
  `min_separation_m` o freio comeca a pedir parada):

  | margem | separacao minima real | erro |
  |---|---|---|
  | 0,0 m | 0,45 m | -55% (momento do clipe de caminhada carrega alem do pedido) |
  | 0,5 m | 0,45 m | -55% -- **identico bit a bit** ao de 0,0 m |
  | **1,0 m** | **1,009 m** | **<1%** |
  | 2,0 m (limiar > distancia inicial) | -- | nem chegam a andar (freio dispara ja no 1o passo, correto) |

  O salto de "nenhum efeito" (0,5m) pra "quase exato" (1,0m) em vez de uma
  melhora gradual sugere que `full_navigation_agent.generate_new_frames`
  planeja em BLOCOS de ~0,53s (`get_controller_dt()*2`) -- pedir pra frear
  uns quadros mais cedo dentro do MESMO bloco nao muda nada; so um salto
  grande o bastante desloca o pedido pro bloco anterior. **`--brake-margin
  1.0` (o novo padrao) e o valor calibrado** -- recalibrar (testar 0/0.5/1/2
  de novo) se o checkpoint ou o passo de tempo mudar.

**O que o diretor NAO faz, e por que nao e defeito dele**: nao produz contato
fisico de verdade (aperto de mao, abraco, luta) -- o checkpoint publico so
tem clipes de locomocao, entao os dois agentes so conseguem parar perto um do
outro e encarar-se. Nenhuma quantidade de coordenacao em tempo real inventa
um clipe de gesto que nao existe no checkpoint.

## Adaptador que ainda falta

1. ~~Um controlador próprio que leia `commands[]`~~ — feito (acima).
2. Retarget do qpos do G1 (36 DOF) para o rig do personagem e exportação
   FBX/GLB ou sequência de poses.
3. Usar essa sequência como guia de movimento no gerador de vídeo — hoje só o
   LTX 2.3 tem pose real (IC-LoRA union-control); no 2.5 e no MiniMax a única
   entrada de movimento é o texto, que é o que `motion_prompt` já cobre.
4. Um checkpoint com clipes de upper-body (gesto, alcançar, contato) ligaria
   `speak`/`gesture`/`reach`/`contact` a movimento de verdade sem mudar uma
   linha do controlador — só o mapa `PRIMITIVE_TO_CLIP`.

Em mais de um personagem, não execute agentes isolados para abraço, luta,
entrega ou objeto compartilhado. O `formation` reserva posições e o score
marca `contact` (e avisa com 3+ atores); um diretor de cena deve resolver a
trajetória conjunta, colisão e sincronização antes da inferência de cada
agente. Para diálogo sem contato, um agente por personagem mais as âncoras é
suficiente.
