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
`locomote`, `reach`, `contact`.

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

## Limite e adaptador necessário

O MotionBricks controla um esqueleto humanoide (o G1) com primitivas/objetivos
e clips de estilo; não aceita o prompt de cinema nem gera pixels. O que falta
para condicionamento físico real, nesta ordem:

1. Um controlador próprio com a interface de `generate_control_signals(viewer,
   mj_model, mj_data, visualize, control_info)` que leia `commands[]` do
   `motion_plan.json` (âncora, destino, primitiva, duração) em vez do teclado
   ou do aleatório.
2. Retarget do qpos do G1 (36 DOF) para o rig do personagem e exportação
   FBX/GLB ou sequência de poses.
3. Usar essa sequência como guia de movimento no gerador de vídeo — hoje só o
   LTX 2.3 tem pose real (IC-LoRA union-control); no 2.5 e no MiniMax a única
   entrada de movimento é o texto, que é o que `motion_prompt` já cobre.

Em mais de um personagem, não execute agentes isolados para abraço, luta,
entrega ou objeto compartilhado. O `formation` reserva posições e o score
marca `contact` (e avisa com 3+ atores); um diretor de cena deve resolver a
trajetória conjunta, colisão e sincronização antes da inferência de cada
agente. Para diálogo sem contato, um agente por personagem mais as âncoras é
suficiente.
