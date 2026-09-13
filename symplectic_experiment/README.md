# Experimento: dinâmica hamiltoniana latente para vídeo (protótipo isolado)

Status: **protótipo de pesquisa, não integrado a nenhum pipeline de produção
deste repositório.** Nada aqui é chamado por `script_pipeline/`, `ltx25_backend.py`,
`ltx_pipelines_25.py` ou qualquer UI (`*_ui.py`, `web_ui_*`, `music_maker_*`).
Roda isolado, com dados sintéticos, e serve para explorar se a mecânica proposta
(integrador simplético + decoder INR) sequer treina de forma estável — não para
gerar vídeo de qualidade nem para substituir o LTX 2.3/2.5.

## Por que isto NÃO é "transporte de pesos" do LTX

A proposta original pedia para reinterpretar a rede de score do LTX (um DiT de
difusão) como o gradiente de um potencial hamiltoniano estático V(q), congelando
o VAE e "convertendo" o backbone via LoRA. Isso não foi implementado, porque a
premissa central não se sustenta:

1. **O score de difusão não é o gradiente de um potencial fixo.** A rede prediz
   ∇ₓ log p_t(x) para uma *família de distribuições marginais indexadas pelo
   nível de ruído t*. Esse campo muda de direção e magnitude com t; não existe
   um único escalar V(q) do qual ele seja gradiente em todos os níveis de ruído
   simultaneamente. "Fixar σ→0 e usar a rede como -∇V" descarta justamente o
   mecanismo (agendamento de ruído) que faz a difusão funcionar — o resultado
   não é uma força conservativa, é a saída da rede em um regime para o qual ela
   nunca foi treinada a ser consistente ao longo do tempo.
2. **Conservação de energia é a propriedade errada para vídeo real.** Cortes de
   cena, mudança de iluminação, câmera se movendo, objetos entrando/saindo de
   quadro são eventos *não conservativos* por definição. Um treinamento que
   penaliza `|H(t+1) - H(t)|²` empurra o modelo para trajetórias suaves demais
   para conteúdo real.
3. **MiniMax H3 é citado como alvo de LoRA, mas é servido por API fechada** —
   sem acesso a pesos ou ativações internas, não há nada para transplantar.

Este experimento, portanto, valida só a **mecânica do integrador e do
decoder**, com um potencial V aprendido do zero sobre um dataset sintético
simples (não a partir de pesos do LTX). Se um dia a ideia dos autores originais
provar valor real, o próximo passo seria uma revisão da literatura de "neural
ODEs / Hamiltonian generative networks aplicados a vídeo condicionado por
difusão destilada" — não uma reescrita direta do backbone existente.

## O que está implementado

- [`hamiltonian.py`](hamiltonian.py): `KineticEnergy` (massa diagonal via
  softplus), `PotentialEnergy` (MLP com SiLU, C² — necessário porque o
  integrador diferencia V duas vezes), `SymplecticIntegrator` (passo de Verlet
  explícito, gradiente analítico em T(p)).
- [`inr_decoder.py`](inr_decoder.py): decoder INR modulado (Fourier features +
  camadas SIREN moduladas por FiLM a partir de q(t)), resolução de saída
  arbitrária via a malha de coordenadas passada em `forward`.
- [`toy_dataset.py`](toy_dataset.py): gerador sintético de "vídeo" (um disco
  colorido quicando numa caixa 2D com iluminação senoidal) — escolhido porque é
  um sistema *de fato* aproximadamente conservativo (colisões elásticas),
  então é o caso mais favorável para testar se o integrador consegue aprender
  uma trajetória hamiltoniana coerente. Não é representativo de vídeo real.
- [`train_toy.py`](train_toy.py): loop de treino com `L1` de reconstrução +
  `L_H-cons` (conservação de energia) + `L_symp` (resíduo simplético via JVP
  estocástico), CPU-only, poucos parâmetros — feito para rodar em minutos e
  servir de sanity check, não como benchmark de qualidade.

## RESULTADO (2026-09-12): a estrutura hamiltoniana PERDE do baseline

Experimento decisivo em [`train_dynamics_only.py`](train_dynamics_only.py),
rodado na RTX 4070 (Ubuntu, 100.99.13.105), 5 sementes:

| modelo | params | MSE dentro do treino | MSE extrapolando |
|---|---|---|---|
| hamiltoniano (Verlet simplético) | 4.483 | 0,00912 ± 0,00010 | **0,36704 ± 0,02723** |
| baseline compacto SEM física | 4.530 | 0,00018 ± 0,00004 | **0,00011 ± 0,00013** |
| piso trivial (estado congelado) | — | 0,28461 | 0,28461 |

As 5 sementes não se sobrepõem (ham: 0,340–0,398; base: 0,00001–0,00031).
O baseline generaliza melhor do que treina (0,00011 extrapolando contra
0,00018 dentro) — aprendeu a dinâmica real. O hamiltoniano extrapola **pior
que o piso trivial**: diverge.

**Causa**: a colisão com a parede é uma DESCONTINUIDADE (velocidade inverte
instantaneamente). Um potencial suave V(q) não produz reflexão instantânea —
só aproxima com uma parede íngreme, cujo erro compõe e explode na
extrapolação. A rede sem restrições aprende o mapa de reflexão diretamente.

**Peso deste resultado**: é o caso MAIS favorável possível à tese hamiltoniana
— sistema quase-conservativo, sem cortes de câmera, sem dissipação, sem
confundidor de renderização, contagens de parâmetro pareadas em 1%. E mesmo
assim perde por 3 ordens de grandeza. Vídeo real é *mais* descontínuo, não
menos.

**O que este resultado NÃO refuta**: a formulação híbrida (fluxo suave entre
eventos + mapa de salto aprendido `s_{t⁺} = R(s_{t⁻}, e_t)`) e a extensão
dissipativa/port-hamiltoniana (`ṗ = −∇V − D·M⁻¹p + B·u`). Essas são justamente
a resposta à descontinuidade e não foram testadas aqui.

### Duas tentativas anteriores foram INVÁLIDAS — registrado para não repetir

Antes desta versão, dois experimentos produziram números que pareciam
conclusivos e não eram:

1. [`train_toy.py`](train_toy.py) e [`train_compare.py`](train_compare.py)
   aprendiam dinâmica + espaço latente + decoder ao mesmo tempo. Colapsaram na
   solução estática: o decoder vira constante, o gradiente em `q` morre e a
   dinâmica nunca aprende. MEDIDO: predições com variação temporal 0,0001–0,0003
   contra 0,0060 do vídeo real, e L1 (0,0110/0,0112) **pior que o piso trivial
   de desenhar só o fundo** (0,0108). O "empate" entre os dois modelos era
   empate em não ter aprendido nada.
2. Aumentar o raio da bola de 0,18 para 0,45 **não** resolveu — o colapso é do
   acoplamento decoder+dinâmica, não do tamanho do sinal.

A lição (que estava no material de referência e foi ignorada na primeira
implementação): **não substitua dinâmica, espaço latente e renderizador
simultaneamente** — fica impossível saber qual mudança causou o quê. O
`train_compare.py` agora calcula o piso trivial e **aborta sozinho** a
comparação se algum modelo não o superar.

### Dois bugs corrigidos no integrador (2026-09-12)

- **Gradiente do rollout**: `q.detach().requires_grad_(True)` dentro do passo
  de Verlet cortava a dependência da força em relação aos estados anteriores;
  `create_graph=True` não restaura elo cortado. Erro MEDIDO de 0,10% (3 passos,
  potencial fraco) a 104% (24 passos, potencial rígido).
- **Resíduo simplético**: media `sum(v_q*v_p)` com um vetor só, que não é
  invariante — VERIFICADO: sob um mapa simplético exato essa quantidade vai de
  +0,3497 a +2,6369, enquanto ω(u,v)=u_q·v_p−u_p·v_q de dois vetores se
  preserva a 2,4e−07. Com um vetor só seria impossível: vᵀJv ≡ 0.

## Continuação: Probability Flow ODE (a versão matematicamente correta)

A objeção acima ("score não vira força hamiltoniana") tem uma causa mais
precisa do que "não é gradiente" — o score ∇ₓ log p_t(x) **é** gradiente de um
potencial (log p_t). O problema é que ele conflava dois eixos de tempo
diferentes: o nível de ruído da difusão (que indexa uma família de marginais
sobre um vídeo inteiro tratado como um único ponto x) e o tempo físico do
vídeo (frame a frame). Densidade estática de dados não carrega informação de
transição temporal — não tem como um refinamento de engenharia recuperar essa
informação, porque ela nunca esteve codificada ali.

A formulação que resolve isso corretamente, sem essa conflação, é a
**Probability Flow ODE** (Song et al., 2021) com precondicionamento **EDM**
(Karras et al., 2022) — os mesmos princípios por trás de destilação real
(consistency models, LCM) usada em produção, inclusive nos checkpoints
`*-distilled` deste repositório. Implementada em:

- [`score_model.py`](score_model.py): rede de score toy `F_theta(x, sigma)`
  (a rede só vê o nível de ruído, nunca "tempo de frame" — o vídeo inteiro é
  um único ponto x).
- [`pf_ode.py`](pf_ode.py): precondicionamento EDM (`c_skip`, `c_out`,
  `c_in`, `c_noise`), perda de denoising score matching, e o solver de Heun
  (2ª ordem) para integrar `dx/dsigma = (x - D(x;sigma)) / sigma`.
- [`toy_video_distribution.py`](toy_video_distribution.py): distribuição de
  vídeos sintéticos (posição/velocidade/cor iniciais aleatórias) — o score
  model aprende essa distribuição de verdade, não decora um vídeo só.
- [`train_pf_ode.py`](train_pf_ode.py): treina o score model via EDM loss e
  depois demonstra round-trip determinístico dado→ruído→dado, medindo erro de
  reconstrução por número de passos de integração.

**Isto entrega de verdade a propriedade que a versão hamiltoniana só
alegava**: trajetória determinística e invertível entre dado e ruído — medida,
não hipotética. Rodando com 800 passos de treino (`--train-steps 800`), o
erro de round-trip cai de ~0,12 (5 passos) para ~0,002-0,004 (10-30 passos),
confirmando que a ODE integra de forma consistente nos dois sentidos.

**O que isto ainda NÃO é**: um caminho para "vídeo em um passo só" de
qualidade real. Essa parte exige destilação (consistency models / LCM),
treinando um segundo modelo a colapsar a trajetória da ODE em poucos passos —
não implementado aqui. E o dado ainda é sintético (disco quicando); nada
disso toca nos pesos do LTX 2.3/2.5.

## Como rodar

```bash
.venv/Scripts/python.exe symplectic_experiment/train_toy.py --steps 200
```

Para a versão PF-ODE (recomendada — é a matematicamente correta):

```bash
.venv/Scripts/python.exe symplectic_experiment/train_pf_ode.py --train-steps 800
```

Não usa GPU, não usa `.venv` do LTX especificamente — qualquer ambiente com
`torch` funciona (testado com o `torch` já presente em `.venv`).

## Resultado esperado (e o que ele NÃO prova)

Ao final, o script imprime a perda de reconstrução do disco quicando e o
desvio de energia hamiltoniana. Uma perda baixa aqui mostra apenas que o
integrador simplético consegue aprender a dinâmica de um sistema físico
simples de brinquedo — isso **não** é evidência de que a mesma arquitetura
escalaria para vídeo generativo condicionado por texto, nem que "substitui a
difusão". Trate como confirmação de que o código dos blocos matemáticos está
correto, nada além disso.
