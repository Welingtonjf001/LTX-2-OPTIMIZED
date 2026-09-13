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

## ATUALIZAÇÃO (2026-09-13): com dados suficientes, o resultado de 2026-09-12 se INVERTE

[`train_hybrid.py`](train_hybrid.py), RTX 4070 (Ubuntu), 5 sementes, 2000
iterações. Treino em **256 trajetórias** com estado inicial aleatório; teste em
16 trajetórias **novas**, 100 passos (5× o horizonte de treino). Métrica
principal: **tempo de previsão válida** (passos até o erro de posição passar de
0,1). Parâmetros pareados (~4,4k).

| modelo | previsão válida | MSE dentro (trajetórias novas) | desvio \|v\| (mediana) | treino | divergências |
|---|---|---|---|---|---|
| hamiltoniano puro | **32,1 ± 7,4** | 0,00240 | 0,0338 | 82s | 0/80 |
| baseline sem física | 22,4 ± 4,9 | **0,00105** | 0,0503 | **19s** | 0/80 |
| híbrido, guarda aprendida | 23,3 ± 14,1 | 0,02240 | 0,0781 | 137s | 0/80 |
| híbrido-caixa (prior forte) | 34,2 ± 14,5 | 0,00181 | **0,0291** | 95s | 0/80 |
| piso congelado | 1,2 | 0,12434 | — | — | — |
| piso inercial (reta sem colisão) | 9,7 | 0,26830 | — | — | — |

**1. Hamiltoniano puro vs baseline: CONFIRMADO com 20 sementes.**

Com 5 sementes o sinal era sugestivo (4 de 5, t ≈ 2,0, p ≈ 0,11, Wilcoxon
p = 0,125). Com n = 5, porém, os testes exatos nem conseguiam chegar a
p < 0,05: o menor p possível é 0,0625, e com 4 vitórias o melhor é 0,125.
Confirmação a seguir, mesma configuração, só estes dois modelos, sementes 0–19:

| métrica | hamiltoniano | baseline | diferença | sementes a favor | teste pareado |
|---|---|---|---|---|---|
| previsão válida (passos) | **28,2 ± 7,0** | 21,8 ± 4,0 | **+6,3 (+29%)** | 16/20 | t = 3,22 (19 g.l.) p = 0,0045 · troca de sinal p = 0,005 · Wilcoxon p = 0,005 |
| desvio de \|v\| (mediana) | **0,033** | 0,051 | **−35%** | 17/20 | t = −5,20 p = 0,0001 · troca de sinal p = 0,0002 · Wilcoxon p = 0,0001 |
| MSE dentro do horizonte | 0,0028 | **0,0010** | 2,7× pior | — | — |
| custo de treino | 81s | **19s** | 4,3× mais caro | — | — |
| divergências | 0/320 | 0/320 | — | — | — |

Leitura: **com dados suficientes, a estrutura hamiltoniana prevê de forma
coerente por mais tempo** que um aluno do mesmo tamanho sem estrutura, mesmo
sendo menos precisa passo a passo e 4× mais cara de treinar. O mecanismo é
medido, não suposto: ela conserva melhor a velocidade, a fase deriva mais
devagar e o erro demora mais a passar do limiar. É exatamente a pergunta da
linha 3 da tabela experimental da reanálise ("a estrutura melhora coerência e
extrapolação?"), respondida afirmativamente **neste sistema**.

O efeito encolheu da estimativa de 5 sementes (+9,7) para a de 20 (+6,3),
como se espera de uma amostra pequena que por acaso saiu favorável.

**Limites do que isto prova:** sistema de brinquedo em 4 dimensões,
quase-conservativo e com colisão elástica, que é o caso MAIS favorável à
estrutura. Não há renderização, dissipação, cortes nem aparência. Não diz nada
ainda sobre vídeo real.

**A regra de comparação do script estava errada nos dois sentidos.** Com 5
sementes disse "acima do ruído" (p = 0,11); com 20 disse "dentro do ruído"
(p = 0,005). Ela comparava a diferença de médias com o desvio *entre* sementes
sem parear, e é o pareamento que remove a variação entre sementes. Foi
substituída por teste pareado de troca de sinal no `train_hybrid.py`.

**2. O "perde por 3 ordens de grandeza" (seção abaixo) era artefato de falta de
dados, não uma propriedade da estrutura.** Aquele experimento treinava numa
trajetória só: o potencial não tinha cobertura para aprender onde ficam as
paredes e divergia. Com 256 trajetórias, V aprende a parede como potencial
íngreme e o fluxo suave passa a rebater. Chamar aquele resultado de "decisivo"
foi exagero. A causa atribuída (descontinuidade) é real, mas o tamanho do
efeito dependia da escassez de dados.

**3. Híbridos (a pergunta original desta rodada): sem efeito detectável, e o
mecanismo de salto não foi realmente aprendido.**
- `hibrido_caixa` aprendeu limites entre 0,65 e 0,83, contra 0,55 verdadeiro.
  Quem produz o rebote é o potencial; o salto funciona como guarda-corpo
  ocasional. A média mais alta (34,2) não pode ser atribuída ao evento.
- A guarda aprendida é frágil: uma semente não aprendeu nada (1,2 passos,
  perda de treino 0,49), e o custo de treino é 7× o do baseline.
- Uma semente de cada híbrido ficou **individualmente abaixo do piso
  inercial** (1,2 e 8,9 passos). A guarda de validade do script só olha a
  média e não pegou isso.

### A primeira versão do `train_hybrid.py` foi inválida (mesma data)

Com 16 trajetórias de treino, gatilho de evento suave e MSE de horizonte longo
como métrica, **nenhum modelo bateu o piso trivial**. Três defeitos de desenho:
1. `p·(1−2w)` com w entre 0 e 1 encolhia |p| em vez de refletir. O
   `hibrido_caixa` dissipou energia e empurrou as paredes para fora. Corrigido
   com salto duro no forward e gradiente do sigmoide no backward
   (straight-through).
2. MSE de posição em 80 passos satura: erro pequeno de velocidade desloca a
   fase, e a previsão fica tão descorrelacionada quanto um palpite. Trocado
   pelo tempo de previsão válida, com piso inercial.
3. 320 transições não bastavam para generalizar a estados iniciais novos.
   Trocado para 256 trajetórias.

Nessa rodada inválida, o baseline **explodiu em 2 de 5 sementes** (MSE ~1e13)
em trajetórias novas, e os hamiltonianos ficaram limitados. Com 256 trajetórias
nenhum modelo divergiu, então esse sinal de estabilidade apareceu só no regime
de poucos dados.

## RESULTADO (2026-09-12, SUPERADO pela atualização acima): a estrutura hamiltoniana PERDE do baseline

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
