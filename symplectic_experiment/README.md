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

## RESULTADO (2026-09-14): com dissipação, a vantagem da estrutura DESAPARECE

[`train_dissipative.py`](train_dissipative.py), RTX 4070 (Ubuntu), 20 sementes, 2000
iterações, mesmo protocolo do `train_hybrid.py`: 256 trajetórias de treino,
16 novas de teste com 100 passos, parâmetros pareados (~4,5k). O disco agora
tem arrasto linear: a velocidade decai exp(−γ·dt) por passo, com γ = 0,15.

| modelo | previsão válida | desvio \|v\| (mediana) | treino | divergências |
|---|---|---|---|---|
| hamiltoniano conservativo | 18,6 ± 3,2 | 0,336 | 86s | 0/320 |
| **baseline sem física** | **35,7 ± 5,9** | **0,030** | **19s** | 0/320 |
| port-hamiltoniano (D aprendido) | 32,3 ± 11,0 | 0,041 | 95s | 0/320 |
| piso congelado / inercial | 1,2 / 9,4 | — | — | — |

Comparações pareadas (troca de sinal exata, confirmada por t pareado e Wilcoxon):

| par | previsão válida | desvio \|v\| |
|---|---|---|
| hamiltoniano vs baseline | baseline melhor, −17,1 passos, 0/20, p < 0,0001 | baseline melhor, 0/20, p < 0,0001 |
| port-hamiltoniano vs hamiltoniano | port melhor, +13,8 passos, 20/20, p < 0,0001 | port melhor, 20/20, p < 0,0001 |
| **port-hamiltoniano vs baseline** | **sem efeito**: −3,3 passos, 8/20, p = 0,30 | **baseline melhor**: 8/20, p ≈ 0,03 |

Leitura:
1. **A estrutura conservativa pura quebra com perda de energia.** Ela não tem
   como representar a desaceleração: desvio de velocidade 11× o do baseline.
2. **O termo dissipativo conserta a quebra** (+13,8 passos em 20/20 sementes), o
   que confirma a reformulação: impor conservação num sistema dissipativo é viés
   errado.
3. **Mas não sobra vantagem sobre o aluno sem estrutura.** O port-hamiltoniano
   empata na previsão válida, acompanha pior a velocidade, custa 5× mais para
   treinar e varia muito mais entre sementes (±11,0 contra ±5,9).
4. **γ aprendido superestima o atrito de forma sistemática:** média 0,192 contra
   0,15 verdadeiro, acima em todas as 20 sementes (0,157–0,278). As sementes com
   γ mais alto foram as piores (15,9 e 18,0 passos com γ entre 0,21 e 0,28).

**Somando com o caso conservativo:** a estrutura rendeu +29% (p = 0,005) no
sistema mais favorável a ela, e deixou de render assim que o sistema passou a
perder energia, que é o regime comum em vídeo. Sinal negativo para a tese nesta
escala.

**Reteste com o integrador corrigido (terceiro bug, abaixo): o empate se confirma.**
A corrida original usou o integrador em que o primeiro meio-passo de cada
rollout não propagava gradiente para V, um viés contra os modelos hamiltonianos.
Repetida com a correção, só port-hamiltoniano vs baseline, 20 sementes:

| | antes da correção | integrador corrigido |
|---|---|---|
| port-hamiltoniano, previsão válida | 32,3 ± 11,0 | **34,1 ± 11,1** |
| baseline, previsão válida | 35,7 ± 5,9 | 35,7 ± 5,9 (idêntico: corrida determinística) |
| port vs baseline, previsão válida | p = 0,30 | **p = 0,62**, 9/20 sementes |
| port vs baseline, desvio de \|v\| | baseline melhor, p ≈ 0,03 | baseline melhor, **p ≈ 0,02** (permutação 0,019) |
| γ aprendido (verdadeiro 0,15) | 0,192 | 0,191 |

A correção subiu o port-hamiltoniano em 1,8 passo, na direção prevista, sem
tirar do empate. O baseline continua melhor em velocidade, e o viés de atrito
não mudou. **Conclusão robusta: com dissipação, a estrutura não dá vantagem
sobre um aluno igualmente pequeno sem física.**

## CONCLUÍDO (2026-09-14): primeiro teste em vídeo real, no Ubuntu (RTX 4070). Sem ganho da estrutura

**Validação cruzada por famílias derrubou o resultado preliminar (6/6, p = 0,031)
descrito mais abaixo.** 5 dobras, cada família de conteúdo testada uma vez, 3
sementes por dobra, port-hamiltoniano vs baseline, MSE latente em 8 passos
(~2,7 s). Agregado com [`agregar_cv.py`](agregar_cv.py):

| dobra | família de teste | seqs | port-hamiltoniano | baseline | port vs baseline |
|---|---|---|---|---|---|
| 0 | 20260907_ltx_distilled + gguf2 | 76 | **0,695** | 0,766 | **+9,2%**, 3/3 |
| 1 | 20260912_primeiro_tour_msr | 39 | 0,660 | **0,619** | **−6,7%**, 0/3 |
| 2 | 5 corridas duplicadas 20260913_* | 30 | 0,702 (não bate o piso) | **0,637** | **−10,2%**, 0/3 |
| 3 | palácio + lyra + rt_infantil + teste_qualidade | 19 | 1,022 | 1,012 | −1,0%, 1/3 |
| 4 | beatriz_lucas + meili | 18 | **1,001** | 1,128 | **+11,2%**, 3/3 |

**Pares (dobra, semente): port melhor em 7/15, p = 0,44. Dobras: 2/5, p = 0,69.
Nenhuma vantagem geral.** O ganho aparece em algumas famílias e some ou se inverte
em outras. A corrida preliminar testou justamente beatriz_lucas, lyra e
teste_qualidade, e beatriz_lucas volta a favorecer o port na dobra 4.

**Bug no agregador, corrigido antes do número final:** a primeira versão excluía a
dobra quando QUALQUER modelo não batia o piso. Ela descartou a dobra 2, onde o
port falhou e o baseline não: viés de seleção a favor de quem falhou. Com essa
regra o agregado dava 7/12, p = 0,13. Agora a dobra só sai se NENHUM dos dois
bate o piso; se só um falha, conta como derrota dele. Com isso: 7/15, p = 0,44.

**Conclusão da linha de pesquisa, nesta escala (~332k parâmetros, 193 clipes):**

| sistema | estrutura hamiltoniana vs aluno igualmente pequeno sem física |
|---|---|
| brinquedo conservativo | **ajuda**: +29% de previsão válida, p = 0,005 |
| brinquedo dissipativo | conservativo quebra; port-hamiltoniano **empata** (p = 0,62) e custa 5× |
| latentes reais do LTX 2.5 | **não ajuda**: validação cruzada nula (p = 0,44), e em imagem o ranking nem batia com o latente |

A estrutura só rendeu no caso mais favorável a ela. No regime que o vídeo real
ocupa (perda de energia, cortes, conteúdo aberto) ela não superou um aluno do
mesmo tamanho sem restrições. A tese continua testável em escala maior ou com
outro estado, mas **não há evidência a favor dela aqui**.

Lições de método, todas medidas nesta linha:
- MSE latente premia previsão média; o ranking não bateu com o PSNR em imagem.
- Um conjunto de teste fixo pequeno produziu um "6/6, p = 0,031" que não
  generalizou.
- Corridas diferentes continham o mesmo conteúdo (similaridade 1,000); dividir
  por corrida vaza.
- Guardas de validade precisam ser simétricas, senão viram viés de seleção.

## Detalhes do teste em vídeo real (passos 0 a 3)

Máquina `100.99.13.105`, pasta `~/symplectic/vae_real/`. Nada disto usa a 3090
de produção.

**Passo 0: o VAE do LTX 2.5 roda fora do Windows. Passou.** Código do núcleo do
ComfyUI deste repositório (142 MB, sem custom nodes), Python 3.13, torch
2.11+cu128. [`vae_roundtrip.py`](vae_roundtrip.py) num clipe do LTX 2.5
(960x544, 121 quadros):

| medida | valor |
|---|---|
| latente | `(1, 128, 16, 17, 30)`: 128 canais, compressão temporal 8, espacial 32 |
| encode / decode | 5,2 s / 28,7 s |
| pico de VRAM | 5,2 GB |
| PSNR da reconstrução | **31,2 dB** médio (33,4 no 1º quadro, 27,9 no último, mínimo 26,8) |

Os 31,2 dB são a **régua**: nenhum preditor avaliado pelo decoder congelado
passa disso. O clipe de origem já tinha passado por este decoder e por h264, então
o número mede a perda de uma nova ida e volta.

Armadilha encontrada: chamar o VAE fora de `torch.inference_mode()` quebra o
decode em blocos com `Inplace update to inference tensor outside InferenceMode`.
O ComfyUI executa todo node dentro desse modo. O encode passou e só o decode
quebrou.

**Passo 1: dados. Concluído.** [`extract_latents.py`](extract_latents.py) nos 193
clipes do LTX 2.5 em 960x544 (21.249 quadros): **193 latentes, 2.632 passos de
dinâmica, 16 corridas, 353 MB, zero falhas**, em 20 min. O 1º quadro latente
(causal) é separado da sequência de dinâmica.

**Achado: dividir por corrida vazava conteúdo duplicado.** Similaridade de
cosseno do 1º latente de dinâmica (reduzido a 4x8); referência: mediana 0,37
dentro da mesma corrida, 0,19 entre corridas.
- Cinco corridas `20260913_*` (msr_g03, msr_g05, distilled_close, close_v2,
  webui_loras) são **o mesmo plano de 6 tomadas**: mesma lista de quadros por
  plano e similaridade **1,000** entre corridas.
- `palacio_esmeralda_v6_ltx` × `teste_decupagem_w4a8`: **0,989** (stills
  compartilhados).
- Eu suspeitava das duas `20260907_ltx_*` (mesmo roteiro, variantes diferentes)
  e **errei**: máxima 0,856, nenhum par acima de 0,95. Relacionadas, mas não
  cópias.

A divisão treino/teste passou a ser por **família de conteúdo**: união de
corridas com similaridade máxima acima de 0,8. É conservador de propósito e une
também as `20260907_ltx_*` e o `_lora_ic_test_palacio` (0,861).

**Passo 2: preditor.** [`train_latent_dynamics.py`](train_latent_dynamics.py):
baseline convolucional com atalho inercial vs hamiltoniano (V como campo escalar
convolucional, força −∇V) vs port-hamiltoniano, com parâmetros pareados (~332k),
pisos congelado e linear, e teste pareado por semente.

**Passo 3: régua em imagem.** [`decode_predictions.py`](decode_predictions.py)
decodifica os latentes previstos, separa o erro do preditor da perda do VAE e
monta vídeo lado a lado com os pisos.

### Resultado preliminar em latentes reais (`dinamica_v1`, 2026-09-14)

182 sequências com pelo menos 5 passos (2.588 passos). Contexto de 2 latentes,
treino com rollout de 3 passos e avaliação de 8 (~2,7 s). 6 sementes, 3.000
iterações, parâmetros pareados (~332k). Integrador já corrigido (terceiro bug,
abaixo).

| modelo | MSE latente normalizado (passos 1..8) | vs piso congelado | α aprendido | treino |
|---|---|---|---|---|
| **port-hamiltoniano** | **0,811 ± 0,023** | **0,71×** | 0,045 | 58s |
| baseline sem física | 0,870 ± 0,012 | 0,77× | 0,028 | 17s |
| hamiltoniano conservativo | 1,063 ± 0,035 | 0,94×, **reprovado** | 0,032 | 56s |
| piso congelado | 1,136 | — | — | — |
| piso linear | 9,648 | — | — | — |

- **Port-hamiltoniano vs baseline: −0,059 de MSE (−6,8%) em 6 de 6 sementes,
  sem sobreposição** (0,771–0,841 contra 0,852–0,883). Troca de sinal p = 0,031,
  que é o **menor p possível** com 6 sementes.
- **O conservativo repete o padrão do brinquedo dissipativo:** acompanha no curto
  prazo e diverge no longo (1,70 contra 1,57 do piso congelado no passo 8). O
  amortecimento aprendido é o que estabiliza.
- **O α de 0,03 a 0,05** confirma que quase nada da "velocidade" latente
  persiste a 1/3 s por passo, coerente com o piso congelado ser 8× melhor que o
  linear.

**Limites e um bug na divisão:**
1. **Bug:** as listas de treino e teste comparavam o nome da corrida com o nome
   da família. Famílias unidas nunca entravam no teste. O teste real teve **13
   sequências de 3 famílias** (beatriz_lucas, lyra, teste_qualidade_20260912),
   não as 4 impressas no log. **Não houve vazamento:** todas as famílias unidas
   ficaram no treino. Corrigido em `dividir`.
2. **Variância de conteúdo não medida.** As sementes variam só o treino; o teste
   é fixo e pequeno. O efeito pode ser específico desses 13 clipes. Próximo
   passo: validação cruzada por famílias (`--n-folds`).
3. **Em imagem, o ranking do latente NÃO se confirmou nas amostras decodificadas.**
   `decode_predictions.py`, semente 0, 2 sequências de teste (beatriz_lucas,
   planos 0 e 1), 8 latentes previstos (2,7 s). PSNR contra o latente real
   decodificado:

   | | seq 0 | seq 1 |
   |---|---|---|
   | piso congelado | **15,55** | 14,01 |
   | baseline | 13,85 | **15,43** |
   | port-hamiltoniano | 13,48 | 14,42 |
   | hamiltoniano | 12,98 | 14,19 |
   | piso linear | 10,98 | 11,62 |
   | teto (VAE vs original) | 29,0 | 33,5 |

   O baseline supera o port-hamiltoniano nas duas sequências, e na seq 0 o
   congelado supera todos os modelos. Tudo fica em 13–15 dB, perto do nível de
   imagens sem relação: **a 2,7 s de horizonte nenhum preditor gera vídeo útil**,
   e as diferenças são entre modos de falhar. Explicação provável: o MSE sobre
   latente normalizado pesa os 128 canais igualmente e premia previsões "médias",
   que decodificam borradas ou erradas. A amostra é fraca (2 sequências, 1
   semente, 1 família), mas basta para **não** afirmar ganho da estrutura em vídeo
   real só pelo MSE latente. É a mesma armadilha do MEMORIAL §3.18: métrica
   plausível sem inspeção visual. Para decidir, a métrica em imagem precisa cobrir
   o conjunto de teste inteiro.
4. **Custo:** o port-hamiltoniano treina 3,4× mais devagar que o baseline.

### Terceiro bug no integrador (2026-09-14)

`SymplecticIntegrator._force` usava `create_graph=q.requires_grad`. No primeiro
meio-passo de cada rollout, q vem dos dados sem gradiente, e aquela força não
propagava gradiente para os parâmetros de V. O que decide se o grafo é preciso é
estar treinando: agora `create_graph=torch.is_grad_enabled()`. Os resultados de
`train_hybrid.py` e `train_dissipative.py` anteriores à correção perderam 1 de
2K avaliações de força por rollout. O viés é **contra** o hamiltoniano, então não
infla o resultado positivo abaixo.

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
