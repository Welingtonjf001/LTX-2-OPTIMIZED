# Guia do roteiro — o que indicar para o pipeline filmar o que você escreveu

Cada plano é gerado **sem memória** dos outros. O que o texto não diz, o modelo inventa, e
inventa diferente em cada plano. No CERCO EM SEUL (MEMORIAL §3.132) o filme saiu com cara de
thriller, mas não dava para entender quem protegia quem, quem fugia nem como a ameaça
evoluía. A maior parte dos erros nasceu do texto, antes do vídeo.

A auditoria `screenplay_gaps.py` lê o roteiro já estruturado e lista o que falta em
`parse/lacunas.md`. Este guia é a versão preventiva: escreva assim e o relatório sai limpo.

## 1. Lugar

- **Cabeçalho com lugar nomeável e período:** `EXT. AVENIDA EM SEUL, ENTRADA DO HOTEL — DIA`.
  Sem isso, cenas no mesmo lugar não são reconhecidas como o mesmo lugar.
- **Geografia fixa, uma vez por locação:** onde fica a entrada, a rua, o prédio do outro
  lado, a janela alta. Os planos seguintes se orientam por esses pontos. "Janela no quinto
  andar do prédio em frente" é filmável; "uma janela" não é.

## 2. Todo mundo que aparece — inclusive quem não fala

- Apresente **cada pessoa na primeira aparição**, entre parênteses: idade, cabelo, roupa
  com cores e um item distintivo. `O PRESIDENTE (68 anos, cabelo grisalho penteado, terno
  azul-marinho, gravata vermelha, broche de bandeira)`.
- **Figurante que age é elenco.** O Presidente, o motociclista e o atirador agem em metade
  dos planos. Sem descrição, cada plano desenhou outra pessoa.
- **Use sempre o mesmo nome para a mesma pessoa.** "O motociclista" que depois vira "o
  terrorista" é uma pessoa só para o leitor. Diga isso uma vez: `O MOTOCICLISTA (o
  terrorista)`.
- **Gênero explícito** quando o nome não diz: "Ha-eun, atlética, cabelo preso" não informa.
  Escreva "ela". Sem isso, a voz foi escolhida por palpite e saiu masculina.
- Figurino ou estado que **muda**: diga quando e como (tira o capacete, fica molhado, fere o
  braço).

## 3. Ação: quem faz o quê com quem, e como termina

Uma ação por frase, com **agente** (nome), **verbo físico**, **alvo** e **resultado**.

| Em vez de | Escreva |
|---|---|
| "Ela o derruba." | "Ha-eun derruba o terrorista sobre o capô do táxi." |
| "Ji-ho protege." | "Ji-ho joga o Presidente no asfalto e cobre o corpo dele com o próprio corpo." |
| "Seo-yeon atira." | "Seo-yeon atira para o alto enquanto recua." |
| "A placa cai." | "A bala solta a placa de metal do terceiro andar; ela despenca na direção do Presidente." |

- **Posição e direção:** quem está à esquerda de quem, a que distância, para onde se move
  (para a porta, para a câmera, para o outro lado da rua).
- **Objetos:** quem segura o quê e em que mão; quando passa de mão ou cai.
- **Estado que fica:** o carro destruído, as portas de aço fechadas, o terrorista algemado.
  Os planos seguintes precisam saber (o relatório lista isso em "Continuidade a manter").

## 4. Causa, ação, reação, consequência — sem reiniciar o evento

- Cada acontecimento aparece **uma vez**. Se a fala acontece durante a ação, não repita a
  ação na rubrica da fala: diga a **reação** ou a **consequência**. MEDIDO: cinco falas do
  CERCO repetiam a ação do plano anterior, e o evento recomeçava na tela.
- **Fala durante ação física:** um close mostra o rosto, não o corpo. Diga se a fala é
  **voz sobre a ação** ou uma **reação logo depois**. A decupagem filma o close como reação
  e a ação precisa de um plano próprio.
- **Tempo:** "enquanto", "logo depois", "ao mesmo tempo". Sem isso, a ordem vira palpite.

## 5. Fala e atuação

- Quem fala, **para quem**, e o que faz enquanto fala, cabendo no rosto: olhar, respiração,
  mandíbula. Gesto de corpo tira a boca do quadro.
- **Atuação separada da emoção:** "calma" numa perseguição virou um rosto sereno e "alegre"
  virou sorriso durante a derrubada. Indique a intensidade e o esforço: "ofegante, urgente",
  "alívio contido depois da algema".

## 6. Direção de arte e som em linhas separadas

- **ESTILO VISUAL:** o meio (live action, animação 3D, anime), a luz e a época. O parse
  extrai essa linha para a direção de arte, que vai para todo prompt de imagem.
- **SOM:** sirenes, vidro quebrando, tiros, numa linha própria. O parse não separa essa
  linha: ela fica no texto de ação, que o enriquecimento lê como contexto. O que importa é
  **não misturar som no ESTILO VISUAL**. Misturado, ia para todo plano e punha tiro e vidro
  onde não havia nenhum dos dois. A decupagem hoje filtra som e câmera da direção de arte
  (`shot_plan._arte_visual`), mas escrever separado evita depender do filtro.
- **Geografia** também é texto de ação lido pelo enriquecimento. Não é um mapa 3D:
  continuidade espacial com geometria real exige o spec de `SPATIAL_PIPELINE.md`.

## Modelo de cena

```
EXT. AVENIDA EM SEUL, ENTRADA DO HOTEL — DIA
ESTILO VISUAL: live action contemporâneo, luz dura de meio-dia.
SOM: sirenes, multidão, trânsito.
Geografia: hotel à esquerda, avenida ao centro, prédio espelhado do outro lado com janelas
altas.

O PRESIDENTE (68 anos, cabelo grisalho penteado, terno azul-marinho, gravata vermelha) sai
do hotel em direção ao sedã preto blindado, parado na calçada. SEO-YEON (35 anos, ela,
cabelo castanho na altura dos ombros, terninho preto risca de giz, óculos escuros, ponto no
ouvido) caminha à esquerda dele; JI-HO (40 anos, corte militar, terno cinza com colete à
prova de balas) à direita.
Um MOTOCICLISTA (o terrorista; 30 anos, jaqueta de couro preta, capacete fosco, mochila
cinza) para ao lado do sedã, larga a mochila junto à roda traseira e some entre os pedestres.
Seo-yeon vê uma luz vermelha piscando na mochila.

                    SEO-YEON
          (urgente, sem fôlego)
     Presidente, ao chão!

Seo-yeon e Ji-ho jogam o Presidente no asfalto. O sedã explode: a bola de fogo atravessa a
entrada do hotel e os vidros despencam. Ji-ho fica sobre o Presidente, cobrindo-o.
```

## O que o pipeline faz com isso

- `parse/lacunas.md`: o que ainda falta, por plano (sempre gerado; `--lacunas bloquear` para
  a decupagem se houver lacuna crítica).
- `shots/complexity_report.json`: onde texto e still bastam, onde revisar o blocking 2D e
  onde vale um previs 3D de baixa resolução (contato entre corpos, causa e efeito,
  objeto trocando de mão), com o limite de cada motor. Ver `SISTEMA_VIDEO.md`.
