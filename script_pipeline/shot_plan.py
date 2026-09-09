"""Decupagem: transforma cenas + estrutura dramática em uma lista de PLANOS,
cada um com câmera concreta.

A camada que faltava. `parse_screenplay` produz `shot_list`, mas ele só ORDENA
conteúdo -- `{"type":"dialogue","line_index":3}` -- sem dizer como cada coisa é
filmada. Nenhum campo de enquadramento, ângulo ou movimento existia em todo o
caminho até o prompt. O modelo recebia o que está diante da câmera e nunca onde
a câmera está, então caía sempre no default dele: altura dos olhos, plano
médio-aberto. Daí a sensação de ausência de direção.

A RESTRIÇÃO QUE DEFINE ESTE DESENHO

MEDIDO 2026-08-26 (MEMORIAL.md §3.21): o LTX obedece a vocabulário de câmera,
mas **como destino, não como estado**. Pedindo close-up, o frame 0 sai em plano
geral e só o frame 96 chega ao close-up -- ele parte do próprio default e deriva
até a especificação ao longo do clipe.

Duas consequências que mandam na arquitetura:

  1. Enquadramento **não pode** vir do prompt de vídeo. Um plano de 3 segundos
     nunca chega ao destino. Tem que vir da IMAGEM de condicionamento.
  2. O prompt de vídeo descreve o MOVIMENTO, não o enquadramento -- é a única
     coisa que ele controla bem, porque movimento É uma trajetória.

Por isso cada plano sai daqui com DOIS prompts: `storyboard_prompt` (para o
still, que carrega o enquadramento) e `video_prompt` (para o LTX, que carrega o
movimento). Separar os dois é o ponto inteiro deste módulo.

O QUE É DETERMINÍSTICO AQUI

Tudo. Não há LLM neste arquivo. Padrão de cobertura vem da função dramática
(que `story_structure` já anotou), duração vem do perfil de estilo e da tensão,
lado de tela vem da regra dos 180 graus. Estilo, aqui, é um JSON de parâmetros
-- não o nome de um diretor: nome de diretor num prompt de difusão é
condicionamento fraco e não reprodutível, enquanto "câmera travada, corte no
fim da fala, plano médio dominante" é executável.

CLI:
    python -m script_pipeline.shot_plan --run RUN_DIR [--style classico]
    python -m script_pipeline.shot_plan --scenes s.json --structure e.json --style tenso
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from pathlib import Path as pathlib_Path

# --------------------------------------------------------------------------
# vocabulário de plano
# --------------------------------------------------------------------------
# `framing` vai para a IMAGEM; `movement` vai para o VÍDEO. Ver docstring.
FRAMINGS = {
    "wide":     "extreme wide establishing shot, the figure small within a vast frame",
    "full":     "full shot, the whole figure visible head to feet",
    "medium":   "medium shot, framed from the waist up",
    "medium_2": "medium two shot, both figures framed from the waist up",
    # MEDIDO 2026-08-26 (MEMORIAL 3.24): o texto anterior era "the listener's
    # shoulder in the foreground" -- dizia de quem era o OMBRO e nunca de quem
    # era o ROSTO, entao o FLUX escolhia, e escolheu a NUCA justamente na fala
    # mais importante da cena. Over-the-shoulder e um plano de quem esta sendo
    # OLHADO: o rosto no quadro e o do `subject`, e o ombro em primeiro plano e
    # do interlocutor, de costas. Os dois papeis agora estao nomeados -- aqui o
    # de costas, e em SUBJECT_HINTS o de frente.
    "ots":      ("over-the-shoulder shot, framed past the out-of-focus back of another "
                 "person's shoulder and upper arm in the near foreground, that person "
                 "seen from behind with their face away from the camera"),
    "close":    "close-up, the face filling most of the frame",
    # Extreme close-up existe SEPARADO do close porque são gestos diferentes:
    # close mostra a expressão inteira, este isola o olhar. Vale só quando a
    # cena tem o que sustentar nele -- usado à toa vira maneirismo.
    "extreme_close": "extreme close-up on the eyes, only the eyes and brow fill the frame",
    "insert":   "tight insert on the detail, no face in frame",
}
# Complemento do enquadramento, colado na clausula do SUJEITO. Existe so onde o
# enquadramento e ambiguo sobre QUEM aparece: no ots ha duas pessoas no quadro e
# uma delas esta de costas, entao dizer qual e qual e a diferenca entre um plano
# de conversa e uma nuca.
# Enquadramentos que NAO sustentam um rosto falando: os abertos demais, onde o
# rosto nao tem pixel (CLAUDE.md 3.17), e o inserto, que por definicao nao tem
# rosto no quadro (ver SUBJECT_HINTS). Ver _cobertura_de_fala.
SEM_ROSTO = ("wide", "full", "insert")


def _cobertura_de_fala(cobertura: list) -> list:
    """A escada de enquadramentos que serve a um plano de FALA.

    MEDIDO 2026-08-28 no run Lyra: as duas falas do Thoren sairam `wide`, com o
    storyboard dizendo "the figure small within a vast frame" -- para um plano
    de dialogo. O CLAUDE.md 3.17 mede a fisica disso: a 768x512 em quadro
    aberto um rosto ocupa 1,6 x 1,8 celulas latentes e derrete. Nao ha prompt
    que resolva.

    Filtra a cobertura do estilo, mantendo a ORDEM e a intencao dele, e garante
    pelo menos dois degraus -- com um so, o plano-contraplano vira quatro
    planos identicos, que foi o segundo defeito da mesma cena."""
    escada = [c for c in cobertura if c not in SEM_ROSTO]
    if not escada:
        return ["medium", "close"]
    if len(escada) == 1:
        return escada + (["close"] if escada[0] != "close" else ["medium"])
    return escada


# O plano de estabelecimento NAO usa FRAMINGS["wide"]: aquele texto diz "the
# figure small within a vast frame" e pede uma figura, enquanto este plano existe
# justamente para mostrar o LUGAR antes de haver alguem nele.
ESTABELECIMENTO_FRAMING = ("extreme wide establishing shot of the whole location, "
                           "taking in the space and its depth, no one in the foreground")
SUBJECT_HINTS = {
    "ots": "facing the camera in the mid-ground, their face fully visible and in focus",
}
ANGLES = {
    "eye":  "",                     # default do modelo; não gastar prompt com isso
    "low":  "low angle, the camera near the ground looking up",
    "high": "high angle, the camera above looking down",
}
MOVEMENTS = {
    "static": "",                   # ausência é o que mais aproxima de câmera travada
    "push":   "the camera pushes in slowly",
    "pull":   "the camera pulls back slowly",
    "orbit":  "the camera orbits slowly around the subject",
    "handheld": "handheld camera, subtle unsteady movement",
    # NOVO 2026-09-04 -- catalogo trazido pelo usuario (video de referencia com
    # vocabulario tipo "/dollyin", "/orbit", "/steadicam" etc.). Mesma forma dos
    # 5 originais: chave curta -> frase que entra direto no prompt de video. Nao
    # sao parametro estrutural de nenhum motor (LTX/MiniMax nao tem API de
    # camera) -- e so texto testado, igual ao resto. "dollyin" nao virou entrada
    # nova porque já é `push` com outro nome; manter os dois seria dois textos
    # pro mesmo efeito.
    "steadicam": "smooth steadicam movement, gliding alongside the subject",
    "crane_up": "the camera cranes upward, rising away from the subject",
    "crane_down": "the camera cranes downward, descending toward the subject",
    "zoom_in": "the lens zooms in, tightening the frame without the camera moving",
}

# Vocabulario de LUZ, mesma ideia do MOVEMENTS mas para lighting -- ADITIVO ao
# `look` de cada estilo (STYLES[...]["look"]), nao substituto: o `look` carrega
# o humor geral do estilo ("warm soft light" do intimista), e a tag de luz aqui
# e um acento pontual que o LLM so deveria escolher quando o plano especifico
# pede algo que o look de base nao ja diz.
LIGHTING = {
    "": "",
    "rimlight": "rim lighting, subject silhouetted against a bright backlight",
    "volumetric": "volumetric light, visible light rays cutting through haze or smoke",
    "golden_hour": "warm golden hour sunlight, long soft shadows",
    "neon": "neon-lit, saturated color reflections, night city glow",
}

# Efeitos que NAO SAO PROMPT DE GERACAO -- sao instrucao para o POS-processamento
# (segurar o ultimo frame / cortar com transicao) porque nem LTX nem MiniMax tem
# como "congelar no meio" ou "cortar com whip pan" dentro de uma unica geracao;
# isso e propriedade de como o CLIPE PRONTO e usado depois. `freeze` esta
# implementado (render_shots.py segura o ultimo frame do clipe via ffmpeg
# tpad). `whip` fica so REGISTRADO no plano por enquanto -- aplicar de verdade
# exigiria trocar o concat-demuxer (stream-copy, rapido) do assemble_final.py
# por filter_complex com re-encode so nos pares marcados, o que muda a
# arquitetura daquele estagio; nao fiz enquanto nao for pedido.
POST_EFFECTS = {"freeze", "whip"}

# Subconjunto de MOVEMENTS/LIGHTING/POST_EFFECTS que o enriquecimento por LLM
# (enrich_camera_style) pode escolher DENTRO de cada estilo -- nunca a lista
# inteira. Existe pra nao deixar o LLM colocar "/freeze" numa cena de dialogo
# calmo so porque achou o efeito interessante: a permissao já vem cortada pelo
# estilo que a pessoa escolheu antes de qualquer chamada de LLM.
CAMERA_STYLE_VOCAB = {
    "classico":      {"movements": ["static", "push", "pull"], "lighting": [], "post": []},
    "tenso":         {"movements": ["static", "handheld"], "lighting": ["rimlight"], "post": ["whip"]},
    "nervoso":       {"movements": ["handheld", "zoom_in", "static"], "lighting": ["neon"], "post": ["whip", "freeze"]},
    "intimista":     {"movements": ["static", "push", "steadicam"], "lighting": ["rimlight", "golden_hour"], "post": []},
    "contemplativo": {"movements": ["orbit", "push", "pull", "crane_up", "crane_down"], "lighting": ["volumetric", "golden_hour"], "post": []},
}

# --------------------------------------------------------------------------
# perfis de estilo
# --------------------------------------------------------------------------
# Seis parâmetros bastam para separar famílias reconhecíveis. Escritos à mão de
# propósito: para meia dúzia de perfis, RAG sobre texto de cinema devolveria
# prosa SOBRE estilo, e o que a decupagem consome é parâmetro.
STYLES = {
    "classico": {
        "descricao": "decupagem invisível: cobertura convencional, câmera discreta",
        "shot_seconds": 4.5, "tension_speedup": 0.5,
        "movements": {"wide": "static", "medium": "push", "close": "static",
                      "ots": "static", "insert": "static", "full": "static",
                      "medium_2": "static"},
        "angle_bias": "eye",
        "look": "natural cinematic lighting, balanced composition",
    },
    "tenso": {
        "descricao": "thriller travado: quadro simétrico, quase sem movimento, corte seco",
        "shot_seconds": 3.5, "tension_speedup": 0.6,
        "movements": {k: "static" for k in FRAMINGS},
        "angle_bias": "eye",
        "look": "desaturated cool palette, deep shadows, crushed blacks, symmetrical framing",
    },
    "nervoso": {
        "descricao": "handheld: câmera na mão, planos curtos, energia",
        "shot_seconds": 2.5, "tension_speedup": 0.7,
        "movements": {k: "handheld" for k in FRAMINGS},
        "angle_bias": "eye",
        "look": "naturalistic available light, slight grain",
    },
    "intimista": {
        "descricao": "close e olhar: rosto domina o quadro, luz quente, camera quase parada",
        "shot_seconds": 5.5, "tension_speedup": 0.7,
        "movements": {k: "static" for k in FRAMINGS} | {"close": "push", "medium": "push"},
        "angle_bias": "eye",
        "look": "warm soft light, shallow depth of field, intimate close framing",
        # Este estilo SOBREPÕE a cobertura por função: intimismo não é só
        # iluminação, é a decisão de ficar perto mesmo quando a estrutura
        # pediria um plano aberto.
        "coverage": {
            "estabelecimento": ["medium", "close", "close", "ots"],
            "desenvolvimento": ["close", "ots", "close", "medium"],
            "virada":          ["close", "extreme_close", "close"],
            "climax":          ["extreme_close", "close", "close"],
            "respiro":         ["close", "medium"],
            "desfecho":        ["close", "medium"],
        },
    },
    "contemplativo": {
        "descricao": "planos longos e abertos, movimento lento, poucas trocas",
        "shot_seconds": 8.0, "tension_speedup": 0.85,
        "movements": {k: "orbit" for k in FRAMINGS} | {"wide": "push", "insert": "static"},
        "angle_bias": "eye",
        "look": "wide vistas, soft natural light, unhurried",
    },
}

# Cobertura por função dramática. É aqui que a estrutura vira direção: uma
# virada não se cobre como um respiro. `story_structure` anota a função; se ela
# faltar, cai no padrão de desenvolvimento, que é o mais neutro.
COVERAGE = {
    "estabelecimento": ["wide", "medium"],
    "desenvolvimento": ["medium", "ots", "ots"],
    "virada":          ["medium", "close", "close", "insert"],
    "climax":          ["close", "close", "medium_2"],
    "respiro":         ["wide", "full"],
    "desfecho":        ["medium", "wide"],
}
DEFAULT_COVERAGE = COVERAGE["desenvolvimento"]


# --------------------------------------------------------------------------
def assign_screen_sides(characters: list) -> dict:
    """Regra dos 180 graus: cada personagem ganha um lado de tela FIXO.

    ATENÇÃO: a atribuição é GLOBAL, feita uma vez para o filme inteiro, não por
    cena. MEDIDO 2026-08-26: atribuindo pela ordem de aparição DENTRO de cada
    cena, LYRA saía à esquerda na cena 1 e à direita na cena 5, porque na 5
    quem fala primeiro é ORION. Isso é violação de eixo entre cenas -- o
    espectador perde a geografia justamente no retorno ao mesmo cenário, que é
    onde ele mais espera reconhecê-la.

    É o "posicionamento correto dos personagens" -- se A olha da esquerda para a
    direita num plano, no contraplano B tem de olhar da direita para a esquerda,
    ou o espectador perde a geografia. Isto é contabilidade, não interpretação:
    não há por que perguntar a um modelo.

    Vale a ressalva honesta: o LTX não tem cena 3D, então isto é uma DICA no
    prompt, não uma garantia. Melhora a chance, não assegura o resultado."""
    lados = ["left", "right"]
    return {nome: lados[i % 2] for i, nome in enumerate(characters)}


def appearance_only(descriptor: str) -> str:
    """Extrai só a APARÊNCIA de um descritor do cast.

    O descritor vem do texto de ação do roteiro e mistura aparência com o que a
    pessoa está fazendo: "XIAO-LAN (early 20s, graceful, wearing simple
    mint-green silk robes), hurries through the moon-gate archway, clutching her
    skirts". Coladas num prompt de still, essas ações BRIGAM com a ação do
    próprio plano -- o still do close dela ficaria tentando também atravessar um
    arco. Aqui ficam só os dois portadores de aparência que o formato de roteiro
    garante: o parêntese de apresentação e a oração "wearing/vestindo".

    Fora desses dois padrões devolve o descritor inteiro: perder aparência é
    pior que carregar um pouco de ação junto."""
    if not descriptor:
        return ""
    pedacos = []
    m = re.search(r"\(([^)]{3,200})\)", descriptor)
    if m:
        pedacos.append(m.group(1).strip())
    # A oração de figurino para em ")", "." ou fim -- nesta ordem. Sem o ")" ela
    # atravessava o parêntese e arrastava a ação junto ("...robes), hurries
    # through the moon-gate archway, clutching her skirts"). E se o figurino JÁ
    # está dentro do parêntese, não se repete: os dois trechos seriam colados um
    # ao lado do outro no mesmo prompt.
    ja_tem = bool(m) and re.search(r"wearing|dressed in|vestindo",
                                   m.group(1), re.IGNORECASE)
    if not ja_tem:
        w = re.search(r"\b(wearing|dressed in|vestindo)\b([^).]{3,220})",
                      descriptor, re.IGNORECASE)
        if w:
            # SEM parentese de apresentacao, o que vem ANTES do "wearing" nao e
            # acao -- e o resto da aparencia. MEDIDO 2026-08-27: o descritor
            # "Young female samurai with black ponytail, wearing lacquered red
            # armor and a white scarf" chegava ao prompt como so "wearing
            # lacquered red armor and a white scarf": idade, genero e cabelo
            # descartados, justamente as ancoras de identidade que a secao 3.24
            # mostrou serem o que impede a personagem de virar outra a cada
            # plano. COM parentese o prefixo e o nome mais acao, e ali o corte
            # continua valendo.
            inicio = w.start() if m else 0
            pedacos.append(descriptor[inicio:w.end()].strip().rstrip(",;"))
    return ", ".join(pedacos) if pedacos else descriptor.strip()


def _character_in(texto: str, personagens: list) -> str:
    """Primeiro personagem nomeado no texto de ação, se houver.

    Compara sem caixa e sem hífen, porque o cue vem em maiúsculas com hífen
    ("MEI-LI") e a ação escreve como nome próprio ("Mei-Li" ou "Mei Li")."""
    def norm(x: str) -> str:
        return "".join(c for c in x.lower() if c.isalnum())
    alvo = norm(texto or "")
    for nome in personagens:
        if norm(nome) and norm(nome) in alvo:
            return nome
    return ""


# Velocidade de fala usada como proxy de duração. ~14 caracteres por segundo é
# uma taxa confortável em português; serve só para dimensionar o plano, já que a
# duração real vem depois do TTS.
CHARS_PER_SECOND = 14.0

# Respiro somado a duracao do TTS: o plano precisa comecar um pouco antes da
# fala e terminar um pouco depois, ou o corte cai em cima da ultima silaba.
FALA_FOLGA_S = 0.6


def shot_seconds(style: dict, tension: float | None, framing: str,
                 *, text: str = "", is_action: bool = False) -> float:
    """Duração do plano. Tensão alta encurta -- é o mecanismo de ritmo mais
    barato que existe aqui, porque é só `num_frames` na geração.

    Insert é sempre curto; wide sempre um pouco mais longo, porque quadro aberto
    precisa de tempo para ser lido."""
    base = float(style["shot_seconds"])
    if tension is not None:
        # tensão 0 -> duração cheia; tensão 1 -> duração * tension_speedup
        base *= 1.0 - (1.0 - float(style["tension_speedup"])) * float(tension)
    # O CONTEÚDO manda no tempo, não só o estilo. MEDIDO 2026-08-26: sem isto,
    # os dez planos da cena saíam todos com 5,33s -- metrônomo, não montagem.
    # Uma fala de três palavras e uma de trinta não ocupam o mesmo plano.
    if text:
        alvo = len(text) / CHARS_PER_SECOND
        # Fator limitado: a fala dimensiona o plano, não o sequestra.
        base *= max(0.55, min(1.9, alvo / max(base, 0.1)))
    elif is_action:
        base *= 0.65     # gesto é mais curto que fala
    if framing == "insert":
        base *= 0.6
    elif framing == "extreme_close":
        base *= 0.8      # o olhar isolado cansa antes que o rosto inteiro
    elif framing == "wide":
        base *= 1.2
    return round(max(1.5, base), 2)


def frames_for(seconds: float, fps: float) -> int:
    """O modelo só aceita 1 + múltiplo de 8."""
    return 1 + max(1, round((seconds * fps - 1) / 8)) * 8


def _storyboard_prompt(*, framing: str, angle: str, subject: str, location: str,
                       time_of_day: str, look: str, descriptor: str,
                       screen_side: str | None, interior: str = "",
                       pose: str = "") -> str:
    """Prompt do STILL. Carrega enquadramento e ângulo -- que o vídeo não
    consegue estabelecer no frame 0.

    `pose`: MEDIDO 2026-09-08 (usuário assistiu ao filme e apontou) -- até
    aqui este prompt não carregava NADA de ação/gesto, só aparência e
    enquadramento; o still saía sempre um retrato neutro parado, e o vídeo
    tinha que migrar sozinho, sem pose inicial compatível, pro gesto que o
    `_video_prompt` pede (`action`/`beat_visual`). `pose` é o MESMO texto de
    ação já calculado pelo chamador (não duplica enriquecimento) -- aqui
    entra encurtado e fraseado como um instante congelado, não a ação
    inteira em curso (que é o que o vídeo descreve)."""
    partes = [FRAMINGS[framing]]
    if ANGLES.get(angle):
        partes.append(ANGLES[angle])
    if subject:
        clausula = f"{subject}{', ' + descriptor if descriptor else ''}"
        if SUBJECT_HINTS.get(framing):
            clausula += f", {SUBJECT_HINTS[framing]}"
        partes.append(clausula)
    pose_curta = (pose or "").strip().rstrip(".")
    if pose_curta:
        # Recorte na primeira frase -- o still e um instante, nao a acao
        # inteira (que pode cobrir varios segundos de movimento).
        pose_curta = re.split(r"(?<=[.!?])\s+", pose_curta)[0].rstrip(".")
        partes.append(f"caught mid-gesture: {pose_curta}")
    if screen_side and framing in ("ots", "close", "medium"):
        partes.append(f"positioned on the {screen_side} of the frame, looking across it")
    # MEDIDO 2026-08-27: "CORREDOR CURVO COLORIDO" sem o INT. do cabecalho virou
    # uma estrutura pintada AO AR LIVRE no still. O marcador de interior/exterior
    # esta no cabecalho e nao chegava ao prompt -- e "corredor" sozinho nao
    # obriga o modelo a ficar dentro de nada.
    cenario = ", ".join(p for p in (interior, location, time_of_day) if p)
    if cenario:
        partes.append(cenario)
    if look:
        partes.append(look)
    return ". ".join(partes) + "."


# Slug de emocao -> como isso SE VE. A tabela e visual de proposito: o LTX nao
# sabe o que "com_medo" quer dizer, sabe o que olhos arregalados querem dizer.
# Os 17 slugs sao os de `speakers/01_vozes_emotivas`.
EMOCAO_VISIVEL = {
    "raiva":                    "jaw clenched, brow furrowed, sharp forceful gestures",
    "surpresa":                 "eyes widening, head pulling back, eyebrows lifting",
    "desanimo":                 "shoulders dropping, gaze falling, movements slowing",
    "angustia":                 "face tightening, restless hands, shallow quick breathing",
    "tristeza":                 "eyes lowered, mouth drawn, movements heavy and slow",
    "desapontamento":           "gaze dropping away, a small resigned shake of the head",
    "espontanea_entusiasmada":  "face lighting up, quick animated gestures, leaning forward",
    "com_medo":                 "eyes wide and darting, body tensing, breathing fast",
    "alegre":                   "open smile, light quick movements, bright open posture",
    "apaixonada":               "softened gaze held steady, gentle unhurried movement",
    "sensual":                  "slow deliberate movement, gaze held, relaxed shoulders",
    "desdem":                   "chin lifting, a dismissive turn of the head, cool stare",
    "confusa":                  "brow knitting, head tilting, searching glances",
    "excitada":                 "restless energy, quick bright gestures, leaning in",
    "calma":                    "steady gaze, unhurried breathing, still composed posture",
    "neutra":                   "",
    "grito":                    "mouth wide, neck tensed, whole body thrown into the shout",
}


def emocao_visivel(emocao: str | None) -> str:
    """Traducao visual do slug de emocao, para o prompt do video.

    MEDIDO 2026-08-28: a emocao era decidida por fala e usada SO pelo TTS. O
    prompt do video nao a recebia, entao o modelo animava uma pessoa neutra
    dizendo uma fala de panico. Aceita texto livre tambem (parentese do
    roteiro), procurando um slug dentro dele; sem correspondencia devolve vazio
    e nada muda."""
    t = (emocao or "").strip().lower()
    if not t:
        return ""
    if t in EMOCAO_VISIVEL:
        return EMOCAO_VISIVEL[t]
    for slug, visual in EMOCAO_VISIVEL.items():
        if slug in t:
            return visual
    return ""


def _video_prompt(*, action: str, movement: str, descriptor: str, look: str,
                  quote: str | None, subject: str = "", fallback: str = "",
                  emotion: str | None = None) -> str:
    """Prompt do VÍDEO. Descreve MOVIMENTO e ação -- nunca enquadramento, que
    já está fixado pela imagem de condicionamento.

    `quote` (o texto literal da fala) é OPCIONAL e desligado por padrão.
    MEDIDO: `render_scenes.py` injeta `saying: "<texto>"` no prompt, e é esse
    texto que produz LEGENDA QUEIMADA em clipe curto -- o clipe de 19,3s sem
    fala citada saiu limpo. Como o pipeline faz TTS e lip-sync depois, a fala
    citada não é necessária para o resultado final; ela só melhora a
    articulação da boca ANTES do Wav2Lip. Em plano curto, o preço não compensa."""
    # Cadeia de fallback. MEDIDO 2026-08-26: sem enriquecimento, `beat_visual`
    # vem vazio e um plano estático produzia prompt com APENAS o look -- nada
    # sobre quem está em cena nem o que acontece, o que o LTX não sustenta.
    # Movimento vazio é correto para "câmera travada", mas ação vazia não é.
    corpo = (action or "").strip().rstrip(".") or (fallback or "").strip().rstrip(".")
    if not corpo:
        corpo = (f"{subject} stands in the scene, subtle natural movement"
                 if subject else "the scene continues, subtle natural movement")
    partes = [corpo]
    # Logo depois da acao e antes do descritor: e desempenho, nao aparencia.
    visivel = emocao_visivel(emotion)
    if visivel:
        partes.append(visivel)
    if quote:
        partes.append(f'speaking, saying: "{quote}"')
    if descriptor:
        partes.append(descriptor)
    if MOVEMENTS.get(movement):
        partes.append(MOVEMENTS[movement])
    if look:
        partes.append(look)
    return ". ".join(p for p in partes if p) + "."


def _look_e_interior(scene: dict, style: dict) -> tuple[str, str]:
    """O `look` efetivo do plano e o marcador de interior/exterior.

    A DIRECAO DE ARTE VEM PRIMEIRO, antes do acabamento do perfil de estilo:
    ela diz o MEIO da obra (cel animation, 3D storybook, live action) e o perfil
    diz so a luz e a composicao. MEDIDO 2026-08-27: sem canal para o meio, o
    mesmo roteiro na MESMA passada produziu um plano fotorreal e outro ilustrado
    -- a referencia por personagem (MEMORIAL 3.24) ancora a IDENTIDADE, nao o
    acabamento, entao duas imagens do mesmo filme nao pareciam do mesmo filme.
    Um roteiro que nao diz o meio deixa `art_direction` vazio e nada muda."""
    arte = (scene.get("art_direction") or "").strip()
    look = ", ".join(p for p in (arte, style["look"]) if p)
    # MEDIDO 2026-08-27: `startswith` sozinho le "EXT/INT. PALACE ... IMPERIAL
    # BEDCHAMBER" como EXTERIOR e manda "exterior" para o prompt de uma camara
    # fechada. Cabecalho hibrido e o roteirista dizendo que a cena atravessa os
    # dois -- entao a leitura honesta e NAO afirmar nenhum dos dois e deixar o
    # nome do local falar. Afirmar o lado errado e pior que ficar calado.
    cab = (scene.get("heading_raw") or "").strip().upper()
    tem_int = cab.startswith("INT") or "/INT" in cab or "-INT" in cab
    tem_ext = cab.startswith("EXT") or "/EXT" in cab or "-EXT" in cab
    if tem_int and tem_ext:
        interior = ""
    elif tem_int:
        interior = "interior"
    elif tem_ext:
        interior = "exterior"
    else:
        interior = ""
    return look, interior


def _plano_de_estabelecimento(scene: dict, style: dict, tensao: float | None,
                             *, fps: float) -> dict:
    """Um wide do LUGAR, na primeira cena que se passa nele.

    MEDIDO 2026-08-26 (MEMORIAL 3.24): a camara imperial descrita em detalhe no
    roteiro nunca aparecia inteira, porque a cobertura do `intimista` abre em
    `medium`. Coerente com o estilo, errado para uma cena que apresenta um lugar
    novo -- o espectador nunca ve onde a historia acontece. Por isso este plano
    NAO obedece ao estilo: e o unico caso em que a geografia ganha da intencao
    de decupagem, e vale uma vez por local, nao uma vez por cena.

    Nao entra quando a cobertura ja abriu em wide/full: ali o estabelecimento ja
    aconteceu, e um segundo plano aberto seria repeticao paga em GPU."""
    segundos = shot_seconds(style, tensao, "wide", is_action=True)
    movimento = style["movements"].get("wide", "static")
    look, interior = _look_e_interior(scene, style)
    cenario = ", ".join(p for p in (interior, scene.get("location", ""),
                                    scene.get("time_of_day", "")) if p)
    partes = [ESTABELECIMENTO_FRAMING]
    if cenario:
        partes.append(cenario)
    if look:
        partes.append(look)
    corpo = (f"the {scene.get('location')} is revealed, empty and still"
             if scene.get("location") else "the location is revealed, empty and still")
    video = [corpo]
    if MOVEMENTS.get(movimento):
        video.append(MOVEMENTS[movimento])
    if look:
        video.append(look)
    return {
        "scene": scene.get("index"), "position": 0, "type": "establishing",
        "line_index": None, "framing": "wide", "angle": "eye",
        "movement": movimento, "subject": "", "screen_side": None,
        "art_direction": (scene.get("art_direction") or "").strip(),
        "seconds": segundos, "frames": frames_for(segundos, fps),
        "storyboard_prompt": ". ".join(partes) + ".",
        "video_prompt": ". ".join(video) + ".",
        "look_base": look, "interior": interior, "descriptor": "",
        "location": scene.get("location", ""), "time_of_day": scene.get("time_of_day", ""),
        "beat": corpo, "quote": None, "emotion": None, "fallback": corpo,
    }


def plan_scene(scene: dict, struct: dict | None, style: dict, *, fps: float = 24.0,
               descriptors: dict | None = None, include_quotes: bool = False,
               sides: dict | None = None, durations: dict | None = None,
               estabelecer: bool = False, style_por_plano=None) -> list:
    """Decupagem de UMA cena. Sem LLM. `sides` vem de plan_all, global.

    `estabelecer`: esta e a primeira cena neste local -- quem decide e plan_all,
    que e quem enxerga o filme inteiro.

    `style_por_plano`: funcao (posicao) -> (nome, perfil), para o estilo mudar
    DENTRO da cena. Sem ela tudo usa `style`, que e o comportamento anterior.
    Quando ela existe, `style` vale so para o que e decidido antes do laco: o
    plano de estabelecimento e o estilo de referencia da cena."""
    descriptors = descriptors or {}
    if style_por_plano is None:
        _nome0 = next((n for n, v in STYLES.items() if v is style), "")
        def style_por_plano(_pos, _n=_nome0, _s=style):
            return _n, _s
    personagens = list(scene.get("characters") or [])
    lados = sides if sides is not None else assign_screen_sides(personagens)
    funcao = (struct or {}).get("function")
    tensao = (struct or {}).get("tension")

    def cobertura_de(perfil):
        return ((perfil.get("coverage") or COVERAGE).get(funcao)
                or COVERAGE.get(funcao) or DEFAULT_COVERAGE)

    # O conteúdo e sua ORDEM já vêm do shot_list; a decupagem só decide COMO
    # cada item é filmado. Sem shot_list, cai na ordem das falas.
    itens = list(scene.get("shot_list") or [])
    if not itens:
        itens = [{"type": "dialogue", "line_index": i}
                 for i in range(len(scene.get("dialogue") or []))]
    if not itens:
        itens = [{"type": "action", "visual": scene.get("action_text", "")}]

    # POSICAO QUE O USUARIO VE. O plano de estabelecimento e inserido na FRENTE
    # depois do laco, e ai tudo e renumerado -- entao o item de indice 9 pode
    # virar a tomada 10 na tabela. Como a marca de estilo e escrita olhando a
    # tabela ("intimista da 10"), ela tem de casar com o numero de la, nao com o
    # indice interno.
    #
    # Da para saber isto ANTES do laco: em todos os ramos, o item 0 recebe
    # `cobertura[0]` -- fala nova, acao com sujeito ou acao sem sujeito em pos 0
    # caem todos nele. Entao a decisao do estabelecimento e conhecida aqui, e o
    # deslocamento tambem.
    _cob0 = cobertura_de(style)
    vai_estabelecer = bool(estabelecer) and _cob0[0] not in ("wide", "full")
    desloc = 1 if vai_estabelecer else 0

    dialogo = scene.get("dialogue") or []
    planos = []
    ultimo_falante = None
    # Conta as falas da cena. E este ordinal, nao `pos`, que gira a escada de
    # enquadramento: as falas ocupam posicoes de paridade fixa quando ha planos
    # de acao entre elas, e por `pos` o ciclo simplesmente nao gira para elas.
    n_fala = 0
    for pos, item in enumerate(itens):
        # O estilo e resolvido AQUI, por plano. Tudo que decide como o plano e
        # filmado -- cobertura, movimento, duracao, look -- sai deste perfil.
        nome_estilo, estilo = style_por_plano(pos + desloc)
        cobertura = cobertura_de(estilo)
        look, interior = _look_e_interior(scene, estilo)

        if item.get("type") == "dialogue":
            li = item.get("line_index", 0)
            linha = dialogo[li] if 0 <= li < len(dialogo) else {}
            sujeito = linha.get("character") or ""
            acao = linha.get("beat_visual") or scene.get("visual_prompt") or ""
            fala = linha.get("text") or ""
            emocao_da_fala = linha.get("emotion") or linha.get("parenthetical")
            # Alterna a cobertura entre falantes: mesmo padrão repetido em todas
            # as falas é o que hoje deixa a cena parada.
            escada = _cobertura_de_fala(cobertura)
            if sujeito and sujeito != ultimo_falante:
                enquadre = escada[n_fala % len(escada)]
            else:
                # Mesmo falante emendando: fecha, em vez de repetir o degrau.
                enquadre = "close" if "close" in escada else escada[-1]
            ultimo_falante = sujeito or ultimo_falante
            n_fala += 1
        else:
            fala = ""
            emocao_da_fala = None  # plano de acao nao tem fala para colorir
            acao = item.get("visual") or scene.get("action_text") or ""
            # Ação COM personagem nomeado é plano de personagem, não inserto.
            # MEDIDO 2026-08-26: sem esta distinção, "Mei-Li jumps to her feet"
            # e "her face lights up" viravam `insert` -- "sem rosto no quadro" --
            # que é o enquadramento oposto do que a ação pede. Inserto é para
            # detalhe de objeto, e é assim que ele volta a ser usado: só quando
            # a ação não nomeia ninguém.
            sujeito = _character_in(acao, personagens)
            if sujeito:
                enquadre = cobertura[pos % len(cobertura)]
            else:
                enquadre = cobertura[0] if pos == 0 else "insert"

        # Um ots sem sujeito nao tem rosto para enquadrar -- so sobra o ombro de
        # costas, que e exatamente a nuca que o SUBJECT_HINTS existe para evitar.
        if enquadre == "ots" and not sujeito:
            enquadre = "medium"

        # Ângulo: a virada e o clímax ganham inclinação; o resto fica no olho.
        angulo = estilo["angle_bias"]
        if funcao in ("virada", "climax") and enquadre == "close":
            angulo = "low"
        elif funcao == "respiro" and enquadre == "wide":
            angulo = "high"

        # Duracao REAL do TTS quando existir. O proxy por caracteres serve para
        # planejar antes da voz; depois dela, usar o palpite seria escolher o
        # numero pior de proposito -- e um plano curto demais corta a fala no
        # meio, que e o defeito mais visivel que existe.
        real = (durations or {}).get((scene.get("index"), item.get("line_index")))
        if real:
            segundos = round(real + FALA_FOLGA_S, 2)
        else:
            segundos = shot_seconds(estilo, tensao, enquadre, text=fala,
                                    is_action=(item.get("type") != "dialogue"))
        movimento = estilo["movements"].get(enquadre, "static")
        lado = lados.get(sujeito)

        planos.append({
            "scene": scene.get("index"),
            "position": pos,
            "type": item.get("type"),
            "line_index": item.get("line_index"),
            "framing": enquadre,
            "angle": angulo,
            "movement": movimento,
            "subject": sujeito,
            "screen_side": lado,
            "style": nome_estilo,
            "seconds": segundos,
            "frames": frames_for(segundos, fps),
            # Repetido aqui, alem de ja estar dentro do `look`, porque o
            # gerador de still precisa SABER que existe direcao de arte para
            # escolher o prompt negativo (ver generate_storyboards
            # .negative_prompt_for): a lista que proibe "cartoon, anime,
            # illustration" briga com um roteiro em cel animation.
            "art_direction": (scene.get("art_direction") or "").strip(),
            "storyboard_prompt": _storyboard_prompt(
                framing=enquadre, angle=angulo, subject=sujeito,
                location=scene.get("location", ""), time_of_day=scene.get("time_of_day", ""),
                look=look, descriptor=descriptors.get(sujeito, ""),
                screen_side=lado, interior=interior, pose=acao),
            "video_prompt": _video_prompt(
                action=acao, movement=movimento, descriptor=descriptors.get(sujeito, ""),
                look=look, quote=fala if (include_quotes and fala) else None,
                subject=sujeito, fallback=scene.get("action_text", ""),
                emotion=emocao_da_fala),
            # Ingredientes crus, guardados so para o enriquecimento de camera
            # opcional (enrich_camera_style) poder RECONSTRUIR os dois prompts
            # acima depois de trocar movement/look -- sem isto ele teria que
            # adivinhar de volta a partir do texto ja montado, que e o mesmo
            # tipo de erro que binding-com-string-parseada sempre da.
            "look_base": look, "interior": interior, "descriptor": descriptors.get(sujeito, ""),
            "location": scene.get("location", ""), "time_of_day": scene.get("time_of_day", ""),
            "beat": acao, "quote": fala if (include_quotes and fala) else None,
            "emotion": emocao_da_fala, "fallback": scene.get("action_text", ""),
        })

    # O estabelecimento vem na frente e so quando a cobertura ainda nao abriu o
    # quadro sozinha. Renumerar `position` depois mantem a numeracao contigua --
    # a rotacao de cobertura la em cima ja usou a posicao ORIGINAL do item, que e
    # a que importa para alternar enquadramento entre falantes.
    if vai_estabelecer or (estabelecer and not planos):
        planos.insert(0, _plano_de_estabelecimento(scene, style, tensao, fps=fps))
        for i, pl in enumerate(planos):
            pl["position"] = i
    return planos


def parse_style_changes(spec: str | None) -> dict:
    """Marcas de estilo, por CENA ou por PLANO dentro da cena.

        "3:tenso"              -> a partir da CENA 3
        "1.10:intimista"       -> a partir do PLANO 10 da cena 1
        "1:classico,1.10:intimista,2:tenso"   -> combina os dois

    Devolve {(cena, plano): estilo}; marca de cena é (cena, 0), que ordena antes
    de qualquer plano dela.

    POR QUE PLANO E NÃO SÓ CENA

    Uma cena longa não tem um único registro. Uma cena de 20 tomadas pode abrir
    convencional e fechar intimista a partir da décima -- e antes disto a única
    unidade era a cena inteira, o que obriga a quebrar a cena em duas só para
    mudar de tom, quebrando junto a continuidade que a cena é.

    A semântica é a mesma dos dois lados, e é de MUDANÇA DE BOBINA: a marca
    vale dali PARA A FRENTE, atravessando o fim da cena, até aparecer outra.
    Marcar o plano 10 da cena 1 num filme de 3 cenas leva intimista até o fim,
    não até o fim da cena 1."""
    if not spec:
        return {}
    out = {}
    for parte in spec.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if ":" not in parte:
            raise ValueError(
                f"trecho invalido em --style-changes: {parte!r} "
                "(use CENA:estilo ou CENA.PLANO:estilo)")
        onde, nome = parte.split(":", 1)
        nome = nome.strip()
        if nome not in STYLES:
            raise ValueError(f"estilo desconhecido {nome!r}; use {sorted(STYLES)}")
        onde = onde.strip()
        try:
            if "." in onde:
                cena, plano = onde.split(".", 1)
                chave = (int(cena), int(plano))
            else:
                chave = (int(onde), 0)
        except ValueError:
            raise ValueError(
                f"posicao invalida em --style-changes: {onde!r} "
                "(use um numero de CENA, ou CENA.PLANO)") from None
        out[chave] = nome
    return out


def style_for_shot(cena: int, plano: int, base: str, changes: dict) -> str:
    """Estilo VIGENTE num plano: a última marca em posição <= (cena, plano).

    Comparação por tupla, então (1, 10) vem depois de (1, 0) e antes de (2, 0)
    -- que é exatamente a ordem em que os planos são filmados."""
    if not changes:
        return base
    alvo = (cena, plano)
    aplicaveis = [k for k in changes if k <= alvo]
    return changes[max(aplicaveis)] if aplicaveis else base


def style_for_scene(index: int, base: str, changes: dict) -> str:
    """Estilo no INÍCIO de uma cena. Mantido porque o estabelecimento e o
    `look` da cena precisam de um estilo antes do laço de planos começar."""
    return style_for_shot(index, 0, base, changes)


def plan_all(scenes: list, structure: dict | None, *, style_name: str = "classico",
             fps: float = 24.0, descriptors: dict | None = None,
             include_quotes: bool = False, style_changes: dict | None = None,
             durations: dict | None = None) -> dict:
    if style_name not in STYLES:
        raise ValueError(f"estilo desconhecido {style_name!r}; use {sorted(STYLES)}")
    style_changes = style_changes or {}
    por_cena = {s["index"]: s for s in (structure or {}).get("scenes", [])}
    # Lados de tela decididos UMA vez, na ordem de primeira aparição no filme.
    ordem = []
    for sc in scenes:
        for nome in (sc.get("characters") or []):
            if nome not in ordem:
                ordem.append(nome)
    lados = assign_screen_sides(ordem)
    planos = []
    # Locais ja apresentados. A chave e o `location` da cena; cena sem local
    # identificado NAO ganha estabelecimento -- sem identidade de lugar nao da
    # para saber se ele e novo, e um wide de lugar nenhum e so custo de GPU.
    # (Ate 2026-08-27 roteiro em prosa corrida sempre caia aqui, porque o
    # location vinha vazio; corrigido em parse_screenplay._apply_setting.)
    locais_vistos = set()
    for sc in scenes:
        idx = sc.get("index")
        nome = style_for_scene(idx, style_name, style_changes)
        estilo = STYLES[nome]
        local = (sc.get("location") or "").strip().casefold()
        novo_local = bool(local) and local not in locais_vistos
        if local:
            locais_vistos.add(local)
        # Resolvedor por plano: e o que permite "intimista a partir da tomada
        # 10" dentro de uma cena de 20. Sem marca de plano ele devolve sempre o
        # estilo da cena, que e o comportamento anterior.
        def por_plano(pos, _idx=idx):
            n = style_for_shot(_idx, pos, style_name, style_changes)
            return n, STYLES[n]

        novos = plan_scene(sc, por_cena.get(idx), estilo, fps=fps,
                           descriptors=descriptors, include_quotes=include_quotes,
                           sides=lados, durations=durations, estabelecer=novo_local,
                           style_por_plano=por_plano)
        # `style` de cada plano vem do resolvedor, nao daqui: dentro da mesma
        # cena dois planos podem ter nascido sob estilos diferentes.
        for pl in novos:
            pl.setdefault("style", nome)
        planos.extend(novos)
    total = sum(p["seconds"] for p in planos)
    segmentos = []
    for pl in planos:
        if not segmentos or segmentos[-1]["style"] != pl["style"]:
            segmentos.append({"style": pl["style"], "scenes": [pl["scene"]], "seconds": 0.0})
        elif pl["scene"] not in segmentos[-1]["scenes"]:
            segmentos[-1]["scenes"].append(pl["scene"])
        segmentos[-1]["seconds"] = round(segmentos[-1]["seconds"] + pl["seconds"], 2)
    return {"style": style_name, "style_descricao": STYLES[style_name]["descricao"],
            "style_changes": style_changes, "segments": segmentos, "fps": fps,
            "screen_sides": lados, "include_quotes": include_quotes,
            "shots": planos, "total_shots": len(planos), "total_seconds": round(total, 1)}


# --------------------------------------------------------------------------
# enriquecimento por LLM: camera/luz por plano, dentro do vocabulario do estilo
# --------------------------------------------------------------------------
# Pedido do usuario 2026-09-04: dado um Estilo já escolhido, deixar o LLM
# decidir qual movimento de câmera/luz cada PLANO ESPECÍFICO merece -- não
# trocar o Estilo por outra coisa, refinar dentro dele. `plan_all` continua
# 100% determinístico (a estrutura não muda); isto é uma passada OPCIONAL
# depois, que só troca `movement`/`look` quando o LLM escolhe algo do
# vocabulário PERMITIDO para aquele estilo -- nunca fora dele, e a validação
# roda em código, não confia na resposta do modelo.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# MEDIDO 2026-08-26 (MEMORIAL 3.21, ver docstring do módulo): câmera é DESTINO,
# não estado -- um plano curto não termina o movimento que promete. "orbit" é o
# mais caro disso (uma volta inteira pede tempo real); planos abaixo deste
# limiar não recebem orbit do LLM mesmo que o estilo permita, caem no
# determinístico. Não é chute: 5s é o `shot_seconds` do próprio "contemplativo"
# menos folga, o único estilo que já usa orbit por padrão hoje.
ORBIT_MIN_SECONDS = 5.0


def _call_ollama_camera(system: str, user: str, model: str, log=print) -> dict | None:
    payload = {
        "model": model, "stream": False, "format": "json", "think": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    req_bytes = json.dumps(payload).encode("utf-8")
    # RETRY (2026-09-09): mesmo fix de parse_screenplay.py (MEMORIAL 3.65)
    # -- sem isto uma falha HTTP transiente na primeira cena derrubava o
    # --camera-llm da cena inteira, caindo no vocabulario padrao do estilo em
    # silencio, mesmo quando a proxima chamada teria funcionado.
    body = None
    for tentativa in range(3):
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=req_bytes,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.load(r)
            if tentativa:
                log(f"[camera_style] Ollama ok na tentativa {tentativa + 1}/3.")
            break
        except urllib.error.HTTPError as e:
            log(f"[camera_style] Ollama HTTP {e.code} (tentativa {tentativa + 1}/3).")
        except Exception as e:
            log(f"[camera_style] Ollama inacessivel ({type(e).__name__}): {e} (tentativa {tentativa + 1}/3).")
        if tentativa < 2:
            time.sleep(2)
    if body is None:
        return None
    content = (body.get("message") or {}).get("content") or ""
    if not content.strip():
        log("[camera_style] Ollama devolveu content vazio.")
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        log("[camera_style] resposta nao era JSON valido.")
        return None


_CAMERA_SYSTEM_PROMPT = """Voce e um diretor de fotografia. Para cada plano
listado, escolha OPCIONALMENTE um movimento de camera e/ou uma tag de luz
DENTRO das listas permitidas para o estilo daquele plano especifico -- nunca
fora delas. So escolha algo quando o CONTEUDO do plano (o campo "beat")
genuinamente pede -- a maioria dos planos deve ficar com "movement": null e
"lighting": null, mantendo a escolha padrao do estilo. Nao repita o mesmo
movimento nao-default em planos consecutivos da mesma cena sem motivo -- isso
lê como decisão de câmera, não como tique.

"post" (lista) so quando o plano tem um motivo forte: "freeze" para um
instante que pede congelar (auge de um salto, revelação chocante), "whip"
quando o PRÓXIMO plano deveria entrar com corte brusco tipo chicote (mudança
abrupta de lugar/tempo/energia). Raro -- a maioria dos planos deve ter "post": [].

Devolva JSON: {"choices": [{"position": <int>, "movement": <string ou null>,
"lighting": <string ou null>, "post": [<string>, ...]}, ...]} -- um item por
plano recebido, na mesma ordem."""


def enrich_camera_style(plan: dict, *, engine: str, log=print) -> int:
    """Passada OPCIONAL sobre um plano ja construido por `plan_all`: para cada
    plano, pergunta ao LLM se movimento/luz do vocabulario PERMITIDO pelo
    estilo daquele plano [CAMERA_STYLE_VOCAB] descrevem melhor a cena do que o
    padrao determinístico. Muda `shot["movement"]`/`shot["look_base"]` e
    RECONSTROI os dois prompts só quando a escolha valida (dentro da lista
    permitida, dentro do limiar de duração pro orbit). Devolve quantos planos
    mudaram.

    Falha (Ollama fora do ar, resposta invalida) e SILENCIOSA por cena -- essa
    cena fica com o determinístico de `plan_all`, que já é um resultado
    utilizável; isto é acabamento, não estrutura."""
    por_cena: dict[int, list[dict]] = {}
    for sh in plan.get("shots", []):
        por_cena.setdefault(sh["scene"], []).append(sh)

    total_mudados = 0
    for cena_idx, shots in por_cena.items():
        itens = []
        for sh in shots:
            vocab = CAMERA_STYLE_VOCAB.get(sh.get("style"), CAMERA_STYLE_VOCAB["classico"])
            movs = list(vocab["movements"])
            if "orbit" in movs and sh.get("seconds", 0) < ORBIT_MIN_SECONDS:
                movs = [m for m in movs if m != "orbit"]
            itens.append({
                "position": sh["position"], "framing": sh.get("framing"),
                "subject": sh.get("subject") or "(sem sujeito)",
                "beat": (sh.get("beat") or "")[:200],
                "seconds": sh.get("seconds"),
                "movements_permitidos": movs,
                "lighting_permitida": list(vocab["lighting"]),
                "post_permitido": list(vocab["post"]),
            })
        user = json.dumps({"planos": itens}, ensure_ascii=False)
        resp = _call_ollama_camera(_CAMERA_SYSTEM_PROMPT, user, engine, log=log)
        if not resp or not isinstance(resp.get("choices"), list):
            log(f"[camera_style] cena {cena_idx}: sem resposta valida, mantendo padrao do estilo.")
            continue

        by_pos = {sh["position"]: sh for sh in shots}
        allowed_by_pos = {it["position"]: it for it in itens}
        for escolha in resp["choices"]:
            pos = escolha.get("position")
            sh = by_pos.get(pos)
            allowed = allowed_by_pos.get(pos)
            if sh is None or allowed is None:
                continue

            mudou = False
            novo_mov = escolha.get("movement")
            if novo_mov and novo_mov in allowed["movements_permitidos"] and novo_mov != sh["movement"]:
                sh["movement"] = novo_mov
                mudou = True

            nova_luz = escolha.get("lighting")
            look_efetivo = sh.get("look_base", "")
            if nova_luz and nova_luz in allowed["lighting_permitida"] and LIGHTING.get(nova_luz):
                look_efetivo = ", ".join(p for p in (look_efetivo, LIGHTING[nova_luz]) if p)
                mudou = True

            post = [p for p in (escolha.get("post") or [])
                   if p in allowed["post_permitido"] and p in POST_EFFECTS]
            if post:
                sh["post_effects"] = post
                mudou = True

            if mudou:
                total_mudados += 1
                sh["storyboard_prompt"] = _storyboard_prompt(
                    framing=sh["framing"], angle=sh["angle"], subject=sh["subject"],
                    location=sh.get("location", ""), time_of_day=sh.get("time_of_day", ""),
                    look=look_efetivo, descriptor=sh.get("descriptor", ""),
                    screen_side=sh.get("screen_side"), interior=sh.get("interior", ""),
                    pose=sh.get("beat", ""))
                sh["video_prompt"] = _video_prompt(
                    action=sh.get("beat", ""), movement=sh["movement"],
                    descriptor=sh.get("descriptor", ""), look=look_efetivo,
                    quote=sh.get("quote"), subject=sh["subject"],
                    fallback=sh.get("fallback", ""), emotion=sh.get("emotion"))
    log(f"[camera_style] {total_mudados} plano(s) com camera/luz refinada pelo LLM "
        f"(de {sum(len(v) for v in por_cena.values())} no total).")
    return total_mudados


def summary(plan: dict) -> str:
    l = [f"estilo base: {plan['style']} -- {plan['style_descricao']}",
         f"{plan['total_shots']} planos, {plan['total_seconds']}s no total"]
    if len(plan.get("segments") or []) > 1:
        l.append("")
        l.append("segmentos:")
        for seg in plan["segments"]:
            l.append(f"  {seg['style']:<14} cenas {seg['scenes']}  {seg['seconds']}s")
    l += ["", f"{'cena':>4} {'#':>2} {'estilo':<14} {'enquadre':<14} {'ang':<5} "
              f"{'mov':<9} {'s':>5} {'lado':<6} sujeito"]
    for p in plan["shots"]:
        alvo = p["subject"] or (p["video_prompt"][:30] + "...")
        l.append(f"{p['scene']:>4} {p['position']:>2} {p.get('style',''):<14} "
                 f"{p['framing']:<14} {p['angle']:<5} "
                 f"{p['movement']:<9} {p['seconds']:>5} {str(p['screen_side'] or '-'):<6} {alvo[:34]}")
    return "\n".join(l)


def load_tts_durations(path) -> dict:
    """(cena, indice_da_fala) -> duracao em segundos, de dialogue/lines.json."""
    if not path or not pathlib_Path(path).exists():
        return {}
    dados = json.load(open(path, encoding="utf-8"))
    out = {}
    for e in dados:
        if e.get("ok") and e.get("duration_sec"):
            out[(e["scene_index"], e["line_index"])] = float(e["duration_sec"])
    return out


def sp_default_cast(*, sp_run, scenes_path):
    """cast.json costuma ficar em <run>/characters/. Procura ali a partir de
    qualquer um dos dois caminhos que a CLI aceita."""
    if sp_run:
        c = Path(sp_run) / "characters" / "cast.json"
        return c if c.exists() else None
    c = Path(scenes_path).parent.parent / "characters" / "cast.json"
    return c if c.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Decupagem deterministica a partir da estrutura")
    ap.add_argument("--run")
    ap.add_argument("--scenes")
    ap.add_argument("--structure")
    ap.add_argument("--style", default="classico", choices=sorted(STYLES))
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--include-quotes", action="store_true",
                    help="poe o texto da fala no prompt de video (melhora a boca, "
                         "mas causa legenda queimada em plano curto)")
    ap.add_argument("--style-changes", default=None,
                    help='troca de estilo, vigora ate a proxima marca. Por CENA ("3:tenso") ou por TOMADA dentro da cena ("1.10:intimista" = da tomada 10 da cena 1 em diante, atravessando o fim da cena). O numero da tomada e o que aparece na tabela do plano. Ex.: "1:classico,1.10:intimista,2:tenso"')
    ap.add_argument("--cast", default=None,
                    help="characters/cast.json; os descritores entram nos prompts de still")
    ap.add_argument("--dialogue", default=None,
                    help="dialogue/lines.json; usa a duracao REAL do TTS por fala")
    ap.add_argument("--out")
    ap.add_argument("--camera-llm", action="store_true",
                    help="deixa o LLM refinar movimento/luz por plano, dentro do "
                         "vocabulario permitido pelo Estilo escolhido (ver "
                         "CAMERA_STYLE_VOCAB) -- opt-in, custa 1 chamada de Ollama "
                         "por cena. Sem isto, so o determinístico de sempre.")
    ap.add_argument("--engine", default="qwen3.6-35b-a3b:latest",
                    help="tag do Ollama para --camera-llm")
    args = ap.parse_args()
    changes = parse_style_changes(args.style_changes)

    if args.scenes:
        sp = Path(args.scenes)
        stp = Path(args.structure) if args.structure else sp.parent / "story_structure.json"
    elif args.run:
        sp = Path(args.run) / "parse" / "scenes.json"
        stp = Path(args.run) / "parse" / "story_structure.json"
    else:
        ap.error("informe --run ou --scenes")

    data = json.load(open(sp, encoding="utf-8"))
    scenes = data["scenes"] if isinstance(data, dict) and "scenes" in data else data
    structure = json.load(open(stp, encoding="utf-8")) if stp.exists() else None
    if structure is None:
        print(f"[shot_plan] sem {stp.name}: cobertura cai no padrao neutro "
              "(rode story_structure para ter funcao dramatica).")

    # Descritores do cast: só a APARÊNCIA entra, ver appearance_only().
    cast_path = Path(args.cast) if args.cast else sp_default_cast(sp_run=args.run, scenes_path=sp)
    descritores = {}
    if cast_path and cast_path.exists():
        cast = json.load(open(cast_path, encoding="utf-8"))
        descritores = {n: appearance_only(v.get("descriptor", "")) for n, v in cast.items()}
        print(f"[shot_plan] cast: {len(descritores)} descritor(es) de {cast_path.name}")
    else:
        print("[shot_plan] sem cast.json: os stills sairao sem figurino/idade do roteiro.")

    dlg = args.dialogue
    if not dlg and args.run:
        cand = Path(args.run) / "dialogue" / "lines.json"
        dlg = str(cand) if cand.exists() else None
    duracoes = load_tts_durations(dlg)
    if duracoes:
        print(f"[shot_plan] {len(duracoes)} duracao(oes) real(is) de TTS em uso")
    else:
        print("[shot_plan] sem lines.json: duracao das falas por estimativa de caracteres")

    plan = plan_all(scenes, structure, style_name=args.style, fps=args.fps,
                    include_quotes=args.include_quotes, style_changes=changes,
                    descriptors=descritores, durations=duracoes)
    if args.camera_llm:
        enrich_camera_style(plan, engine=args.engine, log=print)
    out = Path(args.out) if args.out else sp.parent / "shot_plan.json"
    json.dump(plan, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(summary(plan))
    print(f"\n[shot_plan] salvo em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
