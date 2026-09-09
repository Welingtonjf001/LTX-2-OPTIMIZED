"""Diagnóstico e reparo temporal pós-geração para clipes do LTX.

O problema: o LTX produz artefatos que duram poucos frames -- rosto ou mão
"derretendo", textura pulsando, um frame isolado destoando dos vizinhos. Regerar
a cena inteira por causa de 3 frames é caro e joga fora o que estava bom.

A ideia central é UMA medida, e vale explicá-la porque tudo depende dela:

    resíduo de fluxo = | warp(frame[t-1] -> t) - frame[t] |

Calcula-se o fluxo óptico de t-1 para t, deforma-se o frame anterior por esse
fluxo e compara-se com o frame real. Movimento normal -- por mais rápido que
seja -- é EXPLICADO pelo fluxo, então o resíduo fica baixo. O que o fluxo não
consegue explicar é justamente o que nos interessa: matéria aparecendo,
sumindo ou trocando de forma. Por isso o resíduo separa "a câmera girou" de
"o rosto derreteu", coisa que diferença de pixel simples não faz.

Duas medidas auxiliares desambiguam o tipo de defeito:

  - delta de luminância: se o brilho salta mas a geometria está estável, é
    flicker de iluminação, não deformação. Não adianta interpolar.
  - delta de alta frequência: textura pulsando sem mudança de forma.

A classificação sai daí, e mapeia direto para o reparo mais barato que resolve:

    1-3 frames, resíduo alto        -> interpolação por fluxo bidirecional
    luminância domina               -> deflicker temporal
    4-24 frames, resíduo alto       -> regeneração parcial com margem
    > 24 frames                     -> reporta, não conserta sozinho

O último caso é deliberado. Defeito longo quase sempre significa que a cena
saiu errada, e remendar 40 frames costuma custar mais que regerar o plano --
essa é uma decisão de quem dirige, não do script.

Sobre a margem na regeneração: reconstruir SÓ o frame defeituoso normalmente
cria duas descontinuidades no lugar de uma. O reparo pega frames bons antes e
depois como âncoras e regenera o miolo, preservando continuidade de movimento.

### O LIMITE QUE IMPORTA: revise antes de reparar.

MEDIDO 2026-08-24, e é o resultado mais importante deste arquivo: o detector
acusou uma MÃO ENTRANDO EM QUADRO -- movimento legítimo e rápido -- e tanto o
RIFE quanto o warp APAGARAM a mão ao interpolar entre as âncoras. As duas
"reparações" melhoraram a métrica (pico de diferença caiu 38%) destruindo
exatamente o conteúdo que a fazia subir.

Nenhuma medida temporal separa com segurança "matéria surgindo porque o modelo
errou" de "matéria surgindo porque entrou em cena". Quem separa é quem olha.
Por isso `--previews` gera uma tira por segmento e `--only` repara apenas os
escolhidos. Reparar tudo às cegas piora vídeo bom.

Ferramentas instaladas (2026-08-24):
  - RIFE ncnn-vulkan em tools/rife (release oficial nihui/rife-ncnn-vulkan
    20221029, modelos rife-v2 a v4.6). Interpolação APRENDIDA: sintetiza
    conteúdo plausível em oclusão, onde o warp só estica pixel. Padrão;
    cai para warp sozinho se o binário sumir.
    ARMADILHA: o binário converte o `-s` com o LOCALE DO SISTEMA. Em pt-BR
    "0.2" vira 0 e ele recusa; "0,2" funciona. O código tenta os dois.
  - RAFT (torchvision, pesos em cache): fluxo melhor que o DIS, ~45x mais
    lento. `--flow raft`. O DIS segue como padrão.
  - Sem detector de rosto/mãos (insightface/SAM ausentes): o ramo "rosto
    instável" da taxonomia cai no reparo por duração.

Uso:
    python video_doctor.py analyze clipe.mp4 --plan p.json --previews tiras/
    python video_doctor.py repair  clipe.mp4 --plan p.json --only 0,2 -o out.mp4
    python video_doctor.py doctor  clipe.mp4 -o out.mp4     # sem revisão, cuidado

Interface: video_doctor_ui.build_doctor_tab() embute a aba nas UIs de geração
(music_maker v2/v3, 2.3 e 2.5); start_video_doctor.bat abre avulso na 7912.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

FFMPEG = os.environ.get("LTX_FFMPEG", "C:/ffmpeg/bin/ffmpeg.exe")
# Só o nome do executável: um replace no caminho inteiro transformaria
# C:/ffmpeg/bin/ffmpeg.exe em C:/ffprobe/bin/ffprobe.exe.
FFPROBE = os.path.join(os.path.dirname(FFMPEG),
                       os.path.basename(FFMPEG).replace("ffmpeg", "ffprobe"))

# Limiares. São z-scores robustos (mediana/MAD), não valores absolutos: o nível
# de resíduo "normal" depende de quanto a cena se mexe, então calibrar por
# constante absoluta quebraria entre um plano parado e uma câmera girando.
Z_RESIDUAL = 4.0        # acima disso o resíduo é anômalo
Z_LUMA = 3.5            # acima disso o salto de brilho é anômalo
MAX_INTERP_LEN = 3      # até aqui, interpolar
MAX_REGEN_LEN = 24      # acima disso, não conserta sozinho
REGEN_MARGIN = 2        # frames bons extras de cada lado, como âncora
GRID = 8                # grade do resíduo local (8x8 células por quadro)
Z_RESIDUAL_LOCAL = 4.5  # limiar do resíduo por célula; mais alto que o global
                        # porque a série da pior célula é mais ruidosa

# PISOS ABSOLUTOS. z-score é relativo ao próprio clipe, então num clipe muito
# estável ele promove ruído a defeito: MEDIDO 2026-08-24, um clipe sadio gerou
# 4 "flickers" cuja oscilação real era 0,5/255 -- meio por cento de brilho,
# invisível. Abaixo destes valores nada é marcado, por mais alto que dê o z.
MIN_LUMA_DELTA = 2.5    # 0-255; ~1% de brilho, na fronteira do perceptível
MIN_RESIDUAL_LOCAL = 12.0  # 0-255; erro médio na pior célula da grade
CONTRAST_WINDOW = 31    # janela da mediana de referência; tem de ser maior que
                        # MAX_REGEN_LEN, ou um defeito longo vira a linha de base


# --------------------------------------------------------------------------
# leitura / escrita
# --------------------------------------------------------------------------
def probe(path: str) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (st["r_frame_rate"].split("/") + ["1"])[:2]
    return {"width": int(st["width"]), "height": int(st["height"]),
            "frames": int(st.get("nb_frames") or 0), "fps": float(num) / float(den)}


def read_frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"Nenhum frame lido de {path}")
    return frames


def write_frames(frames: list, out_path: str, fps: float, audio_from: str | None = None):
    """Escreve via ffmpeg (não cv2.VideoWriter) para poder recolocar o áudio
    original e controlar o codec -- o writer do OpenCV não faz nem um nem outro."""
    tmp = tempfile.mkdtemp(prefix="vdoctor_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"f_{i:06d}.png"), f)
        cmd = [FFMPEG, "-y", "-v", "error", "-framerate", f"{fps}",
               "-i", os.path.join(tmp, "f_%06d.png")]
        if audio_from:
            # "?" torna o mapeamento opcional: clipe sem áudio não faz o comando falhar.
            cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "aac", "-b:a", "256k", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)


# --------------------------------------------------------------------------
# medição
# --------------------------------------------------------------------------
def _flow_engine(kind: str):
    if kind == "raft":
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(dev).eval()

        def flow(a, b):
            def prep(x):
                t = torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                return (t.float() / 127.5 - 1.0).unsqueeze(0).to(dev)
            with torch.no_grad():
                return model(prep(a), prep(b))[-1][0].permute(1, 2, 0).cpu().numpy()
        return flow

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def flow(a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return dis.calc(ga, gb, None)
    return flow


def _block_max(err: np.ndarray, mag: np.ndarray, grid: int) -> tuple:
    """Pior célula da grade por resíduo POR UNIDADE DE MOVIMENTO DELA MESMA.

    Uma cara derretendo é um defeito enorme num pedaço pequeno do quadro; média
    global dilui isso até sumir, e o máximo por célula preserva. A grade é
    grossa de propósito -- célula pequena demais dispara em qualquer borda em
    movimento, que é ruído.

    A escolha é pelo erro BRUTO, sem descontar movimento. Descontar foi uma
    tentativa que se provou errada (2026-08-24): um defeito real gera fluxo
    errático justamente por ser inexplicável, então o movimento alto ali é
    CONSEQUÊNCIA do defeito, não explicação dele. Dividir por ele apagava
    exatamente o que se procura -- uma mão derretendo perdia a seleção para
    uma célula qualquer de fundo parado. O que separa defeito de movimento
    legítimo é a duração, e disso cuida local_contrast()."""
    h, w = err.shape
    ch, cw = max(1, h // grid), max(1, w // grid)
    best, by, bx = -1.0, 0, 0
    for y in range(0, h - ch + 1, ch):
        for x in range(0, w - cw + 1, cw):
            v = float(err[y:y + ch, x:x + cw].mean())
            if v > best:
                best, by, bx = v, y, x
    return best, best, by, bx


def warp(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(frame, gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def high_freq(gray: np.ndarray) -> float:
    F = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float64))))
    h, w = F.shape
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - h // 2) ** 2 + (X - w // 2) ** 2)
    tot = F.sum()
    return float(F[r > min(h, w) * 0.25].sum() / tot) if tot else 0.0


def color_hist(img) -> "np.ndarray":
    """Histograma BGR 8x8x8 normalizado. Grosso de proposito: para distinguir
    PLANO de plano nao interessa detalhe, interessa a paleta."""
    h = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    return cv2.normalize(h, h).flatten().astype(np.float32)


def detect_cuts(metrics: dict, z_thr: float = 6.0, min_dist: float = 0.04) -> list:
    """Frames em que comeca um PLANO NOVO, num filme ja montado.

    MEDIDO 2026-08-28 na cena Lyra/Thoren: dos 10 segmentos apontados no filme
    montado, QUATRO eram as quatro fronteiras de corte -- e eram os de MAIOR
    pontuacao (z_glob 190 a 224 contra mediana de residuo 5,93). Um corte e a
    maior mudanca possivel entre dois quadros; e a definicao de corte. Para tres
    deles o veredito era INTERPOLATE, que num corte significa fabricar um quadro
    misturando dois enquadramentos: nao conserta nada e estraga a montagem.

    **O que separa corte de defeito nao e o tamanho do pico -- e se a mudanca
    PERSISTE.** Um defeito de um quadro volta ao normal no seguinte, entao o
    frame anterior e o posterior se parecem; um corte nao volta, porque o plano
    novo continua. Por isso a decisao usa tres distancias:

        d_lead   = hist(t-2) -> hist(t-1)   PEQUENA se t-1 era o normal estavel
        d_passo  = hist(t-1) -> hist(t)     grande nos dois casos
        d_salto  = hist(t-1) -> hist(t+1)   grande SO no corte

    Comparar magnitude do residuo de fluxo nao resolveria: os dois casos sao
    pico isolado e enorme, que e exatamente por que o classificador se confunde.

    **`d_lead` existe por um caso que so apareceu ao ESCREVER TESTE para esta
    funcao** (nao em producao -- ver tests/test_pipeline_smoke.py): uma
    anomalia de UM quadro produz DUAS transicoes de magnitude igual, entrada e
    saida. `d_salto` sozinho rejeita a de ENTRADA corretamente (hist(t-2) e
    hist(t) sao ambos o normal, ficam parecidos), mas a de SAIDA usa hist(t-1)
    -- que E o proprio quadro anomalo -- como ancora "antes", e comparado dali
    para a frente sempre parece "mudanca que persiste". `d_lead` fecha isso:
    exige que o quadro ANTES da transicao ja fosse ele mesmo estavel, nao a
    cauda de uma anomalia. Validado sem regressao contra os 4 cortes reais da
    cena Lyra (73/202/305/410) e contra os defeitos internos do clipe T2V
    (228-240 sobrevive intacto -- ver MEMORIAL 3.39).

    Sem cortes (clipe unico) devolve lista vazia, e nada muda."""
    hs = metrics.get("_hists") or []
    if len(hs) < 5:
        return []
    d_passo = np.array([1.0 - float(cv2.compareHist(hs[i], hs[i + 1],
                                                    cv2.HISTCMP_CORREL))
                        for i in range(len(hs) - 1)])
    z = robust_z(d_passo)
    cortes = []
    for i in range(1, len(d_passo) - 1):
        if z[i] < z_thr or d_passo[i] < min_dist:
            continue
        d_salto = 1.0 - float(cv2.compareHist(hs[i], hs[i + 2],
                                              cv2.HISTCMP_CORREL))
        # Persistiu? Se o quadro seguinte continua tao diferente do anterior
        # quanto o do pico, mudou de plano. Se voltou, foi defeito.
        if d_salto < 0.6 * d_passo[i]:
            continue
        d_lead = (1.0 - float(cv2.compareHist(hs[i - 1], hs[i], cv2.HISTCMP_CORREL))
                  if i > 0 else 0.0)
        # O quadro ANTES da transicao precisa ja ser estavel -- senao ele
        # proprio e a cauda de uma anomalia curta, e "persistiu" e ilusao.
        if d_lead >= 0.6 * d_passo[i]:
            continue
        cortes.append(i + 1)  # +1: indice t da metrica descreve o frame t
    return cortes


def drop_cut_segments(plan: list, cuts: list, tolerancia: int = 2) -> tuple:
    """Separa do plano os segmentos que sao so fronteira de corte.

    Tolerancia de +-2 quadros porque o corte vem do histograma e o segmento vem
    do residuo de fluxo; os dois marcam o mesmo evento com um quadro de
    diferenca conforme o arredondamento."""
    if not cuts:
        return plan, []
    manter, descartados = [], []
    for seg in plan:
        na_fronteira = any(seg["start"] - tolerancia <= c <= seg["end"] + tolerancia
                           for c in cuts)
        (descartados if na_fronteira else manter).append(seg)
    return manter, descartados


def measure(frames: list, flow_kind: str = "dis", progress=None) -> dict:
    """Série temporal de métricas. Índice t refere-se à transição t-1 -> t,
    então todas as listas têm len(frames)-1 entradas e começam em t=1."""
    flow_fn = _flow_engine(flow_kind)
    residual, residual_local, residual_local_raw = [], [], []
    luma_d, hf_d, mag = [], [], []
    worst_cell = []
    hists = [color_hist(frames[0])]
    prev = frames[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_hf = high_freq(prev_gray)
    for t in range(1, len(frames)):
        cur = frames[t]
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        fl = flow_fn(prev, cur)
        w = cv2.cvtColor(warp(prev, fl), cv2.COLOR_BGR2GRAY)
        # O resíduo é medido em luma: crominância acrescenta ruído sem
        # acrescentar sinal para o tipo de defeito que interessa aqui.
        err = np.abs(w.astype(np.float64) - cur_gray.astype(np.float64))
        residual.append(float(err.mean()))
        # MEDIDO 2026-08-24: a média do quadro NÃO enxerga defeito local. Um
        # rosto derretendo ocupa 0,77% do quadro a 768x512, então some na média.
        # O resíduo por célula resolve: divide em grade e guarda a pior célula,
        # que é justamente onde o defeito está.
        magmap = np.linalg.norm(fl, axis=2)
        score, raw, cy, cx = _block_max(err, magmap, GRID)
        # Guarda-se o valor JÁ normalizado pelo movimento da célula: é ele que
        # entra no z-score. O bruto vai à parte, para os pisos absolutos.
        residual_local.append(float(score))
        residual_local_raw.append(float(raw))
        worst_cell.append([int(cy), int(cx)])
        luma_d.append(float(cur_gray.mean() - prev_gray.mean()))  # COM SINAL, ver _is_flicker
        cur_hf = high_freq(cur_gray)
        hf_d.append(float(abs(cur_hf - prev_hf)))
        mag.append(float(np.linalg.norm(fl, axis=2).mean()))
        # Histograma de cor: serve so para achar CORTE (ver detect_cuts). E
        # barato perto do fluxo optico e evita uma segunda decodificacao.
        hists.append(color_hist(cur))
        prev, prev_gray, prev_hf = cur, cur_gray, cur_hf
        if progress and t % 40 == 0:
            progress(t, len(frames))
    return {"residual": residual, "residual_local": residual_local,
            "residual_local_raw": residual_local_raw,
            "luma_delta": luma_d, "hf_delta": hf_d, "flow_mag": mag,
            "_worst_cell": worst_cell, "_hists": hists}


def _is_flicker(luma_signed: np.ndarray, run: list, span: int = 2) -> bool:
    """Distingue FLICKER de mudança legítima de iluminação.

    MEDIDO 2026-08-24: sem esta checagem, um pôr do sol com a câmera orbitando
    -- brilho mudando de verdade, e bonito -- gerava 17 falsos positivos num
    clipe sadio.

    O que separa os dois é a REVERSÃO. Flicker sobe e volta no quadro seguinte;
    iluminação real deriva na mesma direção por vários frames. Então exige-se
    que o sinal do delta troque na vizinhança do pico."""
    lo = max(0, run[0] - span)
    hi = min(len(luma_signed), run[-1] + span + 1)
    seg = luma_signed[lo:hi]
    if len(seg) < 3:
        return False
    sg = np.sign(seg)
    sg = sg[sg != 0]
    if len(sg) < 3:
        return False
    return bool(np.any(sg[:-1] != sg[1:]))


def robust_z(series: list) -> np.ndarray:
    """z-score por mediana/MAD. Média e desvio seriam contaminados pelos
    próprios defeitos que queremos achar -- um pico grande inflaria o desvio e
    se esconderia atrás dele."""
    a = np.asarray(series, dtype=np.float64)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    scale = mad * 1.4826 if mad > 1e-9 else (a.std() or 1.0)
    return (a - med) / scale


# --------------------------------------------------------------------------
# classificação
# --------------------------------------------------------------------------
def local_contrast(series, window: int = 15):
    """Quanto cada ponto se destaca da PRÓPRIA VIZINHANÇA, não do clipe todo.

    Esta é a medida certa, e chegar nela custou duas tentativas erradas
    (2026-08-24):

      1. z-score contra o clipe inteiro: acusa pan rápido de câmera. O DIS
         subestima deslocamento grande, o resíduo sobe, e não há defeito nenhum.
      2. dividir o resíduo pelo movimento: mata o falso positivo do pan, mas
         mata JUNTO o defeito real -- uma mão derretendo também se move.

    O que separa os dois não é magnitude, é DURAÇÃO. Falha de fluxo por
    movimento de câmera sobe e desce suavemente ao longo de dezenas de frames,
    então a mediana da vizinhança sobe junto e o contraste some. Defeito de
    poucos frames destoa dos vizinhos imediatos por definição.

    A janela precisa ser maior que o defeito mais longo que se quer pegar, ou
    ele vira a própria linha de base e se esconde."""
    a = np.asarray(series, dtype=np.float64)
    half = window // 2
    padded = np.pad(a, half, mode="edge")
    base = np.array([np.median(padded[i:i + window]) for i in range(len(a))])
    return a - base


def classify(metrics: dict, *, z_res=Z_RESIDUAL, z_luma=Z_LUMA,
             z_res_local=Z_RESIDUAL_LOCAL) -> list:
    zr = robust_z(local_contrast(metrics["residual"], CONTRAST_WINDOW))
    zrl = robust_z(local_contrast(
        metrics.get("residual_local_raw") or metrics["residual"], CONTRAST_WINDOW))
    luma_signed = np.asarray(metrics["luma_delta"], dtype=np.float64)
    zl = robust_z(np.abs(luma_signed))
    zh = robust_z(metrics["hf_delta"])
    cells = metrics.get("_worst_cell") or []

    # O piso absoluto entra como AND: z alto e amplitude real. Um dos dois
    # sozinho não basta -- z sozinho marca ruído, amplitude sozinha marcaria
    # toda cena com muito movimento.
    abs_l = np.abs(luma_signed)
    abs_rl = np.asarray(metrics.get("residual_local_raw")
                        or metrics.get("residual_local") or metrics["residual"],
                        dtype=np.float64)
    flagged = sorted(set(np.where(zr > z_res)[0])
                     | set(np.where((zrl > z_res_local) & (abs_rl > MIN_RESIDUAL_LOCAL))[0])
                     | set(np.where((zl > z_luma) & (abs_l > MIN_LUMA_DELTA))[0]))
    segments, run = [], []
    for i in flagged:
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                segments.append(run)
            run = [i]
    if run:
        segments.append(run)

    plan = []
    for run in segments:
        # +1 porque o índice t da métrica descreve a transição para o frame t.
        # int() explícito: np.where devolve int64, que json.dump recusa.
        a, b = int(run[0]) + 1, int(run[-1]) + 1
        length = b - a + 1
        peak_r = float(zr[run].max())
        peak_rl = float(zrl[run].max())
        peak_l = float(zl[run].max())
        peak_h = float(zh[run].max())
        # "onde": célula da grade com pior erro no frame de pico do segmento.
        top = run[int(np.argmax(zrl[run]))]
        where = cells[top] if top < len(cells) else None

        luma_dominates = peak_l > z_luma and peak_r < z_res and peak_rl < z_res_local
        if luma_dominates and not _is_flicker(luma_signed, run):
            # Brilho mudou, mas na mesma direção: é a luz da cena, não defeito.
            continue
        if luma_dominates:
            action, why = "deflicker", "brilho oscila (sobe e volta) com geometria estável"
        elif peak_rl > z_res_local and peak_r < z_res:
            # Só a célula acusou: defeito localizado (rosto, mão, um objeto).
            action = "interpolate" if length <= MAX_INTERP_LEN else (
                "regenerate" if length <= MAX_REGEN_LEN else "report")
            why = f"defeito LOCAL por {length} frame(s) (média do quadro não acusa)"
        elif length <= MAX_INTERP_LEN:
            action, why = "interpolate", f"{length} frame(s), vizinhos sadios"
        elif length <= MAX_REGEN_LEN:
            action, why = "regenerate", f"deformação por {length} frames"
        else:
            action, why = "report", f"{length} frames — longo demais para remendo"

        plan.append({
            "start": a, "end": b, "length": length, "action": action, "reason": why,
            "z_residual": round(peak_r, 2), "z_residual_local": round(peak_rl, 2),
            "z_luma": round(peak_l, 2), "z_hf": round(peak_h, 2),
            "worst_cell_yx": where,
        })
    return plan


def global_flicker(metrics: dict) -> bool:
    """Flicker espalhado pelo clipe inteiro não vira segmento -- ele É a linha
    de base, então nenhum ponto se destaca. Detecta-se pela oscilação: o brilho
    troca de direção quadro a quadro em vez de derivar suavemente."""
    l = np.asarray(metrics["luma_delta"])
    if len(l) < 20:
        return False
    d = np.diff(l)
    flips = np.sum(np.sign(d[:-1]) != np.sign(d[1:])) / max(len(d) - 1, 1)
    return bool(flips > 0.6 and l.mean() > 1.0)


# --------------------------------------------------------------------------
# reparos
# --------------------------------------------------------------------------
RIFE_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tools", "rife", "rife-ncnn-vulkan.exe")
RIFE_MODEL = os.environ.get("LTX_RIFE_MODEL", "rife-v4.6")
_RIFE_DECIMAL = None    # separador decimal aceito pelo binário; ver repair_interpolate_rife


def rife_available() -> bool:
    return os.path.exists(RIFE_EXE)


# CORRIGIDO 2026-09-03: a nota anterior aqui ("RIFE quadricula acima de
# ~1024px") estava ERRADA -- diagnóstico de causa trocada. O quadriculado era
# do upscale (`upscale_video.ps1` com `-Model realesrgan-x4plus`, RRDBNet):
# esse modelo devolve lixo determinístico (grid de blocos sem relação com o
# conteúdo de entrada) neste build do ncnn-vulkan, INDEPENDENTE de resolução,
# tile size (-t, testado 0/200/400/2000 = 1 tile só) ou GPU (0 e 1, mesmo
# hash). `realesrgan-x4plus-anime` (mesma família RRDBNet) quebra igual, com
# outra cara. `realesr-animevideov3` (SRVGG, o padrão original do script,
# que eu tinha sobrescrito pra testar x4plus) sai limpo. RIFE testado DE NOVO
# em par de frames 1536x1024 corretamente upscalados (animevideov3): saiu
# perfeito, sem quadriculado, sem precisar de -u nem gpu específica. Ver
# MEMORIAL.md 3.56 (correção).
RIFE_MAX_LONG_SIDE = None  # sem limiar real conhecido; ver nota acima


def repair_interpolate_rife(frames: list, a: int, b: int, gpu: str = "auto") -> bool:
    """Reconstrói frames[a..b] com RIFE, que é interpolação APRENDIDA.

    Vantagem sobre o warp bidirecional: o RIFE sintetiza conteúdo plausível em
    oclusão -- onde o fluxo não tem correspondência e o warp só consegue
    esticar pixel. É a diferença entre reconstruir uma mão e borrar a mão.

    O binário aceita `-s` (0..1), então cada frame do vão é pedido na sua
    posição fracionária exata entre as duas âncoras, em vez de bisseção
    recursiva -- que só cobriria vãos de 2^k-1 frames.

    Devolve False se o RIFE não estiver instalado ou se falhar, para o
    chamador cair no warp -- mais fraco em oclusão, mas nunca quadricula."""
    if not rife_available():
        return False
    left, right = a - 1, b + 1
    if left < 0 or right >= len(frames):
        return False
    tmp = tempfile.mkdtemp(prefix="vdoc_rife_")
    try:
        p0 = os.path.join(tmp, "a.png")
        p1 = os.path.join(tmp, "b.png")
        cv2.imwrite(p0, frames[left])
        cv2.imwrite(p1, frames[right])
        n = right - left
        for i in range(a, b + 1):
            t = (i - left) / n
            out = os.path.join(tmp, f"o_{i}.png")
            img = None
            # MEDIDO 2026-08-24: o binário converte o `-s` com o LOCALE DO
            # SISTEMA. Nesta máquina (pt-BR) "0.2" vira 0 e ele recusa com
            # "invalid timestep"; "0,2" funciona. Fixar a vírgula quebraria numa
            # máquina em inglês, então tenta-se os dois e memoriza-se o que
            # funcionou -- uma vez por processo, não por frame.
            global _RIFE_DECIMAL
            seps = [_RIFE_DECIMAL] if _RIFE_DECIMAL else [".", ","]
            for sep in seps:
                cmd = [RIFE_EXE, "-0", p0, "-1", p1, "-o", out,
                       "-s", f"{t:.6f}".replace(".", sep), "-m", RIFE_MODEL]
                if gpu != "auto":
                    cmd += ["-g", str(gpu)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if os.path.exists(out):
                    img = cv2.imread(out)
                    if img is not None:
                        _RIFE_DECIMAL = sep
                        break
            if img is None:
                raise RuntimeError(r.stderr.strip()[-400:] or "RIFE não produziu saída")
            frames[i] = img
        return True
    except Exception as e:
        print(f"[doctor] RIFE falhou ({type(e).__name__}: {e}); usando warp")
        return False
    finally:
        for f in os.listdir(tmp):
            try:
                os.remove(os.path.join(tmp, f))
            except OSError:
                pass
        os.rmdir(tmp)


def repair_interpolate(frames: list, a: int, b: int, flow_kind="dis") -> None:
    """Reconstrói frames[a..b] a partir das âncoras boas a-1 e b+1.

    Warp bidirecional: cada frame sintético é a mistura das duas âncoras
    deformadas na proporção da sua posição no vão. Usar só a âncora da esquerda
    faria o movimento congelar e depois saltar no fim do trecho."""
    left, right = a - 1, b + 1
    if left < 0 or right >= len(frames):
        # Sem âncora dos dois lados só resta repetir a que existe.
        src = frames[right] if left < 0 else frames[left]
        for i in range(a, b + 1):
            frames[i] = src.copy()
        return
    flow_fn = _flow_engine(flow_kind)
    f_lr = flow_fn(frames[left], frames[right])
    f_rl = flow_fn(frames[right], frames[left])
    n = right - left
    for i in range(a, b + 1):
        t = (i - left) / n
        wl = warp(frames[left], f_lr * t)
        wr = warp(frames[right], f_rl * (1.0 - t))
        frames[i] = cv2.addWeighted(wl, 1.0 - t, wr, t, 0.0)


def repair_deflicker(frames: list, a: int, b: int, window: int = 15) -> None:
    """Reescala a luminância de cada frame para seguir a curva suavizada da
    vizinhança. Mexe só no ganho: geometria e cor relativa ficam intactas, que
    é o ponto -- flicker é problema fotométrico, não geométrico.

    MEDIANA, não média. MEDIDO 2026-08-24: com média, a curva-alvo é puxada
    pelos próprios frames defeituosos que ela deveria corrigir, e o reparo sai
    pela metade (z_luma 10,6 -> 6,5, em vez de sumir). A mediana ignora o pico.

    A correção é aplicada com margem além do segmento: um salto de brilho no
    frame t estraga a transição t-1->t E t->t+1, então tratar só t deixa a
    segunda metade do degrau no lugar."""
    margin = 2
    a, b = max(0, a - margin), min(len(frames) - 1, b + margin)
    lo, hi = max(0, a - window), min(len(frames), b + window + 1)
    means = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames[lo:hi]])
    half = window // 2
    padded = np.pad(means, half, mode="edge")
    target = np.array([np.median(padded[i:i + window]) for i in range(len(means))])
    for idx in range(a, b + 1):
        j = idx - lo
        if means[j] < 1e-6:
            continue
        gain = float(target[j] / means[j])
        # Ganho é limitado: um valor extremo aqui significa que o "flicker" era
        # na verdade um corte de cena, e clarear isso destruiria o plano.
        gain = max(0.75, min(1.33, gain))
        frames[idx] = np.clip(frames[idx].astype(np.float32) * gain, 0, 255).astype(np.uint8)


def repair_regenerate(frames: list, a: int, b: int, *, prompt: str, fps: float,
                      seed: int, log=print) -> bool:
    """Regenera o trecho pelo LTX, ancorado nos frames bons de cada lado.

    Este é o "frame bom 115 -> regenerar 116-125 -> frame bom 126": as âncoras
    entram como keyframes (LTXVAddGuideAdvanced), que é a mesma máquina do
    storyplay25. Devolve False -- sem alterar nada -- quando não dá para fazer
    com segurança, para o chamador cair no reparo mais barato."""
    import ltx25_backend

    left, right = max(0, a - REGEN_MARGIN - 1), min(len(frames) - 1, b + REGEN_MARGIN + 1)
    span = right - left + 1
    # O modelo só aceita 1 + múltiplo de 8 frames.
    n_frames = 1 + max(1, round((span - 1) / 8)) * 8
    h, w = frames[0].shape[:2]

    tmp = tempfile.mkdtemp(prefix="vdoc_regen_")
    try:
        p_left = os.path.join(tmp, "anchor_left.png")
        p_right = os.path.join(tmp, "anchor_right.png")
        cv2.imwrite(p_left, frames[left])
        cv2.imwrite(p_right, frames[right])
        out = os.path.join(tmp, "regen.mp4")
        log(f"[doctor] regenerando {left}..{right} ({n_frames} frames) via LTX")
        ltx25_backend.generate(
            prompt, out, width=w, height=h, num_frames=n_frames, frame_rate=fps,
            seed=seed, image_path=p_left, image_strength=1.0,
            # A âncora final entra como keyframe no fim do trecho. O índice tem
            # de ser múltiplo de 8 (o nó arredonda para baixo se não for).
            keyframes=[(p_right, (n_frames - 1) // 8 * 8, 1.0)],
            disable_audio=True, log_cb=lambda m: None,
        )
        new = read_frames(out)
        if len(new) < span:
            log(f"[doctor] regeneração devolveu {len(new)} frames para um vão de "
                f"{span}; mantendo o original")
            return False
        for i in range(left, right + 1):
            frames[i] = cv2.resize(new[i - left], (w, h))
        return True
    except Exception as e:
        log(f"[doctor] regeneração falhou ({type(e).__name__}: {e}); mantendo o original")
        return False
    finally:
        for f in os.listdir(tmp):
            try:
                os.remove(os.path.join(tmp, f))
            except OSError:
                pass
        os.rmdir(tmp)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def export_previews(video: str, plan: dict, outdir: str, pad: int = 3) -> list:
    """Uma tira PNG por segmento, com alguns frames bons de cada lado.

    Existe porque o detector NÃO é confiável o bastante para reparar sozinho.
    MEDIDO 2026-08-24: ele marcou como defeito uma mão ENTRANDO EM QUADRO --
    movimento legítimo e rápido -- e tanto o warp quanto o RIFE, ao interpolar
    entre as âncoras, APAGARAM a mão. As duas reparações "melhoraram" a métrica
    (pico de diferença caiu 38%) destruindo justamente o conteúdo que causava a
    diferença.

    Nenhuma métrica temporal separa com segurança "matéria surgindo porque o
    modelo errou" de "matéria surgindo porque entrou em cena". Quem separa é
    quem olha. Daí a tira: o reparo passa a ser escolha informada, não
    automatismo."""
    os.makedirs(outdir, exist_ok=True)
    frames = read_frames(video)
    h = frames[0].shape[0]
    scale = min(1.0, 200.0 / h)
    out = []
    for i, seg in enumerate(plan["segments"]):
        lo = max(0, seg["start"] - pad)
        hi = min(len(frames) - 1, seg["end"] + pad)
        tiles = []
        for t in range(lo, hi + 1):
            im = cv2.resize(frames[t], None, fx=scale, fy=scale)
            # Frames marcados ganham borda: sem isso não dá para saber, olhando
            # a tira, quais o detector acusou e quais são a referência sadia.
            color = (0, 0, 255) if seg["start"] <= t <= seg["end"] else (0, 160, 0)
            im = cv2.copyMakeBorder(im, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=color)
            cv2.putText(im, str(t), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(im)
        path = os.path.join(outdir, f"seg{i:02d}_{seg['start']}-{seg['end']}_{seg['action']}.png")
        cv2.imwrite(path, np.hstack(tiles))
        out.append(path)
    return out


def do_analyze(args) -> dict:
    info = probe(args.video)
    frames = read_frames(args.video)
    print(f"[doctor] {os.path.basename(args.video)}: {len(frames)} frames "
          f"{info['width']}x{info['height']} @ {info['fps']:.3f} fps")
    m = measure(frames, args.flow,
                progress=lambda t, n: print(f"[doctor] medindo {t}/{n}", flush=True))
    plan = classify(m, z_res=args.z_residual, z_luma=args.z_luma,
                    z_res_local=args.z_residual_local)

    # CORTES NAO SAO DEFEITOS. Num filme montado, toda fronteira entre planos
    # aparece como o maior pico de mudanca do arquivo -- e reparar isso mistura
    # dois enquadramentos. Ver detect_cuts.
    if getattr(args, "cuts", None):
        # O pipeline SABE as fronteiras (contagem de frames por clipe); quando
        # ele informa, nao ha o que inferir.
        cortes = sorted(int(x) for x in str(args.cuts).replace(";", ",").split(",") if x.strip())
    elif getattr(args, "no_cut_detection", False):
        cortes = []
    else:
        cortes = detect_cuts(m)
    plan, em_corte = drop_cut_segments(plan, cortes)
    result = {"video": os.path.abspath(args.video), "frames": len(frames),
              "fps": info["fps"], "width": info["width"], "height": info["height"],
              "global_flicker": global_flicker(m), "segments": plan,
              "cuts": cortes, "segments_at_cuts": em_corte,
              "stats": {k: {"mediana": round(float(np.median(v)), 3),
                            "max": round(float(np.max(v)), 3)}
                        for k, v in m.items() if not k.startswith("_")}}
    print()
    print(f"resíduo de fluxo: mediana {result['stats']['residual']['mediana']}, "
          f"pico {result['stats']['residual']['max']}")
    if result["global_flicker"]:
        print("FLICKER GLOBAL detectado (oscilação de brilho no clipe inteiro)")
    if cortes:
        print(f"cortes detectados nos frames {cortes} -- segmentos nessas "
              f"fronteiras NAO sao defeito e foram descartados")
    if em_corte:
        for s in em_corte:
            print(f"  [corte] frames {s['start']:>4}-{s['end']:<4} "
                  f"z_glob={s['z_residual']:>6} -- ignorado")
    if not plan:
        print("nenhum segmento anômalo acima dos limiares — nada a corrigir")
    for s in plan:
        print(f"  frames {s['start']:>4}-{s['end']:<4} ({s['length']:>2}f)  "
              f"z_glob={s['z_residual']:>5}  z_local={s['z_residual_local']:>5}  "
              f"z_luma={s['z_luma']:>5}  -> {s['action'].upper():<12} {s['reason']}")
    if plan:
        print("\nATENCAO: revise antes de reparar. O detector acha MUDANCA, e nem "
              "toda mudanca e defeito -- movimento rapido legitimo (uma mao "
              "entrando em quadro) aparece igual, e repara-lo APAGA o movimento. "
              "Use --previews para gerar as tiras de revisao.")
    if args.plan:
        json.dump(result, open(args.plan, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"plano salvo em {args.plan}")
    if getattr(args, "previews", None):
        paths = export_previews(args.video, result, args.previews)
        print(f"{len(paths)} tira(s) de revisao em {args.previews}")
    return result


def do_repair(args, plan: dict | None = None) -> str:
    plan = plan or json.load(open(args.plan, encoding="utf-8"))
    only = getattr(args, "only", None)
    if only:
        keep = {int(x) for x in str(only).replace(" ", "").split(",") if x != ""}
        plan = dict(plan, segments=[s for i, s in enumerate(plan["segments"]) if i in keep])
        print(f"[doctor] reparando apenas os segmentos {sorted(keep)}")
    frames = read_frames(args.video)
    applied = {"interpolate": 0, "deflicker": 0, "regenerate": 0, "report": 0, "falhou": 0}

    # Do fim para o começo: regeneração pode trocar frames, e trabalhar de trás
    # para frente mantém os índices dos segmentos ainda não tratados válidos.
    for s in sorted(plan["segments"], key=lambda x: -x["start"]):
        a, b, act = s["start"], s["end"], s["action"]
        if act == "interpolate":
            if args.interp == "warp" or not repair_interpolate_rife(frames, a, b):
                repair_interpolate(frames, a, b, args.flow)
        elif act == "deflicker":
            repair_deflicker(frames, a, b)
        elif act == "regenerate":
            if not args.prompt:
                print(f"[doctor] frames {a}-{b} pediam regeneração, mas --prompt não "
                      f"foi dado; caindo para interpolação (resultado inferior)")
                if args.interp == "warp" or not repair_interpolate_rife(frames, a, b):
                    repair_interpolate(frames, a, b, args.flow)
                applied["falhou"] += 1
                continue
            ok = repair_regenerate(frames, a, b, prompt=args.prompt, fps=plan["fps"],
                                   seed=args.seed)
            if not ok:
                applied["falhou"] += 1
                continue
        else:
            print(f"[doctor] frames {a}-{b}: {s['reason']} — não tratado "
                  f"automaticamente, regenere o plano inteiro se incomodar")
        applied[act] = applied.get(act, 0) + 1

    if plan.get("global_flicker") and not args.no_global_deflicker:
        print("[doctor] aplicando deflicker global")
        repair_deflicker(frames, 0, len(frames) - 1)

    write_frames(frames, args.output, plan["fps"], audio_from=args.video)
    print(f"[doctor] pronto -> {args.output}")
    print("[doctor] aplicados: " + ", ".join(f"{k}={v}" for k, v in applied.items() if v))
    return args.output


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnóstico e reparo temporal de clipes LTX")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("video")
        p.add_argument("--flow", choices=["dis", "raft"], default="dis",
                       help="dis = OpenCV, offline. raft = melhor, baixa pesos na 1a vez.")
        p.add_argument("--z-residual", type=float, default=Z_RESIDUAL)
        p.add_argument("--z-luma", type=float, default=Z_LUMA)
        p.add_argument("--z-residual-local", type=float, default=Z_RESIDUAL_LOCAL)
        p.add_argument("--interp", choices=["rife", "warp"], default="rife",
                       help="rife = interpolação aprendida (padrão, cai para warp "
                            "se o binário faltar); warp = fluxo bidirecional")

    a = sub.add_parser("analyze", help="mede e classifica, sem alterar o vídeo")
    common(a)
    a.add_argument("--cuts", default=None,
                   help="frames onde comeca cada plano, separados por virgula. "
                        "Use quando souber (o pipeline sabe: e a contagem de "
                        "frames por clipe). Dispensa a deteccao automatica.")
    a.add_argument("--no-cut-detection", action="store_true",
                   help="nao detectar cortes. Use so em clipe UNICO, onde nao ha "
                        "corte para confundir -- num filme montado, sem isto as "
                        "fronteiras entre planos viram falso positivo de maior "
                        "pontuacao.")
    a.add_argument("--plan", default=None, help="salva o plano em JSON")
    a.add_argument("--previews", default=None,
                   help="pasta para as tiras de revisão (uma por segmento)")

    r = sub.add_parser("repair", help="aplica um plano existente")
    common(r)
    r.add_argument("--plan", required=True)
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--prompt", default=None, help="necessário para regeneração parcial")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--no-global-deflicker", action="store_true")
    r.add_argument("--only", default=None,
                   help="índices dos segmentos a reparar, ex 0,2,5 (padrão: todos)")

    d = sub.add_parser("doctor", help="analisa e repara numa passada")
    common(d)
    d.add_argument("-o", "--output", required=True)
    d.add_argument("--plan", default=None)
    d.add_argument("--prompt", default=None)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--no-global-deflicker", action="store_true")
    d.add_argument("--previews", default=None)
    d.add_argument("--only", default=None)

    args = ap.parse_args()
    if args.cmd == "analyze":
        do_analyze(args)
    elif args.cmd == "repair":
        do_repair(args)
    else:
        plan = do_analyze(args)
        if not plan["segments"] and not plan["global_flicker"]:
            print("[doctor] nada a fazer; vídeo não foi reescrito")
            return 0
        do_repair(args, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
