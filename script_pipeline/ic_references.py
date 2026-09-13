"""Referencias visuais para IC-LoRA na passada de VIDEO: folha de ingredientes e
pseudo-video do MSR (Multiple Subject Reference).

Por que isso existe: o still do FLUX fixa o PRIMEIRO quadro e o personagem deriva
depois (MEMORIAL 3.38: a ancora de still trava o quadro 0, o estilo escorrega ao longo
do clipe). Um IC-LoRA recebe a referencia como tokens de contexto durante a amostragem
inteira -- a folha do personagem continua visivel para o modelo no quadro 200, nao so
no 0.

Dois formatos, porque os dois LoRAs foram treinados com entradas diferentes:

- Ingredients (Lightricks, 2.3 e 2.5): UMA imagem com paineis sobre fundo PRETO, sem
  texto, repetida em todos os quadros. Prompt em duas partes, "Reference sheet: ..." e
  "Generated video: ...".
- MSR V2 (LiconStudio, treinado no 2.3): sequencia curta (17-65 quadros) em que cada
  sujeito ocupa um grupo de quadros alinhado ao latente (o quadro 0 sozinho, depois 8
  por latente), sujeitos contidos sobre fundo BRANCO e o cenario cobrindo o quadro no
  fim. Prompt nomeia "Image 1", "Image 2"... O node do autor (ComfyUI-Licon-MSR) so
  monta essa sequencia de imagens; a mesma alocacao esta reescrita aqui para nao
  instalar codigo de terceiros. O resto sao nos nativos (LTXICLoRALoaderModelOnly +
  LTXAddVideoICLoRAGuide + LTXVCropGuides), ver ltx25_backend._apply_ic_guide.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

MSR_FRAME_COUNTS = (17, 25, 33, 41, 49, 57, 65)


def _latents(frames: int) -> int:
    return max(1, frames) if frames <= 1 else round((frames - 1) / 8) + 1


def _frame_span(lat_ini: int, lat_fim: int) -> tuple[int, int]:
    """Quadros cobertos por um intervalo de latentes: o latente 0 e so o quadro 0; o
    latente k (k >= 1) cobre os quadros 8k-7 .. 8k."""
    ini = 0 if lat_ini <= 0 else 1 + (lat_ini - 1) * 8
    fim = 0 if lat_fim <= 0 else lat_fim * 8
    return ini, fim


def _msr_latent_counts(n: int, budget: int) -> list[int]:
    """Latentes por sujeito. O primeiro ganha mais; os outros chegam a 2 antes de o
    primeiro passar de 3; o que sobra roda em ciclo. E a regra do treino do MSR --
    mudar a distribuicao muda o que o LoRA ve."""
    counts = [1] * n
    extra = budget - n
    if extra > 0:
        counts[0] += 1
        extra -= 1
    i = 1
    while extra > 0 and n > 1 and any(c < 2 for c in counts[1:]):
        if counts[i] < 2:
            counts[i] += 1
            extra -= 1
        i = i + 1 if i + 1 < n else 1
    if extra > 0 and counts[0] < 3:
        counts[0] += 1
        extra -= 1
    i = 0
    while extra > 0:
        counts[i] += 1
        extra -= 1
        i = (i + 1) % n
    return counts


def msr_frame_assignment(n_subjects: int, frame_count: int) -> list[int]:
    """Indice da imagem em cada quadro: 0..n-1 sujeitos, n = cenario."""
    fundo = n_subjects
    quadros = [fundo] * frame_count
    if n_subjects == 0:
        return quadros
    latentes = _latents(frame_count)
    budget = max(0, latentes - 1)  # o ultimo latente sempre fica para o cenario
    if budget >= n_subjects:
        cursor = 0
        for idx, c in enumerate(_msr_latent_counts(n_subjects, budget)):
            ini, fim = _frame_span(cursor, cursor + c - 1)
            cursor += c
            for f in range(max(0, ini), min(frame_count - 1, fim) + 1):
                quadros[f] = idx
    else:
        total = max(1, _frame_span(0, max(0, latentes - 2))[1] + 1)
        for idx in range(n_subjects):
            ini = idx * total // n_subjects
            fim = total - 1 if idx == n_subjects - 1 else (idx + 1) * total // n_subjects - 1
            for f in range(ini, min(frame_count - 1, max(ini, fim)) + 1):
                quadros[f] = idx
    return quadros


def runs(assignment: list[int]) -> list[tuple[int, int]]:
    """[0,0,0,1,1,2] -> [(0,3),(1,2),(2,1)] -- vira RepeatImageBatch + ImageBatch."""
    out: list[tuple[int, int]] = []
    for idx in assignment:
        if out and out[-1][0] == idx:
            out[-1] = (idx, out[-1][1] + 1)
        else:
            out.append((idx, 1))
    return out


def pick_msr_frame_count(num_frames: int) -> int | None:
    """Maior sequencia do treino que cabe no clipe (a guia nao pode passar dele)."""
    cabem = [f for f in MSR_FRAME_COUNTS if f <= num_frames]
    return max(cabem) if cabem else None


def _contain(src: Image.Image, w: int, h: int, fundo: tuple) -> Image.Image:
    escala = min(w / src.width, h / src.height)
    nw = max(1, min(w, round(src.width * escala)))
    nh = max(1, min(h, round(src.height * escala)))
    tela = Image.new("RGB", (w, h), fundo)
    tela.paste(src.resize((nw, nh), Image.LANCZOS), ((w - nw) // 2, (h - nh) // 2))
    return tela


def _cover(src: Image.Image, w: int, h: int) -> Image.Image:
    escala = max(w / src.width, h / src.height)
    nw, nh = max(w, round(src.width * escala)), max(h, round(src.height * escala))
    r = src.resize((nw, nh), Image.LANCZOS)
    esq, topo = (nw - w) // 2, (nh - h) // 2
    return r.crop((esq, topo, esq + w, topo + h))


def build_msr_guide(subjects: list[str], background: str, width: int, height: int,
                    num_frames: int, out_dir: Path) -> tuple[list[tuple[str, int]], int] | None:
    """Prepara as imagens (sujeitos contidos em branco, cenario cobrindo o quadro) e
    devolve (corridas, quadros_da_guia) no formato de ic_lora['frames'] do backend.
    None quando o clipe e curto demais para a menor sequencia do treino."""
    fc = pick_msr_frame_count(num_frames)
    subjects = [s for s in subjects if s][:4]
    if fc is None or not background:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for k, s in enumerate(subjects):
        p = out_dir / f"msr_sujeito{k + 1}.png"
        _contain(Image.open(s).convert("RGB"), width, height, (255, 255, 255)).save(p)
        arquivos.append(str(p))
    pb = out_dir / "msr_cenario.png"
    _cover(Image.open(background).convert("RGB"), width, height).save(pb)
    arquivos.append(str(pb))
    corridas = [(arquivos[idx], n) for idx, n in runs(msr_frame_assignment(len(subjects), fc))]
    return corridas, fc


def build_ingredients_sheet(panels: list[str], width: int, height: int, out_path: Path,
                            gap: int = 16) -> str:
    """Paineis lado a lado sobre preto, cada um contido no seu retangulo, sem texto --
    o formato do model card. Na resolucao do clipe: o no redimensiona a guia, mas no
    aspecto certo nada e esticado."""
    imgs = [Image.open(p).convert("RGB") for p in panels if p]
    if not imgs:
        raise ValueError("folha de ingredientes sem nenhum painel")
    tela = Image.new("RGB", (width, height), (0, 0, 0))
    n = len(imgs)
    larg = (width - gap * (n + 1)) // n
    alt = height - 2 * gap
    for k, im in enumerate(imgs):
        tela.paste(_contain(im, larg, alt, (0, 0, 0)), (gap + k * (larg + gap), gap))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tela.save(out_path)
    return str(out_path)


def _curto(texto: str, limite: int = 160) -> str:
    """Primeira frase do descritor. O video_prompt ja carrega o descritor inteiro; aqui
    so precisa amarrar NOME a IMAGEM sem duplicar o paragrafo."""
    texto = (texto or "").strip()
    frase = texto.split(". ")[0].rstrip(".")
    return frase if len(frase) <= limite else frase[:limite].rsplit(" ", 1)[0]


def msr_prompt(subjects: list[tuple[str, str]], video_prompt: str) -> str:
    partes = [f"Image {k + 1}: {nome}" + (f", {_curto(desc)}" if desc else "")
              for k, (nome, desc) in enumerate(subjects)]
    partes.append(f"Image {len(subjects) + 1}: the location and background")
    return ". ".join(partes) + ". " + video_prompt


def ingredients_prompt(panels: list[str], video_prompt: str) -> str:
    return ("Reference sheet: " + "; ".join(f"panel {k + 1} shows {_curto(d)}" for k, d in enumerate(panels))
            + ". Generated video: " + video_prompt)


# Preferencia por modo: o retreino 2.5 quando a licenca ja foi aceita e o arquivo
# existe; senao o 2.3, que os workflows oficiais 2.5 carregam.
IC_MODE_LORAS = {"ingredients": ("ingredients-2.5", "ingredients-2.3"), "msr": ("msr-2.3-v2",)}


def resolve_ic_lora(mode: str, override: str | None = None) -> str | None:
    import ltx_loras
    if override:
        return ltx_loras.resolve(override)[1]
    for key in IC_MODE_LORAS.get(mode, ()):
        spec = ltx_loras.BY_KEY[key]
        if ltx_loras.installed_path(spec.local):
            return spec.local
    return None


def shot_ic_spec(mode: str, shot: dict, still: str, refs: dict, location_ref: str | None,
                 descriptors: dict | None, *, width: int, height: int, num_frames: int,
                 work_dir: Path, lora: str | None, strength: float = 1.0,
                 guide_strength: float = 1.0,
                 include_location: bool = False) -> tuple[dict | None, str, str]:
    """(ic_lora do backend, prompt final, fragmento da chave de cache) de UM plano.

    Devolve (None, video_prompt, "") quando o plano nao tem personagem com referencia:
    um plano de cenario so tem o still, e o still ja entra como primeiro quadro.

    Locacao FORA da folha por padrao. VISTO na folha do primeiro teste (2026-09-12): o
    "still de locacao" da cena e o primeiro plano aberto, e plano aberto quase sempre
    tem gente -- o painel de locacao da MEI-LI mostrava a XIAO-LAN, e o Ingredients e
    treinado para reproduzir o que esta na folha. O still do proprio plano ja entra como
    primeiro quadro e fixa o cenario; pelo mesmo motivo, e ele o cenario do MSR."""
    video_prompt = shot["video_prompt"]
    if mode == "off" or not lora:
        return None, video_prompt, ""
    sujeitos = [s for s in (shot.get("subject"), shot.get("co_subject")) if s and refs.get(s)]
    if not sujeitos:
        return None, video_prompt, ""
    imagens = [refs[s] for s in sujeitos]
    descs = [(s, (descriptors or {}).get(s, "")) for s in sujeitos]
    locacao = None
    if mode == "msr":
        # O slot de cenario e obrigatorio no MSR.
        r = build_msr_guide(imagens, still, width, height, num_frames, work_dir)
        if r is None:
            return None, video_prompt, ""
        corridas, quadros = r
        ic = {"frames": corridas, "crop": "center",
              "describe": f"MSR: {len(imagens)} sujeito(s) + cenario em {quadros} quadros"}
        prompt = msr_prompt(descs, video_prompt)
    elif mode == "ingredients":
        locacao = location_ref if include_location else None
        paineis = imagens + ([locacao] if locacao else [])
        folha = build_ingredients_sheet(paineis, width, height, work_dir / "ingredients_sheet.png")
        ic = {"frames": [(folha, num_frames)], "crop": "disabled",
              "describe": f"folha de ingredientes com {len(paineis)} painel(is)"}
        rotulos = [f"{n}, {d}" if d else n for n, d in descs] + (["the location"] if locacao else [])
        prompt = ingredients_prompt(rotulos, video_prompt)
    else:
        raise ValueError(f"modo de IC-LoRA desconhecido: {mode!r}")
    ic.update({"lora": lora, "strength": float(strength), "guide_strength": float(guide_strength)})
    chave = "|ic=" + ":".join([mode, lora, f"{strength:g}", f"{guide_strength:g}"]
                              + [Path(p).name for p in imagens]
                              + [Path(locacao).name if locacao else ""])
    return ic, prompt, chave
