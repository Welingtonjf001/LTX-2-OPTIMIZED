"""Importa vozes novas para a biblioteca do XTTS, na convenção que já existe.

A CONVENÇÃO (não é escolha minha -- é o que `voice_library` já lê)

    speakers/01_vozes_emotivas/<ID_DA_VOZ>/<NN>_<emocao>.wav

  - `ID_DA_VOZ` começa com F ou M: `voice_library.voice_gender()` decide o
    gênero por essa letra, e gênero errado é erro audível imediato.
  - `<emocao>` tem de ser um dos 17 slugs conhecidos. `pick_take()` casa o
    nome do arquivo com o slug; um nome fora da lista é INVISÍVEL para ele,
    e a fala cai no neutro em silêncio -- exatamente o defeito que
    `emotion_director` acabou de corrigir do outro lado.
  - `NN` é ordem de exibição, não semântica.

POR QUE UM IMPORTADOR, E NÃO COPIAR NA MÃO

Vozes vindas de fora chegam com o nome que o autor delas quis: "angry.wav",
"Raiva_v2.WAV", "03 - triste.wav", "kristin_happy_take2.wav". Copiar sem
normalizar produz uma pasta que PARECE certa e não funciona -- e o sintoma
aparece só depois, na síntese, como voz sem emoção.

Aqui cada arquivo é classificado contra os slugs (por nome, com sinônimos em
pt/en), e o que não casa é REPORTADO em vez de adivinhado. Adivinhar emoção
errada é pior que deixar de fora: o neutro pelo menos não contradiz a cena.

USO
    python -m script_pipeline.import_voices --from PASTA [--dry-run]
    python -m script_pipeline.import_voices --from PASTA --voice-id F07_nome --apply
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ordem canônica dos 17 takes: define o prefixo NN e é a fonte de verdade dos
# slugs. Copiada da biblioteca existente para não divergir dela em silêncio.
SLUGS_ORDEM = [
    "raiva", "surpresa", "desanimo", "angustia", "tristeza", "desapontamento",
    "espontanea_entusiasmada", "com_medo", "alegre", "apaixonada", "sensual",
    "desdem", "confusa", "excitada", "calma", "neutra", "grito",
]

# Sinônimos por slug, pt e en. Deliberadamente generoso na entrada: o custo de
# reconhecer um arquivo a mais é zero, e o de NÃO reconhecer é uma voz perdida.
SINONIMOS: dict[str, tuple[str, ...]] = {
    "raiva": ("raiva", "irritad", "furios", "brav", "angry", "anger", "furious", "rage", "mad"),
    "surpresa": ("surpres", "espant", "sobressalt", "surprise", "shock", "astonish", "startled"),
    "desanimo": ("desanim", "abatid", "resignad", "discourag", "downcast", "weary"),
    "angustia": ("angust", "aflit", "afflic", "anguish", "distress"),
    "tristeza": ("triste", "chorand", "pesar", "sad", "sorrow", "grief", "crying", "tearful"),
    "desapontamento": ("desapont", "decepcion", "frustrad", "disappoint", "letdown"),
    "espontanea_entusiasmada": ("espontan", "entusiasm", "empolg", "enthusias", "eager", "excited_spontaneous"),
    "com_medo": ("medo", "assustad", "temeros", "afraid", "fear", "scared", "frightened"),
    "alegre": ("alegr", "feliz", "content", "happy", "cheer", "joy", "glad"),
    "apaixonada": ("apaixon", "amoros", "tern", "afetuos", "loving", "tender", "affection", "romantic"),
    "sensual": ("sensual", "sedut", "sultry", "seduct"),
    "desdem": ("desdem", "despre", "ironi", "sarcas", "disdain", "scorn", "contempt"),
    "confusa": ("confus", "perdid", "hesitan", "confused", "puzzled", "uncertain"),
    "excitada": ("excitad", "agitad", "urgent", "eletr", "excited", "agitated", "urgent"),
    "calma": ("calma", "seren", "trangu", "tranqu", "gentil", "calm", "serene", "gentle", "soft"),
    "neutra": ("neutr", "normal", "base", "padrao", "default", "neutral", "flat"),
    "grito": ("grito", "gritand", "berr", "shout", "scream", "yell"),
}

XTTS_SPEAKERS = Path(r"E:\Users\home\Documents\xtts\webui\speakers")
EMOTIVE_DIR = "01_vozes_emotivas"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _fold(t: str) -> str:
    """minúsculas, sem acento, só alfanumérico -- para casar 'Angústia' com
    'angustia' e 'ANGRY_v2' com 'angry'."""
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def classify(nome: str) -> str | None:
    """Slug de emoção a partir do nome do arquivo, ou None.

    Percorre na ORDEM de SLUGS_ORDEM, e dentro dela o primeiro sinônimo que
    aparecer vence. 'grito' vem por último na lista mas é checado à parte
    antes de 'raiva', porque "grito de raiva" é grito -- o take mais extremo
    manda."""
    f = _fold(nome)
    if any(s in f for s in SINONIMOS["grito"]):
        return "grito"
    for slug in SLUGS_ORDEM:
        if slug == "grito":
            continue
        if any(s in f for s in SINONIMOS[slug]):
            return slug
    return None


def scan(origem: Path) -> dict:
    """Agrupa os áudios por PASTA (uma pasta = uma voz) e classifica cada um.

    Áudio solto na raiz vira um grupo `(raiz)`: pode ser uma voz cujo autor não
    criou subpasta, e descartar seria perder material."""
    grupos: dict[str, list[Path]] = {}
    for p in sorted(origem.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            chave = p.parent.name if p.parent != origem else "(raiz)"
            grupos.setdefault(chave, []).append(p)

    out = {}
    for voz, arquivos in grupos.items():
        itens = []
        for a in arquivos:
            # Tenta o nome do arquivo; se não casar, tenta o da pasta -- há
            # coleções em que a emoção está no diretório e o arquivo é "01.wav".
            slug = classify(a.stem) or classify(a.parent.name)
            itens.append({"origem": a, "slug": slug})
        out[voz] = itens
    return out


def sugerir_id(nome_pasta: str, existentes: set[str]) -> str:
    """Propõe um ID na convenção `F07_descricao` / `M07_descricao`.

    O gênero NÃO é adivinhado a partir do nome: se não houver marca explícita,
    devolve com `?` para o usuário decidir. Voz masculina rotulada como
    feminina é erro que só aparece ouvindo, e tarde."""
    f = _fold(nome_pasta)
    genero = None
    if re.match(r"^[fm]\d", nome_pasta.lower()):
        genero = nome_pasta[0].upper()
    elif any(w in f for w in ("female", "feminin", "mulher", "woman", "girl")):
        genero = "F"
    elif any(w in f for w in ("male", "masculin", "homem", "man", "boy")):
        genero = "M"
    prefixo = genero or "?"
    usados = [int(m.group(1)) for e in existentes
              if (m := re.match(rf"^{prefixo}(\d+)", e))] if genero else []
    n = max(usados, default=0) + 1
    limpo = re.sub(r"\s+", "_", f) or "voz"
    return f"{prefixo}{n:02d}_{limpo[:28]}"


def relatorio(grupos: dict, existentes: set[str]) -> str:
    l = []
    for voz, itens in grupos.items():
        casados = [i for i in itens if i["slug"]]
        orfaos = [i for i in itens if not i["slug"]]
        sugerido = sugerir_id(voz, existentes)
        l.append(f"PASTA {voz!r}  ->  id sugerido: {sugerido}")
        l.append(f"  {len(casados)}/{len(itens)} arquivo(s) classificado(s)")
        vistos = {}
        for i in casados:
            vistos.setdefault(i["slug"], []).append(i["origem"].name)
        for slug in SLUGS_ORDEM:
            if slug in vistos:
                extra = f"  (+{len(vistos[slug])-1} repetido)" if len(vistos[slug]) > 1 else ""
                l.append(f"    {slug:<26} <- {vistos[slug][0]}{extra}")
        faltando = [s for s in SLUGS_ORDEM if s not in vistos]
        if faltando:
            l.append(f"    FALTAM: {', '.join(faltando)}")
        for i in orfaos:
            l.append(f"    NAO CLASSIFICADO: {i['origem'].name}")
        if sugerido.startswith("?"):
            l.append("    ATENCAO: genero indefinido -- passe --voice-id F0X_nome ou M0X_nome")
        l.append("")
    return "\n".join(l)


def aplicar(grupos: dict, destino: Path, voice_id: str | None, log=print) -> int:
    copiados = 0
    for voz, itens in grupos.items():
        existentes = {p.name for p in (destino).iterdir()} if destino.exists() else set()
        vid = voice_id or sugerir_id(voz, existentes)
        if vid.startswith("?"):
            log(f"[import_voices] {voz!r}: genero indefinido; pulando. Use --voice-id.")
            continue
        alvo = destino / vid
        alvo.mkdir(parents=True, exist_ok=True)
        usados = set()
        for slug in SLUGS_ORDEM:
            cand = [i for i in itens if i["slug"] == slug]
            if not cand:
                continue
            nn = SLUGS_ORDEM.index(slug) + 1
            src = cand[0]["origem"]
            dst = alvo / f"{nn:02d}_{slug}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            usados.add(id(cand[0]))
            copiados += 1
            log(f"  {src.name}  ->  {vid}/{dst.name}")
        naoclass = [i for i in itens if not i["slug"]]
        if naoclass:
            # Não some com o que não foi entendido: fica ao lado, para o usuario
            # renomear e reimportar. Descartar em silencio seria perder material.
            quar = alvo / "_nao_classificados"
            quar.mkdir(exist_ok=True)
            for i in naoclass:
                shutil.copy2(i["origem"], quar / i["origem"].name)
            log(f"  {len(naoclass)} arquivo(s) sem emocao reconhecida -> {vid}/_nao_classificados/")
    return copiados


def main() -> int:
    ap = argparse.ArgumentParser(description="Importa vozes novas na convencao do XTTS")
    ap.add_argument("--from", dest="origem", required=True)
    ap.add_argument("--speakers", default=str(XTTS_SPEAKERS))
    ap.add_argument("--voice-id", default=None,
                    help="forca o ID (ex: F07_jovem_doce). Use quando o genero nao for obvio.")
    ap.add_argument("--apply", action="store_true", help="copia de fato (sem isto, so relata)")
    args = ap.parse_args()

    origem = Path(args.origem)
    if not origem.exists():
        print(f"[import_voices] pasta nao encontrada: {origem}", file=sys.stderr)
        return 1
    destino = Path(args.speakers) / EMOTIVE_DIR
    existentes = {p.name for p in destino.iterdir() if p.is_dir()} if destino.exists() else set()

    grupos = scan(origem)
    if not grupos:
        print(f"[import_voices] nenhum audio em {origem}")
        return 1
    print(f"[import_voices] {sum(len(v) for v in grupos.values())} audio(s) em "
          f"{len(grupos)} pasta(s). Vozes ja existentes: {len(existentes)}\n")
    print(relatorio(grupos, existentes))

    if not args.apply:
        print("Nada foi copiado (use --apply). Confira os IDs e as emocoes acima primeiro.")
        return 0
    n = aplicar(grupos, destino, args.voice_id)
    print(f"\n[import_voices] {n} arquivo(s) copiado(s) para {destino}")
    print("Rode emotion_director --recast para reconsiderar o elenco com as vozes novas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
