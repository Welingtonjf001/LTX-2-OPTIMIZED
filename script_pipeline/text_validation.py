"""Validacao pos-PARSE, ainda em texto: o `shot_plan.json` inteiro (cenas, personagens,
objetos, locais, falas) bate com o roteiro-FONTE original -- antes de gastar GPU nos stills.

POR QUE EXISTE

MEDIDO 2026-09-29 (REENTRY WINDOW): dois bugs reais so foram achados depois de gerar 71
stills e o usuario/uma auditoria externa olharem as IMAGENS a mao -- (1) o LLM de
reestruturacao (`prose_to_screenplay.py`) colou as falas finais de 3 personagens sob um
cabecalho de cena errado (mistura de local externo/interno numa prosa so); (2) o mesmo LLM
copiou um EXEMPLO de sintaxe do proprio prompt ("ESTILO VISUAL: polished hand-drawn cel
animation...") como se fosse extraido do roteiro. Os dois eram detectaveis em TEXTO, sem
nenhuma imagem -- so ninguem comparava o `shot_plan.json` final contra o roteiro-fonte antes
da geracao comecar. Ate 2026-09-29 as unicas defesas existentes eram pontuais: fidelidade de
DESCRITOR por personagem (`cast_characters._enforce_descriptor_fidelity`), local/estilo
"dado lido vence palpite" (`parse_screenplay._apply_setting`/`extract_art_direction`), e
completude do roteiro-fonte em si (`screenplay_gaps.py`) -- nenhuma olhava o `shot_plan.json`
DERIVADO como um todo contra o texto original.

O QUE FAZ

1. Checagens DETERMINISTICAS (sempre, sem modelo, poucos milissegundos):
   - Todo personagem do cast aparece literalmente no roteiro-fonte (senao, nome inventado
     ou digitado errado em algum lugar da cadeia).
   - Toda fala com "quote" no shot_plan bate (por normalizacao leve, nao exata) com uma fala
     do roteiro-fonte -- pega fala reescrita/alucinada pelo enriquecimento.
   - Cada local do shot_plan tem sobreposicao lexical minima com o roteiro-fonte (pega local
     inventado do zero pela reestruturacao).
   - Contradicao de descritor ENTRE PLANOS DA MESMA CENA (reaproveita
     `cast_characters._attribute_contradiction` -- mesma logica de hoje, aplicada shot-a-shot
     em vez de só na hora de gerar o cast.json).
2. Checagem SEMANTICA por LLM (opcional, `--engine`): para cada cena, manda o trecho de
   texto-fonte + os `storyboard_prompt`/`video_prompt` dos planos derivados dela, pede pra
   apontar objeto/personagem/local que aparece no plano mas NAO no texto (alucinado), ou
   evento importante do texto que sumiu da decupagem. Sempre marcado como SUGESTAO
   INFERIDA -- so relata, nunca reescreve nada sozinho (confianca baixa demais pra corrigir
   sem revisao).
3. CORRECAO automatica de descritor (pedido do usuario 2026-09-29): ao contrario das
   checagens acima, a deriva de descritor ENTRE PLANOS *e* corrigida, nao so relatada --
   depois desta etapa os descritores tem que estar prontos pra gerar still, nao so
   diagnosticados. Todo plano cujo `descriptor`/`co_descriptor` divirja do canonico em
   `cast.json` (que ja passou por `_enforce_descriptor_fidelity`) e sobrescrito pelo
   canonico, e o texto trocado tambem dentro de `storyboard_prompt`/`video_prompt` -- o
   `shot_plan.json` e regravado em disco com a correcao.

Saida: `parse/text_validation.json` e `parse/text_validation.md` (inclui a lista de
descritores corrigidos). Nunca bloqueia por padrao; `--bloquear` devolve 2 se houver
problema critico que NAO foi corrigido automaticamente (falas/locais sem base no roteiro,
personagem inventado).

CLI:
  python -m script_pipeline.text_validation --run-dir DIR [--engine qwen2.5:32b-instruct-q4_K_M] [--bloquear]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from script_pipeline.cast_characters import _attribute_contradiction  # noqa: E402


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _words(text: str) -> set[str]:
    return {w for w in _norm(text).split() if len(w) >= 4}


def _load_scenes_flat(run_dir: Path) -> tuple[dict]:
    scenes = json.loads((run_dir / "parse" / "scenes_enriched.json").read_text(encoding="utf-8")) \
        if (run_dir / "parse" / "scenes_enriched.json").exists() \
        else json.loads((run_dir / "parse" / "scenes.json").read_text(encoding="utf-8"))
    return scenes if isinstance(scenes, list) else scenes.get("scenes", [])


def _source_text(run_dir: Path) -> tuple[str, bool]:
    """O roteiro-FONTE de verdade -- antes de qualquer reestruturacao por LLM. Cai pro
    screenplay_auto.txt (ja reestruturado) so em corridas antigas, de antes desta funcao
    existir, que nunca guardaram o original -- aviso claro nesse caso, pois a comparacao
    fica mais fraca (nao pega bug que a propria reestruturacao introduziu)."""
    original = run_dir / "parse" / "screenplay_original.txt"
    if original.exists():
        return original.read_text(encoding="utf-8", errors="replace"), True
    auto = run_dir / "parse" / "screenplay_auto.txt"
    if auto.exists():
        return auto.read_text(encoding="utf-8", errors="replace"), False
    return "", False


def check_characters(source: str, cast: dict) -> list[dict]:
    problemas = []
    source_low = source.lower()
    for name in cast:
        if name.lower() not in source_low:
            problemas.append({"tipo": "personagem_nao_encontrado", "gravidade": "atencao",
                              "personagem": name,
                              "detalhe": f"'{name}' esta no elenco mas nao aparece no roteiro-fonte "
                                        "-- nome inventado ou grafado diferente em algum estagio."})
    return problemas


def _scene_text_by_index(scenes: list[dict] | None) -> dict:
    if not scenes:
        return {}
    return {s.get("index"): (s.get("action_text") or "") for s in scenes}


def check_quotes(source: str, shots: list[dict], scenes: list[dict] | None = None) -> list[dict]:
    """ACHADO externo 2026-09-29: procurar a fala em QUALQUER lugar do roteiro deixa passar
    o erro que motivou esta validacao em primeiro lugar -- fala de uma cena colada sob o
    cabecalho de outra. Com `scenes`, cada fala e checada primeiro contra o texto da SUA
    PROPRIA cena (`shot["scene"]`); so se falhar ali e que se olha o roteiro inteiro, e nesse
    caso o problema vira `fala_em_cena_errada` (a fala existe, mas nao onde o plano diz que
    esta) em vez de silenciosamente aprovar. Sem `scenes` (compat com chamadas antigas / sem
    scenes.json disponivel), mantem o comportamento anterior -- so checa presenca global."""
    problemas = []
    source_norm = _norm(source)
    scene_norms = _scene_text_by_index(scenes)
    for shot in shots:
        quote = (shot.get("quote") or "").strip()
        if len(quote) < 8:
            continue
        quote_norm = _norm(quote)
        # janela de 6 palavras do inicio da fala -- pega reescrita total, tolera
        # pequenas diferencas de pontuacao/maiusculas que a normalizacao ja tirou.
        janela = " ".join(quote_norm.split()[:6])
        if not janela:
            continue
        scene_idx = shot.get("scene")
        scene_text_norm = _norm(scene_norms[scene_idx]) if scene_idx in scene_norms else None
        if scene_text_norm is not None and janela in scene_text_norm:
            continue  # bate na propria cena -- ok
        if janela not in source_norm:
            problemas.append({"tipo": "fala_nao_encontrada", "gravidade": "critica",
                              "plano": shot.get("index"), "personagem": shot.get("subject"),
                              "detalhe": f"fala do plano nao bate com nada no roteiro-fonte: \"{quote[:80]}\""})
        elif scene_text_norm is not None:
            # existe em algum lugar do roteiro, mas NAO na cena que este plano diz cobrir
            problemas.append({"tipo": "fala_em_cena_errada", "gravidade": "critica",
                              "plano": shot.get("index"), "personagem": shot.get("subject"),
                              "cena": scene_idx,
                              "detalhe": f"fala do plano {shot.get('index')} (cena {scene_idx}) existe no "
                                        f"roteiro-fonte, mas NAO no texto dessa cena -- pode ter sido "
                                        f"atribuida a cena/personagem errado: \"{quote[:80]}\""})
    return problemas


def check_locations(source: str, shots: list[dict], scenes: list[dict] | None = None,
                    *, min_overlap: int = 1) -> list[dict]:
    """Mesma logica de `check_quotes` acima: sobreposicao lexical com o roteiro INTEIRO nao
    prova que o local pertence aquela cena -- so que ele existe em algum lugar do roteiro.
    Com `scenes`, tenta primeiro contra o texto da propria cena do plano; sem sobreposicao ali
    mas com sobreposicao em outra parte do roteiro, vira `local_em_cena_errada` em vez de
    passar como correto."""
    problemas = []
    source_words = _words(source)
    scene_texts = _scene_text_by_index(scenes)
    scene_words_cache: dict = {}
    vistos = set()
    for shot in shots:
        loc = (shot.get("location") or "").strip()
        if not loc:
            continue
        scene_idx = shot.get("scene")
        chave = (loc, scene_idx)
        if chave in vistos:
            continue
        vistos.add(chave)
        loc_words = _words(loc)
        if not loc_words:
            continue
        if scene_idx in scene_texts:
            if scene_idx not in scene_words_cache:
                scene_words_cache[scene_idx] = _words(scene_texts[scene_idx])
            if loc_words & scene_words_cache[scene_idx]:
                continue  # bate no texto da propria cena -- ok
            if loc_words & source_words:
                problemas.append({"tipo": "local_em_cena_errada", "gravidade": "atencao",
                                  "local": loc, "cena": scene_idx,
                                  "detalhe": f"local '{loc}' do plano na cena {scene_idx} nao tem "
                                            "sobreposicao com o texto DESSA cena, so com outra parte "
                                            "do roteiro -- pode ter sido atribuido a cena errada."})
                continue
        elif loc_words & source_words:
            continue  # sem info de cena pra comparar -- mantem comportamento antigo (so global)
        problemas.append({"tipo": "local_nao_encontrado", "gravidade": "atencao",
                          "local": loc,
                          "detalhe": f"local '{loc}' nao tem nenhuma palavra em comum com o "
                                    "roteiro-fonte -- pode ter sido inventado na reestruturacao."})
    return problemas


def check_cross_shot_descriptor_drift(shots: list[dict]) -> list[dict]:
    """Mesmo personagem, comprimento/cor de cabelo ou de roupa DIFERENTES entre dois
    planos da MESMA cena -- sinal de que o descritor usado num still nao e o mesmo do
    outro (deriva silenciosa, nao pega por `_enforce_descriptor_fidelity`, que so compara
    contra o roteiro na hora de criar o cast.json, uma vez so). Usada so pra RELATAR
    quando nao ha cast.json disponivel para corrigir (ver `fix_descriptors` abaixo, que e
    o caminho normal -- corrige contra o cast.json em vez de só comparar planos entre si)."""
    problemas = []
    por_personagem_cena: dict[tuple[str, int], list[dict]] = {}
    for shot in shots:
        for campo in ("subject", "co_subject"):
            nome = shot.get(campo)
            desc = shot.get("descriptor") if campo == "subject" else shot.get("co_descriptor")
            if not nome or not desc:
                continue
            chave = (str(nome), shot.get("scene"))
            por_personagem_cena.setdefault(chave, []).append({"plano": shot.get("index"), "descriptor": desc})
    for (nome, cena), entradas in por_personagem_cena.items():
        base = entradas[0]
        for outro in entradas[1:]:
            motivo = _attribute_contradiction(base["descriptor"], outro["descriptor"])
            if motivo:
                problemas.append({"tipo": "descritor_diverge_entre_planos", "gravidade": "critica",
                                  "personagem": nome, "cena": cena,
                                  "planos": [base["plano"], outro["plano"]],
                                  "detalhe": f"{nome} contradiz {motivo} entre o plano {base['plano']} "
                                            f"e o plano {outro['plano']} da mesma cena."})
    return problemas


def _diff_excerpt(a: str, b: str, *, context: int = 60) -> tuple[str, str]:
    """ACHADO 2026-09-29 (rodada de teste real, cena 'Palace of Emerald Shadows"): truncar
    `de`/`para` nos primeiros N caracteres faz o relatorio mostrar a MESMA string dos dois
    lados sempre que dois descritores compartilham um prefixo longo e so divergem depois do
    corte -- parece correcao inutil ("de X para X"), quando na verdade os textos diferem mais
    adiante. Acha o primeiro indice onde `a` e `b` diferem e recorta uma janela em volta dele
    (com "..." nas pontas quando corta no meio), pra o "de"/"para" do relatorio sempre mostrar
    a parte que realmente mudou."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == len(a) == len(b):
        return a, b  # strings identicas (nao deveria chegar aqui, mas nao quebra se chegar)
    inicio = max(0, i - context)
    prefixo = "..." if inicio > 0 else ""
    fim_a, fim_b = inicio + 2 * context, inicio + 2 * context
    sufixo_a = "..." if fim_a < len(a) else ""
    sufixo_b = "..." if fim_b < len(b) else ""
    return (f"{prefixo}{a[inicio:fim_a]}{sufixo_a}", f"{prefixo}{b[inicio:fim_b]}{sufixo_b}")


def fix_descriptors(shots: list[dict], cast: dict, log=print) -> tuple[list[dict], list[dict]]:
    """ACHADO 2026-09-29 (pedido do usuario): relatar deriva de descritor nao bastava --
    depois desta etapa os descritores tem que estar CORRETOS, prontos pra still. O
    `cast.json` já passou por `_enforce_descriptor_fidelity` (fiel ao roteiro-fonte) e e o
    unico lugar com um descritor "canonico" por personagem; qualquer plano cujo
    `descriptor`/`co_descriptor` divirja dele (mesmo sem cair em `_attribute_contradiction`
    -- aqui a barra e mais alta: qualquer diferenca de atributo, nao so as 3 categorias
    ja cobertas) e SOBRESCRITO pelo canonico, e o mesmo texto e trocado dentro de
    `storyboard_prompt`/`video_prompt` (onde o descritor antigo foi colado literalmente ao
    montar o prompt) para o still nao herdar a versao errada. Devolve os shots corrigidos
    e a lista de correcoes feitas, para o relatorio."""
    correcoes = []
    for shot in shots:
        for campo_nome, campo_desc, campo_prompt_ok in (
            ("subject", "descriptor", True), ("co_subject", "co_descriptor", False),
        ):
            nome = shot.get(campo_nome)
            atual = shot.get(campo_desc)
            if not nome or not atual:
                continue
            canonico = (cast.get(str(nome)) or {}).get("descriptor")
            if not canonico or canonico == atual:
                continue
            motivo = _attribute_contradiction(canonico, atual) or "descritor diferente do cast.json"
            for prompt_campo in ("storyboard_prompt", "video_prompt"):
                texto = shot.get(prompt_campo)
                if texto and atual in texto:
                    shot[prompt_campo] = texto.replace(atual, canonico)
            shot[campo_desc] = canonico
            de_excerto, para_excerto = _diff_excerpt(atual, canonico)
            correcoes.append({"tipo": "descritor_corrigido", "personagem": str(nome),
                              "plano": shot.get("index"), "motivo": motivo,
                              "de": de_excerto, "para": para_excerto})
    if correcoes:
        log(f"[validacao-texto] {len(correcoes)} descritor(es) corrigido(s) contra o cast.json "
            "(descritor + storyboard_prompt/video_prompt atualizados).")
        for c in correcoes[:8]:
            log(f"  CORRIGIDO plano {c['plano']} ({c['personagem']}): {c['motivo']}")
    return shots, correcoes


LLM_SYSTEM_PROMPT = (
    "Voce audita se uma decupagem (lista de planos de video) representa fielmente um trecho "
    "de roteiro. Aponte APENAS o que estiver claramente errado: objeto, personagem ou local "
    "que aparece na decupagem mas NAO existe no texto-fonte (alucinado); ou evento importante "
    "do texto que sumiu da decupagem. Nao invente problema. Responda em JSON: "
    '{"alucinacoes": [{"o_que": str, "onde": str}], "eventos_perdidos": [str]}. '
    "Listas vazias se nao houver problema."
)


def audit_llm(source: str, scenes: list[dict], shots: list[dict], engine: str, log=print) -> list[dict]:
    from script_pipeline.story_structure import _call_ollama

    sugestoes = []
    por_cena: dict[int, list[dict]] = {}
    for shot in shots:
        por_cena.setdefault(shot.get("scene", 0), []).append(shot)
    for scene in scenes:
        idx = scene.get("index", 0)
        planos = por_cena.get(idx, [])
        if not planos:
            continue
        resumo_planos = "\n".join(
            f"- plano {p.get('index')}: {(p.get('storyboard_prompt') or '')[:200]}" for p in planos)
        user = (f"Trecho do roteiro-fonte (cena {idx}):\n{(scene.get('action_text') or '')[:2000]}\n\n"
                f"Planos derivados dessa cena:\n{resumo_planos}")
        payload = _call_ollama(LLM_SYSTEM_PROMPT, user, engine, log=log) or {}
        for aluc in payload.get("alucinacoes") or []:
            if not isinstance(aluc, dict) or not aluc.get("o_que"):
                continue
            sugestoes.append({"tipo": "alucinacao_sugerida_llm", "gravidade": "info", "cena": idx,
                              "detalhe": f"{aluc['o_que']} (em: {aluc.get('onde', '?')})", "inferido": True})
        for evento in payload.get("eventos_perdidos") or []:
            if not str(evento).strip():
                continue
            sugestoes.append({"tipo": "evento_perdido_sugerido_llm", "gravidade": "info", "cena": idx,
                              "detalhe": str(evento), "inferido": True})
    log(f"[validacao-texto] LLM ({engine}): {len(sugestoes)} sugestao(oes)")
    return sugestoes


def render_markdown(relatorio: dict) -> str:
    r = relatorio["resumo"]
    linhas = [f"# Validacao pos-parse — {relatorio.get('fonte', '?')}", "",
             f"**{r['critica']} critica(s), {r['atencao']} de atencao, {r['info']} info.**",
             f"Checagem semantica (LLM): {relatorio.get('checagem_semantica_llm', 'desativada')}.",
             "",
             "> Este relatorio NAO e uma aprovacao de fidelidade ao roteiro -- e o resultado de "
             "checagens deterministicas (sempre) mais uma checagem semantica opcional. Zero "
             "critica com a checagem semantica desativada/pulada/falhada significa apenas que "
             "as checagens deterministicas nao acharam nada, nao que o conteudo foi revisado "
             "por completo.", ""]
    if not relatorio["fonte_confiavel"]:
        linhas += ["> ⚠️ Corrida antiga: sem `screenplay_original.txt` salvo, comparando contra "
                  "o texto JA REESTRUTURADO -- bug introduzido na propria reestruturacao pode "
                  "passar batido aqui.", ""]
    if relatorio.get("correcoes"):
        linhas += ["## Descritores corrigidos automaticamente", ""]
        for c in relatorio["correcoes"]:
            linhas.append(f"- plano {c['plano']} ({c['personagem']}, {c['motivo']}): "
                          f"\"{c['de']}\" → \"{c['para']}\"")
        linhas.append("")
    por_tipo: dict[str, list[dict]] = {}
    for p in relatorio["problemas"]:
        por_tipo.setdefault(p["tipo"], []).append(p)
    titulos = {
        "personagem_nao_encontrado": "Personagens sem base no roteiro",
        "fala_nao_encontrada": "Falas que nao batem com o roteiro",
        "fala_em_cena_errada": "Falas atribuidas a cena/personagem errado",
        "local_nao_encontrado": "Locais sem base no roteiro",
        "local_em_cena_errada": "Locais atribuidos a cena errada",
        "descritor_diverge_entre_planos": "Descritor divergente entre planos da mesma cena",
        "alucinacao_sugerida_llm": "Possivel invencao (sugestao do LLM)",
        "evento_perdido_sugerido_llm": "Possivel evento perdido (sugestao do LLM)",
    }
    for tipo, itens in por_tipo.items():
        linhas += [f"## {titulos.get(tipo, tipo)}", ""]
        for it in itens:
            linhas.append(f"- **{it['gravidade']}**: {it['detalhe']}")
        linhas.append("")
    return "\n".join(linhas)


def audit(run_dir: Path, *, engine: str | None = None, log=print) -> dict:
    run_dir = Path(run_dir)
    source, confiavel = _source_text(run_dir)
    cast_path = run_dir / "characters" / "cast.json"
    cast = json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.exists() else {}
    plan_path = run_dir / "parse" / "shot_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shots = plan.get("shots", plan) if isinstance(plan, dict) else plan
    scenes = _load_scenes_flat(run_dir)
    # ACHADO 2026-09-29 (teste real, prompt LTX-2.3 de cena unica): com 1 cena so, nao
    # existe "outra cena" pra uma fala/local ser atribuida por engano -- qualquer
    # divergencia contra o texto DAQUELA cena e mais provavel ser artefato de traducao
    # (location fica em ingles, action_text as vezes sai traduzido pro portugues) do que
    # erro de atribuicao de verdade. So passa `scenes` pras checagens por-cena quando ha
    # mais de uma cena pra comparar contra.
    scenes_para_comparacao = scenes if len(scenes) > 1 else None

    problemas = []
    if source:
        problemas += check_characters(source, cast)
        problemas += check_quotes(source, shots, scenes_para_comparacao)
        problemas += check_locations(source, shots, scenes_para_comparacao)
    else:
        log("[validacao-texto] AVISO: nenhum roteiro-fonte disponivel (nem original, nem "
            "reestruturado) -- pulando checagens de fidelidade textual.")

    correcoes = []
    if cast:
        shots, correcoes = fix_descriptors(shots, cast, log=log)
        if correcoes:
            plan_out = plan
            if isinstance(plan, dict):
                plan_out["shots"] = shots
            else:
                plan_out = shots
            plan_path.write_text(json.dumps(plan_out, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        problemas += check_cross_shot_descriptor_drift(shots)
        log("[validacao-texto] AVISO: cast.json ausente -- so relatando deriva de descritor "
            "entre planos, sem corrigir (nao ha descritor canonico pra corrigir contra).")

    if not engine:
        semantica_status = "desativada"
    elif not source:
        semantica_status = "pulada (sem roteiro-fonte)"
    else:
        try:
            problemas += audit_llm(source, scenes, shots, engine, log=log)
            semantica_status = "ok"
        except Exception as exc:  # o LLM nunca derruba a auditoria deterministica
            log(f"[validacao-texto] LLM indisponivel ({type(exc).__name__}: {exc}); "
                "so a parte deterministica")
            semantica_status = f"falhou ({type(exc).__name__})"

    resumo = {g: sum(1 for p in problemas if p["gravidade"] == g) for g in ("critica", "atencao", "info")}
    # ACHADO externo 2026-09-29: "relatorio gerado sem critica" != "fidelidade aprovada" --
    # a checagem semantica (a unica que pega alucinacao livre, nao so contradicao textual) e
    # opcional e pode ter sido desativada, pulada ou falhado sem derrubar a corrida. Deixar
    # esse estado explicito no relatorio evita ler "0 critica" como aprovacao quando a etapa
    # que mais pegaria esse tipo de erro nem rodou.
    relatorio = {"resumo": resumo, "problemas": problemas, "correcoes": correcoes, "fonte_confiavel": confiavel,
                "fonte": "screenplay_original.txt" if confiavel else "screenplay_auto.txt (sem original salvo)",
                "checagem_semantica_llm": semantica_status}
    out_dir = run_dir / "parse"
    (out_dir / "text_validation.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    (out_dir / "text_validation.md").write_text(render_markdown(relatorio), encoding="utf-8")
    log(f"[validacao-texto] {resumo['critica']} critica(s), {resumo['atencao']} de atencao, "
        f"{resumo['info']} info -> {out_dir / 'text_validation.md'}")
    for p in [x for x in problemas if x["gravidade"] == "critica"][:8]:
        log(f"  CRITICA: {p['detalhe']}")
    return relatorio


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--engine", default=None, help="modelo do Ollama para a checagem semantica por cena")
    ap.add_argument("--bloquear", action="store_true", help="devolve 2 se houver problema critico")
    args = ap.parse_args(argv)
    relatorio = audit(Path(args.run_dir).resolve(), engine=args.engine)
    return 2 if (args.bloquear and relatorio["resumo"]["critica"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
