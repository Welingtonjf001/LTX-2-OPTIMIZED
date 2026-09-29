"""Validacao pos-parse, ainda em texto (pedido do usuario 2026-09-29, depois dos bugs reais
achados no REENTRY WINDOW): o shot_plan.json inteiro contra o roteiro-fonte, ANTES dos stills.
Descritor divergente entre planos e CORRIGIDO (nao so relatado) contra o canonico do
cast.json. Sem GPU/LLM."""
import json

from script_pipeline import text_validation as tv


def test_check_characters_pega_personagem_sem_base_no_roteiro():
    source = "ANA entra na sala. Ela fala com GIANA."
    cast = {"ANA": {}, "GIANA": {}, "FANTASMA": {}}
    problemas = tv.check_characters(source, cast)
    assert len(problemas) == 1
    assert problemas[0]["personagem"] == "FANTASMA"


def test_check_quotes_pega_fala_reescrita():
    source = 'ANA diz: "Reentry window in sixty seconds."'
    shots = [{"index": 5, "subject": "ANA", "quote": "Reentry window in sixty seconds."},
             {"index": 6, "subject": "ANA", "quote": "This is a completely different sentence entirely."}]
    problemas = tv.check_quotes(source, shots)
    assert len(problemas) == 1
    assert problemas[0]["plano"] == 6


def test_check_quotes_ignora_fala_curta_demais():
    problemas = tv.check_quotes("qualquer coisa", [{"index": 1, "quote": "Oi."}])
    assert problemas == []


def test_check_quotes_pega_fala_na_cena_errada():
    """Achado de revisao externa 2026-09-29: a fala existe no roteiro, so que dentro do
    texto de OUTRA cena -- o comportamento antigo (procurar em qualquer lugar) aprovava
    isso em silencio; e exatamente o bug real que motivou esta validacao (fala de controle
    em terra colada sob o cabecalho de outra cena)."""
    source = "CENA 1. GROUND CONTROL. Ana says: \"Reentry window in sixty seconds.\"\n" \
             "CENA 2. UPPER ATMOSPHERE. Giana says: \"Hull temperature nominal.\""
    scenes = [{"index": 0, "action_text": "GROUND CONTROL. Ana says: \"Reentry window in sixty seconds.\""},
             {"index": 1, "action_text": "UPPER ATMOSPHERE. Giana says: \"Hull temperature nominal.\""}]
    # plano 0 diz que e da cena 1 (UPPER ATMOSPHERE) mas a fala e da cena 0 (GROUND CONTROL)
    shots = [{"index": 0, "scene": 1, "subject": "Ana", "quote": "Reentry window in sixty seconds."}]
    problemas = tv.check_quotes(source, shots, scenes)
    assert len(problemas) == 1
    assert problemas[0]["tipo"] == "fala_em_cena_errada"
    assert problemas[0]["gravidade"] == "critica"


def test_check_quotes_sem_scenes_mantem_comportamento_antigo():
    """Compat: sem passar `scenes`, so checa presenca global -- corridas antigas sem
    scenes.json disponivel nao devem quebrar nem ganhar falso positivo novo."""
    source = 'ANA diz: "Reentry window in sixty seconds."'
    shots = [{"index": 0, "scene": 0, "subject": "ANA", "quote": "Reentry window in sixty seconds."}]
    assert tv.check_quotes(source, shots) == []


def test_check_locations_pega_local_na_cena_errada():
    source = ("CENA 1. The scene takes place in the ship's cabin, with control panels everywhere.\n"
             "CENA 2. The scene takes place in the underwater observation deck.")
    scenes = [{"index": 0, "action_text": "The scene takes place in the ship's cabin, with control panels everywhere."},
             {"index": 1, "action_text": "The scene takes place in the underwater observation deck."}]
    # plano diz cena 1 (observation deck) mas o local citado e da cena 0 (ship cabin)
    shots = [{"index": 0, "scene": 1, "location": "SHIP CABIN"}]
    problemas = tv.check_locations(source, shots, scenes)
    assert len(problemas) == 1
    assert problemas[0]["tipo"] == "local_em_cena_errada"


def test_check_locations_pega_local_inventado():
    source = "The scene takes place in the ship's cabin, with control panels everywhere."
    shots = [{"index": 0, "location": "SHIP CABIN"},
             {"index": 1, "location": "UNDERWATER ATLANTIS PALACE"}]
    problemas = tv.check_locations(source, shots)
    assert len(problemas) == 1
    assert "ATLANTIS" in problemas[0]["local"]


def test_fix_descriptors_corrige_plano_divergente_do_cast():
    cast = {"ANA": {"descriptor": "long dark hair, charcoal pressure suit with blue piping"}}
    shots = [
        {"index": 10, "subject": "ANA", "descriptor": "long dark hair, charcoal pressure suit with blue piping",
         "storyboard_prompt": "close-up. ANA, long dark hair, charcoal pressure suit with blue piping. interior."},
        {"index": 11, "subject": "ANA", "descriptor": "short blonde hair, red jacket",
         "storyboard_prompt": "medium shot. ANA, short blonde hair, red jacket. interior."},
    ]
    fixed, correcoes = tv.fix_descriptors(shots, cast, log=lambda m: None)
    assert len(correcoes) == 1
    assert correcoes[0]["plano"] == 11
    # o plano 11 tem que sair com o descritor CANONICO, nao o divergente
    assert fixed[1]["descriptor"] == cast["ANA"]["descriptor"]
    # e o storyboard_prompt tem que refletir a troca, nao ficar com o texto velho colado
    assert "short blonde hair, red jacket" not in fixed[1]["storyboard_prompt"]
    assert cast["ANA"]["descriptor"] in fixed[1]["storyboard_prompt"]
    # o plano 10 ja batia com o cast -- nao mexe, nao gera correcao a toa
    assert fixed[0]["descriptor"] == cast["ANA"]["descriptor"]


def test_fix_descriptors_nao_mexe_quando_ja_bate():
    cast = {"ANA": {"descriptor": "long dark hair"}}
    shots = [{"index": 0, "subject": "ANA", "descriptor": "long dark hair", "storyboard_prompt": "x"}]
    fixed, correcoes = tv.fix_descriptors(shots, cast, log=lambda m: None)
    assert correcoes == []


def test_fix_descriptors_relatorio_mostra_a_divergencia_de_verdade():
    """Achado real (rodada de teste 2026-09-29, cena do palacio): dois descritores com um
    prefixo comum LONGO (>120 chars) e que so divergem depois do corte antigo faziam o
    relatorio mostrar "de X para X" -- parecia correcao inutil. O excerto tem que cair
    exatamente na parte que MUDOU, nao no prefixo idendico."""
    prefixo = "XIAO-LAN has long, dark raven hair tied in a neat bun, with a few loose strands " \
             "framing her oval face. Her early 20s are "
    atual = prefixo + "noted for a lively energy and quick wit."
    canonico = prefixo + "marked by a serene and graceful demeanor."
    cast = {"XIAO-LAN": {"descriptor": canonico}}
    shots = [{"index": 0, "subject": "XIAO-LAN", "descriptor": atual, "storyboard_prompt": atual}]
    _, correcoes = tv.fix_descriptors(shots, cast, log=lambda m: None)
    assert len(correcoes) == 1
    assert correcoes[0]["de"] != correcoes[0]["para"]
    assert "noted for" in correcoes[0]["de"]
    assert "marked by" in correcoes[0]["para"]


def test_audit_end_to_end_corrige_e_regrava_shot_plan(tmp_path):
    run = tmp_path
    (run / "parse").mkdir()
    (run / "characters").mkdir()
    source = "ANA (long dark hair, charcoal suit) grips the yoke. \"Reentry window now.\""
    (run / "parse" / "screenplay_original.txt").write_text(source, encoding="utf-8")
    (run / "parse" / "scenes.json").write_text(json.dumps([
        {"index": 0, "action_text": source, "characters": ["ANA"], "shot_list": []}
    ]), encoding="utf-8")
    cast = {"ANA": {"descriptor": "long dark hair, charcoal suit"}}
    (run / "characters" / "cast.json").write_text(json.dumps(cast), encoding="utf-8")
    shot_plan = {"shots": [
        {"index": 0, "scene": 0, "subject": "ANA", "descriptor": "long dark hair, charcoal suit",
         "quote": "Reentry window now.", "location": "flight deck",
         "storyboard_prompt": "close. ANA, long dark hair, charcoal suit."},
        {"index": 1, "scene": 0, "subject": "ANA", "descriptor": "totally different look, pink hat",
         "quote": None, "location": "flight deck",
         "storyboard_prompt": "medium. ANA, totally different look, pink hat."},
    ]}
    plan_path = run / "parse" / "shot_plan.json"
    plan_path.write_text(json.dumps(shot_plan), encoding="utf-8")

    relatorio = tv.audit(run, log=lambda m: None)

    assert relatorio["resumo"]["critica"] == 0
    assert len(relatorio["correcoes"]) == 1
    assert relatorio["correcoes"][0]["plano"] == 1
    # confirma que o ARQUIVO em disco foi regravado com a correcao -- nao so o dict em memoria
    releitura = json.loads(plan_path.read_text(encoding="utf-8"))
    assert releitura["shots"][1]["descriptor"] == "long dark hair, charcoal suit"
    assert "pink hat" not in releitura["shots"][1]["storyboard_prompt"]
    assert (run / "parse" / "text_validation.md").exists()


def test_audit_com_1_cena_so_nao_gera_falso_positivo_de_cena_errada(tmp_path):
    """ACHADO 2026-09-29 (teste real, prompt LTX-2.3 de cena unica): com 1 cena so,
    'local/fala em cena errada' e um falso positivo garantido quando o campo location
    fica em ingles mas o action_text da cena sai traduzido pro portugues (mesmo
    documento, sem cabecalhos de cena) -- nao existe OUTRA cena pra atribuir por
    engano. `audit()` so deve repassar `scenes` as checagens por-cena quando ha mais
    de uma."""
    run = tmp_path
    (run / "parse").mkdir()
    (run / "characters").mkdir()
    source = "A cinematic scene in the circular ancient stone chamber. Lyra enters."
    (run / "parse" / "screenplay_original.txt").write_text(source, encoding="utf-8")
    (run / "parse" / "scenes.json").write_text(json.dumps([
        {"index": 0, "action_text": "Uma cena cinematica na camara circular antiga. Lyra entra.",
         "characters": ["LYRA"], "shot_list": []}
    ]), encoding="utf-8")
    (run / "characters" / "cast.json").write_text(json.dumps({}), encoding="utf-8")
    shot_plan = {"shots": [
        {"index": 0, "scene": 0, "subject": "LYRA", "location": "CIRCULAR ANCIENT STONE CHAMBER",
         "storyboard_prompt": "wide. LYRA in the chamber."},
    ]}
    (run / "parse" / "shot_plan.json").write_text(json.dumps(shot_plan), encoding="utf-8")

    relatorio = tv.audit(run, log=lambda m: None)
    tipos = {p["tipo"] for p in relatorio["problemas"]}
    assert "local_em_cena_errada" not in tipos


def test_audit_sem_roteiro_fonte_nao_quebra(tmp_path):
    run = tmp_path
    (run / "parse").mkdir()
    (run / "characters").mkdir()
    (run / "parse" / "scenes.json").write_text(json.dumps([{"index": 0, "action_text": "", "characters": []}]),
                                                encoding="utf-8")
    (run / "characters" / "cast.json").write_text(json.dumps({}), encoding="utf-8")
    (run / "parse" / "shot_plan.json").write_text(json.dumps({"shots": []}), encoding="utf-8")
    relatorio = tv.audit(run, log=lambda m: None)
    assert relatorio["fonte_confiavel"] is False
    assert relatorio["problemas"] == []
