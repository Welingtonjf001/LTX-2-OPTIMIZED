from script_pipeline import emotion_director


def test_retake_uses_original_parenthetical_instead_of_previous_slug(monkeypatch):
    captured = {}

    def fake_ollama(system, user, model, log=print):
        captured["user"] = user
        return {"emocoes": [{"scene": 1, "line": 0, "emocao": "calma"}]}

    monkeypatch.setattr(emotion_director, "_ollama", fake_ollama)
    scenes = [{"index": 1, "dialogue": [{
        "character": "ANA", "text": "Tudo certo.",
        "parenthetical": "desanimo",
        "parenthetical_original": "firme e tranquilizadora",
        "emotion": "desanimo",
    }]}]
    result = emotion_director.direct_emotions(scenes, engine="test")
    assert result[(1, 0)] == "calma"
    assert "firme e tranquilizadora" in captured["user"]
    assert "rubrica: desanimo" not in captured["user"]
