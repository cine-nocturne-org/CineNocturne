import pytest
from api_movie_v2 import interpret_score, normalize_text

def test_interpret_score():
    assert interpret_score(0.9).startswith("🎯")
    assert interpret_score(0.75).startswith("👍")
    assert interpret_score(0.6).startswith("🤔")
    assert interpret_score(0.3).startswith("⚠️")

def test_normalize_text():
    # On s'assure que normalize_text supprime les accents et les espaces
    assert normalize_text("Annabelle") == "annabelle"
    assert normalize_text("ZoMbIeLand") == "zombieland"
    assert normalize_text(" Kpop ") == "kpop"
    assert normalize_text("Été") == "ete"
    assert normalize_text(" café ") == "cafe"
