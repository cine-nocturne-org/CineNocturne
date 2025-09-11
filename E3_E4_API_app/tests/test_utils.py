import pytest
from api_movie_v2 import normalize_text

def test_normalize_text():
    # On s'assure que normalize_text supprime les accents et les espaces
    assert normalize_text("Annabelle") == "annabelle"
    assert normalize_text("ZoMbIeLand") == "zombieland"
    assert normalize_text(" Kpop ") == "kpop"
    assert normalize_text("Été") == "ete"
    assert normalize_text(" café ") == "cafe"

