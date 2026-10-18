import pytest

from rewind_kv.metrics import extract_boxed, is_correct


@pytest.mark.parametrize(
    "pred, gold",
    [
        # Casos reais do piloto 2 que o corretor marcou como errados.
        ("11\\sqrt{2}", "11\\sqrt2"),
        ("90", "90^\\circ"),
        ("\\dfrac{243}{625}", "\\frac{243}{625}"),
        ("\\frac{1}{2}", "\\frac12"),
        ("1,000", "1000"),
        ("p - q", "p - q"),
        ("45\\%", "45"),
    ],
)
def test_equivalent_answers(pred, gold):
    assert is_correct(pred, gold)


@pytest.mark.parametrize("pred, gold", [("240", "720"), (None, "720"), ("\\sqrt{3}", "\\sqrt2")])
def test_wrong_answers(pred, gold):
    assert not is_correct(pred, gold)


def test_extract_boxed_nested():
    assert extract_boxed("so \\boxed{1} then \\boxed{\\frac{3}{56}}.") == "\\frac{3}{56}"
