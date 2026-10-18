import pytest

from rewind_kv.detector import detect_failure, first_sentence


@pytest.mark.parametrize(
    "text",
    [
        "Wait, let me check that again. ",
        "So x = 3.5 works fine. ",
        "Hmm, maybe I should double-check the arithmetic. ",
    ],
)
def test_ignores_normal_reasoning(text):
    assert detect_failure(text) is None


def test_waits_for_sentence_end():
    assert detect_failure("So x = 2. Wait, that's wrong because the denominator") is None


def test_extracts_full_reason_sentence():
    text = "So x = 2. Wait, that's wrong because the denominator becomes 0. Let me"
    signal = detect_failure(text)
    assert signal is not None
    assert signal.reason == "Wait, that's wrong because the denominator becomes 0."
    assert text[: signal.end].endswith("becomes 0.")


def test_decimal_point_does_not_end_sentence():
    assert detect_failure("Hmm, that doesn't work since 3.5") is None


@pytest.mark.parametrize(
    "text",
    [
        "This contradicts the condition x > 0.\n",
        "That doesn’t make sense, the area is negative.\n",
        "I think I made a mistake in step 2. ",
        "Let me recalculate the sum. ",
        "Substituting back, it does not work. ",
    ],
)
def test_detects_markers(text):
    assert detect_failure(text) is not None


def test_first_sentence():
    assert first_sentence("\n  Let's try x = 2. Then 2x = 4.\nMore") == "Let's try x = 2."
