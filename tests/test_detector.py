import pytest

from rewind_kv.detector import clean_reason, detect_failure, first_sentence, informative_sentence, opens_branch


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


# Falsos positivos reais do piloto de 18/10/2026: os dois apagaram respostas corretas.
@pytest.mark.parametrize(
    "text",
    [
        "Wait, so does that mean the original double sum is equal to p - q? "
        "Let me verify this step by step to make sure I didn't make a mistake.\n\n",
        "But just to ensure I haven't made a mistake in the prime factorization or the calculations, "
        "let me recheck each step.\n\n",
        "Let me recheck in case I made an error. ",
        "I did not make a mistake there. ",
    ],
)
def test_ignores_verification_sentences(text):
    assert detect_failure(text) is None


# Piloto 2: suspeitas ("maybe I made a mistake") não são erro confirmado.
@pytest.mark.parametrize(
    "text",
    [
        "Alternatively, maybe I made a mistake in the direction vectors. ",
        "Wait, perhaps I made a mistake in setting up the direction vectors. ",
    ],
)
def test_ignores_hedged_suspicions(text):
    assert detect_failure(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "So, that contradicts my earlier conclusion. ",
        "Wait, 720 ≠ 240, so I must have made a mistake. ",
        "This contradicts x > 0, so maybe I should flip the sign. ",
    ],
)
def test_detects_confirmed_failures_from_pilot(text):
    assert detect_failure(text) is not None


def test_skips_verification_and_finds_later_failure():
    text = "Let me make sure I didn't make a mistake. Hmm, that's wrong, 3*4 is 12. "
    signal = detect_failure(text)
    assert signal is not None
    assert signal.reason == "Hmm, that's wrong, 3*4 is 12."
    assert text[signal.start :].startswith("Hmm")


def test_clean_reason():
    assert clean_reason("Wait, no, actually, that's not correct because x < 0.") == "that's not correct because x < 0."
    assert clean_reason("Now that's wrong.") == "Now that's wrong."


def test_first_sentence():
    assert first_sentence("\n  Let's try x = 2. Then 2x = 4.\nMore") == "Let's try x = 2."


def test_clean_reason_strips_hold_on():
    assert clean_reason("But hold on, I think I made a mistake there.") == "I think I made a mistake there."


# Aberturas reais de caminhos que falharam no Experimento A.
@pytest.mark.parametrize(
    "paragraph",
    [
        "Alternatively, if I didn't treat them as a single block, but instead considered the three of them",
        "Wait, perhaps the points are not in order? Maybe the order is different.",
        "Hmm, maybe the diagonals are BC and AD? Let's see.",
        "Wait, let me try this. Let's denote point D as (x, y).",
        "So, if I let:",
        "\n  Let's try x = 2.",
        "Another approach: count the complement.",
    ],
)
def test_opens_branch(paragraph):
    assert opens_branch(paragraph)


@pytest.mark.parametrize(
    "paragraph",
    [
        "Let me compute the dot product.",
        "So, the octal number is 6 3 2 5, which is 6325.",
        "Wait, so in the first line, direction vector is (1/2, 1/3, -1).",
        "Therefore, a + b = (-3) + 2 = -1",
        "Wait, perhaps when I converted the binary to decimal, I made a mistake.",
        "Iffy step: x = 3.",
    ],
)
def test_does_not_open_branch(paragraph):
    assert not opens_branch(paragraph)


def test_informative_sentence_skips_fillers_and_numbers():
    assert informative_sentence("Wait, hold on, no. Wait, the direction vectors are (3, 2, -6).") == (
        "Wait, the direction vectors are (3, 2, -6)."
    )
    assert informative_sentence("0 1 0 | 1 0 1 | 0 1 0\n\nSo the octal number is 6325") == "So the octal number is 6325"
    assert informative_sentence("Hmm.") == "Hmm."  # sem frase melhor: cai na primeira
