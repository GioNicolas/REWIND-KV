"""O rewind precisa ser invisível para o modelo: depois de crop + âncora, o cache
deve ser idêntico ao de quem gerou só prompt + âncora desde o início (mesmas
posições RoPE, mesmos K/V). Roda na CPU com um Qwen2 minúsculo e pesos aleatórios.
"""

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from rewind_kv.controller import _forward, crop_to, paragraph_rewind_target, rewind_and_inject

ATOL = 1e-5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    return Qwen2ForCausalLM(config).eval()


def fresh_logits(model, ids):
    return _forward(model, DynamicCache(), ids)


def ids(n, seed):
    return torch.randint(0, 64, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@torch.no_grad()
def test_crop_reduces_cache_length(model):
    cache = DynamicCache()
    _forward(model, cache, ids(20, 1))
    crop_to(cache, 12)
    assert cache.get_seq_length() == 12


@torch.no_grad()
@pytest.mark.parametrize("anchor_len", [0, 1, 5])
def test_rewind_matches_fresh_generation(model, anchor_len):
    prompt, attempt, anchor = ids(10, 1), ids(15, 2), ids(anchor_len, 3)
    cache = DynamicCache()
    tokens = prompt + attempt
    _forward(model, cache, tokens)

    logits = rewind_and_inject(model, cache, tokens, checkpoint=len(prompt), anchor_ids=anchor)

    assert tokens == prompt + anchor
    assert cache.get_seq_length() == len(prompt) + anchor_len
    torch.testing.assert_close(logits, fresh_logits(model, prompt + anchor), atol=ATOL, rtol=0)


@torch.no_grad()
def test_generation_continues_correctly_after_rewind(model):
    """Compara os passos seguintes ao corte, não só o primeiro logit: pega erro de posição RoPE."""
    prompt, attempt, anchor, after = ids(10, 1), ids(15, 2), ids(5, 3), ids(4, 4)
    cache = DynamicCache()
    tokens = prompt + attempt
    _forward(model, cache, tokens)
    rewind_and_inject(model, cache, tokens, checkpoint=len(prompt), anchor_ids=anchor)

    for i, t in enumerate(after):
        logits = _forward(model, cache, [t])
        expected = fresh_logits(model, prompt + anchor + after[: i + 1])
        torch.testing.assert_close(logits, expected, atol=ATOL, rtol=0)


@torch.no_grad()
def test_two_consecutive_rewinds(model):
    """Segunda tentativa também falha: a primeira âncora fica protegida antes do novo L_0."""
    prompt, anchor1, anchor2 = ids(10, 1), ids(5, 3), ids(6, 5)
    cache = DynamicCache()
    tokens = prompt + ids(15, 2)
    _forward(model, cache, tokens)
    rewind_and_inject(model, cache, tokens, checkpoint=len(prompt), anchor_ids=anchor1)
    checkpoint = len(tokens)

    second_attempt = ids(12, 6)
    tokens.extend(second_attempt)
    _forward(model, cache, second_attempt)
    logits = rewind_and_inject(model, cache, tokens, checkpoint=checkpoint, anchor_ids=anchor2)

    assert tokens == prompt + anchor1 + anchor2
    torch.testing.assert_close(logits, fresh_logits(model, prompt + anchor1 + anchor2), atol=ATOL, rtol=0)


# Texto do problema 1 do piloto: o erro está no parágrafo da fórmula, e o
# "Wait, no..." abre o parágrafo seguinte.
PILOT_TEXT = (
    "Let me list the possible ways to insert parentheses.\n\n"
    "So, the structure is:\n\n"
    "( (2 * 3) * (4 * 5) + 1 ) ?\n\n"
    "Wait, no, actually, that's not correct because the original expression is a linear chain."
)


def _paragraphs(text, first_token=100):
    """(token, char) de cada parágrafo, fingindo 1 token por caractere a partir de `first_token`."""
    starts = [0] + [i + 2 for i in range(len(text)) if text.startswith("\n\n", i)]
    return [(first_token + c, c) for c in starts]


def test_paragraph_target_cuts_previous_paragraph_when_failure_opens_it():
    paragraphs = _paragraphs(PILOT_TEXT)
    sentence_start = PILOT_TEXT.index("Wait")
    token, char = paragraph_rewind_target(paragraphs, PILOT_TEXT, sentence_start)
    assert PILOT_TEXT[char:].startswith("( (2 * 3)")
    assert token == 100 + char


def test_paragraph_target_cuts_own_paragraph_when_failure_is_mid_paragraph():
    text = "Step one is fine.\n\nThe root is the addition. Wait, no, that's not correct."
    paragraphs = _paragraphs(text)
    _, char = paragraph_rewind_target(paragraphs, text, text.index("Wait"))
    assert text[char:].startswith("The root")


def test_paragraph_target_never_goes_before_checkpoint():
    text = "Wait, that's wrong."
    assert paragraph_rewind_target([(100, 0)], text, 0) == (100, 0)
