"""Âncora e escolha do corte, com um tokenizer de mentira (1 token por palavra)."""

from rewind_kv.controller import AnchorStyle, RewindKVGenerator, RewindScope


class WordTokenizer:
    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        self.vocab = text.split()
        return self._Out(list(range(len(self.vocab))))

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)


def _gen(**kwargs):
    return RewindKVGenerator(None, WordTokenizer(), "rewind_kv", **kwargs)


CUT = (
    "Wait, hold on, no.\n\n"
    "Alternatively, if I fix Pierre's position, Rosa and Thomas sit next to him.\n\n"
    "Calculating that: 2 x 120 = 240."
)


def test_voice_anchor_uses_informative_sentence_and_clean_reason():
    anchor = _gen().build_anchor(CUT, "Wait, but this contradicts my previous answer of 720.")
    assert anchor == (
        "Wait, I already tried this before: \"Alternatively, if I fix Pierre's position, Rosa and Thomas sit next to him.\" "
        "It was wrong: this contradicts my previous answer of 720. "
        "I will not repeat it, so let me do it differently.\n\n"
    )


def test_note_anchor_matches_experiment_a():
    anchor = _gen(anchor_style=AnchorStyle.NOTE).build_anchor(CUT, "Hmm, that doesn't make sense.")
    assert anchor == '[Note: "Wait, hold on, no." was wrong: that doesn\'t make sense. Do not repeat it.]\n\n'


def test_doubt_anchor_does_not_forbid_going_back():
    anchor = _gen(anchor_style=AnchorStyle.DOUBT).build_anchor(CUT, "Wait, but this contradicts my previous answer of 720.")
    assert anchor == (
        "Wait, earlier I tried \"Alternatively, if I fix Pierre's position, Rosa and Thomas sit next to him.\" "
        "and then doubted it: this contradicts my previous answer of 720. "
        "That doubt may or may not be right, so let me check it in a different way before relying on it.\n\n"
    )
    assert "wrong" not in anchor and "not repeat" not in anchor


def test_hypothesis_is_truncated():
    gen = _gen(max_hypothesis_tokens=3)
    assert '"Alternatively, if I..."' in gen.build_anchor(CUT, "That's wrong.")


TEXT = (
    "Treating them as a block gives 720.\n\n"
    "Alternatively, if I fix Pierre's position, they sit next to him.\n\n"
    "Calculating that: 2 x 120 = 240. Wait, but this contradicts 720."
)
PARAGRAPHS = [(100, 0), (200, TEXT.index("Alternatively")), (300, TEXT.index("Calculating"))]


def _target(scope, **kwargs):
    return _gen(rewind_scope=scope, **kwargs).rewind_target(PARAGRAPHS, TEXT, TEXT.index("Wait"), n_tokens=350)


def test_rewind_target_by_scope():
    assert _target(RewindScope.BRANCH) == PARAGRAPHS[1]
    assert _target(RewindScope.PARAGRAPH) == PARAGRAPHS[2]
    assert _target(RewindScope.ATTEMPT) == PARAGRAPHS[0]


def test_branch_longer_than_max_cut_falls_back_to_the_step():
    assert _target(RewindScope.BRANCH, max_cut_tokens=100) == PARAGRAPHS[2]


def test_voice_hypothesis_never_is_the_failure_sentence():
    cut = "0 1 0 | 1 0 1 | 0 1 0\n\nBut hold on, I think I made a mistake there. The leading zero"
    anchor = _gen().build_anchor(cut, "But hold on, I think I made a mistake there.")
    assert anchor.startswith('Wait, I already tried this before: "0 1 0 | 1 0 1 | 0 1 0"')
