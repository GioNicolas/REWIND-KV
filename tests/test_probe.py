"""Reconstrução do contexto de cada corte e escolha do limiar da sonda."""

import numpy as np
import pytest

from rewind_kv.probe import cut_context, reason_token_span, recall_threshold

ANCHOR = 'Wait, earlier I tried "x" and then doubted it. '


def _record():
    # Dois cortes: o 1º com âncora no lugar do trecho cortado, o 2º sem âncora (blind_rewind)
    text = "Step A. " + ANCHOR + "Step C. Step E."
    first = len("Step A. ")
    return {
        "problem_idx": 0,
        "mode": "rewind_kv",
        "text": text,
        "rewinds": [
            {"anchor": ANCHOR, "resume_char": first + len(ANCHOR), "discarded_text": "Step B. Wait, that's wrong."},
            {"anchor": "", "resume_char": len(text) - len("Step E."), "discarded_text": "Step D. Hmm, no."},
        ],
    }


def test_cut_context_rebuilds_text_at_detection():
    record = _record()
    assert cut_context(record, 0) == "Step A. Step B. Wait, that's wrong."
    assert cut_context(record, 1) == "Step A. " + ANCHOR + "Step C. Step D. Hmm, no."


def test_cut_context_checks_anchor_position():
    record = _record()
    record["rewinds"][0]["resume_char"] += 1
    with pytest.raises(ValueError):
        cut_context(record, 0)


def test_reason_token_span_uses_last_occurrence():
    text = "no. ok no. end"
    offsets = [(0, 3), (3, 4), (4, 6), (6, 7), (7, 10), (10, 14)]
    assert reason_token_span(offsets, text, "no.") == (4, 5)


def test_recall_threshold_keeps_minimum_recall():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.95])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    t = recall_threshold(scores, y, 0.8)
    assert t == 0.2 and (scores[y == 1] >= t).mean() == 0.8
