"""Gatilho de falha v0.1: detecção léxica dos marcadores de backtracking do próprio modelo.

Modelos como o DeepSeek-R1 declaram explicitamente quando uma hipótese falhou
("Wait, that's wrong...", "This contradicts..."). O detector procura esses
marcadores no texto da tentativa atual e só dispara quando a frase termina,
para que o motivo do erro esteja completo e possa virar Âncora de Falha.

Na dúvida, o detector prefere NÃO disparar: um falso positivo apaga trabalho
certo (no piloto, apagou duas respostas corretas), enquanto um falso negativo
só deixa o cache como no baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_APOS = "['’]"
# Lookbehinds de largura fixa para "didn't/haven't/did not make a mistake".
_NOT_NEGATED = rf"(?<!n{_APOS}t )(?<!not )"

# "Wait," sozinho NÃO é falha: o R1 usa "Wait, let me check" o tempo todo.
# Só entram frases que declaram que o caminho atual está errado.
FAILURE_MARKERS = [
    rf"that{_APOS}?s (?:not right|not correct|wrong|incorrect)",
    r"that is (?:not right|not correct|wrong|incorrect)",
    r"(?:this|that|which) contradicts",
    r"\bcontradiction\b",
    rf"(?:doesn{_APOS}?t|does not|didn{_APOS}?t|did not) work",
    rf"{_NOT_NEGATED}(?:made|make) (?:a|an) (?:mistake|error)",
    r"let me recalculate",
    rf"(?:that|this) can{_APOS}?t be (?:right|correct)",
    rf"(?:that|this) doesn{_APOS}?t make sense",
]

# Frases de verificação que contêm um marcador mas não declaram erro:
# "let me verify to make sure I made no mistake", "in case I made an error".
_VERIFICATION_RE = re.compile(
    r"\b(?:make sure|ensure|verify|double[- ]check|in case|whether|if I)\b",
    re.IGNORECASE,
)

_MARKER_RE = re.compile("|".join(f"(?:{m})" for m in FAILURE_MARKERS), re.IGNORECASE)
# Ponto seguido de espaço, para não confundir com decimais ("3.5").
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s)|\n")
_SENTENCE_START_RE = re.compile(r"[.!?]\s|\n")
_FILLER_RE = re.compile(r"^(?:(?:wait|hmm+|oh|no|but|actually|so|okay|ok)\b[\s,.!]*)+", re.IGNORECASE)

# Se o modelo não fechar a frase, corta o motivo depois deste tanto de caracteres.
MAX_REASON_CHARS = 200


@dataclass
class FailureSignal:
    marker: str  # trecho que casou com o marcador
    reason: str  # frase inteira em que o modelo declara o erro
    start: int  # offset (em caracteres) do início da frase
    end: int  # offset (em caracteres) logo após o fim da frase


def detect_failure(text: str) -> FailureSignal | None:
    """Retorna o primeiro sinal de falha completo em `text`, ou None.

    Retorna None também quando há marcador mas a frase ainda não terminou:
    o chamador deve tentar de novo no próximo token.
    """
    for m in _MARKER_RE.finditer(text):
        end_m = _SENTENCE_END_RE.search(text, m.end())
        if end_m is not None:
            end = end_m.end()
        elif len(text) - m.end() >= MAX_REASON_CHARS:
            end = m.end() + MAX_REASON_CHARS
        else:
            return None

        start = 0
        for s in _SENTENCE_START_RE.finditer(text, 0, m.start()):
            start = s.end()
        sentence = text[start:end]
        if _VERIFICATION_RE.search(sentence):
            continue
        return FailureSignal(marker=m.group(0), reason=sentence.strip(), start=start, end=end)
    return None


def clean_reason(reason: str) -> str:
    """Tira as interjeições do começo: "Wait, no, actually, that's wrong..." -> "that's wrong..."."""
    return _FILLER_RE.sub("", reason).strip() or reason


def first_sentence(text: str) -> str:
    """Primeira frase não vazia do trecho: a descrição do passo que ele tentou."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            end_m = _SENTENCE_END_RE.search(line + " ")
            return line[: end_m.end()] if end_m else line
    return ""
