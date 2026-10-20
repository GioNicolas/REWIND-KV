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

# Suspeita não é falha: "Alternatively, maybe I made a mistake..." (piloto 2).
# Só olha o trecho ANTES do marcador, para não descartar "This contradicts X, so maybe...".
_HEDGE_RE = re.compile(r"\b(?:maybe|perhaps|might|possibly|probably)\b", re.IGNORECASE)

_MARKER_RE = re.compile("|".join(f"(?:{m})" for m in FAILURE_MARKERS), re.IGNORECASE)

# Modo estrito (v0.3): na validação da v0.2, 4 em cada 10 cortes jogaram fora algo certo
# (docs/resultados-experimentos.md). Estas regras descartam os tipos de frase que causaram isso.
_QUESTION_START_RE = re.compile(
    r"^(?:(?:wait|hmm+|oh|but|okay|ok|so|actually|hold on)\b[\s,.!]*)*"
    r"(?:is|are|was|were|did|do|does|could|can|have|has|am)\b",
    re.IGNORECASE,
)
# Avisos genéricos: "sometimes ... it's easy to make a mistake", "before I conclude, ... any chance".
_GENERIC_RE = re.compile(r"\b(?:sometimes|easy to|any chance|be careful|before I conclude)\b", re.IGNORECASE)
# Eliminar um candidato não é errar: "So, t=1 doesn't work."
_ELIMINATION_RE = re.compile(
    rf"\b[A-Za-z]\s*=\s*-?\d+(?:\.\d+)?\s*,?\s*(?:doesn{_APOS}?t|does not|didn{_APOS}?t|did not) work",
    re.IGNORECASE,
)
# Uma ação, não uma declaração de erro.
_ACTION_MARKER_RE = re.compile(r"let me recalculate", re.IGNORECASE)
# Contradição entre dois resultados: a frase não diz qual lado está errado, e o corte
# sempre levaria o mais recente ("2107, but earlier I got 2037... I made a mistake somewhere").
# Desligada por padrão: nos cortes rotulados, pulava 11 erros reais para evitar 5 cortes do
# lado errado. As palavras não separam os dois casos; isso pede outro sinal.
_INCONSISTENCY_RE = re.compile(
    r"\b(?:earlier|previous(?:ly)?|before|other (?:approach|method|result|way)|(?:first|second) (?:approach|method|calculation)"
    rf"|one of the|somewhere|which one|I got|I{_APOS}?m getting|I am getting|I had)\b",
    re.IGNORECASE,
)


def strict_skip_reason(sentence: str, marker: str, skip_inconsistency: bool = False) -> str | None:
    """Por que o modo estrito NÃO dispararia nesta frase (ou None, se dispararia).

    Categorias: "pergunta", "aviso", "eliminacao", "acao", "contradicao".
    """
    s = sentence.strip()
    if s.endswith("?") or _QUESTION_START_RE.match(s):
        return "pergunta"
    if _GENERIC_RE.search(s):
        return "aviso"
    if _ELIMINATION_RE.search(s):
        return "eliminacao"
    if _ACTION_MARKER_RE.fullmatch(marker.strip()):
        return "acao"
    if skip_inconsistency and _INCONSISTENCY_RE.search(s):
        return "contradicao"
    return None
# Ponto seguido de espaço, para não confundir com decimais ("3.5").
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s)|\n")
_SENTENCE_START_RE = re.compile(r"[.!?]\s|\n")
_FILLER_RE = re.compile(r"^(?:(?:wait|hmm+|oh|no|but|actually|so|okay|ok|hold on)\b[\s,.!]*)+", re.IGNORECASE)

# Frases que abrem um caminho novo ("Alternatively, ...", "Let me try ...", "Suppose ...").
# No Experimento A, cortar só o parágrafo do "errei" deixava as premissas do caminho
# no cache, e o modelo refazia a mesma conta logo depois da âncora.
_BRANCH_OPENER_RE = re.compile(
    r"(?:(?:wait|hmm+|oh|but|okay|ok|so|actually|no|hold on)\b[\s,.!]*)*"
    r"(?:alternatively|another (?:way|approach|method|idea)|instead|suppose|assume|what if|if I"
    r"|let me (?:try|consider|approach|use|think (?:differently|of another|about another))"
    rf"|let{_APOS}?s (?:try|consider)"
    rf"|(?:maybe|perhaps) (?:I should|we should|I can|I could|we can|we could|the|it{_APOS}?s))\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Se o modelo não fechar a frase, corta o motivo depois deste tanto de caracteres.
MAX_REASON_CHARS = 200


@dataclass
class FailureSignal:
    marker: str  # trecho que casou com o marcador
    reason: str  # frase inteira em que o modelo declara o erro
    start: int  # offset (em caracteres) do início da frase
    end: int  # offset (em caracteres) logo após o fim da frase


def detect_failure(text: str, strict: bool = False) -> FailureSignal | None:
    """Retorna o primeiro sinal de falha completo em `text`, ou None.

    Retorna None também quando há marcador mas a frase ainda não terminou:
    o chamador deve tentar de novo no próximo token. Com `strict`, também
    pula as frases que `strict_skip_reason` descarta.
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
        if _VERIFICATION_RE.search(sentence) or _HEDGE_RE.search(text, start, m.start()):
            continue
        if strict and strict_skip_reason(sentence, m.group(0)):
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


def _sentences(text: str):
    for line in text.splitlines():
        line = line.strip()
        start = 0
        for m in _SENTENCE_END_RE.finditer(line + " "):
            yield line[start : m.end()].strip()
            start = m.end()
        if line[start:].strip():
            yield line[start:].strip()


def informative_sentence(text: str, min_words: int = 4) -> str:
    """Primeira frase com conteúdo: pula "Wait, hold on, no." e linhas só de números ("0 1 0 | 1 0 1")."""
    for sentence in _sentences(text):
        if len(_WORD_RE.findall(_FILLER_RE.sub("", sentence))) >= min_words:
            return sentence
    return first_sentence(text)


def opens_branch(paragraph: str) -> bool:
    """O parágrafo começa um caminho novo? ("Alternatively, ...", "Wait, maybe the ...", "Let me try ...")."""
    return _BRANCH_OPENER_RE.match(paragraph.lstrip()) is not None
