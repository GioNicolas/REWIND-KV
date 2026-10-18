"""Métricas do experimento: tamanho do KV-Cache, acurácia e repetição de erros (amnésia)."""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from statistics import mean

import torch

# Duas tentativas descartadas com início tão parecido assim contam como o mesmo erro repetido.
REPEAT_SIMILARITY = 0.8
REPEAT_PREFIX_CHARS = 300


def kv_cache_bytes(config, seq_len: int, dtype_bytes: int = 2) -> int:
    """Bytes ocupados pelo KV-Cache para `seq_len` tokens (K e V, todas as camadas)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    return 2 * config.num_hidden_layers * kv_heads * head_dim * seq_len * dtype_bytes


def peak_vram_bytes() -> int | None:
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None


def extract_boxed(text: str) -> str | None:
    """Conteúdo do último \\boxed{...}, respeitando chaves aninhadas."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def normalize_answer(ans: str) -> str:
    ans = ans.strip().strip("$").rstrip(".")
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\(?:left|right|!|,|;|\s)", "", ans)
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = re.sub(r"\^\{?\\circ\}?|°", "", ans)  # 90^\circ -> 90
    ans = re.sub(r"\\?%$", "", ans)
    ans = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", ans)  # \sqrt2 -> \sqrt{2}
    ans = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", ans)  # \frac12 -> \frac{1}{2}
    ans = re.sub(r"(?<=\d),(?=\d{3}\b)", "", ans)  # 1,000 -> 1000
    return ans.replace(" ", "")


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p == g


def repeated_failures(discarded_texts: list[str]) -> int:
    """Quantas tentativas descartadas repetem quase igual uma tentativa descartada anterior.

    É a medida direta do "Paradoxo da Amnésia": o blind_rewind deve repetir
    muito mais que o rewind_kv.
    """
    prefixes = [t[:REPEAT_PREFIX_CHARS] for t in discarded_texts]
    repeats = 0
    for i, cur in enumerate(prefixes):
        if any(difflib.SequenceMatcher(None, prev, cur).ratio() >= REPEAT_SIMILARITY for prev in prefixes[:i]):
            repeats += 1
    return repeats


def summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    """Agrega os registros de experiment.py por modo."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)

    summary = {}
    for mode, rs in by_mode.items():
        summary[mode] = {
            "n": len(rs),
            "accuracy": mean(r["correct"] for r in rs),
            "final_cache_len": mean(r["final_cache_len"] for r in rs),
            "peak_cache_len": mean(r["peak_cache_len"] for r in rs),
            "generated_tokens": mean(r["generated_tokens"] for r in rs),
            "detections": mean(r["detections"] for r in rs),
            "rewinds": mean(len(r["rewinds"]) for r in rs),
            "discarded_tokens": mean(sum(e["discarded_tokens"] for e in r["rewinds"]) for r in rs),
            "repeated_failures": mean(r["repeated_failures"] for r in rs),
            "hit_token_limit": mean(r["hit_token_limit"] for r in rs),
            "seconds": mean(r["seconds"] for r in rs),
        }
    return summary
