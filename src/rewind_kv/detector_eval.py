"""Avalia o modo estrito do detector nos cortes rotulados, sem GPU.

Para cada corte rotulado, pergunta se o modo estrito ainda dispararia na frase que
causou o corte. É uma aproximação: se o detector pula uma frase, o modelo geraria
um texto diferente dali em diante, e os cortes seguintes poderiam mudar.

Uso: python -m rewind_kv.detector_eval docs/dados/rotulos-cortes-validacao-v02.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .detector import _MARKER_RE, strict_skip_reason

GOOD = {"real"}  # cortes que queremos manter; os outros jogaram fora algo certo
LABEL_ORDER = ["real", "real_colateral", "lado_errado", "falso"]


def evaluate(cuts: list[dict], skip_inconsistency: bool) -> dict:
    kept, skipped = Counter(), Counter()
    reasons: dict[str, Counter] = {}
    for c in cuts:
        marker = _MARKER_RE.search(c["reason"])
        why = strict_skip_reason(c["reason"], marker.group(0) if marker else "", skip_inconsistency)
        if why is None:
            kept[c["label"]] += 1
        else:
            skipped[c["label"]] += 1
            reasons.setdefault(why, Counter())[c["label"]] += 1
    n_kept = sum(kept.values())
    good_total = sum(1 for c in cuts if c["label"] in GOOD)
    return {
        "kept": kept,
        "skipped": skipped,
        "reasons": reasons,
        "precision": sum(kept[l] for l in GOOD) / n_kept if n_kept else float("nan"),
        "recall": sum(kept[l] for l in GOOD) / good_total if good_total else float("nan"),
        "n_kept": n_kept,
    }


def report(name: str, cuts: list[dict]) -> None:
    base_good = sum(1 for c in cuts if c["label"] in GOOD)
    print(f"\n== {name}: {len(cuts)} cortes; v0.2 mantinha todos (precisão {base_good / len(cuts):.0%})")
    for label, skip_inc in [("estrito (padrão)", False), ("estrito + regra de contradição", True)]:
        r = evaluate(cuts, skip_inc)
        kept = ", ".join(f"{l} {r['kept'][l]}" for l in LABEL_ORDER)
        print(f"  {label}: mantém {r['n_kept']} ({kept})")
        print(f"     precisão {r['precision']:.0%} · mantém {r['recall']:.0%} dos erros reais")
        for why, cnt in sorted(r["reasons"].items()):
            print(f"     pulou por {why}: " + ", ".join(f"{l} {cnt[l]}" for l in LABEL_ORDER if cnt[l]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", help="arquivo JSON de rótulos (docs/dados/...)")
    args = parser.parse_args()
    cuts = json.load(open(args.labels, encoding="utf-8"))["cuts"]
    report("todos os cortes", cuts)
    report("só o 1º corte de cada problema", [c for c in cuts if len(c["modes"]) == 2])


if __name__ == "__main__":
    main()
