"""Roda os 3 modos sobre um benchmark e salva um JSONL por execução em results/.

Uso:
    python -m rewind_kv.experiment --dataset gsm8k --n 30
    python -m rewind_kv.experiment --dataset math500 --level 4 --n 30 --modes baseline rewind_kv
    python -m rewind_kv.experiment --dataset math500 --level 4 --n 30 --rewind-scope attempt  # ablação
    python -m rewind_kv.experiment --dataset math500 --level 4 --problems 0 3 6 --modes rewind_kv  # só esses
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .controller import AnchorStyle, Mode, RewindKVGenerator, RewindScope
from .metrics import extract_boxed, is_correct, kv_cache_bytes, peak_vram_bytes, repeated_failures, summarize

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_problems(dataset: str, level: int | None = None) -> list[dict]:
    from datasets import load_dataset

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [
            {"question": r["problem"], "answer": r["answer"]}
            for r in ds
            if level is None or r["level"] == level
        ]
    else:
        raise ValueError(f"dataset desconhecido: {dataset}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], default="gsm8k")
    parser.add_argument("--level", type=int, default=None, help="só para math500: nível de dificuldade (1-5)")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--start", type=int, default=0, help="pula os primeiros problemas (para retomar uma execução que caiu)")
    parser.add_argument("--problems", type=int, nargs="+", help="roda só estes índices (ignora --start e --n)")
    parser.add_argument("--modes", nargs="+", choices=[m.value for m in Mode], default=[m.value for m in Mode])
    parser.add_argument(
        "--rewind-scope",
        choices=[s.value for s in RewindScope],
        default=RewindScope.BRANCH.value,
        help="branch: volta até o início do caminho que falhou (v0.2); paragraph: só o passo errado (Exp. A); "
        "attempt: desde o último checkpoint (v0.1)",
    )
    parser.add_argument(
        "--anchor-style",
        choices=[s.value for s in AnchorStyle],
        default=AnchorStyle.VOICE.value,
        help="voice: na voz do modelo (v0.2); note: [Note: ...] (Exp. A)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    dtype_bytes = torch.finfo(dtype).bits // 8

    problems = load_problems(args.dataset, args.level)
    if args.problems:
        selected = [(i, problems[i]) for i in args.problems]
    else:
        selected = list(enumerate(problems))[args.start : args.start + args.n]
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"{len(selected)} problemas x {len(args.modes)} modos em {device} -> {out_path}")

    records = []
    with out_path.open("w", encoding="utf-8") as f:
        # Com --start, a contagem continua de onde parou ("[N/30]" diz o que já terminou).
        first = 1 if args.problems else args.start + 1
        total = first - 1 + len(selected)
        for pos, (i, problem) in enumerate(selected, start=first):
            for mode in args.modes:
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                gen = RewindKVGenerator(
                    model,
                    tokenizer,
                    mode,
                    rewind_scope=args.rewind_scope,
                    anchor_style=args.anchor_style,
                    max_new_tokens=args.max_new_tokens,
                )
                # Mesma semente em todos os modos: as gerações são idênticas até o primeiro corte.
                result = gen.generate(problem["question"], seed=args.seed + i)
                pred = extract_boxed(result.text)
                record = {
                    "problem_idx": i,
                    **asdict(result),
                    "gold": problem["answer"],
                    "pred": pred,
                    "correct": is_correct(pred, problem["answer"]),
                    "repeated_failures": repeated_failures([e.discarded_text for e in result.rewinds]),
                    "final_cache_mb": kv_cache_bytes(model.config, result.final_cache_len, dtype_bytes) / 2**20,
                    "peak_cache_mb": kv_cache_bytes(model.config, result.peak_cache_len, dtype_bytes) / 2**20,
                    "peak_vram_mb": (peak_vram_bytes() or 0) / 2**20,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                records.append(record)
                print(
                    f"[{pos}/{total}] "
                    f"p{i:<3} {mode:<13} correct={record['correct']!s:<5} "
                    f"cache={result.final_cache_len:>5} gerados={result.generated_tokens:>5} "
                    f"deteccoes={result.detections} rewinds={len(result.rewinds)} {result.seconds:.0f}s"
                )

    print("\nResumo por modo:")
    for mode, stats in summarize(records).items():
        print(f"  {mode}: " + ", ".join(f"{k}={v:.3g}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
