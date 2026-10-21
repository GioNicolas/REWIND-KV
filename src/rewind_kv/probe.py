"""Sonda nos estados internos: o modelo "sabe" se a falha detectada é real?

Teste de viabilidade pré-registrado em docs/resultados-experimentos.md. Para cada
corte rotulado, refaz o contexto que o modelo tinha no instante da detecção e
guarda o estado oculto (último token e média da frase de erro) e a entropia da
próxima palavra. Depois treina uma regressão logística no nível 4 e testa no nível 5.

Uso:
    python -m rewind_kv.probe extract --runs <pasta .jsonl> --labels <rotulos.json> \\
        --problems <MATH-500 test.jsonl> --level 4 --out feats_n4.npz
    python -m rewind_kv.probe evaluate --train feats_n4.npz --test feats_n5.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time

import numpy as np

TARGET = {"real": 1, "falso": 0, "lado_errado": 0}  # real_colateral fica de fora
GRID_LAYERS = [7, 14, 21, 28]
GRID_POOL = ["last", "mean"]
GRID_C = [0.01, 0.1, 1.0]
MIN_RECALL = 0.80  # o limiar é escolhido no treino para manter 80% dos erros reais


def load_runs(folder: str) -> dict[tuple[int, str], dict]:
    runs = {}
    for path in sorted(glob.glob(f"{folder}/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            runs[(r["problem_idx"], r["mode"])] = r
    return runs


def load_problems_jsonl(path: str, level: int) -> list[dict]:
    """O mesmo filtro de experiment.load_problems, lendo o test.jsonl do MATH-500 já baixado."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [{"question": r["problem"], "answer": r["answer"]} for r in rows if r["level"] == level]


def cut_context(record: dict, order: int) -> str:
    """Texto gerado que o modelo tinha quando o detector disparou o corte número `order`.

    O texto final do modo guarda tudo até o ponto do corte (seguido da âncora, se houver);
    o trecho cortado vem do evento. `resume_char` marca o fim da âncora no texto final.
    """
    event = record["rewinds"][order]
    start = event["resume_char"] - len(event["anchor"])
    if record["text"][start : event["resume_char"]] != event["anchor"]:
        raise ValueError(f"âncora fora do lugar em p{record['problem_idx']} {record['mode']} corte {order}")
    return record["text"][:start] + event["discarded_text"]


def labeled_cuts(runs: dict, labels_path: str) -> list[dict]:
    cuts = []
    for c in json.load(open(labels_path, encoding="utf-8"))["cuts"]:
        if c["label"] not in TARGET:
            continue
        mode = c["modes"][0]
        record = runs[(c["problem_idx"], mode)]
        cuts.append({
            "id": c["id"],
            "problem_idx": c["problem_idx"],
            "y": TARGET[c["label"]],
            "context": cut_context(record, c["rewind_order"][mode]),
            "reason": c["reason"],
        })
    return cuts


def reason_token_span(offsets: list[tuple[int, int]], text: str, reason: str) -> tuple[int, int]:
    """Índices [início, fim) dos tokens que cobrem a última ocorrência de `reason` em `text`."""
    start = text.rfind(reason)
    if start < 0:
        raise ValueError(f"frase de erro não encontrada: {reason[:60]!r}")
    end = start + len(reason)
    idx = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    return idx[0], idx[-1] + 1


def extract(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .controller import RewindKVGenerator

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.eval()
    prompter = RewindKVGenerator(None, tok, "baseline")
    problems = load_problems_jsonl(args.problems, args.level)
    runs = load_runs(args.runs)
    cuts = labeled_cuts(runs, args.labels)
    for (p, _), r in runs.items():
        assert r["gold"] == problems[p]["answer"], f"problema {p} não bate com o gabarito"

    last, mean, entropy = [], [], []
    for i, c in enumerate(cuts):
        t0 = time.perf_counter()
        prompt_ids = prompter.build_prompt_ids(problems[c["problem_idx"]]["question"])
        enc = tok(c["context"], add_special_tokens=False, return_offsets_mapping=True)
        s, e = reason_token_span(enc["offset_mapping"], c["context"], c["reason"])
        s, e = s + len(prompt_ids), e + len(prompt_ids)
        ids = torch.tensor([prompt_ids + enc["input_ids"]], device=device)
        n = ids.shape[1]
        # logits só a partir da posição que prevê o 1º token da frase de erro
        out = model(ids, output_hidden_states=True, use_cache=False, logits_to_keep=n - (s - 1))
        # hidden_states: embeddings + uma saída por camada, cada uma (1, n, d)
        last.append(np.stack([h[0, -1].cpu().numpy() for h in out.hidden_states]))
        mean.append(np.stack([h[0, s:e].mean(dim=0).cpu().numpy() for h in out.hidden_states]))
        logp = torch.log_softmax(out.logits[0, : e - s].float(), dim=-1)  # preveem os tokens s..e-1
        entropy.append(float(-(logp.exp() * logp).sum(-1).mean()))
        print(f"[{i + 1}/{len(cuts)}] {c['id']} p{c['problem_idx']} y={c['y']} tokens={n} "
              f"frase={e - s} entropia={entropy[-1]:.3f} {time.perf_counter() - t0:.0f}s", flush=True)

    np.savez_compressed(
        args.out,
        ids=np.array([c["id"] for c in cuts]),
        problem_idx=np.array([c["problem_idx"] for c in cuts]),
        y=np.array([c["y"] for c in cuts]),
        last=np.stack(last).astype(np.float32),
        mean=np.stack(mean).astype(np.float32),
        entropy=np.array(entropy, dtype=np.float32),
    )
    print("salvo em", args.out)


def _features(data, layer: int | None, pool: str) -> np.ndarray:
    if pool == "entropy":
        return data["entropy"][:, None]
    return data[pool][:, layer]


def _pipeline(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float) -> np.ndarray:
    """Probabilidades fora da amostra, deixando um problema de fora por vez."""
    from sklearn.model_selection import LeaveOneGroupOut

    scores = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        scores[te] = _pipeline(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return scores


def recall_threshold(scores: np.ndarray, y: np.ndarray, min_recall: float = MIN_RECALL) -> float:
    """Maior limiar que ainda mantém `min_recall` dos positivos."""
    pos = np.sort(scores[y == 1])[::-1]
    return float(pos[math.ceil(min_recall * len(pos)) - 1])


def precision_recall(scores: np.ndarray, y: np.ndarray, threshold: float) -> tuple[float, float, int]:
    keep = scores >= threshold
    tp = int((keep & (y == 1)).sum())
    return (tp / keep.sum() if keep.sum() else float("nan")), tp / (y == 1).sum(), int(keep.sum())


def bootstrap_auc(scores: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs = []
    while len(aucs) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            aucs.append(roc_auc_score(y[i], scores[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def fit_and_test(train, test, layer, pool, C) -> dict:
    """Protocolo pré-registrado para uma configuração já escolhida."""
    from sklearn.metrics import roc_auc_score

    Xtr, ytr, Xte, yte = _features(train, layer, pool), train["y"], _features(test, layer, pool), test["y"]
    threshold = recall_threshold(oof_scores(Xtr, ytr, train["problem_idx"], C), ytr)
    scores = _pipeline(C).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    precision, recall, kept = precision_recall(scores, yte, threshold)
    return {"threshold": threshold, "precision": precision, "recall": recall, "kept": kept,
            "auc": roc_auc_score(yte, scores), "auc_ci": bootstrap_auc(scores, yte)}


def evaluate(args) -> None:
    from sklearn.metrics import roc_auc_score

    train, test = dict(np.load(args.train)), dict(np.load(args.test))
    ytr, groups = train["y"], train["problem_idx"]
    print(f"treino: {len(ytr)} cortes ({ytr.sum()} reais); teste: {len(test['y'])} cortes ({test['y'].sum()} reais)")
    print(f"cortar tudo no teste: precisão {test['y'].mean():.1%}, recall 100%\n")

    print("Escolha no nível 4 (AUC da validação cruzada, um problema de fora por vez):")
    grid = []
    for layer in GRID_LAYERS:
        for pool in GRID_POOL:
            for C in GRID_C:
                auc = roc_auc_score(ytr, oof_scores(_features(train, layer, pool), ytr, groups, C))
                grid.append((auc, layer, pool, C))
                print(f"  camada {layer:>2} {pool:<4} C={C:<5} AUC={auc:.3f}")
    best_auc, layer, pool, C = max(grid)
    print(f"\nescolhida: camada {layer}, {pool}, C={C} (AUC na validação cruzada {best_auc:.3f})")

    r = fit_and_test(train, test, layer, pool, C)
    ok = r["precision"] >= 0.75 and r["recall"] >= 0.70
    print(f"\nNÍVEL 5 (teste, uma vez só): limiar {r['threshold']:.3f}, mantém {r['kept']} de {len(test['y'])} cortes")
    print(f"  precisão {r['precision']:.1%}  recall {r['recall']:.1%}  AUC {r['auc']:.3f} "
          f"(IC 95% {r['auc_ci'][0]:.3f}–{r['auc_ci'][1]:.3f})")
    print(f"  critério (precisão ≥ 75% e recall ≥ 70%): {'ATINGIDO' if ok else 'NÃO atingido'}")

    print("\nComparação: só a entropia da frase de erro")
    ent = max((roc_auc_score(ytr, oof_scores(_features(train, None, "entropy"), ytr, groups, c)), c) for c in GRID_C)
    e = fit_and_test(train, test, None, "entropy", ent[1])
    print(f"  validação cruzada AUC {ent[0]:.3f}; nível 5: precisão {e['precision']:.1%} recall {e['recall']:.1%} "
          f"AUC {e['auc']:.3f} (IC 95% {e['auc_ci'][0]:.3f}–{e['auc_ci'][1]:.3f})")

    rng = np.random.default_rng(0)
    X = _features(train, layer, pool)
    shuffled = [roc_auc_score(p, oof_scores(X, p, groups, C)) for p in (rng.permutation(ytr) for _ in range(20))]
    print(f"\nControle: rótulos embaralhados no nível 4, AUC médio {np.mean(shuffled):.3f} "
          f"(máx {np.max(shuffled):.3f}, 20 sorteios)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--runs", required=True)
    ex.add_argument("--labels", required=True)
    ex.add_argument("--problems", required=True)
    ex.add_argument("--level", type=int, required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--train", required=True)
    ev.add_argument("--test", required=True)
    args = parser.parse_args()
    extract(args) if args.cmd == "extract" else evaluate(args)


if __name__ == "__main__":
    main()
