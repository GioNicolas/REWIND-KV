"""RewindKVGenerator: decodificação token a token com rollback físico do KV-Cache.

Os três modos do experimento usam exatamente o mesmo laço de geração e o
mesmo detector; só muda o que acontece quando uma falha é detectada:

- baseline:      nada (a falha só é registrada);
- blind_rewind:  corte do KV-Cache sem âncora (ablação para medir a amnésia);
- rewind_kv:     corte do KV-Cache + injeção da Âncora de Falha.

O escopo do corte também é configurável:

- paragraph:  corta só o passo errado (padrão);
- attempt:    corta tudo desde o último checkpoint (v0.1 original; no piloto
              apagava raciocínio certo por causa de um erro local).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch
from transformers import DynamicCache

from .detector import clean_reason, detect_failure, first_sentence

BOXED_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ANCHOR_TEMPLATE = '[Note: "{hypothesis}" was wrong: {reason} Do not repeat it.]\n\n'


class Mode(str, Enum):
    BASELINE = "baseline"
    BLIND_REWIND = "blind_rewind"
    REWIND_KV = "rewind_kv"


class RewindScope(str, Enum):
    PARAGRAPH = "paragraph"
    ATTEMPT = "attempt"


@dataclass
class RewindEvent:
    generated_at: int  # nº de tokens gerados (incluindo descartados) no momento do corte
    checkpoint: int  # L_0 desta reversão: comprimento do cache para onde o corte voltou
    discarded_tokens: int
    discarded_text: str
    reason: str
    anchor: str
    anchor_tokens: int


@dataclass
class GenerationResult:
    mode: str
    rewind_scope: str
    text: str  # o que sobrou no cache depois do prompt (com âncoras, sem rascunhos cortados)
    prompt_tokens: int
    final_cache_len: int
    peak_cache_len: int
    generated_tokens: int  # todos os tokens amostrados, inclusive os que foram cortados
    detections: int  # falhas detectadas (no baseline elas só são registradas)
    hit_token_limit: bool
    seconds: float
    detected_reasons: list[str] = field(default_factory=list)
    rewinds: list[RewindEvent] = field(default_factory=list)


def _forward(model, cache: DynamicCache, ids: list[int]) -> torch.Tensor:
    """Processa `ids` a partir do fim do cache, com posições contínuas. Retorna os logits do próximo token."""
    start = cache.get_seq_length()
    input_ids = torch.tensor([ids], device=model.device)
    cache_position = torch.arange(start, start + len(ids), device=model.device)
    out = model(input_ids=input_ids, past_key_values=cache, cache_position=cache_position, use_cache=True)
    return out.logits[0, -1]


def crop_to(cache: DynamicCache, length: int) -> None:
    """Corta o cache para `length` tokens.

    No transformers >= 5, `crop(n)` com n positivo (tamanho absoluto) dá erro; o
    jeito suportado é `crop(-k)`, que remove os k últimos tokens. A forma negativa
    também funciona nas versões 4.x.
    """
    to_remove = cache.get_seq_length() - length
    if to_remove > 0:
        cache.crop(-to_remove)


@torch.no_grad()
def rewind_and_inject(model, cache: DynamicCache, tokens: list[int], checkpoint: int, anchor_ids: list[int]) -> torch.Tensor:
    """Corta o cache de volta para `checkpoint` (L_0) e injeta a âncora com RoPE contínuo.

    Para amostrar o próximo token é preciso o estado oculto do último token antes
    do corte, e esse estado não fica guardado no cache. Por isso o corte vai até
    L_0 - 1 e o token tokens[L_0 - 1] é reprocessado junto com a âncora, num único
    forward. O resultado é idêntico a ter gerado prompt + âncora do zero.

    Modifica `cache` e `tokens` in place e retorna os logits do próximo token.
    """
    del tokens[checkpoint:]
    crop_to(cache, checkpoint - 1)
    feed = [tokens[-1], *anchor_ids]
    tokens.extend(anchor_ids)
    return _forward(model, cache, feed)


def paragraph_rewind_target(paragraphs: list[tuple[int, int]], text: str, sentence_start: int) -> tuple[int, int]:
    """Escolhe para onde cortar no escopo de parágrafo.

    `paragraphs` lista (índice do token, offset em `text`) do início de cada
    parágrafo da tentativa atual; o primeiro é sempre o próprio checkpoint.
    Se a frase de erro está no meio de um parágrafo, o passo errado é esse
    parágrafo. Se ela abre um parágrafo ("\\n\\nWait, that's wrong..."), o passo
    errado é o parágrafo anterior.
    """
    k = max(i for i, (_, char) in enumerate(paragraphs) if char <= sentence_start)
    if k > 0 and not text[paragraphs[k][1] : sentence_start].strip():
        k -= 1
    return paragraphs[k]


def _sample(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, order = probs.sort(descending=True)
    sorted_probs = sorted_probs * (sorted_probs.cumsum(-1) - sorted_probs < top_p)
    choice = torch.multinomial(sorted_probs, 1, generator=generator)
    return int(order[choice])


class RewindKVGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        mode: Mode | str = Mode.REWIND_KV,
        *,
        rewind_scope: RewindScope | str = RewindScope.PARAGRAPH,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,  # valores recomendados pela DeepSeek para o R1
        top_p: float = 0.95,
        min_cut_tokens: int = 32,  # cortes menores que a âncora não economizam nada
        max_rewinds: int = 8,  # evita loop infinito, principalmente no blind_rewind
        max_hypothesis_tokens: int = 12,
        max_reason_tokens: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = Mode(mode)
        self.rewind_scope = RewindScope(rewind_scope)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_cut_tokens = min_cut_tokens
        self.max_rewinds = max_rewinds
        self.max_hypothesis_tokens = max_hypothesis_tokens
        self.max_reason_tokens = max_reason_tokens

    def build_prompt_ids(self, question: str) -> list[int]:
        messages = [{"role": "user", "content": question + BOXED_INSTRUCTION}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Versões novas do template do R1-Distill já terminam em "<think>\n"; as antigas não.
        if not text.rstrip().endswith("<think>"):
            text += "<think>\n"
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens]).rstrip() + "..."

    def build_anchor(self, cut_text: str, reason: str) -> str:
        """Âncora de Falha = 1ª frase do trecho cortado (o passo errado) + frase do erro, sem as contas do meio."""
        hypothesis = first_sentence(cut_text).replace('"', "'")
        return ANCHOR_TEMPLATE.format(
            hypothesis=self._truncate(hypothesis, self.max_hypothesis_tokens),
            reason=self._truncate(clean_reason(reason), self.max_reason_tokens),
        )

    @torch.no_grad()
    def generate(self, question: str, seed: int = 0) -> GenerationResult:
        t0 = time.perf_counter()
        generator = torch.Generator(device=self.model.device).manual_seed(seed)
        eos_id = self.tokenizer.eos_token_id

        tokens = self.build_prompt_ids(question)
        prompt_len = len(tokens)
        cache = DynamicCache()
        logits = _forward(self.model, cache, tokens)

        # Tudo antes do checkpoint (prompt, âncoras, falhas já registradas) fica protegido.
        checkpoint = prompt_len
        paragraphs = [(checkpoint, 0)]
        peak = len(tokens)
        generated = 0
        detected_reasons: list[str] = []
        rewinds: list[RewindEvent] = []
        in_think = True
        hit_limit = True

        while generated < self.max_new_tokens:
            next_id = _sample(logits, self.temperature, self.top_p, generator)
            tokens.append(next_id)
            generated += 1
            if next_id == eos_id:
                hit_limit = False
                break
            logits = _forward(self.model, cache, [next_id])
            peak = max(peak, len(tokens))

            if not in_think:
                continue
            attempt_text = self.tokenizer.decode(tokens[checkpoint:])
            if "</think>" in attempt_text:
                in_think = False  # só fazemos rollback dentro do raciocínio
                continue
            if attempt_text.endswith("\n\n") and paragraphs[-1][1] != len(attempt_text):
                paragraphs.append((len(tokens), len(attempt_text)))

            signal = detect_failure(attempt_text)
            if signal is None:
                continue
            detected_reasons.append(signal.reason)

            if self.rewind_scope == RewindScope.PARAGRAPH:
                cut_token, cut_char = paragraph_rewind_target(paragraphs, attempt_text, signal.start)
            else:
                cut_token, cut_char = checkpoint, 0
            cut_len = len(tokens) - cut_token
            can_rewind = (
                self.mode != Mode.BASELINE
                and cut_len >= self.min_cut_tokens
                and len(rewinds) < self.max_rewinds
            )
            if not can_rewind:
                checkpoint = len(tokens)  # não detecta a mesma falha de novo
                paragraphs = [(checkpoint, 0)]
                continue

            cut_text = attempt_text[cut_char:]
            anchor = self.build_anchor(cut_text, signal.reason) if self.mode == Mode.REWIND_KV else ""
            anchor_ids = self.tokenizer(anchor, add_special_tokens=False).input_ids if anchor else []
            rewinds.append(
                RewindEvent(
                    generated_at=generated,
                    checkpoint=cut_token,
                    discarded_tokens=cut_len,
                    discarded_text=cut_text,
                    reason=signal.reason,
                    anchor=anchor,
                    anchor_tokens=len(anchor_ids),
                )
            )
            logits = rewind_and_inject(self.model, cache, tokens, cut_token, anchor_ids)
            checkpoint = len(tokens)  # a âncora fica "trancada" antes do novo L_0
            paragraphs = [(checkpoint, 0)]

        return GenerationResult(
            mode=self.mode.value,
            rewind_scope=self.rewind_scope.value,
            text=self.tokenizer.decode(tokens[prompt_len:], skip_special_tokens=True),
            prompt_tokens=prompt_len,
            final_cache_len=cache.get_seq_length(),
            peak_cache_len=peak,
            generated_tokens=generated,
            detections=len(detected_reasons),
            hit_token_limit=hit_limit,
            seconds=time.perf_counter() - t0,
            detected_reasons=detected_reasons,
            rewinds=rewinds,
        )
