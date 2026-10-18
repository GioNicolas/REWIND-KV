"""RewindKVGenerator: decodificação token a token com rollback físico do KV-Cache.

Os três modos do experimento usam exatamente o mesmo laço de geração e o
mesmo detector; só muda o que acontece quando uma falha é detectada:

- baseline:      nada (a falha só é contada e o checkpoint avança);
- blind_rewind:  crop(L_0) sem âncora (ablação para medir a amnésia);
- rewind_kv:     crop(L_0) + injeção da Âncora de Falha.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch
from transformers import DynamicCache

from .detector import detect_failure, first_sentence

BOXED_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ANCHOR_TEMPLATE = '\n[Note: attempt "{hypothesis}" failed: {reason} Try a different approach.]\n'


class Mode(str, Enum):
    BASELINE = "baseline"
    BLIND_REWIND = "blind_rewind"
    REWIND_KV = "rewind_kv"


@dataclass
class RewindEvent:
    generated_at: int  # nº de tokens gerados (incluindo descartados) no momento do corte
    checkpoint: int  # L_0: comprimento do cache para onde o corte voltou
    discarded_tokens: int
    discarded_text: str
    reason: str
    anchor: str
    anchor_tokens: int


@dataclass
class GenerationResult:
    mode: str
    text: str  # o que sobrou no cache depois do prompt (com âncoras, sem rascunhos cortados)
    prompt_tokens: int
    final_cache_len: int
    peak_cache_len: int
    generated_tokens: int  # todos os tokens amostrados, inclusive os que foram cortados
    detections: int  # falhas detectadas (no baseline elas só são contadas)
    hit_token_limit: bool
    seconds: float
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
        max_new_tokens: int = 4096,
        temperature: float = 0.6,  # valores recomendados pela DeepSeek para o R1
        top_p: float = 0.95,
        min_attempt_tokens: int = 32,  # tentativas menores não valem o corte
        max_rewinds: int = 8,  # evita loop infinito, principalmente no blind_rewind
        max_hypothesis_tokens: int = 8,
        max_reason_tokens: int = 16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = Mode(mode)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_attempt_tokens = min_attempt_tokens
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

    def build_anchor(self, attempt_text: str, reason: str) -> str:
        """Âncora de Falha = 1ª frase da tentativa (a hipótese) + frase do erro, sem as contas do meio."""
        hypothesis = first_sentence(attempt_text).replace('"', "'")
        return ANCHOR_TEMPLATE.format(
            hypothesis=self._truncate(hypothesis, self.max_hypothesis_tokens),
            reason=self._truncate(reason, self.max_reason_tokens),
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

        checkpoint = prompt_len  # L_0 da primeira tentativa: início do <think>
        peak = len(tokens)
        generated = detections = 0
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
            signal = detect_failure(attempt_text)
            if signal is None:
                continue

            detections += 1
            attempt_len = len(tokens) - checkpoint
            can_rewind = (
                self.mode != Mode.BASELINE
                and attempt_len >= self.min_attempt_tokens
                and len(rewinds) < self.max_rewinds
            )
            if not can_rewind:
                checkpoint = len(tokens)  # a próxima tentativa começa aqui
                continue

            anchor = self.build_anchor(attempt_text, signal.reason) if self.mode == Mode.REWIND_KV else ""
            anchor_ids = self.tokenizer(anchor, add_special_tokens=False).input_ids if anchor else []
            rewinds.append(
                RewindEvent(
                    generated_at=generated,
                    checkpoint=checkpoint,
                    discarded_tokens=attempt_len,
                    discarded_text=attempt_text,
                    reason=signal.reason,
                    anchor=anchor,
                    anchor_tokens=len(anchor_ids),
                )
            )
            logits = rewind_and_inject(self.model, cache, tokens, checkpoint, anchor_ids)
            checkpoint = len(tokens)  # a âncora fica "trancada" antes do novo L_0

        return GenerationResult(
            mode=self.mode.value,
            text=self.tokenizer.decode(tokens[prompt_len:], skip_special_tokens=True),
            prompt_tokens=prompt_len,
            final_cache_len=cache.get_seq_length(),
            peak_cache_len=peak,
            generated_tokens=generated,
            detections=detections,
            hit_token_limit=hit_limit,
            seconds=time.perf_counter() - t0,
            rewinds=rewinds,
        )
