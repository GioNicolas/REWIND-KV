# Resultados dos Experimentos — RewindKV

Diário dos experimentos: o que rodou, o que deu e o que mudou por causa disso.
Modelo: `DeepSeek-R1-Distill-Qwen-1.5B` (FP16, Colab T4), temperatura 0,6, top-p 0,95, mesma semente por problema em todos os modos.

---

## Experimento A — v0.1 (18/10/2026)

**Configuração:** MATH-500 nível 4, problemas 0–29, limite de 4096 tokens, modos `baseline` e `rewind_kv`, corte por parágrafo (`--rewind-scope paragraph`), âncora `[Note: ...]` (`--anchor-style note`).

| | baseline | rewind_kv |
|---|---|---|
| Acertos (30 problemas) | 17 (56,7%) | 18 (60,0%) |
| Acertos (10 problemas com detecção no baseline) | 4 | 5 |
| Respostas certas que viraram erradas | — | 0 |
| Redução do cache final | — | 1,0% |
| Tokens cortados / tokens de âncora (média por rewind) | — | 66 / 35 |

**Leitura:** efeito nulo. Só 1 problema mudou (p0, errado → certo); com 1 par discordante, McNemar não distingue de acaso.

**Achados:**
1. **Todos os 13 erros do baseline foram por limite de tokens**, nenhum por resposta errada. A 4096 tokens, "acertar" é na prática "terminar a tempo".
2. **7 dos 13 erros não tinham nenhuma declaração de erro.** O RewindKV só pode agir em cerca de 1/3 dos problemas.
3. **O modelo refazia a conta cortada logo depois da âncora** (p3, p6, p13, p24 e p0, lidos à mão). Cortar só o parágrafo do "errei" deixava as premissas do caminho errado no cache:
   - p3: o caminho errado ("fixar a posição do Pierre") começava parágrafos antes; o corte levou só "2 × 120 = 240" e o modelo chegou em 240 de novo.
   - p6: depois da âncora, o modelo escreveu *"Same result again. x=6"*.
   - p22: o modelo disse *"in the earlier step I made a mistake"* e o corte pegou o passo atual.
4. **A âncora `[Note: ...]` era vaga e ignorada.** A "hipótese" era a 1ª frase do trecho cortado, muitas vezes sem conteúdo (`"Wait, hold on, no."`, `"0 1 0 | 1 0 1..."`).
5. **Em contradições, não se sabe qual lado está errado.** "Isso contradiz meu 720" não diz se o erro é o 720 ou o 240; o RewindKV sempre corta o mais recente.
6. **O ganho do p0 ilustra a tese:** no baseline, os vetores errados `(2, 3, -1)` ficaram no contexto e o modelo voltou a eles até estourar o limite; com o rewind, ele os repetiu uma vez, depois se livrou e terminou em 90°.
7. **A economia de memória não se sustenta neste modelo:** os cortes são curtos (66 tokens) e a âncora ocupa metade disso.

**Consequência:** o argumento "muito menos memória" sai do pitch; o foco passa a ser **destravar o modelo quando ele empaca num caminho que já declarou errado**.

---

## v0.2 — mudanças motivadas pelo Experimento A

| Mudança | Motivo (achado) | Flag |
|---|---|---|
| **Corte por caminho:** volta até a frase que abriu o caminho que falhou ("Alternatively, ...", "Let me try ...", "Suppose ..."); sem ela, corta o passo errado; caminho maior que 1024 tokens também corta só o passo | 3 | `--rewind-scope branch` (padrão) |
| **Âncora na voz do modelo:** *Wait, I already tried this before: "…" It was wrong: … I will not repeat it, so let me do it differently.* | 4 | `--anchor-style voice` (padrão) |
| **Hipótese = 1ª frase com conteúdo antes da frase do erro** | 4 | — |
| `resume_char` em cada rewind, para medir se o modelo refaz o trecho cortado | 3 | — |

Simulando o corte nos textos do Experimento A: no p3, o corte passa de 33 caracteres ("2 × 120 = 240") para 1274, começando em *"Alternatively, if I didn't treat them as a single block…"*.

## Experimento A2 — v0.2, conjunto de desenvolvimento (19/10/2026)

**Configuração:** os 10 problemas com detecção no baseline do Experimento A (0, 3, 5, 6, 13, 15, 17, 22, 24, 26), modos `rewind_kv` e `blind_rewind` com a v0.2. O baseline é reaproveitado do Experimento A. A geração foi idêntica ao baseline até o primeiro corte em todos os 20 registros (`prefixo_ok`), então a comparação é pareada.

> **Cuidado:** a v0.2 foi desenhada olhando esses 10 problemas. Os resultados abaixo **não valem como evidência**; só justificam validar em problemas novos.

| | Acertos (de 10) | Ganhou do baseline | Perdeu do baseline |
|---|---|---|---|
| baseline | 4 | — | — |
| rewind_kv v0.1 | 5 | p0 | — |
| **rewind_kv v0.2** | **5** | p0, p6 | p15 |
| blind_rewind v0.2 | 3 | — | p5 |

rewind_kv vs blind_rewind nos problemas que mudaram: 3 a 1 (p0, p5 e p6 contra p15). Não é significativo (teste do sinal, p ≈ 0,63).

**O fato errado volta depois do corte?** Para 6 problemas, anotei à mão o fato que o modelo declarou errado (`(2,3,-1)` no p0, `240` no p3, `-12.5` no p5, `x = 6` no p6, a equação do p22, `A√B = 0` no p24) e procurei esse fato nos 1500 caracteres seguintes a cada corte que o continha:

| | Fato errado reapareceu |
|---|---|
| blind_rewind v0.2 | 9 / 9 cortes |
| rewind_kv v0.1 (Experimento A) | 6 / 8 |
| rewind_kv v0.2 | 4 / 8 |

Dos 4 casos da v0.2, só 1 é o modelo **adotando de novo** o erro (p24). Nos outros, ele cita o fato **para rejeitá-lo**: no p5, *"Maybe I thought the term was -12.5, but that was a miscalculation"*; no p3, *"why is the other approach giving me 240? Because … I fixed Pierre's seat"*. Nos casos de blind_rewind que li, o modelo adotava o erro de novo; os 9 não foram todos conferidos à mão.

Uma medida automática por sobreposição de 4-gramas **não** separou os modos (rewind_kv 0,22, blind 0,18). Ela conta como repetição a reescrita de coisas certas que estavam no trecho cortado e não pega repetições parafraseadas; por isso não foi usada.

**Achados:**
1. **A amnésia existe:** sem âncora, o modelo volta ao mesmo fato errado em todos os cortes. Esta é a primeira evidência, ainda pequena, para a contribuição central (âncora > corte cego).
2. **A âncora amplifica falsos positivos (p15):** com 11/36 já calculado, o modelo duvidou de uma conta certa ("6 + 5 = 11? that doesn't make sense"). A âncora afirmou *"It was wrong"*, o modelo passou a acreditar em 10 resultados e estourou o limite; o blind_rewind acertou. O corte cego só faz esquecer, a âncora **ensina** o erro.
3. **Quando o caminho é inevitável, a âncora não basta (p24):** o enunciado obriga a forma `A(1+√B)+…`, então "fazer diferente" não é possível e o modelo refaz a mesma conta.
4. **Acerto com 4096 tokens (p0) é válido:** o modelo escreveu "**Final Answer** … \boxed{90}" e fechou o `</think>`; o limite acabou no resumo final. Nenhum dos 43 acertos do A e do A2 veio de um `\boxed` no meio do raciocínio sem conclusão.

---

## Validação da v0.2 — pré-registro (19/10/2026, escrito antes de rodar)

**Por que pré-registrar:** a v0.2 foi desenhada olhando os problemas 0–29. Para o resultado valer, o teste tem que ser em problemas novos, com as hipóteses e a análise fixadas **antes** de ver os dados. Nada abaixo será mudado depois de ver o resultado.

**Código congelado:** tag `validacao-v0.2`. Nenhuma mudança no detector, no corte ou na âncora até a análise terminar, inclusive a correção do falso positivo do p15 (fica para a v0.3).

**Configuração:**
- MATH-500 nível 4, **problemas 30–127** (98 problemas; nenhum foi lido durante o desenho da v0.2).
- Modos `baseline`, `rewind_kv` e `blind_rewind`, com `--rewind-scope branch --anchor-style voice --max-new-tokens 4096`; semente = índice do problema.
- `--skip-untriggered`: quando o baseline não detecta nenhuma falha, o resultado é copiado para os outros dois modos (a geração seria idêntica, como mostraram o Experimento A e o `prefixo_ok` do A2).
- Kaggle, 2 × T4, metade dos problemas em cada GPU (`notebooks/kaggle_validacao_v02.ipynb`).

**Hipóteses** (nos problemas em que o baseline detectou pelo menos 1 falha, os únicos em que os modos podem diferir):
- **H1 (principal):** `rewind_kv` acerta mais que `blind_rewind`. É a contribuição central: a âncora evita a amnésia.
- **H2:** `rewind_kv` acerta mais que o `baseline`.

**Teste:** teste do sinal exato, bilateral, nos pares em que só um dos dois modos acertou; significativo se p < 0,05. Nenhum problema é excluído. Se a execução cair, retoma-se de onde parou e todos os 98 entram na análise.

**Secundárias (descritivas, sem teste):** tokens gerados, fração que bate o limite de 4096, número de rewinds, tamanho final do cache.

**Expectativa honesta:** no Experimento A, ~1/3 dos problemas teve detecção; aqui devem ser ~30, com talvez 5 a 10 pares discordantes. Com isso, só um efeito grande (por exemplo, 8 a 1) seria significativo. Um resultado na direção certa mas sem significância será reportado como tal, e não como confirmação.
