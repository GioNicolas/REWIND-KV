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

## Validação da v0.2 — resultado (19/10/2026)

Rodou no Kaggle (2 × T4, 3h31min), sem erros, nos 98 problemas. A análise abaixo é exatamente a pré-registrada, recalculada localmente a partir dos `.jsonl` (bate com a saída do notebook). A geração foi idêntica ao baseline até o primeiro corte em todos os registros.

| | Acurácia (98 problemas) | Acurácia (32 com detecção) | Bateu no limite de 4096 (32 com detecção) |
|---|---|---|---|
| baseline | 61,2% | 31,2% | 72% |
| blind_rewind | 62,2% | 34,4% | 62% |
| **rewind_kv** | **58,2%** | **21,9%** | **75%** |

| Hipótese | Vitórias | Derrotas | p (sinal, bilateral) | Resultado |
|---|---|---|---|---|
| **H1:** rewind_kv > blind_rewind | 1 | 5 | 0,22 | **Não confirmada.** A direção é a **oposta** à prevista. |
| **H2:** rewind_kv > baseline | 1 | 4 | 0,38 | **Não confirmada.** Direção oposta. |
| (extra, não pré-registrada) blind_rewind vs baseline | 2 | 1 | 1,0 | sem diferença |

**Leitura:** a v0.2 **não** generalizou. Nos problemas novos, a âncora não ajudou e, na direção, atrapalhou. O 3 a 1 do A2 não se repetiu: era ajuste ao conjunto de desenvolvimento, como o pré-registro previa que poderia ser.

**Por que a âncora perdeu** (análise exploratória, feita depois de ver os dados). Li os 6 problemas discordantes do H1:

| Problema | Vencedor | O que foi cortado | O que a âncora fez |
|---|---|---|---|
| p30 | blind | no 3º corte, a conta certa (`f(3) = 8`); a frase do "erro" culpava a **outra** abordagem | *"It was wrong"* sobre o caminho certo; o modelo ficou alternando entre 8 e −4 até o limite |
| p36 | blind | a integral que confirmava a resposta certa (1/4); gatilho: *"let me just think about this again because sometimes … it's easy to make a mistake"* | o modelo passou a duvidar da área e estourou o limite |
| p62 | blind | o cálculo certo (`−I`) | âncora *"It was wrong: that contradicts"*; o modelo refez e não fechou |
| p80 | blind | `9 × 7 = 63` (certo); gatilho: a **pergunta** *"Is there any chance I made a mistake in the counting?"* | *"It was wrong"* sobre 63; o modelo foi atrás de outras contagens (35) e estourou o limite |
| p116 | blind | a descoberta de `n = 4` (certa) | âncora *"Wait, let me verify that." It was wrong*; o modelo abandonou o teste de valores e foi para logaritmos. O blind refez o teste e achou 4 |
| p108 | **rewind_kv** | uma conta de inclusão-exclusão errada (64 ≠ 60) | erro de verdade; a âncora funcionou como planejado |

Nos 5 casos em que o blind_rewind venceu, **o corte jogou fora um raciocínio certo** (falso positivo do detector). No único caso em que o rewind_kv venceu, o corte pegou um erro de verdade. O rewind_kv fez 52 cortes nos 32 problemas com detecção; nos 22 problemas com gabarito numérico ou fração simples (43 cortes), a resposta-gabarito aparecia no texto cortado em pelo menos 10. É uma contagem por busca de texto, que subestima os falsos positivos.

**Achados:**
1. **A âncora é uma amplificadora: ela piora os falsos positivos e só ajuda nos verdadeiros.** Depois de um corte errado, o blind_rewind só esquece, e o modelo muitas vezes **redescobre** a resposta certa (p30, p36, p80, p116). A âncora afirma *"It was wrong … I will not repeat it"* e **proíbe** essa redescoberta. É o achado 2 do A2 (p15), agora como padrão: 5 de 5 derrotas.
2. **O detector léxico tem precisão baixa nos problemas novos.** Os gatilhos dos falsos positivos são dúvidas e verificações, não declarações de erro: perguntas (*"Is there any chance I made a mistake?"*), avisos genéricos (*"sometimes … it's easy to make a mistake"*) e contradições entre duas abordagens em que a cortada era a certa (achado 5 do Experimento A).
3. **O teto continua sendo o limite de tokens.** 72% dos problemas com detecção batem no limite no baseline; dos erros com detecção, só 1 (p65) é uma resposta errada terminada, e foi igual nos 3 modos.
4. **O corte cego (blind_rewind) é neutro** (2 a 1 contra o baseline). Cortar sozinho não ajuda nem atrapalha; o efeito, para o bem e para o mal, vem da âncora.

### Precisão do detector: rotulagem dos cortes (20/10/2026, exploratória)

Para medir o tamanho do achado 2, cada um dos **73 cortes únicos** da validação foi rotulado (o 1º corte de cada problema é igual nos dois modos e conta uma vez). Os rótulos estão em [`dados/rotulos-cortes-validacao-v02.json`](dados/rotulos-cortes-validacao-v02.json).

| Rótulo | Definição | Todos (73) | Só o 1º corte (31) |
|---|---|---|---|
| **Erro real** | o que o modelo declarou errado está errado, e o corte não levou nada certo que importasse | 44 (60%) | 16 |
| **Real, mas levou algo certo** | erro real, mas o trecho cortado também tinha um resultado certo (às vezes a resposta) | 4 | 3 |
| **Cortou o lado errado** | contradição real, mas o trecho cortado era o lado certo ou a correção do erro | 10 | 3 |
| **Falso alarme** | nada estava errado | 15 | 9 |

**4 em cada 10 cortes jogaram fora algo certo** (29 de 73). No 1º corte, que não depende do modo, foram 15 de 31.

**De onde vêm os erros do detector:**
- **Falsos alarmes:** perguntas (*"Is there any chance I made a mistake?"*), avisos genéricos (*"it's easy to make a mistake"*), eliminação de candidatos (*"t = 1 doesn't work"*, cortado 3 vezes seguidas pelo blind_rewind no p63) e dúvidas sobre afirmações certas (as 8 retas do p35; "nenhum arranjo fica fixo numa rotação de 90°" no p93). Um detector que entenda contexto poderia evitar boa parte.
- **Lado errado:** em contradições ("2107, mas antes eu tinha 2037"), o corte sempre leva o resultado **mais recente**, que em 10 casos era o certo (6 deles no p37). A frase de erro não diz qual lado está errado, então **nenhuma lista de frases resolve isso**; é preciso outro sinal (verificar o valor, estados internos do modelo, ou esperar o modelo se decidir).

**Limites:** os rótulos são do assistente; o autor conferiu os 16 de confiança baixa ou média e concordou com todos, numa leitura rápida (~2 min), que não substitui uma rotulagem independente. A análise foi feita depois de ver os dados e não estava no pré-registro. Os 57 de confiança alta não foram conferidos por uma segunda pessoa.

### Detector estrito: primeira avaliação, sem GPU (20/10/2026)

Regras novas (`detect_failure(..., strict=True)`, em `detector.py`). Cada uma descarta um tipo de frase que, na rotulagem, não era declaração de erro:

| Regra | Exemplo descartado |
|---|---|
| **pergunta** | *"Is there any chance I made a mistake in the counting?"* |
| **aviso genérico** | *"sometimes … it's easy to make a mistake"*, *"before I conclude … any chance"* |
| **eliminação de candidato** | *"So, t=1 doesn't work."* |
| **ação, não declaração** | *"Let me recalculate D with another point."* |

Avaliação nos cortes rotulados (`python -m rewind_kv.detector_eval docs/dados/rotulos-cortes-validacao-v02.json`):

| | Cortes mantidos | Erros reais mantidos | Precisão |
|---|---|---|---|
| v0.2 | 73 | 44 de 44 | 60% |
| **Estrito** | 63 | **44 de 44** | **70%** |
| Estrito + regra de contradição | 47 | 33 de 44 | 70% |
| (só o 1º corte) v0.2 → estrito | 31 → 25 | 16 → 16 | 52% → 64% |

**Leitura:**
1. **As 4 regras tiram 8 falsos alarmes e 2 cortes do lado errado sem perder nenhum erro real.**
2. **A regra de contradição não funciona** e ficou desligada: ela pulava as frases que comparam dois resultados ("earlier", "somewhere", "I got"…), e isso tirava 11 erros reais para evitar 5 cortes do lado errado. As palavras são as mesmas nos dois casos; confirma que o "lado errado" pede outro sinal.
3. **Sobram 7 falsos alarmes e 8 cortes do lado errado** que nenhuma regra de palavras separa. Os falsos que sobram são dúvidas sobre afirmações certas (*"Wait, actually, no, that's not correct"* sobre as 8 retas certas do p35).

**Limites:** as regras foram escritas olhando esses mesmos 73 cortes, então os 70% são otimistas (é o erro medido nos dados de treino). É uma aproximação: se o detector pula uma frase, o texto seguinte mudaria, e os cortes seguintes também. O número que vale é o de um conjunto novo.

**Consequência:** a tese "a âncora evita a amnésia" (A2, achado 1) continua de pé no mecanismo, mas só vale quando a falha detectada é real. A pergunta de pesquisa passa a ser **a precisão do gatilho**: com o detector léxico atual, a âncora faz mais mal que bem. O argumento de acurácia sai do pitch até haver uma versão que vença em problemas novos.

---

## Validação da v0.3 — pré-registro (20/10/2026, escrito antes de rodar)

**Por que um conjunto novo:** a v0.3 foi desenhada olhando o nível 4 inteiro (problemas 0–29 no A/A2, 30–127 na rotulagem dos cortes). O nível 4 agora é desenvolvimento. O teste é no **nível 5**, que nunca foi rodado nem lido.

**O que muda em relação à v0.2** (o corte por caminho é o mesmo):
1. **Detector estrito** (`--strict-detector`): não corta em perguntas, avisos genéricos, eliminação de candidatos e "let me recalculate".
2. **Âncora de dúvida** (`--anchor-style doubt`): *Wait, earlier I tried "…" and then doubted it: … That doubt may or may not be right, so let me check it in a different way before relying on it.* Ela informa o que foi cortado sem afirmar que estava errado nem proibir o caminho, porque na v0.2 a certeza da âncora causou as 5 derrotas.

**Código congelado:** tag `validacao-v0.3`. Nada no detector, no corte ou na âncora muda até a análise terminar.

**Configuração:**
- MATH-500 nível 5, **todos os 134 problemas** (índices 0–133).
- Modos `baseline`, `rewind_kv` e `blind_rewind`, com `--rewind-scope branch --anchor-style doubt --strict-detector --max-new-tokens 4096`; semente = índice do problema; `--skip-untriggered`.
- Kaggle, 2 × T4, 67 problemas em cada GPU (`notebooks/kaggle_validacao_v03.ipynb`).

**Hipóteses** (nos problemas em que o baseline detectou pelo menos 1 falha com o detector estrito):
- **H1 (principal):** `rewind_kv` acerta mais que `blind_rewind`.
- **H2:** `rewind_kv` acerta mais que o `baseline`.

**Teste:** o mesmo da v0.2. Teste do sinal exato, bilateral, nos pares discordantes; significativo se p < 0,05. Nenhum problema é excluído; se a execução cair, retoma-se de onde parou.

**Secundárias (descritivas, sem teste):** tokens gerados, fração que bate o limite de 4096, número de rewinds, tamanho final do cache, e `blind_rewind` vs `baseline`.

**Análise exploratória planejada:** rotular os cortes do nível 5 com os mesmos 4 rótulos, para medir a precisão do detector estrito em dados novos e comparar com os 64–70% medidos no nível 4.

**Expectativa honesta:**
- O nível 5 é mais difícil, e a maior parte do baseline deve bater no limite de 4096 tokens. Isso deixa pouco espaço para qualquer modo melhorar.
- A âncora de dúvida pode acabar se comportando como o corte sem âncora. Nesse caso, ela deixaria de atrapalhar, mas o H1 seria nulo.
- Com dezenas de problemas com detecção e talvez 5 a 15 pares discordantes, só um efeito grande seria significativo. Um resultado nulo ou na direção oposta será reportado como tal.

## Validação da v0.3 — resultado (21/10/2026)

Rodou no Kaggle (2 × T4), sem erros, nos 134 problemas. A análise abaixo é exatamente a pré-registrada, recalculada localmente a partir dos `.jsonl` (bate com a saída do notebook). A geração foi idêntica ao baseline até o primeiro corte em todos os registros.

| | Acurácia (134 problemas) | Acurácia (40 com detecção) | Bateu no limite de 4096 (40 com detecção) |
|---|---|---|---|
| baseline | 41,8% | 20,0% | 85% |
| blind_rewind | 41,0% | 17,5% | 80% |
| **rewind_kv** | **41,0%** | **17,5%** | **80%** |

| Hipótese | Vitórias | Derrotas | p (sinal, bilateral) | Resultado |
|---|---|---|---|---|
| **H1:** rewind_kv > blind_rewind | 3 | 3 | 1,0 | **Não confirmada.** Empate. |
| **H2:** rewind_kv > baseline | 1 | 2 | 1,0 | **Não confirmada.** |
| (extra, não pré-registrada) blind_rewind vs baseline | 1 | 2 | 1,0 | sem diferença |

**Leitura:** a v0.3 **não** melhora a acurácia. A âncora de dúvida tirou o dano sistemático da v0.2 (de 1 a 5 para 3 a 3), como o pré-registro previa que poderia acontecer, mas não trouxe ganho.

**Os 6 problemas discordantes do H1** (exploratório, com os rótulos abaixo):

| Problema | Vencedor | O que aconteceu |
|---|---|---|
| p4 | rewind_kv | 1º corte falso (dúvida sobre a resposta certa 6 − 5i); depois dele, o rewind_kv terminou certo e o blind bateu no limite |
| p42 | rewind_kv | 1º corte falso (eliminação certa "x = −2 didn't work"); só o rewind_kv acertou, inclusive contra o baseline |
| p64 | rewind_kv | 1º corte real, mas levou junto o "perímetro + 2πr" certo; o blind fez mais um corte e bateu no limite |
| p53 | blind | depois da âncora, o rewind_kv cortou duas vezes a resposta certa 16/27 (dois falsos) |
| p65 | blind | 1º corte falso (igual nos dois); depois dele, só o rewind_kv errou |
| p94 | blind | o rewind_kv cortou a conta certa 150/1,32 = 113,6 (lado errado) |

As vitórias também vêm de cortes falsos: nenhuma delas é a âncora corrigindo um erro real. Com a âncora de dúvida, o que acontece depois do corte parece variar como uma nova amostragem, para os dois lados.

### Precisão do detector estrito em dados novos (rotulagem planejada no pré-registro)

Os **88 cortes únicos** foram rotulados com os mesmos 4 rótulos da v0.2 ([`dados/rotulos-cortes-validacao-v03.json`](dados/rotulos-cortes-validacao-v03.json)).

| Rótulo | Todos (88) | Só o 1º corte (38) |
|---|---|---|
| Erro real | 49 (56%) | 19 (50%) |
| Real, mas levou algo certo | 4 | 3 |
| Cortou o lado errado | 10 | 2 |
| Falso alarme | 25 | 14 |

| Precisão | Nível 4 (desenvolvimento) | **Nível 5 (novo)** |
|---|---|---|
| Detector da v0.2 | 60% | — |
| Detector estrito, estimado | 70% (1º corte: 64%) | — |
| **Detector estrito, medido** | — | **56% (1º corte: 50%)** |

**O detector estrito não generalizou.** Os 70% eram ajuste aos 73 cortes usados para escrever as regras, como o próprio limite da avaliação avisava. No nível 5, a precisão ficou abaixo até da do detector antigo no nível 4.

**De onde vêm os erros agora:**
- **Falsos alarmes (25):**
  - 10 são **eliminações de caso** escritas de outro jeito, que as regras não cobrem: *"1 ≠ 7, so it doesn't work"*, *"So, that's a contradiction"* no fim de uma prova por absurdo correta, *"that order doesn't work either"*.
  - Os outros 15 são **dúvidas sobre afirmações certas**, como o pé da altura (p28, duas vezes), o limite E → 2 (p44) e a resposta 16/27 (p53, duas vezes).
- **Lado errado (10):** além das contradições entre dois resultados, apareceu um tipo novo: **o corte leva a própria correção**. Em *"83 − 68 = 15, not 11. So, I think I made a mistake there"* (p115, nos dois modos), a frase de erro **é** a descoberta do erro, e o corte apaga a descoberta. No p126, os dois modos cortaram a resposta certa (1/201) e terminaram errado.

**Limites:** os rótulos são do assistente e ainda não foram conferidos pelo autor (32 de 88 têm confiança média ou baixa). Os 56 de confiança alta dão uma proporção parecida (32 reais, 17 falsos, 7 lado errado).

### Achados

1. **A âncora de dúvida resolve o dano, mas também perde o efeito.** Na v0.2, a âncora que proibia (*"It was wrong … I will not repeat it"*) amplificava os falsos positivos. A de dúvida (*"may or may not be right"*) não amplifica, mas também não impede a repetição: o rewind_kv refez o mesmo erro mesmo com a âncora: três vezes no p100 (r = 7) e quatro no p110 (a fórmula errada do ortocentro). Já o blind_rewind repetiu o mesmo 7,5 seis vezes no p88, e o rewind_kv só uma vez. **A âncora que impede repetir é a mesma que impede redescobrir o certo.** Qual das duas ganha depende só da precisão do detector.
2. **O detector léxico é o gargalo, e regras de frase não generalizam.** Cerca de metade dos cortes em dados novos joga fora algo certo ou não tinha erro. Cada regra nova tapa um jeito de escrever e o modelo usa outro.
3. **No nível 5, o teto é o limite de tokens.** 85% dos problemas com detecção batem no limite de 4096 no baseline. Dos 78 erros do baseline nos 134 problemas, 71 são por limite. Os cortes economizam cerca de 10% do cache final nos problemas com detecção (3.638 vs 4.060 tokens), pouco para mudar quem termina.
4. **Detalhe de implementação:** em 2 problemas (p52, p57) houve detecção, mas nenhum corte, porque o corte teria menos de 32 tokens (`min_cut_tokens`). Nesses casos, os três modos foram iguais.

**Consequência:** duas validações pré-registradas seguidas sem efeito na acurácia. Com um detector léxico, a âncora não tem como ajudar mais do que atrapalha. O que poderia mudar o quadro, a decidir:
- um sinal que não seja de palavras para decidir se o corte é real, como verificar o valor cortado ou usar uma sonda nos estados internos;
- um orçamento maior de tokens, ou um conjunto em que os erros terminem antes do limite;
- registrar o resultado negativo como contribuição: o rollback com âncora funciona como mecanismo, mas o gatilho léxico tem precisão de cerca de 50 a 60% em dados novos.

## Sonda nos estados internos: teste de viabilidade — pré-registro (21/10/2026, escrito antes de rodar)

**Pergunta:** o estado interno do modelo no momento da detecção separa um **erro real** de um **falso alarme / lado errado** melhor do que as palavras? Se não separar, o caminho da sonda morre aqui e o resultado negativo vira a contribuição.

**Dados (só os que já existem, sem gerar nada novo):**
- **Treino:** os 73 cortes rotulados do nível 4 (validação da v0.2).
- **Teste:** os 88 cortes rotulados do nível 5 (validação da v0.3), usados **uma vez só**, no fim.
- **Alvo:** `real` = 1; `falso` e `lado_errado` = 0. Os `real_colateral` (4 + 4) ficam de fora. Ficam 44 contra 25 no treino e 49 contra 35 no teste.
- **Contexto de cada corte:** prompt + texto que o modelo tinha naquele momento (com as âncoras anteriores) até o fim do texto cortado, que é o instante em que o detector disparou. O texto é retokenizado, então pode haver pequenas diferenças em relação aos tokens amostrados.

**Atributos:**
- **Estado oculto** do modelo (`hidden_states`), em duas versões: no último token e na média dos tokens da frase de erro.
- **Entropia** da próxima palavra, na média da frase de erro. É o sinal mais simples de "incerteza", usado como comparação.

**Classificador:** regressão logística com regularização L2, atributos padronizados.

**Escolha do modelo (só no nível 4):**
- Validação cruzada deixando um problema de fora por vez.
- Grade: camada ∈ {7, 14, 21, 28} × posição ∈ {último token, média da frase} × C ∈ {0,01; 0,1; 1}.
- Vence a configuração de maior AUC.
- **Limiar:** o maior que mantém ≥ 80% dos erros reais nas previsões da validação cruzada.
- Depois, o classificador é treinado de novo nos 69 cortes do nível 4 e aplicado ao nível 5 sem nenhum ajuste.

**Critério de sucesso (no nível 5):** **precisão ≥ 75% e recall ≥ 70%** (fração dos erros reais que continuam sendo cortados). Para comparação, cortar tudo dá precisão de 58% (49 de 84) e recall de 100%. Se só um dos dois limites for atingido, o resultado conta como fracasso.

**Relatado também, sem critério:**
- AUC no nível 5, com intervalo de bootstrap.
- O mesmo procedimento usando só a entropia.
- O AUC da validação cruzada com rótulos embaralhados, como controle.

**Limites conhecidos:**
- São 69 exemplos de treino em 1.536 dimensões. O teste diz se existe sinal, não o tamanho final do efeito.
- Os rótulos são do assistente, e os do nível 5 não foram conferidos pelo autor.
- Os dois níveis já foram vistos na rotulagem, mas a sonda nunca viu o nível 5.

## Sonda nos estados internos — resultado (21/10/2026)

Rodou no Colab (T4) com o código da tag `sonda-v1` (`notebooks/colab_sonda.ipynb`), exatamente como pré-registrado.

**Escolha no nível 4:**
- Venceu a **camada 21, último token, C = 0,1**, com AUC 0,692 na validação cruzada.
- As médias da frase de erro ficaram entre 0,46 e 0,58 em todas as camadas.
- O último token das camadas 21 e 28 ficou entre 0,67 e 0,69.

| Nível 5 (teste, uma vez só) | Precisão | Recall | AUC (IC 95%) |
|---|---|---|---|
| Cortar tudo (sem filtro) | 58,3% (49 de 84) | 100% | — |
| **Sonda** (limiar 0,496, mantém 61 de 84) | **63,9%** | 79,6% | **0,605** (0,483–0,721) |
| Só a entropia da frase de erro | 56,5% | 79,6% | 0,539 (0,410–0,670) |
| **Critério pré-registrado** | **≥ 75%** | ≥ 70% | |

**Resultado: critério NÃO atingido.**
- A sonda tirou 22 cortes, mas só 12 deles eram ruins; os outros 10 eram erros reais.
- O intervalo do AUC inclui 0,5.
- **Controle:** com os rótulos do nível 4 embaralhados, o AUC médio foi 0,476, mas o maior dos 20 sorteios chegou a 0,662. O 0,692 escolhido mal se distingue de sorte, já que a escolha foi feita entre 24 configurações com 69 exemplos.
- A entropia, que é o sinal de "incerteza" mais simples, não separa nada.

**Leitura:** com estes dados, o estado interno no instante da detecção **não** separa um erro real de um falso alarme ou corte do lado errado bem o bastante para filtrar os cortes.

**Limites:**
- **Poucos dados:** 69 exemplos de treino.
- **Rótulos de um só rotulador:** o assistente.
- **A pergunta é mais difícil que a dos trabalhos vizinhos.** Eles preveem se uma resposta está certa (Zhang et al., 2025, arXiv:2504.05419; ver também arXiv:2605.09502). Aqui, a sonda precisaria saber, por exemplo, qual de duas contas anteriores estava certa.

Uma sonda com muito mais dados rotulados talvez funcione, mas isso seria outro projeto, não um ajuste deste. Como combinado antes do teste, o resultado negativo passa a ser a contribuição.

## Teste do oráculo: e se o gatilho fosse perfeito? (21/10/2026, exploratório)

Ideia sugerida numa revisão da literatura feita com o Gemini: antes de buscar um gatilho melhor, ver o que acontece nos problemas em que o gatilho atual **já acertou**. São os problemas em que todos os cortes do rewind_kv foram erro real, segundo os rótulos.

| | Problemas | Acertos baseline / blind / rewind_kv | Baseline no limite de 4096 | rewind_kv vs baseline |
|---|---|---|---|---|
| Nível 4 (v0.2) | 14 | 3 / 3 / 4 | 12 | 1–0 |
| Nível 5 (v0.3) | 14 | 1 / 1 / 1 | 13 | 0–0 |
| (só o 1º corte real, nível 4) | 16 | 4 / 4 / 4 | 13 | 1–1 |
| (só o 1º corte real, nível 5) | 19 | 2 / 3 / 1 | 18 | 0–1 |

**Leitura:** mesmo quando o corte pega um erro de verdade, a acurácia praticamente não muda. Quase todos esses problemas batem no limite de tokens, nos três modos. **Um gatilho melhor não teria salvado estas validações:** o gargalo aqui é o modelo de 1,5B não conseguir terminar os problemas difíceis, não o rascunho errado no cache. São poucos problemas (28), então isso é uma indicação, não uma prova.

**Detalhe de desenho que pesa nisso:** o limite de 4096 conta todos os tokens gerados, inclusive os que depois foram cortados (`generated` em `controller.py` nunca diminui). O corte libera memória, mas **não devolve orçamento de tokens**. Com orçamento fixo, o rewind só pode ajudar se o texto que vem depois do corte for melhor, e não por dar mais espaço para pensar.

---

## Candidatos para a v0.3 (anotados durante a validação; não mudam o código congelado)

1. **Falsos positivos amplificados pela âncora (p15 do A2):** uma dúvida sobre uma conta certa vira *"It was wrong"* na âncora, e o modelo passa a acreditar no erro.
2. **Rewind reversível** (ideia vinda do "zoom" do UniiChat; ver o [Mapa da Literatura](mapa-da-literatura.md), seção 6): guardar o trecho cortado fora da GPU e restaurá-lo se o modelo depois contradisser a âncora. Seria uma resposta direta ao item 1.
3. **Depois da validação:** o item 1 virou o problema principal (5 de 5 derrotas do H1). Qualquer v0.3 precisa ser desenhada e testada nos problemas 0–127 já vistos como **desenvolvimento**, e validada em problemas novos (outro nível do MATH-500 ou outro benchmark), nunca nos mesmos 30–127.
