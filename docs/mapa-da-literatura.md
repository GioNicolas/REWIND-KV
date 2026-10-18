# Mapa da Literatura — RewindKV (levantamento de 18/10/2026)

Levantamento feito depois dos dois pilotos, para decidir os caminhos **A** (cortar caminhos errados → acurácia) e **B** (cortar verificações repetitivas → memória).

> **Limite deste levantamento:** a maioria dos trabalhos foi lida só pelo resumo (*abstract*). Só o LPSR teve a seção de método conferida. Antes de submeter um artigo, ler na íntegra pelo menos LPSR, Thought Rollback, Structured Thoughts e DEER.

---

## 1. Veredito

| Caminho | Situação na literatura | Decisão |
|---|---|---|
| **A** — cortar o ramo que falhou + Âncora de Falha | **Ainda há espaço**, mas há vizinhos próximos que precisam ser citados e diferenciados (LPSR, Thought Rollback, Structured Thoughts). | ✅ Seguir. |
| **B** — cortar verificações repetitivas depois da resposta | **Área muito ocupada.** *Early exit* (DEER, Dynasor, NEAT, PUMA) e supressão de verificação (Self-Verification Dilemma) já atacam isso, e parar a geração é mais simples que cortar e continuar. | ⚠️ Não seguir como foi pensado. Ver o ângulo alternativo na seção 4. |

---

## 2. Vizinhos diretos do caminho A (rollback durante a geração)

| Trabalho | O que faz | Diferença para o RewindKV |
|---|---|---|
| **LPSR — Latent Phase-Shift Rollback** (Gupta & Kumar, abr/2026, [arXiv:2604.18567](https://arxiv.org/abs/2604.18567)) | *Training-free.* Monitora o *residual stream*, detecta "mudanças de fase" e volta **1 token** no KV-Cache, injetando um vetor de direção (*steering*). MATH-500: 28,8% → 44,0% com Llama-3-8B-Instruct. | Volta só 1 token (a ablação deles mostra que 2–3 tokens pioram); a correção é latente, sem texto; não testa modelos de raciocínio com `<think>`; não fala de memória (custa ~3× mais tokens). O RewindKV corta **ramos inteiros**, deixa uma **âncora em texto**, mira modelos R1 e **economiza** cache. ⚠️ É o concorrente mais próximo da **v0.2** (*latent probe*). |
| **Thought Rollback** (Chen & Li, ICML 2024, [arXiv:2412.19707](https://arxiv.org/abs/2412.19707)) | Por *prompting*, o modelo analisa a própria cadeia, volta para antes do primeiro passo errado e **adiciona a tentativa falha ao prompt como "experiência"**. GPT-4, MATH +9%. | **É a ideia da Âncora de Falha no nível de prompt**, com várias chamadas ao modelo. O RewindKV faz o equivalente **dentro de uma única geração**, no nível do KV-Cache, usando os marcadores que o próprio modelo emite. Precisa ser citado como antecessor conceitual. |
| **Aborted but Not Forgotten** (Zhang & Yang, ago/2026, [arXiv:2608.15939](https://arxiv.org/abs/2608.15939)) | Mostra que ramos abortados que continuam no KV-Cache seguem influenciando o modelo (7 famílias de modelos; 25 de 63 casos). | É a **motivação** do RewindKV, no contexto de segurança de agentes. Não propõe um algoritmo de raciocínio. |
| **ArborKV** (Chen et al., mai/2026, [arXiv:2605.22106](https://arxiv.org/abs/2605.22106)) | Gerencia o KV-Cache em buscas em árvore, mantendo o caminho ativo. | Exige orquestração em árvore externa; o RewindKV atua no fluxo linear nativo do R1. |

## 3. "Podar o rascunho e manter um resumo" — todos exigem treino

| Trabalho | Mecanismo | Treino? |
|---|---|---|
| **Structured Thoughts** (jul/2026, [arXiv:2607.10386](https://arxiv.org/abs/2607.10386)) | Blocos *try*/*outcome*; o *try* é podado depois de cada passo e o *outcome* fica. ~85% de economia de memória, com −8,67% de desempenho. | Sim (*fine-tuning*). O *outcome* é análogo à nossa âncora, mas para **todo** passo, não só para falhas. |
| **InftyThink / InftyThink+** ([arXiv:2503.06692](https://arxiv.org/abs/2503.06692); ICML 2026) | Gera, resume, descarta o raciocínio anterior e mantém só o último resumo. | Sim (SFT + RL). |
| **LightThinker** (EMNLP 2025, [arXiv:2502.15589](https://arxiv.org/abs/2502.15589)) | Comprime pensamentos em *gist tokens*. | Sim. |
| **PENCIL** ([arXiv:2503.14337](https://arxiv.org/abs/2503.14337)) | Tokens especiais disparam regras de "redução" que apagam pensamentos intermediários. | Sim (modelos pequenos, quebra-cabeças). |
| **Neural Garbage Collection** (Stanford, NeurIPS 2026, [arXiv:2604.18002](https://arxiv.org/abs/2604.18002)) | O modelo aprende por RL o que despejar do cache. Compressão de 2–3×. | Sim (RL). |
| **Free()LM** ([arXiv:2602.08030](https://arxiv.org/abs/2602.08030)) | Auto-esquecimento com adaptador LoRA. | Sim. |

**Diferencial do RewindKV aqui:** é *training-free* e só age quando o próprio modelo declara uma falha. É um gatilho semântico, não um orçamento fixo de memória.

## 4. O caminho B e por que ele está ocupado

**Early exit (parar quando a resposta estabiliza):**
- **DEER** ([arXiv:2504.15895](https://arxiv.org/abs/2504.15895)): *training-free*, testado em R1-Distill-Qwen, raciocínio 31–43% menor e acurácia +1,7 a +5,7 pp. **Observação importante do próprio DEER:** o modelo pode ficar preso num **laço de verificação quando dois métodos discordam**.
- **Dynasor**, **Answer Convergence** ([arXiv:2506.02536](https://arxiv.org/abs/2506.02536)), **NEAT** ([arXiv:2602.02010](https://arxiv.org/abs/2602.02010)), **PUMA** ([arXiv:2605.17672](https://arxiv.org/abs/2605.17672)).

**Suprimir verificações:**
- **Self-Verification Dilemma** ([arXiv:2602.03485](https://arxiv.org/abs/2602.03485)): a maioria das re-verificações não muda o resultado; suprime as desnecessárias (−20,3% de tokens).
- **ReCo** ([arXiv:2608.04771](https://arxiv.org/abs/2608.04771)): penaliza reflexão e para cedo quando há confiança.

**Compressão de KV-Cache para raciocínio** (seleção token a token, *training-free*): **R-KV** ([arXiv:2505.24133](https://arxiv.org/abs/2505.24133)), **InfoKV** ([arXiv:2606.26875](https://arxiv.org/abs/2606.26875)), **VaSE** ([arXiv:2606.03928](https://arxiv.org/abs/2606.03928)), **Random Attention** ([arXiv:2609.03430](https://arxiv.org/abs/2609.03430)), **KV-Rescue** ([arXiv:2608.15797](https://arxiv.org/abs/2608.15797)), **AvoKV-E** ([arXiv:2610.03007](https://arxiv.org/abs/2610.03007)).

**Ângulo alternativo para B (hipótese, ainda não testada):** o *early exit* só dispara quando a resposta **estabiliza**. Ele não ajuda quando o modelo fica preso entre duas respostas conflitantes, exatamente o laço que o DEER relata e que vimos no piloto 2 (problema 3: 720 vs 240, 88% do raciocínio gasto nisso, sem terminar). Remover do cache o ramo que gerou a resposta errada pode **destravar** esse laço. Isso é território do RewindKV (gatilho `contradicts`) e seria **complementar** ao *early exit*, não concorrente.

## 5. Posicionamento atualizado

> **RewindKV:** rollback *training-free*, durante a geração, de **ramos inteiros** do KV-Cache em modelos de raciocínio, disparado pelas **declarações de falha do próprio modelo**, com uma **Âncora de Falha em texto** no lugar do ramo removido.

| Eixo | RewindKV | Mais próximo |
|---|---|---|
| Granularidade do corte | ramo / parágrafo | LPSR: 1 token |
| O que fica no lugar | âncora em texto, legível | LPSR: vetor latente; Thought Rollback: texto, mas no prompt |
| Precisa de treino? | não | Structured Thoughts, InftyThink, LightThinker, NGC: sim |
| Chamadas ao modelo | 1 geração contínua | Thought Rollback: várias |
| Modelos-alvo | raciocínio com `<think>` (R1) | LPSR: Llama-3-8B-Instruct |

**Implicações para o artigo:**
1. Trocar "o primeiro algoritmo que…" por "até onde verificamos…", citando LPSR e Thought Rollback.
2. **Baselines que um revisor vai pedir:** *Blind Rewind* (já temos), DEER (*early exit*) e, se possível, uma variante estilo Thought Rollback.
3. Testar a combinação **RewindKV + early exit**: um corta os ramos que falharam, o outro para quando a resposta estabiliza.
