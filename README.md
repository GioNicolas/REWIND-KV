# RewindKV: Rollback Cirúrgico de KV-Cache em Modelos de Raciocínio

**Nome oficial do algoritmo/pesquisa:** **RewindKV** (*In-Stream KV-Cache Rollback with Failure Anchors for Reasoning Models*)  
**Autor:** Giorgio Nicolas de A. Bertino  
**Data de início:** 18 de outubro de 2026  
**Status:** Projeto de pesquisa em fase de implementação do protótipo experimental.

---

## Documentos deste Repositório

- 📘 **[Guia de Estudo, Explicação Simples e Pitch (Caso Alguém Pergunte)](docs/guia-de-estudo-e-pitch.md):** documento feito para estudar a pesquisa do zero, entender como funciona sem jargão e responder perguntas de recrutadores, professores ou bancas técnicas.
- 🗺️ **[Mapa da Literatura (18/10/2026)](docs/mapa-da-literatura.md):** vizinhos diretos (LPSR, Thought Rollback, Structured Thoughts), por que o caminho "cortar verificações" está ocupado e como o RewindKV se diferencia.
- 🧪 **[Resultados dos Experimentos](docs/resultados-experimentos.md):** o que cada experimento mostrou (Experimento A: efeito nulo e por quê), as mudanças da v0.2 e a validação pré-registrada (H1 não confirmada: a âncora amplifica os falsos positivos do detector).
- ⚙️ **[Especificação Técnica e Matemática](docs/especificacao-tecnica.md):** equações, funcionamento do `DynamicCache.crop()`, alinhamento de posição RoPE, estrutura de código e protocolo experimental.

---

## 1. Resumo Executivo (O que é o RewindKV e por que importa)

Modelos modernos de raciocínio (como *DeepSeek-R1*, *Qwen-QwQ*, *OpenAI o1/o3*) resolvem problemas complexos de matemática, lógica e programação gerando longas cadeias de pensamento (`<think>...</think>`). Durante esse processo, o modelo frequentemente explora uma hipótese, percebe que ela levou a um erro ou contradição (*"Wait, that's wrong, let me recalculate..."*) e inicia uma nova tentativa.

**O defeito estrutural atual:**  
Como a geração nos Transformers é estritamente aditiva (*append-only*), **todos os centenas ou milhares de tokens do rascunho errado permanecem gravados na memória ativa da placa de vídeo (o KV-Cache)** até o fim da geração. Isso causa dois problemas graves:
1. **Explosão de Memória e Latência:** O KV-Cache acumula rascunhos mortos, devorando memória VRAM e deixando cada palavra nova mais lenta.
2. **Contaminação Causal de Atenção (*Context Poisoning*):** Enquanto tenta calcular o caminho certo na segunda tentativa, o mecanismo de atenção da IA continua esbarrando nos números e passos errados da primeira tentativa que ficaram no KV-Cache, induzindo novos erros.

**A Solução RewindKV:**  
Um algoritmo de intervenção em tempo de inferência (*training-free* na Fase 1, e guiado por um detector discriminativo ultraleve na Fase 2) que:
1. Marca **Checkpoints Lógicos ($L_0$)** no KV-Cache no início de cada hipótese/passo do raciocínio;
2. Detecta o instante em que a tentativa atual falha ou entra em contradição;
3. **Dá um "Rewind" (Rebobina fisicamente o KV-Cache via `DynamicCache.crop(L_0)`)**, apagando da memória da GPU todos os tokens do cálculo errado em frações de milissegundo;
4. Injeta no lugar apenas uma **Âncora de Falha curta (10 a 20 tokens)** resumindo o motivo do erro (ex.: `[Nota: A tentativa X falhou porque violou Y. Não repetir.]`), impedindo que o modelo sofra de amnésia e repita o mesmo erro.

---

## 2. Posicionamento Científico e Literatura de Ponta (2025–2026)

*Nota: Verificamos no arXiv em 18/10/2026 que o nome **RewindKV** é 100% inédito na literatura (diferente de `AnchorKV`, que já foi usado para quantização/compressão estática em [arXiv:2608.02901](https://arxiv.org/abs/2608.02901)).*

| Artigo Recente (Verificado) | O que descobriu / fez | O que deixou em aberto (A Lacuna do RewindKV) |
| :--- | :--- | :--- |
| **R-KV** — Cai et al., NeurIPS 2025 ([arXiv:2505.24133](https://arxiv.org/abs/2505.24133)) | Comprime o KV-Cache de modelos de raciocínio removendo tokens redundantes baseados em atenção. | Apenas poda palavras soltas; **não faz rollback estrutural** de uma tentativa inteira que deu errado nem cria âncoras de erro. |
| **InfoKV** — Kai et al., jun/2026 ([arXiv:2606.26875](https://arxiv.org/abs/2606.26875)) | Usa incerteza preditiva (*Forward Influence*) para comprimir KV-Cache em raciocínio longo. | Mantém a estrutura linear de erros; avaliado apenas em passagem contínua sem *backtracking* físico do cache. |
| **Models Take Notes at Prefill** — Bojie Li, jun/2026 ([arXiv:2606.17107](https://arxiv.org/abs/2606.17107)) | Provou que os vetores KV funcionam como "notas de rascunho" editáveis e componíveis com reposicionamento RoPE. | Focado na fase de *prefill* (leitura do prompt), não na auto-limpeza dinâmica durante a geração do `<think>`. |
| **ArborKV** — Chen et al., mai/2026 ([arXiv:2605.22106](https://arxiv.org/abs/2605.22106)) | Gerencia KV-Cache em buscas em árvore externas (*Tree-of-Thoughts*). | Exige um orquestrador externo gerando múltiplas árvores paralelas caras; não opera *in-stream* dentro do fluxo nativo de um modelo R1. |
| **LPSR — Latent Phase-Shift Rollback** — Gupta & Kumar, abr/2026 ([arXiv:2604.18567](https://arxiv.org/abs/2604.18567)) | Rollback *training-free* do KV-Cache durante a geração, disparado por um sinal no *residual stream*, com correção por vetor latente (MATH-500: 28,8% → 44,0%). | Volta só **1 token**, sem âncora em texto, não testa modelos de raciocínio (`<think>`) e não economiza memória (custa ~3× mais tokens). |
| **Thought Rollback** — Chen & Li, ICML 2024 ([arXiv:2412.19707](https://arxiv.org/abs/2412.19707)) | Volta para antes do passo errado e adiciona a tentativa falha ao prompt como "experiência". | É a ideia da âncora no nível de *prompt*, com várias chamadas ao modelo; não opera no KV-Cache dentro de uma única geração. |
| **Aborted but Not Forgotten** — Zhang & Yang, ago/2026 ([arXiv:2608.15939](https://arxiv.org/abs/2608.15939)) | **Prova a nossa motivação:** mostrou que ramos abortados que continuam retidos no KV-Cache contaminam a atenção futura do modelo (*rollback inconsistency*). | Diagnosticou o problema de segurança/consistência em agentes, mas **não propôs o algoritmo de poda de raciocínio com substituição por Âncora de Falha**. |

---

## 3. POR ONDE COMEÇAR (Roteiro Prático para o "Dia 1")

### Recursos e Ambiente
* **Máquina local:** Windows com Python 3.12.10 e placa integrada AMD Radeon (usada para desenvolver os módulos em `src/`, testes unitários da estrutura do cache e scripts de análise).
* **Execução em GPU (Gratuita):** Para rodar a bateria de avaliação rapidamente, o repositório inclui geração de notebook `.ipynb` para rodar gratuitamente em **Google Colab (GPU NVIDIA T4 - 16GB VRAM)** ou **Kaggle**.
* **Modelo Base Oficial do Experimento:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (ocupa ~3 GB de VRAM em FP16, pensa nativamente com `<think>` e faz *backtracking* explícito).

### O Experimento Decisivo (A construir primeiro)
Criar a pasta `src/rewind_kv/` com um script enxuto que rode **20 a 50 questões difíceis de matemática** (do benchmark **MATH-500** ou **GSM8K**) comparando exatamente **3 Modos**:

1. **Modo 1 — `Baseline (Standard CoT)`:** Geração padrão sem mexer no KV-Cache.
2. **Modo 2 — `Blind Rewind (Ablação sem Âncora)`:** Quando detecta erro no pensamento, dá `.crop(L_0)` no KV-Cache mas **não** coloca a Âncora de Falha (para medir quantas vezes o modelo sofre de amnésia e repete o mesmo erro).
3. **Modo 3 — `RewindKV (Rollback + Âncora de Falha)`:** Quando detecta erro no pensamento, extrai a Âncora curta, dá `.crop(L_0)` no KV-Cache, injeta a Âncora e continua.

### Como Rodar (Fase 1 implementada)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU local
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q          # testes do crop/RoPE e do detector (CPU, ~10 s)
```

O experimento com o modelo de verdade roda na GPU pelo notebook [`notebooks/colab_experimento_rewind_kv.ipynb`](notebooks/colab_experimento_rewind_kv.ipynb) (Colab) ou [`notebooks/kaggle_validacao_v02.ipynb`](notebooks/kaggle_validacao_v02.ipynb) (Kaggle, 2 GPUs, validação pré-registrada), ou direto com:

```bash
python -m rewind_kv.experiment --dataset math500 --level 5 --n 30
```

### Primeiros Comandos para Pedir ao Assistente ao Abrir um Chat Aqui:
> *"Leia o `README.md` e o `docs/especificacao-tecnica.md`. Vamos iniciar a implementação da Fase 1 do RewindKV: crie a estrutura em `src/rewind_kv/` com o controlador usando `DynamicCache.crop()`, os 3 modos experimentais (Baseline, Blind Rewind e RewindKV) e gere o Notebook pronto para rodar o teste com o `DeepSeek-R1-Distill-Qwen-1.5B`."*
