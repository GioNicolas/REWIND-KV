# S.E.E.D.-KV: Rollback Cirúrgico de KV-Cache com Âncoras de Falha em Modelos de Raciocínio

**Sigla sugerida:** **S.E.E.D.-KV** (*Self-Erasing Error Drafts in KV-Cache*) — alternativa de título para artigo: **AnchorKV** ou **RewindKV**.  
**Autor:** Giorgio Nicolas de A. Bertino  
**Data de registro:** 18 de outubro de 2026  
**Status:** Nova direção principal de pesquisa científica (baixo custo computacional, alto impacto em LLMs de raciocínio).

---

## 1. Resumo Executivo (O que é e por que importa)

Modelos modernos de raciocínio (como *DeepSeek-R1*, *Qwen-QwQ*, *OpenAI o1/o3*) resolvem problemas complexos de matemática, lógica e programação gerando longas cadeias de pensamento (`<think>...</think>`). Durante esse processo, o modelo frequentemente explora uma hipótese, percebe que ela levou a um erro ou contradição (*"Wait, that's wrong, let me recalculate..."*) e inicia uma nova tentativa.

**O defeito estrutural atual:**  
Como a geração autorregressiva nos Transformers é estritamente aditiva (*append-only*), **todos os centenas ou milhares de tokens do rascunho errado permanecem gravados na memória ativa da placa de vídeo (o KV-Cache)** até o fim da geração. Isso causa dois problemas graves:
1. **Explosão de Memória e Latência:** O KV-Cache acumula rascunhos mortos, devorando VRAM e deixando cada novo token mais lento.
2. **Contaminação Causal de Atenção (*Context Poisoning*):** Enquanto tenta calcular o caminho certo na segunda tentativa, o mecanismo de atenção da IA continua consultando os números e passos errados da primeira tentativa que ficaram no KV-Cache, induzindo novos erros.

**A Nossa Solução (S.E.E.D.-KV):**  
Um algoritmo de intervenção em tempo de inferência (*training-free* ou guiado por um detector discriminativo ultraleve) que:
1. Marca **Checkpoints Lógicos** no KV-Cache no início de cada hipótese/passo do raciocínio;
2. Detecta o instante em que a tentativa atual falha ou entra em contradição;
3. **Executa um Rollback Físico no KV-Cache** (`DynamicCache.crop()`), apagando da memória da GPU todos os tokens do cálculo errado em frações de milissegundo;
4. Injeta no lugar apenas uma **Âncora Negativa curta (10 a 20 tokens)** resumindo o motivo da falha (ex.: `[Nota: A tentativa X falhou porque violou Y. Não repetir.]`), impedindo que o modelo entre em loop e repita o mesmo erro.

> **Por que o resultado esperado é de altíssimo impacto:**  
> Diferente dos compressores de KV-Cache tradicionais (que apagam memória tentando "perder o mínimo possível de precisão"), o **S.E.E.D.-KV** propõe **reduzir a memória/latência E aumentar a inteligência (acurácia final) simultaneamente**, ao limpar o "lixo tóxico" dos ramos fracassados da frente da atenção do modelo.

---

## 2. Posicionamento Científico e Literatura de Ponta (2025–2026)

Esta pesquisa ataca uma lacuna exata aberta pelos artigos mais recentes publicados entre maio e agosto de 2026:

| Artigo Recente (Verificado) | O que descobriu / fez | O que deixou em aberto (Nossa Lacuna) |
| :--- | :--- | :--- |
| **R-KV** — Cai et al., NeurIPS 2025 ([arXiv:2505.24133](https://arxiv.org/abs/2505.24133)) | Comprime o KV-Cache de modelos de raciocínio removendo tokens redundantes baseados em atenção. | Apenas poda palavras soltas; **não faz rollback estrutural** de uma tentativa inteira que deu errado nem cria âncoras de erro. |
| **InfoKV** — Kai et al., jun/2026 ([arXiv:2606.26875](https://arxiv.org/abs/2606.26875)) | Usa incerteza preditiva (*Forward Influence*) para comprimir KV-Cache em raciocínio longo. | Mantém a estrutura linear de erros; avaliado apenas em passagem contínua sem *backtracking* físico do cache. |
| **Models Take Notes at Prefill** — Bojie Li, jun/2026 ([arXiv:2606.17107](https://arxiv.org/abs/2606.17107)) | Provou que os vetores KV funcionam como "notas de rascunho" editáveis e componíveis com reposicionamento RoPE. | Focado na fase de *prefill* (leitura do prompt), não na auto-limpeza dinâmica durante a geração do `<think>`. |
| **ArborKV** — Chen et al., ICML 2026 ([arXiv:2605.05689](https://arxiv.org/abs/2605.05689)) | Gerencia KV-Cache em buscas em árvore externas (*Tree-of-Thoughts*). | Exige um orquestrador externo gerando múltiplas árvores paralelas caras; não opera *in-stream* dentro do fluxo nativo de um modelo R1. |
| **Aborted but Not Forgotten** — Zhang & Yang, ago/2026 ([arXiv:2608.15939](https://arxiv.org/abs/2608.15939)) | **Prova o nosso ponto central:** mostrou que ramos abortados que continuam retidos no KV-Cache contaminam a atenção futura do modelo (*rollback inconsistency*). | Diagnosticou o problema de segurança/consistência em agentes, mas **não propôs o algoritmo de poda de raciocínio com substituição por Âncora Negativa**. |

---

## 3. Como Funciona a Arquitetura e o Algoritmo (Os 3 Estágios)

O sistema une a filosofia **Neuro-Simbólica** do projeto S.E.E.D. (Intuição Neural + Controle/Âncoras + Operário Discriminativo Rápido):

```text
[Fluxo de Pensamento da LLM: <think>]
       │
       ├──► 1. CHECKPOINT: Início de nova hipótese/passo em L_0 (salva índice do KV-Cache)
       │         "Vou tentar resolver assumindo x = 5... [800 tokens de contas]..."
       │
       ├──► 2. DETECÇÃO DE FALHA (Gatilho Léxico ou Operário Discriminativo em ms)
       │         A IA percebe contradição: "Espera, isso dá 10 != 0, está errado!"
       │
       └──► 3. CIRURGIA S.E.E.D.-KV:
                 (a) Extrai a Âncora Negativa curta: "[Falha: x=5 viola a condição Z]"
                 (b) Executa Rollback Físico: kv_cache.crop(L_0) -> Apaga os 800 tokens!
                 (c) Injeta os ~12 tokens da Âncora Negativa na posição L_0
                 (d) Retoma a geração com o KV-Cache limpo e protegido contra repetição!
```

### Evolução em Duas Fases do Detector (Estágio 2):
* **Fase 1 (Protótipo Imediato — *Self-Triggered*):** Usa os próprios marcadores naturais de *backtracking* que modelos como `DeepSeek-R1-Distill` já emitem quando percebem que erraram (*"Wait"*, *"However, this contradicts"*, *"Let me recalculate"*, *"That doesn't work"*), confirmados por uma verificação rápida.
* **Fase 2 (Operário Discriminativo Estilo Jev — *Latent Probe*):** Um classificador ultraleve que monitora a entropia dos *logits* e as ativações internas a cada bloco de tokens, emitindo primitivas rápidas (`Boolean`: abortar ramo? / `Choice`: erro de conta vs. caminho sem saída) **antes** mesmo de o modelo gastar centenas de tokens escrevendo a conta errada até o fim.

---

## 4. Desafios Técnicos e Cuidados (Onde pode dar problema e como resolver)

1. **Paradoxo da Amnésia (Por que o "Corte Cego" falha):**  
   Se você simplesmente apagar os 800 tokens errados do KV-Cache e voltar o cursor para a posição $L_0$ sem deixar nenhuma nota, a IA vai gerar exatamente os mesmos 800 tokens errados de novo (entrando em loop infinito, especialmente com temperatura baixa)!  
   * **Solução:** É por isso que a **Âncora Negativa** é o coração da invenção. Ela ocupa apenas ~15 tokens, mas altera o contexto em $L_0$ para que a IA saiba que aquele caminho já foi testado e falhou.
2. **Alinhamento de Posição (*RoPE — Rotary Position Embeddings*):**  
   Quando você corta o KV-Cache na posição $L_0$ e insere os 15 tokens da Âncora Negativa, os novos tokens devem receber os índices de posição contínuos $L_0 + 1, L_0 + 2, \dots$ a partir do ponto de corte. Usando `DynamicCache.crop(L_0)` do HuggingFace `transformers` e re-executando o *forward pass* apenas nos 15 tokens da âncora, os embeddings posicionais (RoPE) ficam 100% contínuos e matematicamente exatos, sem precisar reprocessar o prompt inicial!
3. **Falsos Positivos no Gatilho:**  
   Às vezes o modelo diz *"Wait, let me check if this is right... yes, it is!"* (ele só conferiu, mas não estava errado).  
   * **Solução:** O gatilho só dispara o rollback se a frase de hesitação for seguida de descarte real da hipótese, e não de confirmação.

---

## 5. POR ONDE COMEÇAR (Roteiro Prático para o "Dia 1" no Próximo Chat)

### Recursos e Ambiente
* **Máquina local:** Windows com Python 3.12.10 e placa integrada AMD Radeon (ideal para escrever o código, testes unitários da estrutura do cache e scripts de análise).
* **Execução em GPU (Gratuita):** Como modelos de raciocínio de 1.5B a 7B rodam muito mais rápido em GPU CUDA, todo experimento pesado será exportado também como um notebook `.ipynb` pronto para rodar gratuitamente em **Google Colab (GPU NVIDIA T4 - 16GB VRAM)** ou **Kaggle (2x GPU T4)**.
* **Modelo Base Oficial do Experimento:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (super leve, ocupa apenas ~3 GB de VRAM em FP16, pensa nativamente com `<think>` e faz *backtracking* explícito).

### O Experimento Decisivo (A construir primeiro)
Criar a pasta `src/seed_kv/` com um script enxuto que rode **20 a 50 questões difíceis de matemática** (do benchmark **MATH-500** ou **GSM8K**) comparando exatamente **3 Modos**:

1. **Modo 1 — `Baseline (Standard CoT)`:** Geração padrão sem mexer no KV-Cache.
2. **Modo 2 — `Blind Rollback (Ablação sem Âncora)`:** Quando detecta erro no pensamento, dá `.crop()` no KV-Cache mas **não** coloca a Âncora Negativa (com temperatura 0.6 para medir quantas vezes ele repete o mesmo erro).
3. **Modo 3 — `S.E.E.D.-KV (Rollback + Âncora Negativa)`:** Quando detecta erro no pensamento, extrai a Âncora curta, dá `.crop(L_0)` no KV-Cache, injeta a Âncora e continua.

### As 4 Métricas que Vamos Medir na Tela
1. **Pico de Memória do KV-Cache** (em número máximo de tokens retidos na VRAM e em MB);
2. **Total de Tokens Gerados e Tempo de Parede (Segundos)** até chegar na resposta final;
3. **Taxa de Acerto (Acurácia %)** nas questões da prova;
4. **Taxa de Repetição de Erro (Loop Rate)** comparando o Modo 2 vs. Modo 3.

### Primeiros Comandos para Pedir ao Assistente no Novo Chat:
> *"Leia o arquivo `docs/seed-kv-pesquisa.md`. Vamos iniciar a implementação da Fase 1 do S.E.E.D.-KV: crie a estrutura em `src/seed_kv/` com o gerenciador de geração customizado usando `DynamicCache.crop()` do HuggingFace Transformers (implementando os 3 modos: Baseline, Blind Rollback e S.E.E.D.-KV com Âncora Negativa) e gere também um Notebook pronto para rodar o primeiro teste com o `DeepSeek-R1-Distill-Qwen-1.5B` em problemas de matemática."*
