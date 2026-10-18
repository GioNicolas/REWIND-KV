# Especificação Técnica e Plano de Implementação — RewindKV

**Projeto:** **RewindKV** (*In-Stream KV-Cache Rollback with Failure Anchors for Reasoning Models*)  
**Autor:** Giorgio Nicolas de A. Bertino  
**Data:** 18 de outubro de 2026

---

## 1. A Mecânica Exata da Memória (Por que o KV-Cache Contamina o Raciocínio)

Em um Transformer autorregressivo padrão, ao gerar o token $t$, o estado de atenção de cada camada $l$ consulta todas as chaves e valores passados:

$$\mathbf{K}_{1:t}^{(l)}, \mathbf{V}_{1:t}^{(l)} \in \mathbb{R}^{t \times d_k}$$

Suponha que no índice $t = L_0$ o modelo inicie uma tentativa de resolução (Hipótese A) dentro do bloco `<think>`:
* Entre $t = L_0 + 1$ e $t = L_{\text{erro}}$, o modelo gera 800 tokens de passos algébricos ou lógicos.
* No índice $L_{\text{erro}}$, o modelo encontra uma contradição (ex.: $12 = 0$ ou violação de uma restrição do enunciado) e escreve: *"Wait, that's wrong. Let me try another approach."*
* **O problema físico atual:** Quando o modelo tenta a Hipótese B a partir de $t > L_{\text{erro}}$, os 800 vetores $(\mathbf{K}_{L_0:L_{\text{erro}}}, \mathbf{V}_{L_0:L_{\text{erro}}})$ **continuam ativos na matriz de atenção**.
  1. **Custo de Memória e Banda:** Cada novo token precisa calcular o produto escalar $\mathbf{q}_t \mathbf{K}_{1:t}^\top$ contra os 800 tokens mortos.
  2. **Contaminação de Atenção:** Como os números intermediários da Hipótese A têm alta similaridade semântica com as variáveis do problema, as cabeças de atenção (*attention heads*) puxam valores incorretos da Hipótese A para dentro da Hipótese B (fenômeno demonstrado em [arXiv:2608.15939](https://arxiv.org/abs/2608.15939)).

---

## 2. O Algoritmo RewindKV Passo a Passo

### Passo 2.1 — Marcação de Checkpoint ($L_0$)
Durante a geração dentro de `<think>`, o controlador registra um ponteiro de checkpoint $L_0$ sempre que um novo bloco de tentativa começa (após o prompt inicial ou após uma âncora anterior).

### Passo 2.2 — Detecção de Falha da Tentativa
* **Versão v0.1 (Gatilho de Reversão Nativo):** Monitora o *stream* de tokens buscando marcadores de invalidação de hipótese emitidos pelo próprio modelo de raciocínio (ex.: `"Wait, that"`, `"However, this contradicts"`, `"No, that's incorrect"`, `"Let me recalculate"`, `"doesn't work"`), seguidos até o fim da sentença onde o modelo declara o motivo do erro.
* **Versão v0.2 (Operário Discriminativo Ultrarrápido — *Latent Probe*):** Um classificador não-autorregressivo que lê as últimas ativações ocultas $\mathbf{h}_t$ e a entropia dos *logits* $H(p_t)$, emitindo uma decisão primitiva booleana (`Abortar_Ramo: Sim/Não`) e uma escolha (`Tipo_Falha: Conta | Lógica | Loop`) em menos de 5 ms.

### Passo 2.3 — Síntese da Âncora de Falha ($\mathcal{A}_{\text{fail}}$)
Antes de apagar o rascunho fracassado $[L_0 \dots L_{\text{erro}}]$, o algoritmo extrai uma **Âncora de Falha** compacta de no máximo 15 a 25 tokens:

$$\mathcal{A}_{\text{fail}} = \text{``[Note: Previous attempt using } \langle\text{hipótese}\rangle \text{ failed due to } \langle\text{contradição}\rangle\text{. Trying a different path:]''}$$

* **Como extrair $\mathcal{A}_{\text{fail}}$ sem custo extra:** Basta concatenar a primeira linha da tentativa (logo após $L_0$) com a linha da conclusão do erro (em $L_{\text{erro}}$), descartando todo o recheio de contas intermediárias!

### Passo 2.4 — Rewind Físico do KV-Cache e Injeção da Âncora
1. **Corte físico na GPU (`crop`):**
   ```python
   past_key_values.crop(L_0)
   ```
   Isso reduz instantaneamente os tensores $\mathbf{K}$ e $\mathbf{V}$ em todas as camadas para o comprimento exato $L_0$. Os 800 tokens errados deixam de existir na memória.

   > **Atenção (transformers ≥ 5):** `crop(n)` com `n` positivo (tamanho absoluto) passou a lançar `ValueError`. A forma suportada é `crop(-k)`, que remove os `k` últimos tokens e funciona também na 4.x. No código isso fica encapsulado em `crop_to(cache, L_0)` (`src/rewind_kv/controller.py`). Na prática o corte vai até $L_0 - 1$ e o token $L_0 - 1$ é reprocessado junto com a âncora, porque o estado oculto dele (necessário para amostrar o próximo token) não fica no cache.
2. **Injeção da Âncora com RoPE Contínuo:**
   Os tokens de $\mathcal{A}_{\text{fail}}$ (comprimento $m \ll 800$) são passados em um único *forward pass* rápido a partir de `past_key_values` (com `cache_position` indo de $L_0$ até $L_0 + m$).
3. **Atualização do Checkpoint:**
   O novo checkpoint passa a ser $L_0^{\text{novo}} = L_0 + m$, garantindo que a Âncora de Falha fique **protegida (trancada)** no cache enquanto o modelo inicia a próxima tentativa!

---

## 3. Arquitetura de Pastas a Criar no Dia 1

```text
REWIND-KV/
├── README.md                           # Visão geral e guia de início rápido
├── docs/
│   ├── guia-de-estudo-e-pitch.md       # Guia de estudo e respostas para entrevistas/bancas
│   └── especificacao-tecnica.md        # Este documento
├── src/
│   └── rewind_kv/
│       ├── __init__.py
│       ├── controller.py               # Classe RewindKVGenerator com DynamicCache.crop() e Âncoras
│       ├── detector.py                 # Gatilho de falha (v0.1 léxico + v0.2 probe discriminativo)
│       └── metrics.py                  # Medição de VRAM, tamanho de KV-Cache, tokens e acurácia
├── notebooks/
│   └── colab_experimento_rewind_kv.ipynb # Notebook pronto para rodar na GPU T4 do Google Colab
└── tests/
    └── test_kv_crop.py                 # Teste unitário local (roda na CPU em 2 segundos)
```

---

## 4. Critério de Sucesso do Primeiro Experimento (Go / No-Go)

Ao rodar 30 problemas do benchmark **MATH** / **GSM8K** onde o modelo `DeepSeek-R1-Distill-Qwen-1.5B` faz pelo menos 1 *backtracking*:
* **Sucesso Forte:** O modo `RewindKV` reduz o tamanho final do KV-Cache das tentativas erradas em **$>50\%$**, evita os loops de repetição do modo `Blind Rewind` (sem âncora) e **mantém ou aumenta a acurácia final** em relação ao `Baseline` padrão.
