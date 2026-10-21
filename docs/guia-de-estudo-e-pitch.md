# Guia de Estudo e Pitch do RewindKV

Este guia é para você estudar a sua própria pesquisa e saber explicá-la a qualquer pessoa: recrutador, professor, banca ou colega. Ele conta a história **como ela aconteceu**, inclusive o resultado negativo, porque é isso que você vai precisar defender.

> **Atualizado em 21/10/2026**, depois das duas validações pré-registradas, da sonda e do teste do oráculo. A versão anterior prometia ganho de acurácia e de memória que os dados não confirmaram.

---

## 1. O pitch de 30 segundos

> *"Eu estudei se dá para melhorar modelos de raciocínio, como o DeepSeek-R1, apagando da memória deles (o KV-cache) os trechos em que eles mesmos dizem que erraram, e deixando no lugar um bilhete curto sobre o que deu errado. Implementei o mecanismo num modelo de 1,5 bilhão de parâmetros e testei com rigor: duas validações com o critério escrito antes de rodar. Não melhorou a acurácia, e eu descobri por quê. Quando o modelo diz 'errei', só metade das vezes ele errou de verdade. E mesmo quando o corte pega um erro real, nos problemas difíceis o modelo continua ficando sem espaço para pensar. O gargalo não é a memória: é saber quando o modelo errou."*

**Se perguntarem "então deu errado?":**

> *"A hipótese não se confirmou, e isso é um resultado. Eu testei de forma que não desse para me enganar: critério escrito antes, problemas que o método nunca tinha visto, teste estatístico combinado antes. E consegui explicar o fracasso em três camadas, com dados. Isso é útil para quem for tentar a mesma ideia."*

---

## 2. Os conceitos, na ordem em que você deve estudar

### 2.1 Como um modelo de linguagem escreve
- Escreve **um token por vez** (um token é um pedaço de palavra). A cada passo, olha tudo o que já foi escrito e sorteia o próximo token entre os mais prováveis.
- O sorteio usa uma **semente**: mesma semente, mesmo texto. Usamos a mesma semente nos três modos, então o texto é idêntico até o primeiro corte e a comparação é justa.

### 2.2 Modelos de raciocínio
- Pensam em voz alta dentro de `<think>…</think>` antes de responder. Nesse rascunho eles duvidam e voltam atrás: *"Wait, that's not right"*, *"Alternatively…"*.
- O nosso modelo é o `DeepSeek-R1-Distill-Qwen-1.5B`. Ele cabe numa GPU gratuita (T4 do Colab/Kaggle).
- Testamos no **MATH-500**: 500 problemas de competição em 5 níveis de dificuldade. Usamos os níveis 4 e 5, os mais difíceis.

### 2.3 KV-cache
- É a **memória de curto prazo** do modelo. Para cada token já escrito, guarda dois vetores (K e V), para não recalcular tudo a cada token novo.
- No nosso modelo são **28 KB por token**; 4000 tokens ≈ 110 MB, enquanto o modelo ocupa ~3,5 GB. Por isso a economia de memória do corte é pequena aqui.
- **Cortar o cache tem o mesmo efeito que apagar o fim do texto e continuar dali.** O cache só torna isso barato (`DynamicCache.crop()`).
- **RoPE** (a posição de cada token): como o corte é sempre no fim, as posições continuam em sequência e nada se embaralha.

### 2.4 A ideia do RewindKV
1. Marcar onde começa cada tentativa (*checkpoint*).
2. Detectar quando o modelo declara um erro (frases como *"I made a mistake"*).
3. Cortar o KV-cache até o início do caminho que falhou.
4. Colocar no lugar uma **âncora**: um bilhete curto com o que foi tentado e por que deu errado.
5. Deixar o modelo continuar.

**Por que a âncora?** Sem ela, o modelo esquece o que tentou e pode repetir o mesmo erro (amnésia). No nível 5, o corte sem âncora refez a mesma conta errada 6 vezes no problema 88; com âncora, uma vez.

### 2.5 O experimento: três modos
| Modo | O que faz | Para que serve |
|---|---|---|
| `baseline` | nada | o normal |
| `blind_rewind` | corta, sem âncora | isola o efeito de cortar |
| `rewind_kv` | corta e coloca a âncora | comparado com o blind, isola o efeito da âncora (**H1, a hipótese principal**) |

### 2.6 Pré-registro e teste do sinal
- **Pré-registro:** antes de rodar, escrevemos e commitamos no git as hipóteses, os problemas e o teste. Ninguém pode dizer que mudamos a regra depois de ver o resultado.
- **Por que importa:** no conjunto de desenvolvimento (A2), a v0.2 ganhou de 3 a 1. Em problemas novos, perdeu de 1 a 5. O 3 a 1 era ajuste aos problemas que tínhamos olhado.
- **Teste do sinal:** só contam os problemas em que um modo acertou e o outro errou. Se não houvesse diferença, cada um seria cara ou coroa. O **p-valor** é a chance de ver um placar tão desequilibrado só por sorte. Combinamos que só p < 0,05 contaria.

### 2.7 Os resultados
| Experimento | Problemas | H1: rewind_kv vs blind | p |
|---|---|---|---|
| Experimento A (v0.1) | 30 do nível 4 | (sem blind) efeito nulo | — |
| A2 (v0.2, desenvolvimento) | 10 do nível 4 | 3 a 1 | 0,63 |
| **Validação v0.2** | 98 novos do nível 4 | **1 a 5** | 0,22 |
| **Validação v0.3** | 134 do nível 5 | **3 a 3** | 1,0 |

### 2.8 Por que deu errado: três camadas
1. **O gatilho é ruim.** Rotulamos à mão 161 cortes. Em problemas novos, só **56%** eram erro real. Os outros eram:
   - **falso alarme:** dúvida sobre uma conta certa, ou eliminação legítima de um caso (*"1 ≠ 7, so it doesn't work"*);
   - **lado errado:** em *"deu 113, mas antes deu 136"*, o corte apaga o mais recente, que às vezes era o certo; ou o corte apaga a própria correção (*"83 − 68 = 15, not 11"*).

   Regras de palavras mais rígidas chegaram a 70% nos problemas em que foram escritas e caíram para 56% nos novos.
2. **A âncora amplifica o que o gatilho decide.** A âncora que proíbe (*"It was wrong, I will not repeat it"*) impede o modelo de redescobrir o que estava certo: as 5 derrotas da v0.2 foram todas cortes de coisa certa. A âncora de dúvida (v0.3) não atrapalha, mas também não impede repetir.
3. **Mesmo com o gatilho perfeito, não haveria ganho aqui (teste do oráculo).** Nos 28 problemas em que todos os cortes foram erro real, a acurácia praticamente não mudou. Quase todos acabam sem tokens nos três modos. E o limite de 4096 conta os tokens cortados, então o corte não devolve espaço para pensar.

### 2.9 A sonda (probe)
- Tentamos ler o **estado interno** do modelo (os 1.536 números de cada camada) no momento do "errei", em vez das palavras.
- Uma regressão logística, treinada em 69 cortes do nível 4 e testada uma vez em 84 do nível 5, com critério escrito antes: **precisão ≥ 75% e recall ≥ 70%**.
- Resultado: **precisão 63,9%** (cortar tudo dá 58%), **AUC 0,605** (0,5 é chute, 1,0 é perfeito). Não passou.
- **Precisão:** dos cortes mantidos, quantos eram erro real. **Recall:** dos erros reais, quantos foram mantidos.

---

## 3. O que é contribuição (e o que não é)

**É contribuição:**
1. O mecanismo de corte do KV-cache com âncora em texto, dentro do `<think>`, implementado e testado (código aberto, 84 testes).
2. **A medida de quanto valem os autoavisos de erro do modelo:** 161 cortes rotulados, com cerca de 50–60% de precisão em problemas novos e uma classificação dos tipos de erro do gatilho.
3. **A âncora como amplificadora:** proibir evita repetir, mas impede redescobrir o certo; duvidar é seguro, mas não impede repetir.
4. Dois testes que mostram onde está o gargalo: a sonda e o oráculo.
5. O método: duas validações pré-registradas, com resultado negativo explicado.

**Não é contribuição (não diga):**
- Que o RewindKV melhora a acurácia.
- Que economiza muita memória (neste modelo, ~10% do cache, que já é pequeno perto do modelo).
- Que é "o único" ou "o primeiro" a fazer rollback de KV-cache. Há vizinhos próximos: LPSR, Thought Rollback e sondas de correção de raciocínio (ver o [Mapa da Literatura](mapa-da-literatura.md)).

---

## 4. Perguntas difíceis (e respostas honestas)

**"Cortar o KV-cache não bagunça as posições (RoPE)?"**
> "Não. O corte é sempre no fim do cache. A âncora entra nas posições seguintes, em sequência, como se o texto tivesse sido escrito assim desde o começo. Testamos isso: o resultado é idêntico a gerar o texto editado do zero."

**"Qual a diferença para compressores de KV-cache, como R-KV ou H2O?"**
> "Eles apagam tokens soltos que recebem pouca atenção, para economizar memória. A gente apagava um raciocínio inteiro que o próprio modelo declarou errado e deixava uma explicação no lugar. O objetivo era a qualidade do raciocínio, não a memória."

**"Como vocês sabiam quando cortar?"**
> "Por frases de erro do próprio modelo, como 'I made a mistake'. E esse foi o problema: medimos à mão, e só cerca de metade dessas frases eram erro de verdade."

**"Por que não usaram um modelo maior?"**
> "Para caber numa GPU gratuita e permitir validações grandes. É uma limitação: em modelos maiores, os avisos de erro podem ser mais confiáveis. Fica como trabalho futuro."

**"Se o detector fosse perfeito, funcionaria?"**
> "Nos nossos dados, não. No teste do oráculo, mesmo nos problemas em que todos os cortes pegaram erros reais, a acurácia não mudou, porque o modelo ficava sem tokens nos três modos. Talvez funcione em outro cenário: mais tokens, ou problemas em que o modelo termina antes do limite."

**"Precisou treinar o modelo?"**
> "Não. O RewindKV atua só na geração, sem mudar os pesos. A única coisa treinada foi a sonda, uma regressão logística pequena."
