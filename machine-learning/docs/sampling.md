# 🗳️ Métodos de Amostragem e Validação de Modelos

Este documento detalha a formulação dos métodos de partição e avaliação implementados em `src/sampling.py`.

---

## 1. Método Holdout (train_test_split)
Consiste em dividir o conjunto de dados em partes disjuntas de treino e teste. Se os dados forem embaralhados de forma aleatória e uniforme:

*   **Tamanho do Teste ($N_{\text{test}}$)**: $\lfloor N \times \alpha \rfloor$
*   **Tamanho do Treino ($N_{\text{train}}$)**: $N - N_{\text{test}}$

Onde $\alpha \in (0, 1)$ é o parâmetro `test_size`.

---

## 2. Validação Cruzada K-Fold (K-Fold Cross-Validation)
A validação cruzada K-Fold reduz a variância da estimativa de erro dividindo os dados em $K$ partições mutuamente exclusivas e de tamanhos aproximadamente iguais:

$$X = S_1 \cup S_2 \cup \dots \cup S_K, \quad S_i \cap S_j = \emptyset \quad \forall i \neq j$$

Para cada iteração $k = 1, \dots, K$:
1.  **Conjunto de Teste**: $X_{\text{test}} = S_k$
2.  **Conjunto de Treinamento**: $X_{\text{train}} = \bigcup_{j \neq k} S_j$
3.  O modelo é ajustado em $X_{\text{train}}$ e avaliado em $X_{\text{test}}$, obtendo-se uma métrica de desempenho $M_k$.

Ao final das $K$ rodadas, calcula-se a média e o desvio padrão das métricas observadas:

$$\mu_M = \frac{1}{K} \sum_{k=1}^K M_k$$

$$\sigma_M = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (M_k - \mu_M)^2}$$

---

## 3. Validação por Bootstrap (Out-of-Bag Evaluation)
O bootstrap gera amostras aleatórias de tamanho $N$ a partir do dataset original de tamanho $N$ via reamostragem com reposição.

### Probabilidade de Amostras Fora da Bolsa (Out-of-Bag - OOB)
Em cada rodada do bootstrap, alguns pontos do conjunto de dados original nunca serão selecionados. A probabilidade de um determinado ponto de dados *não* ser selecionado em $N$ sorteios com reposição é dada por:

$$p = \left( 1 - \frac{1}{N} \right)^N$$

No limite termodinâmico (para $N \to \infty$):

$$\lim_{N \to \infty} \left( 1 - \frac{1}{N} \right)^N = \frac{1}{e} \approx 0.36787$$

*   **Amostra de Treino**: Contém aproximadamente $63.2\%$ dos pontos originais distintos (alguns repetidos).
*   **Amostra de Teste (Out-of-Bag)**: Contém os aproximadamente $36.8\%$ pontos restantes que não foram sorteados.

A avaliação por bootstrap repete esse processo $B$ vezes, fornecendo uma estimativa robusta das métricas gerais do modelo baseada no conjunto OOB de cada rodada.
