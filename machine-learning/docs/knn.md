# 🤝 Vizinhos Mais Próximos (K-Nearest Neighbors - KNN)

Este documento apresenta a fundamentação matemática do algoritmo KNN implementado no módulo `src/knn.py`.

---

## 1. Métricas de Distância
O algoritmo KNN baseia-se na noção de proximidade geométrica. A distância entre dois vetores de atributos $\mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^D$ é calculada via distância de Minkowski de ordem $p$:

$$d_p(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{d=1}^{D} |x_{id} - x_{jd}|^p \right)^{\frac{1}{p}}$$

Métricas comuns de distância são casos especiais da distância de Minkowski:
1.  **Distância de Manhattan** ($p=1$):
    $$d_1(\mathbf{x}_i, \mathbf{x}_j) = \sum_{d=1}^{D} |x_{id} - x_{jd}|$$
2.  **Distância Euclidiana** ($p=2$):
    $$d_2(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{d=1}^{D} (x_{id} - x_{jd})^2}$$
3.  **Distância de Chebyshev** ($p \to \infty$):
    $$d_\infty(\mathbf{x}_i, \mathbf{x}_j) = \max_{d} |x_{id} - x_{jd}|$$

---

## 2. KNN para Classificação
Seja $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ o conjunto de treinamento. Para uma nova amostra de entrada $\mathbf{x}$, identificamos a vizinhança $N_K(\mathbf{x})$ composta pelos $K$ exemplos em $\mathcal{D}$ mais próximos de $\mathbf{x}$ segundo a distância escolhida.

A classe atribuída $\hat{y}$ é definida pelo voto majoritário dos vizinhos:

$$\hat{y}(\mathbf{x}) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(\mathbf{x})} \mathbb{I}(y_i = c)$$

Onde:
*   $\mathcal{C}$ é o conjunto de classes disponíveis.
*   $\mathbb{I}(\cdot)$ é a função indicadora, que retorna $1$ se a condição interna for verdadeira, e $0$ caso contrário.

---

## 3. KNN para Regressão
Para problemas de regressão, em vez de votação, a resposta predita $\hat{y}(\mathbf{x})$ é calculada como a média aritmética simples dos valores contínuos dos $K$ vizinhos mais próximos:

$$\hat{y}(\mathbf{x}) = \frac{1}{K} \sum_{i \in N_K(\mathbf{x})} y_i$$
