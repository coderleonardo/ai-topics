# 🧬 Agrupamento (Clustering)

Este documento descreve as formulações dos algoritmos de agrupamento e validação de agrupamentos implementados no módulo `src/clustering.py`.

---

## 1. K-Means
O algoritmo K-Means agrupa amostras $\mathbf{X} = \{\mathbf{x}_n\}_{n=1}^N$ em $K$ grupos disjuntos representados por centroides $\{\mathbf{\mu}_k\}_{k=1}^K$ minimizando a inércia (soma das distâncias quadráticas internas):

$$J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \mathbf{\mu}_k\|^2$$

Onde $r_{nk} \in \{0, 1\}$ é a variável de atribuição binária de pertencer ao cluster.
O algoritmo é executado em duas fases iterativas alternadas (Algoritmo Lloyd):
1.  **Fase de Atribuição (Passo E)**:
    $$r_{nk} = \begin{cases} 1 & \text{se } k = \arg\min_j \|\mathbf{x}_n - \mathbf{\mu}_j\|^2 \\ 0 & \text{caso contrário} \end{cases}$$
2.  **Fase de Atualização (Passo M)**:
    $$\mathbf{\mu}_k = \frac{\sum_{n=1}^N r_{nk} \mathbf{x}_n}{\sum_{n=1}^N r_{nk}}$$

### Inicialização K-Means++
Para mitigar a sensibilidade à inicialização aleatória, a inicialização K-Means++ escolhe centroides distantes entre si:
1.  Seleciona o primeiro centroide $\mathbf{\mu}_1$ uniformemente ao acaso das amostras de $\mathbf{X}$.
2.  Para cada centroide subsequente $k = 2, \dots, K$, seleciona uma nova amostra $\mathbf{x}_n$ com probabilidade proporcional à distância quadrática mínima até o centroide mais próximo já selecionado:
    $$P(\mathbf{x}_n) = \frac{D(\mathbf{x}_n)^2}{\sum_{m=1}^N D(\mathbf{x}_m)^2}$$
    Onde $D(\mathbf{x}) = \min_{j < k} \|\mathbf{x} - \mathbf{\mu}_j\|_2$.

---

## 2. DBSCAN (Agrupamento Baseado em Densidade)
O DBSCAN identifica clusters com formatos arbitrários com base na densidade local de pontos e isola ruídos.

### Definições de Densidade
*   **$\epsilon$-Vizinhança**: A vizinhança esférica de raio $\epsilon$ em torno de um ponto $\mathbf{x}_p$:
    $$N_\epsilon(\mathbf{x}_p) = \{ \mathbf{x}_q \in \mathbf{X} \mid \|\mathbf{x}_p - \mathbf{\mu}_q\|_2 \le \epsilon \}$$
*   **Ponto de Núcleo (Core Point)**: Um ponto $\mathbf{x}_p$ tal que a quantidade de vizinhos em seu raio $\epsilon$ é pelo menos um limite mínimo de amostras ($MinPts$):
    $$|N_\epsilon(\mathbf{x}_p)| \ge MinPts$$
*   **Ponto de Borda (Border Point)**: Um ponto $\mathbf{x}_q$ que não é ponto de núcleo, mas é vizinho de algum ponto de núcleo $\mathbf{x}_p$ (ou seja, $\mathbf{x}_q \in N_\epsilon(\mathbf{x}_p)$).
*   **Ponto de Ruído (Noise Point)**: Qualquer ponto que não seja classificado nem como núcleo nem como borda.

---

## 3. Agrupamento Espectral (Spectral Clustering)
O Agrupamento Espectral é uma abordagem baseada em grafos que modela o conjunto de dados como um grafo não direcionado ponderado $G = (V, E)$, onde os vértices $V$ são os pontos de dados e as arestas $E$ possuem pesos que refletem a similaridade entre os pontos.

### Formulação Matemática
1.  **Matriz de Adjacência Ponderada (Similaridade RBF)**:
    $$W_{ij} = \begin{cases} \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) & \text{se } i \neq j \\ 0 & \text{se } i = j \end{cases}$$
2.  **Matriz de Grau ($D$)**:
    Matriz diagonal onde cada entrada representa o grau de conectividade do vértice $i$:
    $$D_{ii} = d_i = \sum_{j=1}^N W_{ij}$$
3.  **Matriz Laplaciana do Grafo ($L$)**:
    *   **Não Normalizada**:
        $$L = D - W$$
    *   **Normalizada Simétrica**:
        $$L_{\text{sym}} = \mathbf{I} - D^{-1/2} W D^{-1/2}$$
4.  **Projeção Espectral**:
    Calculamos os autovetores $\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_K$ correspondentes aos $K$ menores autovalores da matriz Laplaciana:
    $$L_{\text{sym}} \mathbf{u}_k = \lambda_k \mathbf{u}_k$$
    Ordenando de forma crescente: $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N$.
    A matriz $\mathbf{U} \in \mathbb{R}^{N \times K}$ é formada por esses $K$ autovetores como colunas.
5.  **Normalização de Linhas (Filtro Espectral)**:
    Projeta-se cada linha da matriz $\mathbf{U}$ para a esfera unitária para normalizar a influência de cada ponto:
    $$Y_{ij} = \frac{U_{ij}}{\sqrt{\sum_{l=1}^K U_{il}^2}}$$
6.  **Agrupamento**:
    Aplica-se o K-Means sobre as linhas de $\mathbf{Y}$ (ou $\mathbf{U}$) para obter as designações finais de clusters dos exemplos originais.

---

## 4. Coeficiente de Silhueta (Silhouette Coefficient)
O coeficiente de silhueta avalia a qualidade do agrupamento de forma intrínseca. Para cada amostra $i$:

### Distância Intra-Cluster Média ($a_i$)
Distância média entre a amostra $i$ e todas as outras amostras pertencentes ao mesmo cluster $C_A$:

$$a_i = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \neq i} d(\mathbf{x}_i, \mathbf{x}_j)$$

### Distância Inter-Cluster Média Mínima ($b_i$)
A distância média mínima da amostra $i$ para qualquer outro cluster vizinho $C_B$:

$$b_i = \min_{C_B \neq C_A} \frac{1}{|C_B|} \sum_{j \in C_B} d(\mathbf{x}_i, \mathbf{x}_j)$$

### Coeficiente de Silhueta Individual ($s_i$)
$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

Onde $s_i \in [-1, 1]$. Valores próximos a $1$ indicam que o ponto está bem agrupado e distante de outros clusters.
O coeficiente de silhueta médio global do dataset é a média de $s_i$ para todas as amostras.
