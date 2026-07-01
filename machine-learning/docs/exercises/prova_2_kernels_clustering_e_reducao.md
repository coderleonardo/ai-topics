# 📝 Prova Corretiva 2: SVM, Agrupamentos e Redução de Dimensionalidade (Feito à Mão)

Este guia prático foi desenhado no formato de uma prova resolvida passo a passo "na mão", cobrindo métodos baseados em margem, algoritmos de clusterização e projeções lineares.

---

## Questão 1: Máquinas de Vetores de Suporte (SVM), Kernels e SVR

### 1.1 Dedução da Formulação Dual de SVM
Dada a maximização da margem suave com variáveis de folga $\xi_n \ge 0$, o Lagrangeano primal é:
$$L_p(\mathbf{w}, b, \boldsymbol{\xi}, \mathbf{a}, \boldsymbol{\mu}) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{n=1}^N \xi_n - \sum_{n=1}^N a_n \{y_n (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) - 1 + \xi_n\} - \sum_{n=1}^N \mu_n \xi_n$$

Ao tomarmos as derivadas em relação a $\mathbf{w}$, $b$ e $\xi_n$ e igualarmos a zero:
1.  $\mathbf{w} = \sum_{n=1}^N a_n y_n \boldsymbol{\phi}(\mathbf{x}_n)$
2.  $\sum_{n=1}^N a_n y_n = 0$
3.  $a_n = C - \mu_n \implies 0 \le a_n \le C$ (pois $\mu_n \ge 0$)

Substituindo essas relações de volta no Lagrangeano, obtemos a formulação dual max-dual:
$$\max_{\mathbf{a}} \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m y_n y_m K(\mathbf{x}_n, \mathbf{x}_m)$$
Sujeito a: $0 \le a_n \le C$ e $\sum_{n=1}^N a_n y_n = 0$.

---

### 1.2 SVM Linear na Mão (Classificação)
**Exemplo Prático**: Dado o conjunto de dados linearmente separável em 1D:
*   $x_1 = 1, y_1 = -1$ (Classe Negativa)
*   $x_2 = 2, y_2 = -1$ (Classe Negativa)
*   $x_3 = 4, y_3 = 1$  (Classe Positiva)

Encontre o hiperplano separador de margem máxima $w x + b = 0$ na mão.
*   **Análise Visual**: O espaço separável está entre o maior valor negativo ($x=2$) e o menor valor positivo ($x=4$). O ponto médio de corte ótimo é $x = 3$.
*   A fronteira de decisão é:
    $$w x + b = 0 \implies x - 3 = 0 \implies w = 1, \quad b = -3$$
*   **Verificação das Restrições**:
    *   Para $x_1=1$: $y_1(w x_1 + b) = -1(1 \times 1 - 3) = -1(-2) = 2 \ge 1$ (Satisfeito)
    *   Para $x_2=2$: $y_2(w x_2 + b) = -1(1 \times 2 - 3) = -1(-1) = 1 \ge 1$ (Satisfeito - Vetor de Suporte!)
    *   Para $x_3=4$: $y_3(w x_3 + b) = 1(1 \times 4 - 3) = 1(1) = 1 \ge 1$ (Satisfeito - Vetor de Suporte!)
*   Os vetores de suporte são $x_2=2$ e $x_3=4$. A margem geométrica é $\gamma = \frac{1}{\|w\|} = 1$.

---

### 1.3 Quando usar e quando não usar variações de SVM

| Algoritmo | Situação para Usar | Situação para NÃO Usar |
| :--- | :--- | :--- |
| **SVM (Classificação)** | Margens de decisão bem definidas e dados de alta dimensionalidade (com truque do kernel). | Datasets muito grandes (complexidade quadrática a cúbica com o número de amostras) ou com muito ruído. |
| **SVR (Regressão)** | Relações não-lineares complexas em problemas de regressão de pequeno a médio porte. | Quando a interpretabilidade física dos parâmetros do modelo é crucial. |
| **One-Class SVM** | Detecção de anomalias/novidades em cenários não supervisionados (apenas dados da classe normal). | Quando dados rotulados das duas classes estão abundantemente disponíveis (classificadores binários são melhores). |

---

## Questão 2: Agrupamentos K-Means, DBSCAN e Agrupamento Espectral

### 2.1 K-Means 1D na Mão
**Exemplo Prático**: Dado o dataset 1D: $X = \{1, 2, 5, 6\}$.
Inicialize os centroides em: $\mu_1 = 1$, $\mu_2 = 3$. Faça uma iteração completa do algoritmo.

#### Passo 1: Atribuição dos Clusters
Calculamos a distância absoluta de cada ponto aos centroides $\mu_1$ e $\mu_2$:
*   $x_1=1$: $d(1, \mu_1)=0$, $d(1, \mu_2)=2 \implies$ Pertence ao Cluster 1.
*   $x_2=2$: $d(2, \mu_1)=1$, $d(2, \mu_2)=1 \implies$ Desempate arbitrário $\implies$ Cluster 1.
*   $x_3=5$: $d(5, \mu_1)=4$, $d(5, \mu_2)=2 \implies$ Pertence ao Cluster 2.
*   $x_4=6$: $d(6, \mu_1)=5$, $d(6, \mu_2)=3 \implies$ Pertence ao Cluster 2.

*Grupos Resultantes*: $C_1 = \{1, 2\}$, $C_2 = \{5, 6\}$.

#### Passo 2: Atualização dos Centroides
*   Novo $\mu_1 = \frac{1 + 2}{2} = 1.5$
*   Novo $\mu_2 = \frac{5 + 6}{2} = 5.5$

---

### 2.2 DBSCAN 1D na Mão
**Exemplo Prático**: Dado o dataset 1D: $X = \{1, 2, 3, 10\}$.
Sejam os parâmetros $\epsilon = 1.5$ e $\text{MinPts} = 2$. Classifique a natureza dos pontos.
*   **Ponto $1$**: Vizinhos na distância $\le 1.5$: $N_1 = \{1, 2\}$ (tamanho 2 $\ge$ MinPts) $\implies$ **Ponto de Núcleo (Core)**.
*   **Ponto $2$**: Vizinhos: $N_2 = \{1, 2, 3\}$ (tamanho 3 $\ge$ MinPts) $\implies$ **Ponto de Núcleo (Core)**.
*   **Ponto $3$**: Vizinhos: $N_3 = \{2, 3\}$ (tamanho 2 $\ge$ MinPts) $\implies$ **Ponto de Núcleo (Core)**.
*   **Ponto $10$**: Vizinhos: $N_{10} = \{10\}$ (tamanho 1 $<$ MinPts) $\implies$ **Ruído (Noise / Outlier)**.

---

### 2.3 Agrupamento Espectral (Spectral Clustering) na Mão
**Exemplo Prático**: Seja um grafo com 3 nós em linha ($1 - 2 - 3$) com pesos de arestas unitários ($W_{12}=1, W_{23}=1, W_{13}=0$).
*   **Matriz de Adjacência $W$**:
    $$W = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
*   **Matriz de Grau $D$**:
    $$D = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
*   **Matriz Laplaciana $L = D - W$**:
    $$L = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{bmatrix}$$
*   O menor autovalor é sempre $\lambda_1 = 0$, correspondendo ao autovetor trivial $\mathbf{v}_1 = [1, 1, 1]^T$.
*   O segundo menor autovalor $\lambda_2$ (Autovalor de Fiedler) fornece o **Vetor de Fiedler** $\mathbf{v}_2$, cujos sinais definem o corte de partição ideal do grafo em subgrupos desconectados por similaridade espectral.

---

### 2.4 Quando usar e quando não usar métodos de agrupamento

| Método | Situação para Usar | Situação para NÃO Usar |
| :--- | :--- | :--- |
| **K-Means** | Grupos esféricos, bem separados e de variância uniforme. | Formatos não-convexos/não-esféricos, tamanhos desproporcionais ou outliers presentes. |
| **DBSCAN** | Geometrias arbitrárias ou presença de ruído/anomalias espaciais claras. | Densidades muito variáveis nos dados (onde um único raio $\epsilon$ falha). |
| **Agrupamento Espectral** | Agrupamento em grafos de afinidade, variedades não-lineares intrincadas. | Problemas massivos com limitações de memória para decomposição de autovalores. |

---

## Questão 3: Redução de Dimensionalidade (PCA e SVD)

### 3.1 Equivalência Teórica entre PCA e SVD
Dada a matriz centralizada $\mathbf{X}_c \in \mathbb{R}^{N \times D}$.
A matriz de covariância amostral é $\mathbf{S} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c$.

Fazendo a SVD de $\mathbf{X}_c = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$, substituímos na covariância:
$$\mathbf{S} = \frac{1}{N - 1} (\mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T)^T (\mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T) = \frac{1}{N-1} \mathbf{V} \boldsymbol{\Sigma}^T \mathbf{U}^T \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
Como os vetores singulares à esquerda são ortonormais ($\mathbf{U}^T \mathbf{U} = \mathbf{I}$):
$$\mathbf{S} = \mathbf{V} \left( \frac{\boldsymbol{\Sigma}^2}{N - 1} \right) \mathbf{V}^T$$
Esta expressão é exatamente a decomposição espectral da matriz de covariância $\mathbf{S} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T$. Logo:
1.  Os autovetores de $\mathbf{S}$ são as colunas de $\mathbf{V}$ (vetores singulares à direita de $\mathbf{X}_c$).
2.  A relação exata entre autovalores e valores singulares é:
    $$\lambda_i = \frac{\sigma_i^2}{N - 1}$$

---

### 3.2 Projeção PCA na Mão
**Exemplo Prático**: Dada a matriz de covariância amostral calculada:
$$\mathbf{S} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$
Ache os autovalores e o autovetor principal para projetar os dados em 1D na mão.

#### Passo 1: Equação Característica
$$\det(\mathbf{S} - \lambda \mathbf{I}) = \det\left(\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}\right) = 0$$
$$(2-\lambda)^2 - 1 = 0 \implies \lambda^2 - 4\lambda + 3 = 0 \implies (\lambda-3)(\lambda-1) = 0$$
Os autovalores são $\lambda_1 = 3$ e $\lambda_2 = 1$. O maior autovalor (variância explicada principal) é $\lambda_1 = 3$.

#### Passo 2: Encontrar o Autovetor Principal para $\lambda_1 = 3$
$$(\mathbf{S} - 3\mathbf{I})\mathbf{u} = \mathbf{0} \implies \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -u_1 + u_2 = 0 \implies u_1 = u_2$$
Normalizando o vetor para ter norma unitária ($\|\mathbf{u}\| = 1$):
$$\mathbf{u}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
*   **Uso**: Qualquer ponto de dados centralizado $\mathbf{x} = [x, y]^T$ é projetado no subespaço de 1D por:
    $$z = \mathbf{u}_1^T \mathbf{x} = \frac{x + y}{\sqrt{2}}$$

---

### 3.3 Quando usar e quando não usar PCA e SVD

*   **Quando Usar**: Para reduzir multicolinearidade em regressões, comprimir imagens/matrizes ou visualizar dados multidimensionais em gráficos bidimensionais.
*   **Quando NÃO Usar**: Quando a estrutura intrínseca dos dados reside em uma variedade não-linear complexa (nesse caso, manifold learning ou autoencoders são mais indicados) ou quando a interpretabilidade direta das variáveis originais deve ser mantida.
