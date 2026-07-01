# 📉 Técnicas Lineares de Redução de Dimensionalidade (PCA e SVD)

Este documento detalha as formulações matemáticas para Análise de Componentes Principais (PCA) e Decomposição em Valores Singulares (SVD) implementadas em `src/dimensionality_reduction.py`.

---

## 1. Análise de Componentes Principais (PCA)
O PCA busca projetar os dados em um subespaço linear de menor dimensão, maximizando a variância projetada ou, equivalentemente, minimizando a soma das distâncias quadráticas de projeção.

### Formulação Matemática
Seja $\mathbf{X} \in \mathbb{R}^{N \times D}$ a matriz de dados originais.
1.  **Centralização dos Dados**:
    Subtraímos a média amostral de cada atributo:
    $$\mathbf{\mu}_x = \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \in \mathbb{R}^D$$
    $$\mathbf{X}_c = \mathbf{X} - \mathbf{1}\mathbf{\mu}_x^T$$
2.  **Cálculo da Matriz de Covariância Amostral ($S$)**:
    $$\mathbf{S} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{D \times D}$$
3.  **Decomposição Espectral (Eigenvalue Decomposition)**:
    Determinamos os autovalores $\lambda_i$ e autovetores correspondentes $\mathbf{u}_i$ da matriz de covariância simétrica $\mathbf{S}$:
    $$\mathbf{S} \mathbf{u}_i = \lambda_i \mathbf{u}_i, \quad i = 1, \dots, D$$
    Ordenamos os autovetores de forma decrescente com relação aos autovalores: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$.
4.  **Projeção Linear**:
    Selecionamos os $M < D$ primeiros autovetores para formar a matriz de projeção $\mathbf{U}_M = [\mathbf{u}_1, \dots, \mathbf{u}_M] \in \mathbb{R}^{D \times M}$. A representação reduzida $\mathbf{Y}$ é dada por:
    $$\mathbf{Y} = \mathbf{X}_c \mathbf{U}_M \in \mathbb{R}^{N \times M}$$
5.  **Variância Explicada**:
    A razão de variância explicada pela componente principal $i$ é:
    $$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^D \lambda_j}$$

---

## 2. Decomposição em Valores Singulares (SVD)
A decomposição SVD generaliza a diagonalização espectral para matrizes não retangulares. Qualquer matriz real $\mathbf{X} \in \mathbb{R}^{N \times D}$ pode ser decomposta como:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Onde:
*   $\mathbf{U} \in \mathbb{R}^{N \times N}$ é uma matriz ortogonal contendo os vetores singulares à esquerda de $\mathbf{X}$ (autovetores de $\mathbf{X}\mathbf{X}^T$).
*   $\mathbf{V} \in \mathbb{R}^{D \times D}$ é uma matriz ortogonal contendo os vetores singulares à direita de $\mathbf{X}$ (autovetores de $\mathbf{X}^T\mathbf{X}$).
*   $\mathbf{\Sigma} \in \mathbb{R}^{N \times D}$ é uma matriz diagonal retangular com os valores singulares não-negativos ordenados: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(N, D)} \ge 0$, onde $\sigma_i = \sqrt{\lambda_i}$.

### SVD Estável Truncada (svd_estavel_truncada)
Para fins computacionais de estabilidade quando $N \gg D$, realizamos a decomposição espectral de $\mathbf{X}^T\mathbf{X}$:

$$\mathbf{X}^T \mathbf{X} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

1.  Os autovetores $\mathbf{v}_i$ formam a matriz de rotação $\mathbf{V}_k \in \mathbb{R}^{D \times k}$ para os $k$ maiores valores singulares.
2.  A matriz de valores singulares é dada por $\mathbf{\Sigma}_k = \text{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_k}) \in \mathbb{R}^{k \times k}$.
3.  Os vetores singulares à esquerda correspondentes são calculados de forma estável por:
    $$\mathbf{U}_k = \mathbf{X} \mathbf{V}_k \mathbf{\Sigma}_k^{-1} \in \mathbb{R}^{N \times k}$$
4.  A reconstrução de classificação truncada da matriz $\mathbf{X}$ é:
    $$\hat{\mathbf{X}}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
    Esta reconstrução $\hat{\mathbf{X}}_k$ é a melhor aproximação de posto inferior $k$ da matriz original em termos de norma de Frobenius (Teorema de Eckart-Young-Mirsky).
