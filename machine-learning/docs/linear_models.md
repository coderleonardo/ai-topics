# 📈 Modelos Lineares e Métodos de Regularização

Este documento apresenta a formulação matemática dos modelos lineares, regularização L1/L2 e testes estatísticos de qualidade implementados em `src/linear_models.py`.

---

## 1. Mínimos Quadrados Ordinários (OLS) via Equação Normal
O problema de Mínimos Quadrados Ordinários busca determinar o vetor de parâmetros $\mathbf{w}$ que minimiza a soma dos erros quadráticos residuais:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2$$

Na formulação matricial, com a matriz de design $\mathbf{X} \in \mathbb{R}^{N \times D}$ e o vetor de alvos $\mathbf{y} \in \mathbb{R}^N$:

$$E(\mathbf{w}) = \frac{1}{2} (\mathbf{X}\mathbf{w} - \mathbf{y})^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$

Minimizando em relação a $\mathbf{w}$ (derivando e igualando a zero):

$$\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{X}^T \mathbf{X}\mathbf{w} - \mathbf{X}^T \mathbf{y} = \mathbf{0}$$

Obtemos a **Equação Normal**:

$$\mathbf{w}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

## 2. Regressão Ridge (Regularização L2)
A Regressão Ridge impõe uma penalidade quadrática no vetor de pesos (regularização de Tikhonov) para reduzir a variância e evitar sobreajuste:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2$$

Diferenciando com relação a $\mathbf{w}$, obtemos a solução analítica:

$$\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

Onde $\mathbf{I}$ é a matriz identidade de dimensão $D \times D$.

---

## 3. Regressão Lasso (Regularização L1)
A Regressão Lasso impõe uma penalidade L1 nos coeficientes, promovendo esparsidade (alguns pesos tornam-se exatamente zero):

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2 + \alpha \sum_{j=1}^D |w_j|$$

Como a penalidade L1 não é diferenciável na origem, utilizamos o algoritmo de **Descida de Coordenadas (Coordinate Descent)**. Atualizamos sequencialmente cada coordenada $j$:

$$w_j \leftarrow \frac{S(\rho_j, \alpha)}{\sum_{i=1}^N x_{ij}^2}$$

Onde:
*   $\rho_j = \sum_{i=1}^N x_{ij} \left( y_i - \sum_{k \neq j} w_k x_{ik} \right)$ é o resíduo sem a contribuição do atributo $j$.
*   $S(z, \lambda)$ é o operador de limiarização suave (**Soft-Thresholding**):
    $$S(z, \lambda) = \text{sign}(z) \max(0, |z| - \lambda)$$

---

## 4. Diagnósticos de Regressão Linear Simples
Para o modelo de regressão simples $y = \beta_0 + \beta_1 x + \epsilon$ com erros $\epsilon \sim \mathcal{N}(0, \sigma^2)$:

### Estimativa da Variância Residual ($\hat{\sigma}^2$)
A estimativa sem viés da variância do termo de erro com $N-2$ graus de liberdade é dada por:

$$\hat{\sigma}^2 = \frac{\text{SQE}}{N - 2} = \frac{1}{N - 2} \sum_{n=1}^N (y_n - \hat{y}_n)^2$$

### Teste de Utilidade de Regressão ($H_0: \beta_1 = 0$)
Deseja-se testar se a variável independente $x$ é estatisticamente significante.
*   Erro padrão de $\hat{\beta}_1$:
    $$\text{SE}(\hat{\beta}_1) = \sqrt{\frac{\hat{\sigma}^2}{S_{xx}}}$$
    Onde $S_{xx} = \sum_{n=1}^N (x_n - \bar{x})^2$.
*   Estatística de teste $t$:
    $$t = \frac{\hat{\beta}_1}{\text{SE}(\hat{\beta}_1)}$$
*   Sob $H_0$, a estatística $t$ segue a distribuição t de Student com $N-2$ graus de liberdade. O valor-p bicaudal é:
    $$\text{p-valor} = 2 \cdot P(T_{N-2} > |t|)$$

### Análise de Variância (ANOVA) e R²
A variabilidade total dos dados é dividida em variabilidade explicada pela regressão e variabilidade residual:

$$\text{STQT} = \text{SQR} + \text{SQE}$$

*   **Soma dos Quadrados Totais (STQT)**: $\sum_{n=1}^N (y_n - \bar{y})^2$
*   **Soma dos Quadrados devido à Regressão (SQR)**: $\sum_{n=1}^N (\hat{y}_n - \bar{y})^2$
*   **Soma dos Quadrados dos Erros (SQE)**: $\sum_{n=1}^N (y_n - \hat{y}_n)^2$
*   **Coeficiente de Determinação ($R^2$)**:
    $$R^2 = \frac{\text{SQR}}{\text{STQT}} = 1 - \frac{\text{SQE}}{\text{STQT}}$$
