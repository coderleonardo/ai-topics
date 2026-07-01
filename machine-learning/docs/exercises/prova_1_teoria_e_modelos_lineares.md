# 📝 Prova Corretiva 1: Teoria Estatística, Amostragem e Modelos Lineares (Feito à Mão)

Este guia prático foi desenhado no formato de uma prova resolvida passo a passo "na mão", servindo como material de estudo para fixar os cálculos aritméticos, conceitos e decisões de modelagem.

---

## Questão 1: Distribuições, Máxima Verossimilhança e Métodos de Amostragem

### 1.1 Demonstração do Estimador de Variância (MLE)
Seja $\{x_1, \dots, x_N\}$ uma amostra i.i.d. de $x_n \sim \mathcal{N}(\mu, \sigma^2)$.

A função de log-verossimilhança é:
$$\ln L(\mu, \sigma^2) = -\frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2$$

Para achar o estimador MLE da variância $\sigma^2_{\text{ML}}$, definimos $\theta = \sigma^2$, derivamos em relação a $\theta$ e igualamos a zero:
$$\frac{\partial \ln L}{\partial \theta} = -\frac{N}{2\theta} + \frac{1}{2\theta^2} \sum_{n=1}^N (x_n - \mu)^2 = 0 \implies \sigma^2_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N (x_n - \mu)^2$$

**Prova do Viés**: A esperança matemática da soma quadrática média é:
$$\mathbb{E}[\sigma^2_{\text{ML}}] = \frac{N-1}{N}\sigma^2$$
Como $\mathbb{E}[\sigma^2_{\text{ML}}] \neq \sigma^2$, o estimador é **enviesado** para baixo. A correção de Bessel divide por $N-1$ para neutralizar o viés.

---

### 1.2 Métodos de Amostragem e Validação (Holdout, K-Fold e Bootstrap)

#### A) Divisão de Partições de K-Fold na Mão
**Exemplo Prático**: Dado o dataset $\mathcal{D} = \{A, B, C, D, E, F\}$ ($N=6$) e $K=3$ folds. Como são geradas as partições?
*   Tamanho de cada fold: $N / K = 6 / 3 = 2$ amostras por fold.
*   **Fold 1**: Teste = $\{A, B\}$; Treino = $\{C, D, E, F\}$
*   **Fold 2**: Teste = $\{C, D\}$; Treino = $\{A, B, E, F\}$
*   **Fold 3**: Teste = $\{E, F\}$; Treino = $\{A, B, C, D\}$

#### B) Probabilidade Out-of-Bag (OOB) no Bootstrap na Mão
**Exemplo Prático**: No Bootstrap, amostramos $N$ elementos com reposição. Para $N=3$ elementos ($A, B, C$), qual é a probabilidade exata de o elemento $A$ ficar fora da amostra (OOB)?
*   A probabilidade de $A$ não ser escolhido em um único sorteio é de $2/3$.
*   Fazendo $N=3$ sorteios independentes com reposição:
    $$P(A \text{ é OOB}) = \left(\frac{2}{3}\right)^3 = \frac{8}{27} \approx 0.296 \quad (29.6\%)$$
*   *Nota*: Para $N \to \infty$, essa probabilidade converge para $1/e \approx 36.8\%$.

---

### 1.3 Quando usar e quando não usar os métodos de amostragem

| Método | Situação para Usar | Situação para NÃO Usar |
| :--- | :--- | :--- |
| **Holdout (Treino/Teste)** | Datasets muito grandes (onde rodar K-Fold é inviável computacionalmente). | Datasets muito pequenos (o teste fica com alta variância e o treino perde dados valiosos). |
| **K-Fold Cross-Validation** | Avaliação robusta de hiperparâmetros em dados de tamanho médio. | Séries temporais ou dados sequenciais (onde embaralhar quebra a dependência temporal). |
| **Bootstrap (OOB)** | Datasets pequenos e modelos de ensemble (como Random Forests) para estimar incertezas. | Quando o custo computacional de retreinar o modelo centenas de vezes é proibitivo. |

---

## Questão 2: Regressão Linear Simples, ANOVA e Teste t de Hipóteses na Mão

### 2.1 Cálculos Passo a Passo com Pequenos Números
Dada a amostra com $N=4$ pontos:
$$\mathcal{D} = \{(1, 2), (2, 4), (3, 5), (4, 7)\}$$

#### Passo 1: Médias
$$\bar{x} = \frac{1+2+3+4}{4} = 2.5, \quad \bar{y} = \frac{2+4+5+7}{4} = 4.5$$

#### Passo 2: $S_{xx}$ e $S_{xy}$
*   $S_{xx} = \sum (x_i - \bar{x})^2 = (-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0$
*   $S_{xy} = \sum (x_i - \bar{x})(y_i - \bar{y}) = (-1.5)(-2.5) + (-0.5)(-0.5) + (0.5)(0.5) + (1.5)(2.5) = 3.75 + 0.25 + 0.25 + 3.75 = 8.0$

#### Passo 3: Parâmetros $\beta_1$ e $\beta_0$
*   $\beta_1 = \frac{S_{xy}}{S_{xx}} = \frac{8.0}{5.0} = 1.6$
*   $\beta_0 = \bar{y} - \beta_1\bar{x} = 4.5 - 1.6(2.5) = 0.5$
*   *Reta Estimada*: $\hat{y} = 0.5 + 1.6x$

#### Passo 4: Valores Ajustados e Resíduos
*   $\hat{y} = [2.1, 3.7, 5.3, 6.9]$
*   $e = y - \hat{y} = [2-2.1, 4-3.7, 5-5.3, 7-6.9] = [-0.1, 0.3, -0.3, 0.1]$

#### Passo 5: ANOVA e $R^2$
*   **SQE** (Soma dos Quadrados dos Erros): $\sum e_i^2 = (-0.1)^2 + 0.3^2 + (-0.3)^2 + 0.1^2 = 0.01 + 0.09 + 0.09 + 0.01 = 0.20$
*   **STQT** (Soma dos Quadrados Totais): $\sum (y_i - \bar{y})^2 = (-2.5)^2 + (-0.5)^2 + 0.5^2 + 2.5^2 = 6.25 + 0.25 + 0.25 + 6.25 = 13.00$
*   **SQR** (Soma dos Quadrados da Regressão): $\text{STQT} - \text{SQE} = 13.00 - 0.20 = 12.80$
*   **$R^2$**: $\frac{\text{SQR}}{\text{STQT}} = \frac{12.80}{13.00} \approx 0.9846 \quad (98.46\%)$

#### Passo 6: Variância e Erro Padrão
*   $\hat{\sigma}^2 = \frac{\text{SQE}}{N - 2} = \frac{0.20}{2} = 0.10$
*   $\text{SE}(\beta_1) = \sqrt{\frac{\hat{\sigma}^2}{S_{xx}}} = \sqrt{\frac{0.10}{5.0}} = \sqrt{0.02} \approx 0.1414$

#### Passo 7: Estatística $t$ de Utilidade
*   $t = \frac{\beta_1}{\text{SE}(\beta_1)} = \frac{1.6}{0.1414} \approx 11.31$
*   Como $|t| = 11.31 > t_{\text{crit}} = 4.303$ (para 2 graus de liberdade a $\alpha=0.05$), rejeitamos $H_0$. O modelo é estatisticamente útil.

---

### 2.2 Quando usar e quando não usar a Regressão Linear

*   **Quando Usar**: Quando a relação entre a variável resposta e os preditores é aproximadamente linear e deseja-se interpretar a magnitude do impacto de cada feature (inferência estatística).
*   **Quando NÃO Usar**: Quando os resíduos apresentam comportamento heterocedástico (variância não constante) ou não-linearidade acentuada (ex: crescimento exponencial).

---

## Questão 3: Regularização (Ridge) e KNN na Mão

### 3.1 Regressão Ridge 1D na Mão
Sem intercepto, o coeficiente Ridge $\beta_{\text{Ridge}}$ para dados unidimensionais com penalidade $\lambda$ resolve a minimização de:
$$E(w) = \frac{1}{2} \sum_{i=1}^N (y_i - w x_i)^2 + \frac{\lambda}{2} w^2 \implies w_{\text{Ridge}} = \frac{\sum_{i=1}^N x_i y_i}{\sum_{i=1}^N x_i^2 + \lambda}$$

**Exemplo Prático**: Dados os pontos $(1, 2)$ e $(2, 3)$. Calcule $w_{\text{Ridge}}$ com $\lambda = 1.0$:
*   $\sum x_i y_i = (1 \times 2) + (2 \times 3) = 2 + 6 = 8$
*   $\sum x_i^2 = 1^2 + 2^2 = 5$
*   $$w_{\text{Ridge}} = \frac{8}{5 + 1.0} = \frac{8}{6} \approx 1.33$$
*   *Nota (Comparação OLS)*: Sem regularização ($\lambda=0$), teríamos $w_{\text{OLS}} = 8/5 = 1.60$. A penalidade Ridge encolheu o coeficiente para $1.33$.

---

### 3.2 Vizinhos Mais Próximos (KNN) na Mão
**Exemplo Prático**: Dado o conjunto de treino abaixo:
*   $\mathbf{x}_1 = (1, 1), y_1 = 0$
*   $\mathbf{x}_2 = (1, 2), y_2 = 0$
*   $\mathbf{x}_3 = (3, 3), y_3 = 1$
*   $\mathbf{x}_4 = (4, 3), y_4 = 1$

Classifique o ponto de consulta $\mathbf{x}^* = (2, 2)$ usando $K=3$ vizinhos com a **Distância Euclidiana ($L_2$)**.
*   $d_2(\mathbf{x}^*, \mathbf{x}_1) = \sqrt{(2-1)^2 + (2-1)^2} = \sqrt{2} \approx 1.414$ (Classe 0)
*   $d_2(\mathbf{x}^*, \mathbf{x}_2) = \sqrt{(2-1)^2 + (2-2)^2} = 1.000$ (Classe 0)
*   $d_2(\mathbf{x}^*, \mathbf{x}_3) = \sqrt{(2-3)^2 + (2-3)^2} = \sqrt{2} \approx 1.414$ (Classe 1)
*   $d_2(\mathbf{x}^*, \mathbf{x}_4) = \sqrt{(2-4)^2 + (2-3)^2} = \sqrt{5} \approx 2.236$ (Classe 1)

Os 3 vizinhos mais próximos são $\mathbf{x}_2$ ($d=1.0$), $\mathbf{x}_1$ ($d\approx 1.414$) e $\mathbf{x}_3$ ($d\approx 1.414$).
*   Votos: Classe 0 (2 votos), Classe 1 (1 voto).
*   **Resultado**: Classe predita para $\mathbf{x}^*$ é **0**.

---

### 3.3 Quando usar e quando não usar Ridge, Lasso e KNN

| Algoritmo | Situação para Usar | Situação para NÃO Usar |
| :--- | :--- | :--- |
| **Ridge (L2)** | Multicolinearidade acentuada (atributos altamente correlacionados) e muitas features. | Quando é fundamental selecionar atributos e zerar coeficientes irrelevantes. |
| **Lasso (L1)** | Seleção automática de atributos (esparsidade) em problemas com muitas variáveis irrelevantes. | Quando temos variáveis altamente correlacionadas (Lasso tende a escolher uma aleatória e descartar o resto). |
| **KNN** | Padrões de decisão não-lineares locais em baixa dimensionalidade. | Alta dimensionalidade (Mal da Dimensionalidade) ou datasets massivos (lento na predição). |
