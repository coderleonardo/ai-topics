# 📏 Métricas de Desempenho e Avaliação de Modelos

Este documento detalha as formulações matemáticas das métricas de avaliação de modelos implementadas em `src/metrics.py`.

---

## 1. Métricas de Classificação Binária
Seja $y$ o rótulo real da classe e $\hat{y}$ o rótulo predito, assumindo valores no conjunto $\{0, 1\}$ (ou $\{-1, +1\}$).

### Matriz de Confusão
Estrutura tabular que quantifica as previsões corretas e incorretas:
*   **Verdadeiros Positivos ($TP$)**: $y_n = 1 \land \hat{y}_n = 1$
*   **Falsos Positivos ($FP$)**: $y_n = 0 \land \hat{y}_n = 1$
*   **Falsos Negativos ($FN$)**: $y_n = 1 \land \hat{y}_n = 0$
*   **Verdadeiros Negativos ($TN$)**: $y_n = 0 \land \hat{y}_n = 0$

### Acurácia (Accuracy)
Mede a proporção total de classificações corretas:

$$\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1}{N} \sum_{n=1}^N \mathbb{I}(y_n = \hat{y}_n)$$

### Precisão (Precision)
Mede a proporção de previsões positivas que são de fato verdadeiras. Crucial para cenários onde falsos positivos são caros (ex: filtros de spam):

$$\text{Precisão} = \frac{TP}{TP + FP}$$

### Revocação / Sensibilidade (Recall)
Mede a proporção de exemplos positivos reais que foram identificados pelo modelo. Crucial para cenários onde falsos negativos são perigosos (ex: diagnósticos médicos):

$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score
Média harmônica entre Precisão e Revocação, fornecendo uma métrica robusta de balanço entre as duas, especialmente sob desbalanceamento de classes:

$$F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

---

## 2. Métricas de Regressão
Sejam $\mathbf{y} \in \mathbb{R}^N$ os valores reais e $\mathbf{\hat{y}} \in \mathbb{R}^N$ os valores previstos pelo modelo.

### Erro Quadrático Médio (Mean Squared Error - MSE)
Penaliza erros maiores de forma mais acentuada devido ao termo quadrático:

$$\text{MSE} = \frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2$$

### Erro Absoluto Médio (Mean Absolute Error - MAE)
Representa a magnitude média do erro residual absoluto sem penalizar desproporcionalmente os outliers:

$$\text{MAE} = \frac{1}{N} \sum_{n=1}^N |y_n - \hat{y}_n|$$

### Coeficiente de Determinação ($R^2$)
Proporção da variância dos dados reais explicada pelas previsões do modelo:

$$R^2 = 1 - \frac{\text{SQE}}{\text{STQT}} = 1 - \frac{\sum_{n=1}^N (y_n - \hat{y}_n)^2}{\sum_{n=1}^N (y_n - \bar{y})^2}$$

Onde $\bar{y} = \frac{1}{N} \sum_{n=1}^N y_n$ é a média global observada dos alvos reais.
*   $R^2 = 1$: Ajuste perfeito.
*   $R^2 = 0$: O modelo prevê a média de $y$ para todos os exemplos.
*   $R^2 < 0$: O desempenho do modelo é pior do que prever a média constante.
